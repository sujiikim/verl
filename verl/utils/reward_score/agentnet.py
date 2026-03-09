import re
import math
import json
from tqdm import tqdm


def format_reward(predict_str: str) -> float:
    """Return 1.0 if predict_str contains exactly one <tool_call>...</tool_call> block, else 0.0."""
    pattern = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
    matches = pattern.findall(predict_str)
    reward = 1.0 if len(matches) >= 1 else 0.0
    return reward, matches


def type_reward(answer: str, ground_truth: str) -> float:
    """Return 1.0 if answer contains the same text as ground_truth, else 0.0."""
    return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0


def key_reward(answer: list[str], ground_truth: list[str]) -> float:
    """Return 1.0 if answer contains the same keys as ground_truth, else 0.0."""
    ### TODO 순서 맞춰서 비교 해야 함
    answer = [key.strip().lower() for key in answer]
    ground_truth = [key.strip().lower() for key in ground_truth]
    rewards = []
    for key in ground_truth:
        if key in answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return sum(rewards) / len(rewards)


def coordinate_reward(answer: list[tuple[int, int]], ground_truth: list[tuple[int, int]], alpha: float = 0.01) -> float:
    # TODO threshold 추가 필요
    # length mismatch penalty
    if len(answer) != len(ground_truth):
        return 0.0

    if len(answer) == 0:
        return 0.0

    rewards = []

    for (x_pred, y_pred), (x_gt, y_gt) in zip(answer, ground_truth):
        dist = math.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
        point_reward = math.exp(-alpha * dist)
        rewards.append(point_reward)

    # average reward
    return sum(rewards) / len(rewards)


def pixels_reward(answer: int, ground_truth: int, alpha: float = 0.05) -> float:
    d = abs(answer - ground_truth)
    return math.exp(-alpha * d)


def parse_tool_call(tool_call: str) -> dict:
    """Parse the tool call string into a dictionary."""
    return json.loads(tool_call.replace("<tool_call>", "").replace("</tool_call>", "").strip())


def compute_score(predict_str: str, ground_truth: str) -> float:
    format_score, tool_calls = format_reward(predict_str)
    _, gt_tool_calls = format_reward(ground_truth)
    try:
        parsed_tool_calls = [parse_tool_call(tool_call)['arguments'] for tool_call in tool_calls]
    except Exception as e:
        return 0.0
    parsed_gt_tool_calls = [parse_tool_call(tool_call)['arguments'] for tool_call in gt_tool_calls]
    
    scores = []
    for t_call, gt_call in zip(parsed_tool_calls, parsed_gt_tool_calls):
        if t_call['action'] != gt_call['action']:
            scores.append(0.0)
            continue
        else:
            action_score = 1.0
        if t_call['action'] == 'type':
            arguments_score = type_reward(t_call['text'], gt_call['text'])
        elif t_call['action'] == 'key':
            arguments_score = key_reward(t_call['keys'], gt_call['keys'])
        elif 'arguments' in t_call and 'start_coordinate' in t_call['arguments']:
            arguments_score = coordinate_reward([t_call['arguments']['start_coordinate'], t_call['arguments']['coordinate']], [gt_call['arguments']['start_coordinate'], gt_call['arguments']['coordinate']])
        elif 'arguments' in t_call and 'coordinate' in t_call['arguments']:
            arguments_score = coordinate_reward([t_call['arguments']['coordinate']], [gt_call['arguments']['coordinate']])
        elif 'arguments' in t_call and 'pixels' in t_call['arguments']:
            arguments_score = pixels_reward(t_call['arguments']['pixels'], gt_call['arguments']['pixels'])
        else:
            arguments_score = 1.0
        scores.append(0.5 * action_score + 0.5 * arguments_score)
    if len(scores) == 0:
        call_score = 0.0
    else:
        call_score = sum(scores) / len(scores)
    return 0.9 * call_score + 0.1 * format_score
#!/usr/bin/env python3
"""
6_generate_dataset_evocua-chat.py

필터링된 JSONL 파일에서 특정 도메인의 데이터를 추출하여
LlamaFactory 학습용 JSON (sharegpt chat) 포맷으로 변환합니다.

각 traj step마다 하나의 샘플을 생성합니다.
S2/evocua 스타일의 system prompt + tool_call 포맷을 사용합니다.

History 구성:
  - 최근 history_len 개의 step은 multi-turn (이미지 + assistant output)으로 포함
  - 그 이전 step들은 첫 user 메시지에 텍스트 "Step {idx}: {action}" 으로 포함

출력 포맷:
  [
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "<image>\\n\\n...instruction + prev actions..."},
        {"role": "assistant", "content": "...</think>\n\nAction: ...\\n<tool_call>\\n{...}\\n</tool_call>"},
        {"role": "user", "content": "<image>"},
        {"role": "assistant", "content": "...</think>\n\nAction: ...\\n<tool_call>\\n{...}\\n</tool_call>"},
        ...
      ],
      "images": ["절대경로/image1.png", "절대경로/image2.png", ...]
    },
    ...
  ]
"""

import json
import os
import re
import sys
sys.path.append("/c1/suji/workspace/agent/trajectory_distill/LlamaFactory/data_src/")
from evocua_prompt import (
    S2_SYSTEM_PROMPT,
    S2_DESCRIPTION_PROMPT_TEMPLATE,
    build_s2_tools_def,
)

# ──────────────────────────────────────────────
# System Prompt 생성
# ──────────────────────────────────────────────
resolution_info = "* The screen's resolution is 1000x1000."
description_prompt = S2_DESCRIPTION_PROMPT_TEMPLATE.format(resolution_info=resolution_info)
tools_def = build_s2_tools_def(description_prompt)
SYSTEM_PROMPT = S2_SYSTEM_PROMPT.format(tools_xml=json.dumps(tools_def))

# ──────────────────────────────────────────────
# User prompt template
# ──────────────────────────────────────────────
USER_FIRST_TEMPLATE = """Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions_str}"""

# ──────────────────────────────────────────────
# Code → tool_call 변환
# (evocua_agent._parse_response_s2 와 역호환)
# ──────────────────────────────────────────────
RESOLUTION = 1000  # 1000x1000
MAX_COORD = RESOLUTION - 1  # 0..999


def _coord(x: float, y: float) -> list:
    """Normalized (0-1) 좌표를 pixel 좌표로 변환 (0..999 범위 클램핑)"""
    px = max(0, min(MAX_COORD, round(x * RESOLUTION)))
    py = max(0, min(MAX_COORD, round(y * RESOLUTION)))
    return [px, py]


def _parse_coord_kwargs(s: str) -> tuple:
    """'x=0.02, y=0.503' 등에서 x, y 추출 (음수 좌표도 지원)"""
    mx = re.search(r'x\s*=\s*(-?[0-9.]+)', s)
    my = re.search(r'y\s*=\s*(-?[0-9.]+)', s)
    if mx and my:
        return float(mx.group(1)), float(my.group(1))
    return None, None


def _extract_write_text(line: str) -> str:
    """pyautogui.write(message='...') 에서 텍스트 추출 (따옴표 내 이스케이프 처리)"""
    # message= 이후 첫 따옴표부터 마지막 ) 직전 따옴표까지 추출
    m = re.search(r"message=(['\"])(.*)\1\s*\)", line, re.DOTALL)
    if m:
        return m.group(2)
    return ""


def _make_tool_call(args: dict) -> str:
    """tool_call JSON 문자열 생성"""
    return json.dumps({"name": "computer_use", "arguments": args})


def convert_code_to_tool_call(code: str) -> str:
    """
    pyautogui / computer 코드 → <tool_call> JSON 문자열 변환.
    evocua_agent._parse_response_s2 에서 역파싱 가능한 포맷을 생성합니다.
    멀티라인 코드의 경우 주요 액션을 추출하여 변환합니다.
    """
    code = code.strip()
    lines = [l.strip() for l in code.split('\n') if l.strip()]

    # ── 단일 라인 처리 ──
    if len(lines) == 1:
        return _convert_single_line(lines[0])

    # ── 멀티 라인 처리 ──
    # moveTo + dragTo  →  mouse_move + left_click_drag 를 하나의 left_click_drag로 변환
    # (agent 추론 시에는 mouse_move → left_click_drag 두 스텝으로 수행)
    if len(lines) >= 2 and 'moveTo' in lines[0] and 'dragTo' in lines[1]:
        x_start, y_start = _parse_coord_kwargs(lines[0])
        x_end, y_end = _parse_coord_kwargs(lines[1])
        if x_start is not None and x_end is not None:
            args = {
                "action": "left_click_drag",
                "start_coordinate": _coord(x_start, y_start),
                "coordinate": _coord(x_end, y_end),
            }
            return _make_tool_call(args)

    # dragTo + hotkey  →  left_click_drag (hotkey는 action text에서 설명)
    if len(lines) >= 2 and 'dragTo' in lines[0]:
        return _convert_single_line(lines[0])

    # 기타: 첫 줄만 변환
    print(f"  Warning: Could not convert code: {lines}")
    return _convert_single_line(lines[0])


def _convert_single_line(line: str) -> str:
    """단일 라인 코드를 tool_call JSON으로 변환"""

    # pyautogui.click(x=..., y=...)
    m = re.match(r'pyautogui\.click\((.+)\)', line)
    if m:
        x, y = _parse_coord_kwargs(m.group(1))
        return _make_tool_call({"action": "left_click", "coordinate": _coord(x, y)})

    # pyautogui.doubleClick(x=..., y=...)
    m = re.match(r'pyautogui\.doubleClick\((.+)\)', line)
    if m:
        x, y = _parse_coord_kwargs(m.group(1))
        return _make_tool_call({"action": "double_click", "coordinate": _coord(x, y)})

    # pyautogui.rightClick(x=..., y=...)
    m = re.match(r'pyautogui\.rightClick\((.+)\)', line)
    if m:
        x, y = _parse_coord_kwargs(m.group(1))
        return _make_tool_call({"action": "right_click", "coordinate": _coord(x, y)})

    # pyautogui.moveTo(x=..., y=...)  (standalone — 뒤에 dragTo 없는 경우)
    m = re.match(r'pyautogui\.moveTo\((.+)\)', line)
    if m:
        x, y = _parse_coord_kwargs(m.group(1))
        return _make_tool_call({"action": "mouse_move", "coordinate": _coord(x, y)})

    # pyautogui.dragTo(x=..., y=..., button='left')
    m = re.match(r'pyautogui\.dragTo\((.+)\)', line)
    if m:
        x, y = _parse_coord_kwargs(m.group(1))
        return _make_tool_call({"action": "left_click_drag", "coordinate": _coord(x, y)})

    # pyautogui.write(message='...')
    m = re.match(r'pyautogui\.write\(', line)
    if m:
        text = _extract_write_text(line)
        return _make_tool_call({"action": "type", "text": text})

    # pyautogui.press('key')
    m = re.match(r"pyautogui\.press\(['\"](.+?)['\"]\)", line)
    if m:
        return _make_tool_call({"action": "key", "keys": [m.group(1)]})

    # pyautogui.hotkey(['key1', 'key2', ...])
    m = re.match(r'pyautogui\.hotkey\(\[(.+?)\]\)', line)
    if m:
        keys_str = m.group(1)
        keys = [k.strip().strip("'\"") for k in keys_str.split(',')]
        return _make_tool_call({"action": "key", "keys": keys})

    # pyautogui.scroll(n)
    m = re.match(r'pyautogui\.scroll\(([^)]+)\)', line)
    if m:
        pixels = int(m.group(1))
        return _make_tool_call({"action": "scroll", "pixels": pixels})

    # computer.terminate(status='success'|'failure')
    m = re.match(r"computer\.terminate\(status=['\"](\w+)['\"]\)", line)
    if m:
        return _make_tool_call({"action": "terminate", "status": m.group(1)})

    # computer.wait()
    if line.strip() == 'computer.wait()':
        return _make_tool_call({"action": "wait"})

    # computer.tripleClick(x=..., y=...)
    m = re.match(r'computer\.tripleClick\((.+)\)', line)
    if m:
        x, y = _parse_coord_kwargs(m.group(1))
        return _make_tool_call({"action": "triple_click", "coordinate": _coord(x, y)})

    # Fallback: 변환 불가
    print(f"  Warning: Could not convert code: {line}")
    return _make_tool_call({"action": "unknown", "raw_code": line})


# ──────────────────────────────────────────────
# Assistant content 생성
# ──────────────────────────────────────────────
def build_assistant_content(summarized_thought: str, action_desc: str, code: str) -> str:
    """
    Assistant response 생성 (inference 포맷과 동일):
      {summarized_thought}
      </think>

      Action: {action description}
      <tool_call>
      {tool_call_json}
      </tool_call>
    """
    tool_call_json = convert_code_to_tool_call(code)
    return (
        f"{summarized_thought}\n"
        f"</think>\n\n"
        f"Action: {action_desc}\n"
        f"<tool_call>\n{tool_call_json}\n</tool_call>"
    )


# ──────────────────────────────────────────────
# History / Message 구성
# ──────────────────────────────────────────────
def build_sample_messages(
    instruction: str,
    traj: list,
    current_idx: int,
    history_len: int,
    image_base_dir: str,
) -> tuple:
    """
    하나의 training sample에 대한 messages와 images를 생성합니다.

    Returns:
        (messages: list[dict], images: list[str])
    """
    history_start = max(0, current_idx - history_len)

    # ── Text-only Previous Actions (history window 이전) ──
    text_actions = []
    for i in range(history_start):
        step_value = traj[i].get("value", {})
        action_desc = step_value.get("action", "")
        text_actions.append(f"Step {i + 1}: {action_desc}")
    previous_actions_str = "\n".join(text_actions) if text_actions else "None"

    # ── Messages 구성 시작 ──
    messages = []
    images = []

    # System message
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # Multi-turn history steps + current step
    # 범위: history_start ~ current_idx (inclusive)
    turn_steps = list(range(history_start, current_idx + 1))

    for turn_i, step_idx in enumerate(turn_steps):
        step = traj[step_idx]
        step_value = step.get("value", {})
        image_name = step.get("image", "")
        image_path = os.path.join(image_base_dir, image_name)
        action_desc = step_value.get("action", "")
        code = step_value.get("code", "")
        summarized_thought = step_value.get("summarized_thought", "")

        images.append(image_path)

        # User message
        if turn_i == 0:
            # 첫 번째 turn: instruction + previous actions 포함
            user_text = USER_FIRST_TEMPLATE.format(
                instruction=instruction,
                previous_actions_str=previous_actions_str,
            )
            messages.append({"role": "user", "content": f"<image>\n{user_text}"})
        else:
            # 이후 turn: 이미지만
            messages.append({"role": "user", "content": "<image>"})

        # Assistant message
        assistant_content = build_assistant_content(summarized_thought, action_desc, code)
        messages.append({"role": "assistant", "content": assistant_content})

    return messages, images


# ──────────────────────────────────────────────
# 메인 데이터셋 생성
# ──────────────────────────────────────────────
def generate_dataset(
    input_path: str,
    output_path: str,
    domain: str,
    history_len: int,
    image_base_dir: str,
):
    """메인 데이터셋 생성 함수"""

    print("=" * 80)
    print("Dataset Generation (evocua-chat format)")
    print("=" * 80)
    print(f"  Input:          {input_path}")
    print(f"  Output:         {output_path}")
    print(f"  Domain:         {domain}")
    print(f"  History len:    {history_len}")
    print(f"  Image base dir: {image_base_dir}")
    print()

    # 데이터 읽기
    data_items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data_items.append(data)
            except json.JSONDecodeError as e:
                print(f"  Warning: Error parsing line: {e}")

    print(f"  Total items loaded: {len(data_items)}")

    # 도메인 필터링
    if domain == "all":
        filtered_items = data_items
        print(f"  Domain is 'all': using all {len(filtered_items)} items (no filtering)")
    else:
        filtered_items = []
        for item in data_items:
            item_domain = item.get("domain", "")
            meta_domain = item.get("meta_data", {}).get("domains", "")
            if item_domain == domain or meta_domain == domain:
                filtered_items.append(item)
        print(f"  Items matching domain '{domain}': {len(filtered_items)}")

    if not filtered_items:
        print("  No items found for the specified domain. Exiting.")
        return

    # summarized_thought 유무 확인
    has_summarized = False
    for item in filtered_items:
        for step in item.get("traj", []):
            if step.get("value", {}).get("summarized_thought"):
                has_summarized = True
                break
        if has_summarized:
            break
    if has_summarized:
        print("  Using 'summarized_thought' field (available)")
    else:
        raise ValueError("'summarized_thought' not found")

    # 데이터셋 생성
    dataset = []
    skipped_no_image = 0
    convert_errors = 0

    for item in filtered_items:
        instruction = item.get("natural_language_task", "") or item.get("instruction", "")
        if not instruction:
            raise ValueError("instruction not found")
        traj = item.get("traj", [])

        for step_idx in range(len(traj)):
            step = traj[step_idx]
            value = step.get("value", {})
            image_name = step.get("image", "")

            if not image_name:
                skipped_no_image += 1
                continue

            # 이미지 절대 경로 확인
            image_path = os.path.join(image_base_dir, image_name)
            if not os.path.exists(image_path):
                print(f"  Warning: Image not found: {image_path}")
                continue

            # 메시지 구성
            messages, images = build_sample_messages(
                instruction=instruction,
                traj=traj,
                current_idx=step_idx,
                history_len=history_len,
                image_base_dir=image_base_dir,
            )

            # history 이미지 존재 확인
            all_images_exist = all(os.path.exists(img) for img in images)
            if not all_images_exist:
                missing = [img for img in images if not os.path.exists(img)]
                print(f"  Warning: Missing history images: {missing}")
                continue

            sample = {
                "messages": messages,
                "images": images,
            }
            dataset.append(sample)

    print(f"\n  Total samples generated: {len(dataset)}")
    if skipped_no_image:
        print(f"  Skipped steps (no image): {skipped_no_image}")
    if convert_errors:
        print(f"  Code conversion errors: {convert_errors}")

    # 저장
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {file_size:.1f} MB")
    print("=" * 80)


def main():
    # === 설정 ===
    suji_path = "/c1/suji/workspace/agent/trajectory_distill/LlamaFactory/data_src/"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # domain = "Chrome"
    # domainid = "chrome"
    domain = "VScode"
    domainid = "vscode"
    history_len = 1
    image_base_dir = "/d1/dataset/AgentNet/ubuntu_images/"

    input_filename = "agentnet_ubuntu_5k_with_metadata_filtered_summary-gpt-5-mini.jsonl"
    input_path = os.path.join(suji_path, input_filename)

    output_filename = f"evocua-chat_agentnet_ubuntu_filtered_{domainid}_hist{history_len}.json"
    output_path = os.path.join(base_dir, output_filename)

    generate_dataset(input_path, output_path, domain, history_len, image_base_dir)


if __name__ == "__main__":
    main()

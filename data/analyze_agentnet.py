import json
import random
random.seed(42)

ubuntu_data = []
with open("/d1/dataset/AgentNet/agentnet_ubuntu_5k.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        ubuntu_data.append(data)

print(ubuntu_data[0].keys()) # dict_keys(['task_id', 'instruction', 'task_completed', 'alignment_score', 'efficiency_score', 'task_difficulty', 'reason', 'natural_language_task', 'actual_task', 'traj', 'domain'])

win_mac_data = []
with open("/d1/dataset/AgentNet/agentnet_win_mac_18k.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        win_mac_data.append(data)

print(win_mac_data[0].keys()) # dict_keys(['task_id', 'instruction', 'task_completed', 'alignment_score', 'efficiency_score', 'task_difficulty', 'reason', 'natural_language_task', 'actual_task', 'traj'])

total_data = ubuntu_data #+ win_mac_data

meta_data = []
with open("/d1/dataset/AgentNet/meta_data_merged.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        meta_data.append(data)
vscode_meta_data = [data for data in meta_data if data["domains"] == "VScode"]

vscode_data = []
for meta in vscode_meta_data:
    matched = None
    meta_instruction = meta["instruction"]
    for data in total_data:
        if data["instruction"].lower() == meta["instruction"].lower():
            matched = data
            vscode_data.append(data)
            break
    assert matched is not None

print(len(vscode_data)) # 605
train_vscode_data = random.sample(vscode_data, int(len(vscode_data) * 0.9))
test_vscode_data = [data for data in vscode_data if data not in train_vscode_data]

print(train_vscode_data[0]['traj'])
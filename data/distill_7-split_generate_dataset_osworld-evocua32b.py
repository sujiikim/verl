#!/usr/bin/env python3
"""
Generate LlamaFactory ShareGPT-style dataset from OSWorld EvoCUA-32B rollouts.

- Uses ALL episodes (successful or not).
- Uses rollout traj.jsonl as the trajectory source (ignores any prebuilt input jsonl).
- Builds samples in the same "messages/images" format as `6_generate_dataset_evocua-chat.py`.

Output:
  data/evocua-chat_osworld-evocua32b_{domainid}_hist{history_len}.json

Also writes one example sample to:
  data/sample.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Fixed root (from your note)
ROOT32 = "/data-vol1/suji/Trajectory/EvoCUA-32B-20260105"
SPLITS_PATH = os.path.join(os.path.dirname(__file__), "osworld_splits.json")

DATA_DIR = "../dataset"

from evocua_prompt import (  # noqa: E402
    S2_DESCRIPTION_PROMPT_TEMPLATE,
    S2_SYSTEM_PROMPT,
    build_s2_tools_def,
)


# ──────────────────────────────────────────────
# System Prompt (same logic as 6_generate_dataset_evocua-chat.py)
# ──────────────────────────────────────────────
# resolution_info = "* The screen's resolution is 1000x1000."
# description_prompt = S2_DESCRIPTION_PROMPT_TEMPLATE.format(resolution_info=resolution_info)
description_prompt = S2_DESCRIPTION_PROMPT_TEMPLATE
tools_def = build_s2_tools_def(description_prompt)
SYSTEM_PROMPT = S2_SYSTEM_PROMPT.format(tools_xml=json.dumps(tools_def))


USER_FIRST_TEMPLATE = """Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions_str}"""


# Domain mapping: user passes domainid; we map to rollout folder name under ROOT32.
DOMAINID_TO_FOLDER = {
    "chrome": "chrome",
    "gimp": "gimp",
    "calc": "libreoffice_calc",
    "impress": "libreoffice_impress",
    "writer": "libreoffice_writer",
    "os": "os",
    "thunderbird": "thunderbird",
    "vlc": "vlc",
    "vscode": "vs_code",
    "multiapp": "multi_apps",
    "all": "all",
}

# Allow also passing folder-style names directly
FOLDER_TO_DOMAINID = {v: k for k, v in DOMAINID_TO_FOLDER.items() if v != "all"}


def _safe_read_text(path: str, max_chars: int = 500_000) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except FileNotFoundError:
        return None


def _safe_read_float(path: str) -> Optional[float]:
    txt = _safe_read_text(path, max_chars=2000)
    if not txt:
        return None
    s = txt.strip()
    try:
        return float(s)
    except ValueError:
        return None


def _is_success(result_txt_path: str) -> bool:
    v = _safe_read_float(result_txt_path)
    return (v is not None) and (v != 0.0)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_instruction_from_runtime_log(runtime_log_path: str) -> Optional[str]:
    """
    runtime.log contains repeated blocks like:
      Instruction:
      <instruction line>
    We'll take the first instruction we find.
    """
    txt = _safe_read_text(runtime_log_path)
    if not txt:
        return None
    # tolerate Windows newlines; capture the first non-empty line after "Instruction:"
    m = re.search(r"Instruction:\s*\r?\n([^\r\n]+)", txt)
    if not m:
        return None
    instr = m.group(1).strip()
    return instr or None


def _normalize_domainid(domain: str) -> str:
    d = domain.strip().lower()
    if d in DOMAINID_TO_FOLDER:
        return d
    if d in FOLDER_TO_DOMAINID:
        return FOLDER_TO_DOMAINID[d]
    raise ValueError(f"Unknown domain '{domain}'. Use one of: {', '.join(sorted(DOMAINID_TO_FOLDER.keys()))}")


def load_splits() -> Dict[str, List[Dict[str, List[str]]]]:
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Episode:
    domain_folder: str
    episode_id: str
    episode_dir: str


def _iter_episodes(root32: str, domain_folder: Optional[str], train_episodes: Optional[List[str]] = None) -> Iterable[Episode]:
    root32 = os.path.abspath(root32)
    for dirpath, _dirnames, filenames in os.walk(root32):
        if "traj.jsonl" not in filenames:
            continue
        rel_dir = os.path.relpath(dirpath, root32)
        parts = rel_dir.split(os.sep) if rel_dir else []
        dom = parts[0] if parts else "unknown"
        if domain_folder and dom != domain_folder:
            continue
        episode_id = parts[1] if len(parts) >= 2 else os.path.basename(dirpath)
        if train_episodes is not None and episode_id not in train_episodes:
            continue
        yield Episode(domain_folder=dom, episode_id=episode_id, episode_dir=dirpath)


def build_sample_messages_osworld(
    instruction: str,
    steps: List[Dict[str, Any]],
    current_idx: int,
    history_len: int,
    episode_dir: str,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    OSWorld traj.jsonl step schema (observed):
      - screenshot_file: "step_1_....png"
      - response: assistant content (already contains </think> + Action + <tool_call> JSON)
      - action: raw action/code (used for previous-actions text)
    """
    history_start = max(0, current_idx - history_len)

    text_actions: List[str] = []
    for i in range(history_start):
        act = steps[i].get("action", "")
        act = str(act).strip()
        text_actions.append(f"Step {i + 1}: {act}" if act else f"Step {i + 1}: (missing action)")
    previous_actions_str = "\n".join(text_actions) if text_actions else "None"

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    images: List[str] = []

    turn_steps = list(range(history_start, current_idx + 1))
    for turn_i, step_idx in enumerate(turn_steps):
        step = steps[step_idx]
        screenshot = step.get("screenshot_file", "")
        if not screenshot:
            raise ValueError(f"Missing screenshot_file at step_idx={step_idx}")
        img_path = os.path.join(episode_dir, screenshot)
        images.append(img_path)

        if turn_i == 0:
            user_text = USER_FIRST_TEMPLATE.format(instruction=instruction, previous_actions_str=previous_actions_str)
            messages.append({"role": "user", "content": f"<image>\n{user_text}"})
        else:
            messages.append({"role": "user", "content": "<image>"})

        assistant_content = step.get("response", "")
        if not assistant_content:
            raise ValueError(f"Missing response at step_idx={step_idx}")
        messages.append({"role": "assistant", "content": str(assistant_content)})

    return messages, images


def generate_dataset_for_split(domainid: str, splitid: int, train_episodes: List[str], history_len: int) -> Tuple[List[Dict[str, Any]], str]:
    domainid = _normalize_domainid(domainid)
    domain_folder = None if domainid == "all" else DOMAINID_TO_FOLDER[domainid]

    out_name = f"evocua-chat_osworld-evocua32b_{domainid}-split{splitid+1}_hist{history_len}.json"
    out_path = os.path.join(DATA_DIR, out_name)

    episodes = list(_iter_episodes(ROOT32, domain_folder=domain_folder, train_episodes=train_episodes))
    episodes.sort(key=lambda e: (e.domain_folder, e.episode_id))

    dataset: List[Dict[str, Any]] = []
    skipped_missing_img = 0
    skipped_no_instruction = 0

    for ep in episodes:
        traj_path = os.path.join(ep.episode_dir, "traj.jsonl")
        runtime_log_path = os.path.join(ep.episode_dir, "runtime.log")
        instruction = _extract_instruction_from_runtime_log(runtime_log_path)
        if not instruction:
            skipped_no_instruction += 1
            continue

        steps = list(_iter_jsonl(traj_path))
        if not steps:
            continue

        for step_idx in range(len(steps)):
            try:
                messages, images = build_sample_messages_osworld(
                    instruction=instruction,
                    steps=steps,
                    current_idx=step_idx,
                    history_len=history_len,
                    episode_dir=ep.episode_dir,
                )
            except ValueError:
                continue

            if not all(os.path.exists(p) for p in images):
                skipped_missing_img += 1
                continue

            dataset.append({"messages": messages, "images": images})

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Dataset Generation (OSWorld EvoCUA-32B -> evocua-chat format)")
    print("=" * 80)
    print(f"  ROOT32:        {ROOT32}")
    print(f"  Domain:        {domainid} (folder={domain_folder or 'ALL'})")
    print(f"  Split:         {splitid}")
    print(f"  History len:   {history_len}")
    print(f"  Episodes:      {len(episodes)}")
    print(f"  Samples made:  {len(dataset)}")
    if skipped_no_instruction:
        print(f"  Skipped eps (no instruction): {skipped_no_instruction}")
    if skipped_missing_img:
        print(f"  Skipped samples (missing imgs): {skipped_missing_img}")
    print(f"  Saved:         {out_path}")
    print("=" * 80)

    return dataset, out_path


def main() -> None:
    history_len = 2
    splits = load_splits()
    domain_list = [
        # # (domain, domainid)
        # ("Chrome", "chrome"),
        # ("Gimp", "gimp"),
        ("libreoffice_calc", "calc"),
        ("libreoffice_impress", "impress"),
        # ("libreoffice_writer", "writer"),
        # ("OS", "os"),
        # ("Thunderbird", "thunderbird"),
        # ("VLC", "vlc"),
        ("VScode", "vscode"),
    ]
    for domain, domainid in domain_list:
        domain_folder = DOMAINID_TO_FOLDER[domainid]
        if domain_folder not in splits:
            print(f"Warning: {domain_folder} not in splits.json")
            continue
        domain_splits = splits[domain_folder]
        for splitid in range(len(domain_splits)):
            train_episodes = domain_splits[splitid]["train"]
            generate_dataset_for_split(domainid=domainid, splitid=splitid, train_episodes=train_episodes, history_len=history_len)


if __name__ == "__main__":
    main()



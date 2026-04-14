#!/usr/bin/env python3
"""
data_preprocess.py

读取 ESConv.json 并生成两种 JSON 格式：
 1) vanilla: 仅保留对话（speaker 和 content），用于微调（保存为一个数组文件）。
 2) strategy: 在保留对话的同时保留 supporter 回复的 strategy 字段。

输出路径默认：
  - deepseek_vanilla/data/vanilla.json
  - deepseek_strategy/data/strategy.json

用法：
  python data_preprocess.py --input /path/to/ESConv.json

假设与说明：
  - 输入文件为 JSON 数组，每项包含字段 `dialog`，其为按时间顺序的 turn 列表，turn 包含 `speaker`, `content`, `annotation`。
  - 脚本会将 supporter turn 中的 `annotation.strategy`（若存在）保存在 strategy 输出中。
  - 脚本默认把所有会话写入单个输出文件；如果需要按会话拆分请告诉我，我可以改进为每会话单文件。
"""

import json
import os
import argparse
from typing import List, Dict, Any


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_conversations(convs: List[Dict[str, Any]]):
    vanilla = []
    strategy = []

    for idx, conv in enumerate(convs):
        dialog = conv.get('dialog', [])

        # Build vanilla dialog: list of {speaker, content}
        vanilla_dialog = []
        # Build strategy dialog: include strategy for supporter turns when available
        strategy_dialog = []

        for turn in dialog:
            speaker = turn.get('speaker', '').strip()
            content = turn.get('content', '')
            # Normalize content: strip trailing newlines and surrounding spaces
            if isinstance(content, str):
                content = content.strip()
            else:
                content = str(content)

            vanilla_dialog.append({'speaker': speaker, 'content': content})

            ann = turn.get('annotation', {}) or {}
            strat = None
            if isinstance(ann, dict):
                strat = ann.get('strategy')

            # For strategy output we keep same fields but include strategy (may be None)
            sd = {'speaker': speaker, 'content': content}
            if speaker.lower() == 'supporter':
                sd['strategy'] = strat
            strategy_dialog.append(sd)

        # Optionally include high-level meta for future use
        meta = {
            'id': idx,
            'experience_type': conv.get('experience_type'),
            'emotion_type': conv.get('emotion_type'),
            'problem_type': conv.get('problem_type'),
            'situation': conv.get('situation'),
            'initial_emotion_intensity': conv.get('survey_score', {}).get('seeker', {}).get('initial_emotion_intensity'),
            'final_emotion_intensity': conv.get('survey_score', {}).get('seeker', {}).get('final_emotion_intensity')
        }

        vanilla.append({'meta': meta, 'dialog': vanilla_dialog})
        strategy.append({'meta': meta, 'dialog': strategy_dialog})

    return vanilla, strategy


def main():
    parser = argparse.ArgumentParser(description='Preprocess ESConv.json into vanilla and strategy JSONs')
    parser.add_argument('--input', '-i', default='../ESConv.json', help='Path to ESConv.json')
    parser.add_argument('--vanilla_out', default='deepseek_vanilla/data/vanilla.json', help='Output path for vanilla JSON')
    parser.add_argument('--strategy_out', default='deepseek_strategy/data/strategy.json', help='Output path for strategy JSON')
    parser.add_argument('--min_turns', type=int, default=0, help='过滤掉 turns 少于该值的会话（0 表示不过滤）')

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'输入文件不存在: {input_path}')

    print(f'加载 {input_path} ...')
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f'共加载 {len(data)} 个会话，开始处理...')
    vanilla, strategy = process_conversations(data)

    # 可选过滤
    if args.min_turns and args.min_turns > 0:
        orig_v = len(vanilla)
        vanilla = [c for c in vanilla if len(c['dialog']) >= args.min_turns]
        strategy = [c for c in strategy if len(c['dialog']) >= args.min_turns]
        print(f'按 min_turns={args.min_turns} 过滤: {orig_v} -> {len(vanilla)}')

    # 写入目录
    vanilla_out = args.vanilla_out
    strategy_out = args.strategy_out

    ensure_dir(os.path.dirname(vanilla_out))
    ensure_dir(os.path.dirname(strategy_out))

    print(f'写入 vanilla -> {vanilla_out}')
    with open(vanilla_out, 'w', encoding='utf-8') as f:
        json.dump(vanilla, f, ensure_ascii=False, indent=2)

    print(f'写入 strategy -> {strategy_out}')
    with open(strategy_out, 'w', encoding='utf-8') as f:
        json.dump(strategy, f, ensure_ascii=False, indent=2)

    print('完成。')


if __name__ == '__main__':
    main()

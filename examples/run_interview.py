#!/usr/bin/env python3
"""デプスインタビュー実行スクリプト。

AIインタビュアーがペルソナに対して深掘り質問を行い、定性的なインサイトを発掘する。

Usage:
    python examples/run_interview.py
    python examples/run_interview.py --max-turns 5 --concurrent 3
    python examples/run_interview.py --config /path/to/config.yaml
"""
import argparse
import sys

# srcディレクトリをパスに追加
sys.path.insert(0, "src")

from persona_sim.config import load_config
from persona_sim.interview import InterviewRunner


def main():
    parser = argparse.ArgumentParser(description="デプスインタビュー実行")
    parser.add_argument("--config", type=str, default=None, help="設定ファイルパス（デフォルト: ./config.yaml）")
    parser.add_argument("--input", type=str, default=None, help="入力JSONファイル（config.yamlを上書き）")
    parser.add_argument("--output", type=str, default=None, help="出力CSVファイル（config.yamlを上書き）")
    parser.add_argument("--max-turns", type=int, default=None, help="最大ターン数（config.yamlを上書き）")
    parser.add_argument("--concurrent", type=int, default=None, help="並列実行数（config.yamlを上書き）")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.input:
        config["interview"]["input_file"] = args.input
    if args.output:
        config["interview"]["output_file"] = args.output
    if args.max_turns:
        config["interview"]["max_turns"] = args.max_turns
    if args.concurrent:
        config["interview"]["concurrent_limit"] = args.concurrent

    runner = InterviewRunner(config)
    runner.run()


if __name__ == "__main__":
    main()

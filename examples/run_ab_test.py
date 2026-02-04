#!/usr/bin/env python3
"""A/Bテスト実行スクリプト。

2つの案（プランA vs プランB）に対する受容性を比較検証する。

Usage:
    python examples/run_ab_test.py
    python examples/run_ab_test.py --concurrent 5
    python examples/run_ab_test.py --config /path/to/config.yaml
"""
import argparse
import sys

# srcディレクトリをパスに追加
sys.path.insert(0, "src")

from persona_sim.ab_test import ABTestRunner
from persona_sim.config import load_config


def main():
    parser = argparse.ArgumentParser(description="A/Bテスト実行")
    parser.add_argument("--config", type=str, default=None, help="設定ファイルパス（デフォルト: ./config.yaml）")
    parser.add_argument("--input", type=str, default=None, help="入力JSONファイル（config.yamlを上書き）")
    parser.add_argument("--output", type=str, default=None, help="出力CSVファイル（config.yamlを上書き）")
    parser.add_argument("--concurrent", type=int, default=None, help="並列実行数（config.yamlを上書き）")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.input:
        config["ab_test"]["input_file"] = args.input
    if args.output:
        config["ab_test"]["output_file"] = args.output
    if args.concurrent:
        config["concurrent_limit"] = args.concurrent

    runner = ABTestRunner(config)
    runner.run()


if __name__ == "__main__":
    main()

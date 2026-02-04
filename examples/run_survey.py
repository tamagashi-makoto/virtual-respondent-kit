#!/usr/bin/env python3
"""アンケート調査実行スクリプト。

ペルソナに対してシンプルな一問一答のアンケートを実施する。

Usage:
    python examples/run_survey.py
    python examples/run_survey.py --input data/personas_50.json --output output/survey_50.csv
    python examples/run_survey.py --config /path/to/config.yaml
"""
import argparse
import sys

# srcディレクトリをパスに追加
sys.path.insert(0, "src")

from persona_sim.config import load_config
from persona_sim.survey import SurveyRunner


def main():
    parser = argparse.ArgumentParser(description="アンケート調査実行")
    parser.add_argument("--config", type=str, default=None, help="設定ファイルパス（デフォルト: ./config.yaml）")
    parser.add_argument("--input", type=str, default=None, help="入力JSONファイル（config.yamlを上書き）")
    parser.add_argument("--output", type=str, default=None, help="出力CSVファイル（config.yamlを上書き）")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.input:
        config["survey"]["input_file"] = args.input
    if args.output:
        # 出力ファイルパスが指定された場合、output_dirを無視して直接使用
        config["survey"]["output_file"] = args.output

    runner = SurveyRunner(config)
    runner.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ペルソナデータ準備スクリプト。

Hugging Faceからペルソナデータをダウンロード・サンプリングしてJSON形式で保存する。

Usage:
    python examples/prepare_data.py
    python examples/prepare_data.py --sample-size 50 --output data/personas_50.json
"""
import argparse
import sys

# srcディレクトリをパスに追加
sys.path.insert(0, "src")

from persona_sim.data import prepare_persona_data


def main():
    parser = argparse.ArgumentParser(description="ペルソナデータ準備")
    parser.add_argument("--sample-size", type=int, default=100, help="サンプリング数（デフォルト: 100）")
    parser.add_argument("--output", type=str, default="data/personas_100.json", help="出力ファイルパス")
    args = parser.parse_args()

    prepare_persona_data(sample_size=args.sample_size, output_path=args.output)


if __name__ == "__main__":
    main()

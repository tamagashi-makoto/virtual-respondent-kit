"""データ準備モジュール。

Hugging Faceからペルソナデータをダウンロード・サンプリングし、JSON形式で保存する。
"""
import json
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


def prepare_persona_data(
    sample_size: int = 100,
    output_path: str = "data/personas_100.json",
    show_progress: bool = True,
) -> list[dict]:
    """Hugging Faceからペルソナデータをダウンロード・サンプリングして保存する。

    Args:
        sample_size: サンプリング数（デフォルト: 100）
        output_path: 出力ファイルパス（デフォルト: data/personas_100.json）
        show_progress: 進捗バーを表示するか（デフォルト: True）

    Returns:
        list[dict]: サンプリングされたペルソナデータリスト
    """
    output_file = Path(output_path)

    # 出力ディレクトリが存在しない場合は作成
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print("📥 データセットをHugging Faceからダウンロード中...")

    # NVIDIAのデータセットをロード
    dataset = load_dataset("nvidia/Nemotron-Personas-Japan", split="train")

    total_count = len(dataset)
    if show_progress:
        print(f"✅ データセットロード完了 (全 {total_count} 件)")

    # ランダムにサンプリング
    if show_progress:
        print(f"🎲 ランダムに {sample_size} 件を抽出中...")

    random_indices = random.sample(range(total_count), sample_size)

    # データセットから抽出してリスト化
    sampled_personas = [dataset[i] for i in random_indices]

    # JSONとして保存
    if show_progress:
        print(f"💾 '{output_path}' に保存中...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_personas, f, ensure_ascii=False, indent=4)

    if show_progress:
        print("✨ 完了しました。")

    return sampled_personas


def load_personas(input_path: str) -> list[dict]:
    """JSONファイルからペルソナデータを読み込む。

    Args:
        input_path: 入力JSONファイルパス

    Returns:
        list[dict]: ペルソナデータリスト

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        json.JSONDecodeError: JSON形式が不正な場合
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"❌ エラー: {input_path} が見つかりません。")

    print(f"📖 {input_path} を読み込み中...")

    with open(input_file, "r", encoding="utf-8") as f:
        personas = json.load(f)

    print(f"✅ {len(personas)} 人のペルソナを読み込みました。")

    return personas

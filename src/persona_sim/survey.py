"""アンケート調査モジュール。

ペルソナに対してシンプルな一問一答のアンケートを実施する。
"""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm

from .config import load_config
from .llm import create_llm
from .prompts import get_persona_system_prompt


class SurveyRunner:
    """アンケート調査実行クラス。

    ペルソナデータを読み込み、質問に対する回答を生成してCSVに出力する。
    """

    def __init__(self, config: Optional[dict] = None):
        """初期化。

        Args:
            config: 設定辞書（オプション）。指定しない場合はconfig.yamlから読み込む。
        """
        self.config = config or load_config()

        # LLM初期化（プロバイダーに応じて切り替え）
        self.llm = create_llm(self.config)

        # 設定値
        self.input_file = self.config["survey"]["input_file"]
        self.output_dir = self.config["survey"]["output_dir"]
        self.output_file = self.config["survey"]["output_file"]
        self.survey_question = self.config["survey"]["question"]

    def run(
        self,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        question: Optional[str] = None,
    ) -> pd.DataFrame:
        """全ペルソナに対してアンケートを実行する。

        Args:
            input_file: 入力JSONファイルパス（オプション）
            output_file: 出力CSVファイルパス（オプション）
            question: アンケート質問文（オプション）

        Returns:
            pd.DataFrame: アンケート結果
        """
        # パス設定
        input_file = input_file or self.input_file
        output_file = output_file or str(Path(self.output_dir) / self.output_file)
        question = question or self.survey_question

        # 出力ディレクトリ作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # ペルソナデータ読み込み
        from .data import load_personas

        personas = load_personas(input_file)

        results = []
        print(f"🚀 {len(personas)} 人のペルソナに対してアンケートを開始します...")

        # ループ処理でAPIコール
        for persona in tqdm(personas, desc="Progress"):
            try:
                result = self.run_single(persona, question)
                results.append(result)
            except Exception as e:
                print(f"Error (ID: {persona.get('uuid')}): {e}")

        # 結果の保存
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n✅ 全処理完了。結果を '{output_file}' に保存しました。")

        return df

    def run_single(self, persona: dict, question: str) -> dict:
        """単一ペルソナに対してアンケートを実行する。

        Args:
            persona: ペルソナプロフィール
            question: 質問文

        Returns:
            dict: 回答結果
        """
        system_prompt = self._create_system_prompt(persona)

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        )

        # AIMessageからcontentを取得
        if isinstance(response, AIMessage):
            answer = response.content
        else:
            answer = str(response)

        return {
            "ID": persona.get("uuid"),
            "Age": persona.get("age"),
            "Sex": persona.get("sex"),
            "Occupation": persona.get("occupation"),
            "Prefecture": persona.get("prefecture"),
            "Context_Summary": persona.get("persona", "")[:30] + "...",
            "Survey_Answer": answer,
        }

    def _create_system_prompt(self, persona: dict) -> str:
        """プロンプトを生成する。

        Args:
            persona: ペルソナプロフィール

        Returns:
            str: システムプロンプト
        """
        return get_persona_system_prompt(persona, detailed=True)

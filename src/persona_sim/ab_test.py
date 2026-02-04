"""A/Bテストモジュール。

2つの案（プランA vs プランB）に対する受容性を比較検証する。
LangGraphを使用して段階的な思考プロセスをシミュレーションする。
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from tqdm.asyncio import tqdm
from typing_extensions import TypedDict

from .config import load_config
from .llm import create_llm
from .prompts import get_persona_system_prompt


class ABTestState(TypedDict):
    """A/Bテストの状態。"""

    persona_profile: dict
    eval_a: Optional[str]  # Aの評価コメント
    score_a: Optional[int]  # Aのスコア
    eval_b: Optional[str]  # Bの評価コメント
    score_b: Optional[int]  # Bのスコア
    winner: Optional[str]  # "A" or "B"
    final_reason: Optional[str]  # 決定理由


class ABTestRunner:
    """A/Bテスト実行クラス。

    LangGraphを使用して、ペルソナに2つのプランを評価させ、最終的にどちらを選択するかを決定させる。
    """

    def __init__(self, config: Optional[dict] = None):
        """初期化。

        Args:
            config: 設定辞書（オプション）。指定しない場合はconfig.yamlから読み込む。
        """
        self.config = config or load_config()

        # LLM初期化
        self.llm = create_llm(self.config)

        # 設定値
        self.input_file = self.config["ab_test"]["input_file"]
        self.output_file = self.config["ab_test"]["output_file"]
        self.plan_a = self.config["ab_test"]["plan_a"]
        self.plan_b = self.config["ab_test"]["plan_b"]
        self.concurrent_limit = self.config.get("concurrent_limit", 10)

        # LangGraphワークフロー構築
        self.app = self._build_workflow()

    def _build_workflow(self) -> "CompiledStateGraph":
        """LangGraphワークフローを構築する。

        Returns:
            CompiledStateGraph: コンパイルされたワークフロー
        """
        workflow = StateGraph(ABTestState)

        workflow.add_node("evaluate_a", self._evaluate_a_node)
        workflow.add_node("evaluate_b", self._evaluate_b_node)
        workflow.add_node("decision", self._decision_node)

        # フロー: 評価A → 評価B → 決定（順次実行）
        workflow.add_edge(START, "evaluate_a")
        workflow.add_edge("evaluate_a", "evaluate_b")
        workflow.add_edge("evaluate_b", "decision")
        workflow.add_edge(decision, END)

        return workflow.compile()

    async def _evaluate_a_node(self, state: ABTestState) -> dict:
        """プランAを評価するノード。

        Args:
            state: 現在の状態

        Returns:
            dict: 更新された状態（eval_a, score_a）
        """
        prompt = get_persona_system_prompt(state["persona_profile"], detailed=False)
        user_msg = f"""Please look at the following ad copy, rate it out of 10, and state your reason in one sentence.

{self.plan_a}

Answer Format:
Score: (Number only)
Impression: (Impression)
"""

        response = await self.llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=user_msg)])
        content = response.content

        try:
            score_line = [line for line in content.split("\n") if "Score" in line or "点数" in line]
            if score_line:
                score = int(score_line[0].split(":")[-1].strip())
            else:
                score = 5
        except Exception:
            score = 5

        return {"eval_a": content, "score_a": score}

    async def _evaluate_b_node(self, state: ABTestState) -> dict:
        """プランBを評価するノード。

        Args:
            state: 現在の状態

        Returns:
            dict: 更新された状態（eval_b, score_b）
        """
        prompt = get_persona_system_prompt(state["persona_profile"], detailed=False)
        user_msg = f"""Please look at the following ad copy, rate it out of 10, and state your reason in one sentence.

{self.plan_b}

Answer Format:
Score: (Number only)
Impression: (Impression)
"""

        response = await self.llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=user_msg)])
        content = response.content

        try:
            score_line = [line for line in content.split("\n") if "Score" in line or "点数" in line]
            if score_line:
                score = int(score_line[0].split(":")[-1].strip())
            else:
                score = 5
        except Exception:
            score = 5

        return {"eval_b": content, "score_b": score}

    async def _decision_node(self, state: ABTestState) -> dict:
        """最終決定を行うノード。

        Args:
            state: 現在の状態

        Returns:
            dict: 更新された状態（winner, final_reason）
        """
        prompt = get_persona_system_prompt(state["persona_profile"], detailed=False)

        user_msg = f"""You have evaluated two plans.

【Your Evaluation of Plan A】
{state["eval_a"]}

【Your Evaluation of Plan B】
{state["eval_b"]}

Ultimately, which one do you find more attractive and want to purchase for your lifestyle and occupation?
Please answer clearly with "A" or "B" and state the decisive reason.

Answer Format:
Winner: (A or B)
Reason: (Reason text)
"""

        response = await self.llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=user_msg)])
        content = response.content

        winner = "A" if "Winner: A" in content or "勝者: A" in content else "B"
        return {"winner": winner, "final_reason": content}

    async def _run_single_test(self, persona: dict, semaphore: asyncio.Semaphore) -> Optional[dict]:
        """単一ペルソナに対してA/Bテストを実行する。

        Args:
            persona: ペルソナプロフィール
            semaphore: 並列実行制御用セマフォ

        Returns:
            Optional[dict]: テスト結果
        """
        async with semaphore:
            try:
                initial_state = {"persona_profile": persona}
                final_state = await self.app.ainvoke(initial_state)

                return {
                    "ID": persona.get("uuid"),
                    "Age": persona.get("age"),
                    "Occupation": persona.get("occupation"),
                    "Hobbies": str(persona.get("hobbies_and_interests"))[:30] + "...",
                    "Score_A": final_state.get("score_a"),
                    "Score_B": final_state.get("score_b"),
                    "Winner": final_state.get("winner"),
                    "Reason": str(final_state.get("final_reason")).replace("\n", " ")[:100] + "...",
                }
            except Exception as e:
                print(f"Error {persona.get("uuid")}: {e}")
                return None

    async def run_async(
        self,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        concurrent_limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """全ペルソナに対してA/Bテストを非同期で実行する。

        Args:
            input_file: 入力JSONファイルパス（オプション）
            output_file: 出力CSVファイルパス（オプション）
            concurrent_limit: 並列実行数（オプション）

        Returns:
            pd.DataFrame: テスト結果
        """
        # パス設定
        input_file = input_file or self.input_file
        output_file = output_file or self.output_file
        concurrent_limit = concurrent_limit or self.concurrent_limit

        # 出力ディレクトリ作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # ペルソナデータ読み込み
        from .data import load_personas

        personas = load_personas(input_file)

        print(f"⚖️  AB Test Start: {len(personas)} people (Plan A vs Plan B)")

        semaphore = asyncio.Semaphore(concurrent_limit)
        tasks = [self._run_single_test(p, semaphore) for p in personas]

        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            res = await f
            if res:
                results.append(res)

        # 結果の保存
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n✅ Test Completed. Saved to '{output_file}'.")

        # 集計結果表示
        if not df.empty and "Winner" in df.columns:
            win_a = len(df[df["Winner"] == "A"])
            win_b = len(df[df["Winner"] == "B"])
            print("\n=== Aggregation Result ===")
            print(f"🏆 Plan A Wins: {win_a}")
            print(f"🏆 Plan B Wins: {win_b}")

            # 職業別トレンド（スニペット）
            if "Occupation" in df.columns:
                print("\n=== Trend by Occupation (Top 5) ===")
                print(df.groupby("Winner")["Occupation"].value_counts().head(5))

        return df

    def run(
        self,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        concurrent_limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """全ペルソナに対してA/Bテストを実行する。

        Args:
            input_file: 入力JSONファイルパス（オプション）
            output_file: 出力CSVファイルパス（オプション）
            concurrent_limit: 並列実行数（オプション）

        Returns:
            pd.DataFrame: テスト結果
        """
        return asyncio.run(self.run_async(input_file, output_file, concurrent_limit))

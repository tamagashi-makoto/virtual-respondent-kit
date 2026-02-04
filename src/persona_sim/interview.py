"""デプスインタビューモジュール。

AIインタビュアーがペルソナに対して深掘り質問を行い、定性的なインサイトを発掘する。
LangGraphを使用したマルチエージェント対話システム。
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from tqdm.asyncio import tqdm
from typing_extensions import TypedDict
import operator

from .config import load_config
from .llm import create_llm
from .prompts import get_interviewer_system_prompt, get_persona_system_prompt


class InterviewState(TypedDict):
    """インタビューの状態。"""

    messages: Annotated[List[BaseMessage], operator.add]  # 追記専用メッセージ履歴
    persona_profile: dict
    turn_count: int


class InterviewRunner:
    """デプスインタビュー実行クラス。

    AIインタビュアーとペルソナエージェントの対話をシミュレートする。
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
        self.input_file = self.config["interview"]["input_file"]
        self.output_file = self.config["interview"]["output_file"]
        self.max_turns = self.config["interview"]["max_turns"]
        self.concurrent_limit = self.config["interview"]["concurrent_limit"]
        self.initial_question = self.config["interview"]["initial_question"]

        # LangGraphワークフロー構築
        self.app = self._build_workflow()

    def _build_workflow(self) -> "CompiledStateGraph":
        """LangGraphワークフローを構築する。

        Returns:
            CompiledStateGraph: コンパイルされたワークフロー
        """
        workflow = StateGraph(InterviewState)

        workflow.add_node("interviewer", self._interviewer_node)
        workflow.add_node("persona", self._persona_node)

        workflow.add_edge(START, "persona")

        def should_continue(state: InterviewState) -> str:
            """インタビューを続けるか判定する。

            Args:
                state: 現在の状態

            Returns:
                str: 次のノード名（"interviewer" or END）
            """
            if state["turn_count"] >= self.max_turns:
                return END
            return "interviewer"

        workflow.add_conditional_edges("persona", should_continue)
        workflow.add_edge("interviewer", "persona")

        return workflow.compile()

    async def _persona_node(self, state: InterviewState) -> dict:
        """ペルソナノード。

        Args:
            state: 現在の状態

        Returns:
            dict: 更新された状態
        """
        profile = state["persona_profile"]

        system_prompt = f"""You are a real Japanese person with the following profile.

## Your Profile
- Age: {profile.get("age")} / Sex: {profile.get("sex")}
- Occupation: {profile.get("occupation")}
- Region: {profile.get("prefecture")}

## Detailed Persona & Values
- Personality: {profile.get("persona")}
- Professional Stance: {profile.get("professional_persona")}
- Hobbies: {profile.get("hobbies_and_interests")}

Please answer the interviewer's questions acting fully as this person.
Speak your "honest feelings" and "concerns" based on your daily life reality, not just shallow polite answers.
"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await self.llm.ainvoke(messages)

        return {"messages": [response], "turn_count": 0}

    async def _interviewer_node(self, state: InterviewState) -> dict:
        """インタビュアーノード。

        Args:
            state: 現在の状態

        Returns:
            dict: 更新された状態
        """
        last_answer = state["messages"][-1].content

        system_prompt = get_interviewer_system_prompt()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Respondent's Answer: {last_answer}\n\nCreate ONE deep-dive question for this."),
        ]

        response = await self.llm.ainvoke(messages)

        return {"messages": [response], "turn_count": 1}

    async def _run_single_interview(self, persona: dict, semaphore: asyncio.Semaphore) -> Optional[dict]:
        """単一ペルソナに対してインタビューを実行する。

        Args:
            persona: ペルソナプロフィール
            semaphore: 並列実行制御用セマフォ

        Returns:
            Optional[dict]: インタビュー結果
        """
        async with semaphore:
            try:
                initial_state = {
                    "messages": [HumanMessage(content=self.initial_question)],
                    "persona_profile": persona,
                    "turn_count": 0,
                }

                final_state = await self.app.ainvoke(initial_state)

                # トランスクリプト作成
                transcript = self._create_transcript(final_state["messages"])
                final_answer = final_state["messages"][-1].content

                return {
                    "ID": persona.get("uuid"),
                    "Occupation": persona.get("occupation"),
                    "Age": persona.get("age"),
                    "Conversation_Log": transcript,
                    "Final_Answer": final_answer,
                }

            except Exception as e:
                print(f"Error processing {persona.get("uuid")}: {e}")
                return None

    def _create_transcript(self, messages: List[BaseMessage]) -> str:
        """メッセージ履歴からトランスクリプトを作成する。

        Args:
            messages: メッセージリスト

        Returns:
            str: トランスクリプト
        """
        transcript = ""

        for idx, msg in enumerate(messages):
            if idx == 0:
                role = "【Initial Question】"
            elif idx % 2 != 0:
                role = "【Persona Answer】"
            else:
                role = "【Interviewer Question】"

            transcript += f"{role}\n{msg.content}\n\n"

        return transcript

    async def run_async(
        self,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        max_turns: Optional[int] = None,
        concurrent_limit: Optional[int] = None,
        initial_question: Optional[str] = None,
    ) -> pd.DataFrame:
        """全ペルソナに対してインタビューを非同期で実行する。

        Args:
            input_file: 入力JSONファイルパス（オプション）
            output_file: 出力CSVファイルパス（オプション）
            max_turns: 最大ターン数（オプション）
            concurrent_limit: 並列実行数（オプション）
            initial_question: 初期質問（オプション）

        Returns:
            pd.DataFrame: インタビュー結果
        """
        # パス設定
        input_file = input_file or self.input_file
        output_file = output_file or self.output_file
        max_turns = max_turns or self.max_turns
        concurrent_limit = concurrent_limit or self.concurrent_limit
        initial_question = initial_question or self.initial_question

        # 出力ディレクトリ作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # ペルソナデータ読み込み
        from .data import load_personas

        personas = load_personas(input_file)

        print(f"🚀 LangGraph Interview Start: {len(personas)} people (Concurrent: {concurrent_limit})")

        semaphore = asyncio.Semaphore(concurrent_limit)

        tasks = [self._run_single_interview(p, semaphore) for p in personas]

        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            res = await f
            if res:
                results.append(res)

        # 結果の保存
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n✅ Interview Completed. Saved to '{output_file}'.")

        if not df.empty:
            print("\n=== Sample Log (Top 1) ===")
            print(df.iloc[0]["Conversation_Log"][:1000] + "...")

        return df

    def run(
        self,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        max_turns: Optional[int] = None,
        concurrent_limit: Optional[int] = None,
        initial_question: Optional[str] = None,
    ) -> pd.DataFrame:
        """全ペルソナに対してインタビューを実行する。

        Args:
            input_file: 入力JSONファイルパス（オプション）
            output_file: 出力CSVファイルパス（オプション）
            max_turns: 最大ターン数（オプション）
            concurrent_limit: 並列実行数（オプション）
            initial_question: 初期質問（オプション）

        Returns:
            pd.DataFrame: インタビュー結果
        """
        return asyncio.run(
            self.run_async(
                input_file=input_file,
                output_file=output_file,
                max_turns=max_turns,
                concurrent_limit=concurrent_limit,
                initial_question=initial_question,
            )
        )

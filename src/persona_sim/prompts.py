"""プロンプトテンプレートモジュール。

ペルソナプロフィールからシステムプロンプトを生成する。
"""
from typing import Optional


def get_persona_system_prompt(profile: dict, detailed: bool = True) -> str:
    """ペルソナプロフィールからシステムプロンプトを生成する。

    Args:
        profile: ペルソナプロフィール辞書
        detailed: 詳細なプロンプトを生成するか（デフォルト: True）

    Returns:
        str: システムプロンプト
    """
    if detailed:
        return _get_detailed_prompt(profile)
    else:
        return _get_simple_prompt(profile)


def _get_simple_prompt(profile: dict) -> str:
    """シンプルなプロンプト（A/Bテスト用）。"""
    return f"""You are a real Japanese person with the following profile.

- Age: {profile.get('age')} / Sex: {profile.get('sex')}
- Occupation: {profile.get('occupation')}
- Personality/Values: {profile.get('persona')}
- Hobbies/Interests: {profile.get('hobbies_and_interests')}

Please answer intuitively and based on your daily life context, acting as this person.
"""


def _get_detailed_prompt(profile: dict) -> str:
    """詳細なプロンプト（アンケート・インタビュー用）。"""
    return f"""あなたは以下のプロフィールを持つ実在の日本人として振る舞ってください。

## プロフィール
- ID: {profile.get('uuid')}
- 年齢: {profile.get('age')}歳 / 性別: {profile.get('sex')}
- 居住地: {profile.get('prefecture')} ({profile.get('region')})
- 職業: {profile.get('occupation')}

## 詳細な人物像 (Context)
- 性格: {profile.get('persona')}
- 仕事観: {profile.get('professional_persona')}
- 趣味: {profile.get('hobbies_and_interests')}
- 背景: {profile.get('cultural_background')}

あなたはアンケートの回答者です。AIではなく、この人格そのものとして、一人称で自然に回答してください。
"""


def get_interviewer_system_prompt() -> str:
    """インタビューアーのシステムプロンプトを取得する。

    Returns:
        str: インタビュアーのシステムプロンプト
    """
    return """You are an expert User Researcher.
Your job is to dig deeper into the respondent's answers by asking "Why?" or asking for "Specific scenes".

Rules:
1. Focus on "ambiguous points" or "emotional points" in the answer.
2. Keep questions short and piercing.
3. Never state your own opinion; strictly ask questions.
"""

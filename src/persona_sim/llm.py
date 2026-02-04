"""LLMクライアント初期化モジュール。

複数のLLMプロバイダー（Azure OpenAI、OpenAI、Gemini、Anthropic、Groq）に対応し、
設定ファイルで簡単に切り替えられるようにする。
"""
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_model import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .config import (
    get_anthropic_config,
    get_azure_openai_config,
    get_gemini_config,
    get_groq_config,
    get_llm_provider,
    get_openai_config,
    load_config,
)

# プロバイダーとクラスのマッピング
PROVIDER_MAP = {
    "azure_openai": AzureChatOpenAI,
    "openai": ChatOpenAI,
    "gemini": ChatGoogleGenerativeAI,
    "anthropic": ChatAnthropic,
    "groq": ChatGroq,
}


def create_llm(config: Optional[dict] = None) -> BaseChatModel:
    """LLMを初期化する（プロバイダーに応じて切り替え）。

    Args:
        config: 設定辞書（オプション）。指定しない場合はconfig.yamlから読み込む。

    Returns:
        BaseChatModel: 初期化されたLLMクライアント

    Raises:
        ValueError: サポートされていないプロバイダーが指定された場合
    """
    provider = get_llm_provider(config)

    if provider == "azure_openai":
        return _create_azure_openai_llm(config)
    elif provider == "openai":
        return _create_openai_llm(config)
    elif provider == "gemini":
        return _create_gemini_llm(config)
    elif provider == "anthropic":
        return _create_anthropic_llm(config)
    elif provider == "groq":
        return _create_groq_llm(config)
    else:
        supported = ", ".join(PROVIDER_MAP.keys())
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {supported}")


def _get_model_params(config: Optional[dict] = None) -> dict:
    """共通のモデルパラメータを取得する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        dict: モデルパラメータ（temperature, max_tokens等）
    """
    if config is None:
        config = load_config()

    model_params = config.get("model_params", {})

    # パラメータ名の正規化
    params = {}
    if "temperature" in model_params:
        params["temperature"] = model_params["temperature"]
    if "max_completion_tokens" in model_params:
        params["max_tokens"] = model_params["max_completion_tokens"]
    elif "max_tokens" in model_params:
        params["max_tokens"] = model_params["max_tokens"]

    return params


def _create_azure_openai_llm(config: Optional[dict] = None) -> AzureChatOpenAI:
    """Azure OpenAI Chatモデルを初期化する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        AzureChatOpenAI: 初期化されたLLMクライアント
    """
    provider_config = get_azure_openai_config(config)
    model_params = _get_model_params(config)

    return AzureChatOpenAI(
        azure_endpoint=provider_config["endpoint"],
        api_key=provider_config["api_key"],
        api_version=provider_config["api_version"],
        deployment_name=provider_config["deployment_name"],
        **model_params,
    )


def _create_openai_llm(config: Optional[dict] = None) -> ChatOpenAI:
    """OpenAI Chatモデルを初期化する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        ChatOpenAI: 初期化されたLLMクライアント
    """
    provider_config = get_openai_config(config)
    model_params = _get_model_params(config)

    kwargs = {}
    if provider_config.get("base_url"):
        kwargs["base_url"] = provider_config["base_url"]

    return ChatOpenAI(
        api_key=provider_config["api_key"],
        model=provider_config.get("model", "gpt-4o"),
        **model_params,
        **kwargs,
    )


def _create_gemini_llm(config: Optional[dict] = None) -> ChatGoogleGenerativeAI:
    """Gemini Chatモデルを初期化する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        ChatGoogleGenerativeAI: 初期化されたLLMクライアント
    """
    provider_config = get_gemini_config(config)
    model_params = _get_model_params(config)

    return ChatGoogleGenerativeAI(
        api_key=provider_config["api_key"],
        model=provider_config.get("model", "gemini-2.0-flash-exp"),
        **model_params,
    )


def _create_anthropic_llm(config: Optional[dict] = None) -> ChatAnthropic:
    """Anthropic Chatモデルを初期化する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        ChatAnthropic: 初期化されたLLMクライアント
    """
    provider_config = get_anthropic_config(config)
    model_params = _get_model_params(config)

    # Anthropicはmax_tokensを必須とする
    if "max_tokens" not in model_params:
        model_params["max_tokens"] = provider_config.get("max_tokens", 8192)

    return ChatAnthropic(
        api_key=provider_config["api_key"],
        model=provider_config.get("model", "claude-sonnet-4-20250514"),
        **model_params,
    )


def _create_groq_llm(config: Optional[dict] = None) -> ChatGroq:
    """Groq Chatモデルを初期化する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        ChatGroq: 初期化されたLLMクライアント
    """
    provider_config = get_groq_config(config)
    model_params = _get_model_params(config)

    return ChatGroq(
        api_key=provider_config["api_key"],
        model=provider_config.get("model", "gpt-oss-120b"),
        **model_params,
    )

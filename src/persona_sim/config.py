"""設定ファイル読み込みモジュール。

環境変数 CONFIG_PATH から設定ファイルパスを解決し、YAML設定を読み込む。
"""
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# デフォルト設定ファイルパス
DEFAULT_CONFIG_PATH = "./config.yaml"

# 環境変数をロード（.envファイルがあれば）
load_dotenv()


def get_config_path(config_path: Optional[str] = None) -> Path:
    """設定ファイルパスを取得する。

    優先順位:
    1. 引数で指定されたパス
    2. 環境変数 CONFIG_PATH
    3. デフォルトパス (./config.yaml)

    Args:
        config_path: 設定ファイルパス（オプション）

    Returns:
        Path: 設定ファイルのパス
    """
    if config_path:
        return Path(config_path)

    env_config_path = os.getenv("CONFIG_PATH")
    if env_config_path:
        return Path(env_config_path)

    return Path(DEFAULT_CONFIG_PATH)


def load_config(config_path: Optional[str] = None) -> dict:
    """YAML設定ファイルを読み込む。

    Args:
        config_path: 設定ファイルパス（オプション）

    Returns:
        dict: 設定内容

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合
    """
    config_file = get_config_path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_file}' not found. "
            f"Please create a config.yaml file or set CONFIG_PATH environment variable."
        )

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_llm_provider(config: Optional[dict] = None) -> str:
    """使用するLLMプロバイダーを取得する。

    優先順位:
    1. 環境変数 LLM_PROVIDER
    2. config.yamlの llm_provider
    3. デフォルト値 (azure_openai)

    Args:
        config: 設定辞書（オプション）

    Returns:
        str: LLMプロバイダー名 (azure_openai, openai, gemini, anthropic, groq)
    """
    if config is None:
        config = load_config()

    # 環境変数で上書き
    if env_provider := os.getenv("LLM_PROVIDER"):
        return env_provider

    # config.yamlから取得
    provider = config.get("llm_provider", "azure_openai")

    # 後方互換性: azure -> azure_openai
    if provider == "azure":
        provider = "azure_openai"

    return provider


def _get_provider_config(provider: str, config: Optional[dict] = None) -> dict:
    """プロバイダー設定を取得する（共通ロジック）。

    Args:
        provider: プロバイダー名
        config: 設定辞書（オプション）

    Returns:
        dict: プロバイダー設定
    """
    if config is None:
        config = load_config()

    provider_config = config.get(provider, {}).copy()

    # 後方互換性: azure -> azure_openai
    if provider == "azure_openai" and not provider_config:
        provider_config = config.get("azure", {}).copy()

    return provider_config


def get_azure_openai_config(config: Optional[dict] = None) -> dict:
    """Azure OpenAI設定を取得する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        dict: Azure OpenAI設定（endpoint, api_key, api_version, deployment_name）
    """
    provider_config = _get_provider_config("azure_openai", config)

    # 環境変数で上書き
    if env_endpoint := os.getenv("AZURE_OPENAI_ENDPOINT"):
        provider_config["endpoint"] = env_endpoint
    if env_key := os.getenv("AZURE_OPENAI_API_KEY"):
        provider_config["api_key"] = env_key
    if env_version := os.getenv("AZURE_OPENAI_API_VERSION"):
        provider_config["api_version"] = env_version
    if env_deployment := os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
        provider_config["deployment_name"] = env_deployment

    return provider_config


def get_openai_config(config: Optional[dict] = None) -> dict:
    """OpenAI設定を取得する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        dict: OpenAI設定（api_key, model, base_url）
    """
    provider_config = _get_provider_config("openai", config)

    # 環境変数で上書き
    if env_key := os.getenv("OPENAI_API_KEY"):
        provider_config["api_key"] = env_key

    return provider_config


def get_gemini_config(config: Optional[dict] = None) -> dict:
    """Gemini設定を取得する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        dict: Gemini設定（api_key, model）
    """
    provider_config = _get_provider_config("gemini", config)

    # 環境変数で上書き
    if env_key := os.getenv("GEMINI_API_KEY"):
        provider_config["api_key"] = env_key

    return provider_config


def get_anthropic_config(config: Optional[dict] = None) -> dict:
    """Anthropic設定を取得する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        dict: Anthropic設定（api_key, model, max_tokens）
    """
    provider_config = _get_provider_config("anthropic", config)

    # 環境変数で上書き
    if env_key := os.getenv("ANTHROPIC_API_KEY"):
        provider_config["api_key"] = env_key

    return provider_config


def get_groq_config(config: Optional[dict] = None) -> dict:
    """Groq設定を取得する。

    Args:
        config: 設定辞書（オプション）

    Returns:
        dict: Groq設定（api_key, model）
    """
    provider_config = _get_provider_config("groq", config)

    # 環境変数で上書き
    if env_key := os.getenv("GROQ_API_KEY"):
        provider_config["api_key"] = env_key

    return provider_config


# 後方互換性のためのエイリアス
get_azure_config = get_azure_openai_config

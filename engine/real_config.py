from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

PLACEHOLDER_VALUES = {
    "",
    "fill_me",
    "placeholder",
    "todo",
    "change_me",
    "changeme",
    "none",
    "null",
}

OPENAI_MAIN_KEY_NAMES = ("OPENAI_API_KEY", "openai-api-key")
OPENAI_JUDGE_KEY_NAMES = ("OPENAI_JUDGE_API_KEY", "openai_api_key")
OPENROUTER_KEY_NAMES = ("OPENROUTER_API_KEY", "openrouter_api_key")


@dataclass(frozen=True)
class RuntimeConfig:
    run_mode: str
    allow_fallback: bool
    real_case_limit: int
    main_model: str
    openai_judge_model: str
    deepseek_judge_model: str
    embedding_provider: str
    embedding_model: str
    openai_api_key: str
    openai_judge_api_key: str
    openrouter_api_key: str


def _lookup(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    return value.strip() if isinstance(value, str) else ""


def is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    cleaned = value.strip()
    lowered = cleaned.lower()
    return lowered in PLACEHOLDER_VALUES or lowered.endswith("...")


def first_env_value(names: tuple[str, ...], environ: Mapping[str, str] | None = None) -> str:
    env = environ or os.environ
    for name in names:
        value = _lookup(env, name)
        if value and not is_placeholder(value):
            return value
    return ""


def raw_present_value(names: tuple[str, ...], environ: Mapping[str, str] | None = None) -> str:
    env = environ or os.environ
    for name in names:
        value = _lookup(env, name)
        if value:
            return value
    return ""


def bool_env(name: str, default: bool = False, environ: Mapping[str, str] | None = None) -> bool:
    env = environ or os.environ
    value = _lookup(env, name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def int_env(name: str, default: int, environ: Mapping[str, str] | None = None) -> int:
    env = environ or os.environ
    value = _lookup(env, name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_runtime_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    env = environ or os.environ
    run_mode = _lookup(env, "EVAL_MODE").lower() or "local"
    return RuntimeConfig(
        run_mode=run_mode,
        allow_fallback=bool_env("ALLOW_FALLBACK", default=run_mode != "real", environ=env),
        real_case_limit=int_env("REAL_EVAL_CASE_LIMIT", 30, environ=env),
        main_model=_lookup(env, "MAIN_MODEL") or "gpt-4o-mini",
        openai_judge_model=_lookup(env, "OPENAI_JUDGE_MODEL") or "gpt-4o-mini",
        deepseek_judge_model=_lookup(env, "OPENROUTER_JUDGE_MODEL") or "deepseek/deepseek-v4-flash",
        embedding_provider=(_lookup(env, "EMBEDDING_PROVIDER") or "mock").lower(),
        embedding_model=_lookup(env, "EMBEDDING_MODEL") or "openai/text-embedding-3-small",
        openai_api_key=first_env_value(OPENAI_MAIN_KEY_NAMES, env),
        openai_judge_api_key=first_env_value(OPENAI_JUDGE_KEY_NAMES, env),
        openrouter_api_key=first_env_value(OPENROUTER_KEY_NAMES, env),
    )


def validate_real_env(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    env = environ or os.environ
    config = get_runtime_config(env)
    if config.run_mode != "real":
        return config

    missing: list[str] = []
    raw_main = raw_present_value(OPENAI_MAIN_KEY_NAMES, env)
    raw_judge = raw_present_value(OPENAI_JUDGE_KEY_NAMES, env)
    raw_openrouter = raw_present_value(OPENROUTER_KEY_NAMES, env)
    if not config.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not config.openai_judge_api_key:
        missing.append("OPENAI_JUDGE_API_KEY")
    if not config.openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    placeholders = [
        name
        for name, value in {
            "OPENAI_API_KEY": raw_main,
            "OPENAI_JUDGE_API_KEY": raw_judge,
            "OPENROUTER_API_KEY": raw_openrouter,
        }.items()
        if value and is_placeholder(value)
    ]

    if missing or placeholders:
        details = []
        if missing:
            details.append("missing or placeholder: " + ", ".join(sorted(set(missing))))
        if placeholders:
            details.append("explicit placeholder values: " + ", ".join(sorted(set(placeholders))))
        raise RuntimeError("Real evaluation requires valid API keys (" + "; ".join(details) + ").")
    if config.allow_fallback:
        raise RuntimeError("Real evaluation requires ALLOW_FALLBACK=false.")
    return config

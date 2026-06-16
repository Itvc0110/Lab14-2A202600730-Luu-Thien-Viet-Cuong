from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.real_config import validate_real_env


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    try:
        config = validate_real_env()
    except RuntimeError as exc:
        print(f"REAL ENV CHECK: FAIL - {exc}")
        return 1

    print("REAL ENV CHECK: PASS")
    print(f"run_mode={config.run_mode}")
    print(f"main_model={config.main_model}")
    print(f"openai_judge_model={config.openai_judge_model}")
    print(f"deepseek_judge_model={config.deepseek_judge_model}")
    print(f"embedding_provider={config.embedding_provider}")
    print(f"embedding_model={config.embedding_model}")
    print(f"real_case_limit={config.real_case_limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

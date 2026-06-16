from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic_gen import CORPUS_PATH, generate_lab_data
from engine.real_config import validate_real_env
from engine.vector_store import DEFAULT_VECTOR_INDEX_PATH, build_vector_index


def load_corpus_rows() -> list[dict]:
    if not CORPUS_PATH.exists():
        generate_lab_data()
    return [
        json.loads(line)
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    config = validate_real_env()
    rows = load_corpus_rows()
    records = build_vector_index(rows, index_path=DEFAULT_VECTOR_INDEX_PATH)
    print(f"VECTOR INDEX: PASS")
    print(f"path={DEFAULT_VECTOR_INDEX_PATH}")
    print(f"rows={len(records)}")
    print(f"embedding_provider={config.embedding_provider}")
    print(f"embedding_model={config.embedding_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A-123 탐색 실행기.

기존 팀 저장소에는 A-123 관련 진단 스크립트가 이미 있으므로,
이 파일은 중복 코드를 복사하지 않고 기존 스크립트를 호출하는 얇은 진입점으로 둔다.

- counters: 소모품 카운터 기반 A-123 진단
- allzero: 전체-0 센서 패턴 분석

실행 예:
    python src/run_a123.py --mode counters
    python src/run_a123.py --mode allzero
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "counters": ROOT / "hyp_a123_counters.py",
    "allzero": ROOT / "hyp_allzero_a123.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the repo's existing A-123 diagnostics.")
    parser.add_argument(
        "--mode",
        choices=sorted(SCRIPTS),
        default="counters",
        help="Which existing A-123 diagnostic script to run.",
    )
    args = parser.parse_args()

    script = SCRIPTS[args.mode]
    if not script.exists():
        raise FileNotFoundError(f"Missing A-123 script: {script}")

    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

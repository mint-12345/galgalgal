# drift-k-cmp

PHM 2016 CMP 데이터로 웨이퍼 MRR 가상계측 모델과 샘플링 정책을 비교하는 프로젝트. 작업 규칙은 [CLAUDE.md](CLAUDE.md).

## 환경 설치

Python 3.13 기준 (`requirements.txt`는 Python 3.13.2에서 `pip freeze`로 만들었다).

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 데이터 준비와 집계

1. 원본 데이터를 CLAUDE.md 4절 구조대로 `data/raw/`와 `data/locked/`에 둔다 (둘 다 git에서 제외됨).
2. 웨이퍼 단위 집계표를 만든다.

   ```bash
   python src/aggregate.py
   ```

   `data/processed/wafer_table.csv`가 생긴다. 이미 있으면 덮어쓰지 않고 멈추니, 다시 만들 때는 먼저 지운다. 컬럼 설명은 [data/processed/COLUMNS.md](data/processed/COLUMNS.md).

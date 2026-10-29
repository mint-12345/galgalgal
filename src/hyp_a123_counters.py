"""A-123 웨이퍼의 소모품 카운터 6개가 A-123이 실제로 쓰는 부품을 세는지 (사전 확정 문서 5.0절, 진단 CLAUDE.md 14.4).

카운터마다:
1. 웨이퍼 안 증가: 웨이퍼 마지막 값 − 첫 값 > 0인 웨이퍼 비율, 웨이퍼당 증가량 (A-123 / A-456 / B-456).
   A-123은 전체와 기록 누락 여부(features_v1의 allzero)로 나눠서도 낸다.
2. 공유: 스테이지 A에서 A-123 웨이퍼 첫 행 시각에 진행 중인 A-456 행의 카운터 값과 같은지 (±2초 안의 가장 가까운 행).
   그리고 시작 시각 순으로 늘어놓았을 때 A-123 시작값이 앞뒤 A-456 시작값 사이에 들어가는지 (한 줄로 이어지는지).
   참고로 A-456과 B-456도 같은 방식으로 비교한다.
3. 리셋: 그룹별 웨이퍼 시작값 순열에서 aggregate.py와 같은 규칙(스테이지 내 범위의 RESET_FRAC 이상 감소)으로 감지하고,
   A-123 리셋마다 가장 가까운 A-456·B-456 리셋과의 시간 차를 낸다.
4. 웨이퍼 1장 처리 중 증가량 비교 (중앙값).
5. (2에서 카운터가 장비 전체 공유로 나와서 추가) 정상 A-123 웨이퍼 중 다른 456 웨이퍼와 시간이 겹치지 않는 웨이퍼에서도
   증가하는지, 그리고 증가가 A-123 웨이퍼 자신의 어느 챔버·어떤 행(압력 > 0 / 0)에서 일어나는지.

판정 (요청 기준): A-123 처리 중 증가한 웨이퍼 비율 ≥ 0.8 → 자기 부품,
  ≤ 0.2 이고 A-456 쪽에서 증가(≥ 0.8) → 자기 부품 아님, 그 외 → 판단 보류.
  규칙은 A-123 전체 웨이퍼에 그대로 적용하고, 기록 누락 웨이퍼를 뺀 값은 참고로 함께 낸다.

데이터: train·val 원본 센서 (완전 중복 제거), data/processed/features_v1.csv (allzero, 키).
MRR·라벨 파일·test·data/locked/는 읽지 않는다. 이상치 4장(A-123)은 features_v1에 없어 allzero 구분에서는 빠지고 전체에는 들어간다.

실행: python src/hyp_a123_counters.py
출력: results/diagnostics/2026-10-28_김혜연_a123_counters_{wafer.csv, wafer.yaml, tables.md, time.png}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.font_manager import FontProperties, findSystemFonts  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import PRESSURE, RESET_FRAC, USAGE, dedup  # noqa: E402
from load import KEY, ROOT, load_sensors  # noqa: E402

DATE = "2026-10-28"
AUTHOR = "김혜연"
DIAG = ROOT / "results" / "diagnostics"
STEM = f"{DATE}_{AUTHOR}_a123_counters"
OUT_W = DIAG / f"{STEM}_wafer.csv"
OUT_MD = DIAG / f"{STEM}_tables.md"
FIG = DIAG / f"{STEM}_time.png"
FEATURES = ROOT / "data" / "processed" / "features_v1.csv"
GROUPS = ["A-123", "A-456", "B-456"]
NAME_KO = {"BACKING_FILM": "백킹 필름", "DRESSER": "드레서", "POLISH_TABLE": "폴리싱 테이블",
           "DRESSER_TABLE": "드레서 테이블", "MEMBRANE": "멤브레인", "PSHEET": "가압 시트"}
SELF_MIN, NOT_SELF_MAX = 0.8, 0.2
TOL_S = 2.0          # 같은 시각 비교: 가장 가까운 행이 이 안에 있어야 한다 (초)
ZOOM_DAYS = (30.0, 32.0)


def korean_font() -> str:
    want = ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic"]
    have = {FontProperties(fname=f).get_name() for f in findSystemFonts()}
    return next((n for n in want if n in have), plt.rcParams["font.family"][0])


def load_raw() -> pd.DataFrame:
    raw = pd.concat([dedup(load_sensors(s))[0] for s in ("train", "val")], ignore_index=True)
    # load_sensors 결과는 숫자 열도 object 타입이다 (값은 모두 float). merge_asof를 위해 숫자로 바꾼다.
    num = [c for c in raw.columns if c not in ("STAGE", "split")]
    raw[num] = raw[num].astype(float)
    raw["WAFER_ID"] = raw["WAFER_ID"].astype("int64")
    raw = raw.sort_values(KEY + ["TIMESTAMP"], kind="stable").reset_index(drop=True)
    grp = raw.groupby(KEY)["CHAMBER"].transform("min")
    raw["group"] = raw["STAGE"] + np.where(grp >= 4, "-456", "-123")
    return raw


def wafer_table(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(KEY)
    parts = [g["group"].first(), g["TIMESTAMP"].first().rename("t0"), g["TIMESTAMP"].last().rename("t1")]
    for col, ab in USAGE.items():
        parts += [g[col].first().rename(f"s_{ab}"), g[col].last().rename(f"e_{ab}")]
    w = pd.concat(parts, axis=1).reset_index()
    for ab in USAGE.values():
        w[f"d_{ab}"] = w[f"e_{ab}"] - w[f"s_{ab}"]
    az = pd.read_csv(FEATURES, usecols=KEY + ["allzero"])
    w = w.merge(az, on=KEY, how="left")
    w["day"] = (w["t0"] - w["t0"].min()) / 86400
    return w


def same_time(raw: pd.DataFrame, w: pd.DataFrame, src: str, ref: str, stage_only: bool) -> pd.DataFrame:
    """src 그룹 웨이퍼의 첫 행 시각에 진행 중인 ref 그룹 행의 카운터 값 (±TOL_S 안 가장 가까운 행)."""
    r = raw[raw["group"] == ref].sort_values("TIMESTAMP")
    q = w[w["group"] == src].sort_values("t0")
    m = pd.merge_asof(q[KEY + ["t0"] + [f"s_{ab}" for ab in USAGE.values()]], r[["TIMESTAMP"] + list(USAGE)],
                      left_on="t0", right_on="TIMESTAMP", direction="nearest", tolerance=TOL_S)
    out = {"n": len(q), "matched": int(m["TIMESTAMP"].notna().sum())}
    for col, ab in USAGE.items():
        diff = (m[f"s_{ab}"] - m[col]).dropna()
        out[ab] = diff
    return out


def between(w: pd.DataFrame, src: str, ref: str) -> dict:
    """src 웨이퍼 시작값이 시간상 앞뒤 ref 웨이퍼 시작값 사이에 있는지 (그 사이에 리셋이 없는 경우만)."""
    q = w[w["group"] == src].sort_values("t0")
    r = w[w["group"] == ref].sort_values("t0").reset_index(drop=True)
    idx = np.searchsorted(r["t0"].to_numpy(), q["t0"].to_numpy(), side="right")
    ok = (idx > 0) & (idx < len(r))
    q, idx = q[ok], idx[ok]
    prev, nxt = r.iloc[idx - 1].reset_index(drop=True), r.iloc[idx].reset_index(drop=True)
    q = q.reset_index(drop=True)
    res = {}
    for ab in USAGE.values():
        lo, hi, v = prev[f"s_{ab}"], nxt[f"s_{ab}"], q[f"s_{ab}"]
        noreset = hi >= lo
        frac = ((v >= lo) & (v <= hi))[noreset].mean()
        tt = (q["t0"] - prev["t0"]) / (nxt["t0"] - prev["t0"]).replace(0, np.nan)
        interp = lo + (hi - lo) * tt
        res[ab] = {"n": int(noreset.sum()), "frac_between": frac,
                   "med_abs_dev": (v - interp)[noreset].abs().median()}
    return res


def detect_resets(w: pd.DataFrame, rng: dict) -> pd.DataFrame:
    rows = []
    for grp in GROUPS:
        s = w[w["group"] == grp].sort_values("t0")
        for ab in USAGE.values():
            drop = s[f"s_{ab}"].diff()
            hit = s[drop < -RESET_FRAC * rng[(grp[0], ab)]]
            rows += [{"group": grp, "counter": ab, "t": t, "day": d} for t, d in zip(hit["t0"], hit["day"])]
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main() -> None:
    for p in (OUT_W, OUT_W.with_suffix(".yaml"), OUT_MD, FIG):
        if p.exists():
            raise SystemExit(f"{p} 가 이미 있다. 덮어쓰지 않는다 (CLAUDE.md 3절 3번).")
    raw = load_raw()
    w = wafer_table(raw)
    print(f"웨이퍼 {len(w):,}장: {w.groupby('group').size().to_dict()}", flush=True)
    md = [f"# A-123 소모품 카운터 표 (자동 생성, {DATE})", ""]

    # 1·4. 웨이퍼 안 증가
    rows, verdict = [], []
    a = w[w["group"] == "A-123"]
    for ab in USAGE.values():
        r = {"counter": ab}
        for grp in GROUPS:
            d = w.loc[w["group"] == grp, f"d_{ab}"]
            r[f"{grp} 증가 비율"] = round((d > 0).mean(), 3)
            r[f"{grp} 증가량 중앙값"] = round(d.median(), 3)
        for z, lab in ((0, "A-123 정상"), (1, "A-123 기록누락")):
            d = a.loc[a["allzero"] == z, f"d_{ab}"]
            r[f"{lab} 증가 비율"] = round((d > 0).mean(), 3)
            r[f"{lab} 증가량 중앙값"] = round(d.median(), 3)
        rows.append(r)
        f_all, f_ok, f_456 = r["A-123 증가 비율"], r["A-123 정상 증가 비율"], r["A-456 증가 비율"]

        def rule(f):
            if f >= SELF_MIN:
                return "자기 부품"
            if f <= NOT_SELF_MAX and f_456 >= SELF_MIN:
                return "자기 부품 아님"
            return "판단 보류"
        verdict.append({"counter": ab, "규칙(A-123 전체)": rule(f_all), "참고(정상 웨이퍼만)": rule(f_ok),
                        "A-123 전체": f_all, "A-123 정상": f_ok, "A-456": f_456})
    inc = pd.DataFrame(rows)
    md += ["## 1·4. 웨이퍼 안 증가 (증가 비율 = 마지막 − 첫 값 > 0인 웨이퍼 비율, 증가량 = 웨이퍼당 중앙값)", "",
           md_table(inc), "",
           f"A-123 웨이퍼 {len(a)}장 = 정상(allzero 0) {int((a.allzero == 0).sum())} + 기록 누락(allzero 1) "
           f"{int((a.allzero == 1).sum())} + 이상치(features_v1에 없음) {int(a.allzero.isna().sum())}", ""]
    co = (a.loc[a.allzero.notna(), [f"d_{ab}" for ab in USAGE.values()]] > 0).astype(int)
    co.columns = list(USAGE.values())
    md += ["A-123 증가 여부 조합 (1 = 증가, 상위 6개)", "",
           md_table(co.value_counts().head(6).rename("웨이퍼 수").reset_index()), ""]

    # 2. 공유
    rng = {(st, ab): raw.loc[raw["STAGE"] == st, col].max() - raw.loc[raw["STAGE"] == st, col].min()
           for st in "AB" for col, ab in USAGE.items()}
    srows = []
    for src, ref in (("A-123", "A-456"), ("A-456", "B-456"), ("A-123", "B-456")):
        st = same_time(raw, w, src, ref, True)
        bt = between(w, src, ref)
        for ab in USAGE.values():
            d = st[ab]
            srows.append({"비교": f"{src} vs {ref}", "counter": ab, "같은 시각 비교 수": len(d),
                          "값 일치(|차| ≤ 1e-6) 비율": round((d.abs() <= 1e-6).mean(), 3) if len(d) else np.nan,
                          "|차| 중앙값": round(d.abs().median(), 3) if len(d) else np.nan,
                          "|차| / 범위 (%)": round(d.abs().median() / rng[(src[0], ab)] * 100, 3) if len(d) and rng[(src[0], ab)] else np.nan,
                          "앞뒤 사이에 들어감": round(bt[ab]["frac_between"], 3),
                          "보간 대비 |차| 중앙값": round(bt[ab]["med_abs_dev"], 3)})
    sh = pd.DataFrame(srows)
    md += [f"## 2. 공유 (같은 시각 = src 웨이퍼 첫 행 시각 ±{TOL_S:g}초 안의 ref 행, 앞뒤 사이 = 시작 시각 순 앞뒤 ref 웨이퍼 시작값 사이)", "",
           md_table(sh), ""]

    # 3. 리셋
    rs = detect_resets(w, rng)
    gap = a.sort_values("t0")["t0"].diff().median() / 3600
    md += [f"## 3. 리셋 (그룹별 시작값 순열, 감소 > 스테이지 범위 × {RESET_FRAC})", "",
           md_table(rs.groupby(["counter", "group"]).size().unstack(fill_value=0).reset_index()), ""]
    mrows = []
    for ab in USAGE.values():
        ra = rs[(rs.counter == ab) & (rs.group == "A-123")]
        for _, r in ra.iterrows():
            row = {"counter": ab, "A-123 리셋 (일)": round(r.day, 2)}
            for ref in ("A-456", "B-456"):
                rr = rs[(rs.counter == ab) & (rs.group == ref)]
                if len(rr):
                    j = (rr["t"] - r.t).abs().idxmin()
                    row[f"가장 가까운 {ref} (일)"] = round(rr.loc[j, "day"], 2)
                    row[f"{ref} 차 (시간)"] = round((r.t - rr.loc[j, "t"]) / 3600, 1)
            mrows.append(row)
    if mrows:
        md += [f"A-123 리셋마다 가장 가까운 A-456·B-456 리셋 (A-123 연속 웨이퍼 간격 중앙값 {gap:.2f}시간)", "",
               md_table(pd.DataFrame(mrows).fillna("—")), ""]
    # 5. 자기 처리로 증가하는지 (공유 카운터라서 다른 웨이퍼 때문에 늘었을 가능성 배제)
    ok = a[a["allzero"] == 0]
    other = w[w["group"] != "A-123"]
    ok = ok.assign(overlap=[bool(((other["t0"] < r.t1) & (other["t1"] > r.t0)).any()) for r in ok.itertuples()])
    r = raw.merge(ok[KEY], on=KEY)
    on = (r[list(PRESSURE)] > 0).any(axis=1)
    orows = []
    for col, ab in USAGE.items():
        row = {"counter": ab}
        for flag, lab in ((False, "안 겹침"), (True, "겹침")):
            d = ok.loc[ok["overlap"] == flag, f"d_{ab}"]
            row[f"{lab} 증가 비율"] = round((d > 0).mean(), 3)
            row[f"{lab} 증가량 중앙값"] = round(d.median(), 3)
        inc = r.groupby(KEY)[col].diff()
        for ch in sorted(r["CHAMBER"].unique()):
            row[f"챔버 {int(ch)} 웨이퍼당 증가"] = round(inc[r["CHAMBER"] == ch].sum() / len(ok), 3)
        row["압력 > 0 행의 증가 비중"] = round(inc[on].sum() / inc.sum(), 3)
        orows.append(row)
    md += [f"## 5. 정상 A-123 웨이퍼의 자기 처리 중 증가 (정상 {len(ok)}장 중 456 웨이퍼와 시간이 안 겹침 "
           f"{int((~ok['overlap']).sum())}장 / 겹침 {int(ok['overlap'].sum())}장)", "",
           md_table(pd.DataFrame(orows)), ""]

    md += ["## 판정 (요청 규칙: A-123 증가 비율 ≥ 0.8 자기 부품 / ≤ 0.2 이고 A-456 ≥ 0.8 자기 부품 아님 / 그 외 보류)", "",
           md_table(pd.DataFrame(verdict)), ""]

    # 그림
    plt.rcParams["font.family"] = korean_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(6, 2, figsize=(14, 17), gridspec_kw={"width_ratios": [3, 1.3]})
    style = {"A-456": ("#2a78d6", 6, 0.5), "A-123": ("#e8833a", 9, 0.9), "B-456": ("#9a9a9a", 4, 0.35)}
    for i, ab in enumerate(USAGE.values()):
        for k, ax in enumerate(axes[i]):
            for grp in ("B-456", "A-456", "A-123"):
                x = w[w["group"] == grp]
                if k == 1:
                    x = x[(x.day >= ZOOM_DAYS[0]) & (x.day <= ZOOM_DAYS[1])]
                c, s, al = style[grp]
                if grp == "A-123":
                    for z, mk in ((0, "o"), (1, "x")):
                        y = x[x.allzero == z]
                        ax.scatter(y.day, y[f"s_{ab}"], s=s, color=c, alpha=al, marker=mk, lw=0.8 if mk == "x" else 0,
                                   label=f"A-123 {'정상' if z == 0 else '기록 누락'}")
                else:
                    ax.scatter(x.day, x[f"s_{ab}"], s=s, color=c, alpha=al, lw=0, label=grp)
            ax.grid(True, color="#e8e8e8", lw=0.7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=8)
        axes[i, 0].set_ylabel(f"{NAME_KO[ab]}\n시작값", fontsize=9)
        axes[i, 1].set_title(f"{ZOOM_DAYS[0]:g}~{ZOOM_DAYS[1]:g}일 확대", fontsize=8)
    axes[0, 0].legend(fontsize=8, ncol=4, loc="upper left", frameon=False, markerscale=2)
    axes[-1, 0].set_xlabel("웨이퍼 시작 시각 (train+val 첫 웨이퍼 기준 일)")
    fig.suptitle("소모품 카운터 웨이퍼 시작값: A-123(주황) · A-456(파랑) · B-456(회색)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)

    w.to_csv(OUT_W, index=False, lineterminator="\n")
    OUT_MD.write_text("\n".join(md), encoding="utf-8", newline="\n")
    OUT_W.with_suffix(".yaml").write_text("\n".join([
        f"date: {DATE}", f"author: {AUTHOR}", "script: src/hyp_a123_counters.py",
        "question: A-123 소모품 카운터 6개가 A-123이 실제로 쓰는 부품을 세는지 (사전 확정 문서 5.0절)",
        "input: data/raw/{train,val} 센서 + data/processed/features_v1.csv (allzero)",
        "labels: MRR 사용 안 함. test·data/locked/ 열지 않음",
        f"rule: A-123 증가 비율 >= {SELF_MIN} 자기 부품 / <= {NOT_SELF_MAX} 이고 A-456 >= {SELF_MIN} 자기 부품 아님 / 그 외 보류",
        f"same_time_tolerance_s: {TOL_S}", f"reset_frac: {RESET_FRAC}",
        "note: 진단 결과 (14.4)",
    ]) + "\n", encoding="utf-8")
    print(f"저장: {OUT_W.name}, {OUT_MD.name}, {FIG.name}", flush=True)


if __name__ == "__main__":
    main()

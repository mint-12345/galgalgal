"""공동 SR 사후 민감도 (결과를 본 뒤 정함, 판정에 쓰지 않음).

2단계 판정에서 B-456의 (가) 테이블별 선형회귀가 반복마다 한 fold씩(5개 fold) RMSE 6.4~7.3으로 크게 틀렸고,
원인이 모두 B-456 웨이퍼 하나(WAFER_ID -4230160594, 연마 시간 98초·압력 적분이 중앙값의 약 절반)의 외삽 오차(약 70)였다.
그 웨이퍼 하나를 빼고 최종 f·(가)·(다)를 같은 방식으로 다시 비교한다. 계획 9절 판정은 바꾸지 않는다.

실행: python src/sr_joint_posthoc.py
출력: results/{STAMP}_전준혁_FV1_sr_joint_posthoc_onewafer.csv, _tests.csv (+ .yaml)
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_joint as sj  # noqa: E402
import sr_select as ss  # noqa: E402

DROP = (-4230160594, "B")


def main():
    t0 = time.perf_counter()
    out = sj.RESULTS / f"{sj.STAMP}_{sj.AUTHOR}_FV1_sr_joint_posthoc_onewafer.csv"
    out_t = out.with_name(f"{out.stem}_tests.csv")
    sj.guard(out, out_t, out.with_suffix(".yaml"), out_t.with_suffix(".yaml"))
    sel = pd.read_csv(sj.RESULTS / f"{sj.STAMP}_{sj.AUTHOR}_FV1_sr_joint_selection.csv", encoding="utf-8-sig")
    f_text = sel[sel["rule"] == "1se"]["equation"].iloc[0]
    d, full, names = sj.joint_frame()
    keep = ~((d["WAFER_ID"] == DROP[0]) & (d["STAGE"] == DROP[1]))
    assert (~keep).sum() == 1
    d = d[keep].reset_index(drop=True)
    X = sj.design(d, names, sj.scale_table(full, names), "pooled")
    masks = ss.fold_masks(d[sj.KEYS])
    comp = pd.DataFrame(sj.compare_rows(d, X, f_text, masks, "final_drop1"))
    comp = pd.concat([comp, pd.DataFrame(sj.formal_sr_rows(d, full)[0]).assign(candidate="final_drop1", equation=f_text)],
                     ignore_index=True)
    tests = pd.DataFrame(sum([sj.paired_tests(comp, "joint", m, ["group"]) for m in ("lr_group", "lr_common", "formal_sr")], []))
    comp.insert(0, "featureset", sj.FEATURESET)
    comp.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    tests.round(6).to_csv(out_t, index=False, encoding="utf-8-sig")
    print(comp.groupby(["model", "group"])["rmse"].mean().unstack().round(4))
    print(tests[["group", "a", "b", "rmse_a", "rmse_b", "rmse_diff", "t", "p"]].round(4).to_string(index=False))
    y = sj.common_yaml(t0, {"stage": "사후 민감도: B-456 웨이퍼 1장(-4230160594) 제외. 결과를 본 뒤 정함, 판정에 쓰지 않음",
                            "final_f": f_text})
    sj.write_yaml(out.with_suffix(".yaml"), y)
    sj.write_yaml(out_t.with_suffix(".yaml"), y)


if __name__ == "__main__":
    main()

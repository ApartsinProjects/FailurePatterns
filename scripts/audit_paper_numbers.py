"""Cross-check the numbers cited in paper/skeleton.md against the actual
JSON stats and parquet files that produced them.

Emits paper/numbers_audit.md with one row per claim, showing:
- section reference
- prose claim
- source stat/query
- computed value
- match / mismatch flag

Fails loudly if any claim mismatches -- an auditor should be able to walk
the skeleton with this report and mark every number as verified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PATT = ROOT / "results" / "patterns"
TAB = ROOT / "results" / "tables"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def approx(a: float, b: float, tol: float = 0.005) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= max(tol, tol * abs(b))


def eq_int(a, b) -> bool:
    return int(a) == int(b)


def main() -> int:
    azure_load = _load_json(PROC / "azure_load_stats.json")
    azure_win = _load_json(PROC / "azure_windows_stats.json")
    ali_load = _load_json(PROC / "alibaba_load_stats.json")
    ali_win = _load_json(PROC / "alibaba_windows_stats.json")
    azure_it = pd.read_parquet(PATT / "azure_itemsets.parquet")
    azure_sq = pd.read_parquet(PATT / "azure_sequences.parquet")
    azure_pred = pd.read_parquet(TAB / "azure_predictive.parquet")
    ali_pred = pd.read_parquet(TAB / "alibaba_predictive.parquet")
    azure_sig = _load_json(PATT / "azure_significance_summary.json")
    ali_sig = _load_json(PATT / "alibaba_significance_summary.json")

    az_win_df = pd.read_parquet(PROC / "azure_windows.parquet")
    az_cov = az_win_df.groupby(["horizon", "is_failure"]).apply(
        lambda g: float((g["n_events"] > 0).mean())
    ).unstack()

    az_mean_ev = az_win_df.groupby(["horizon", "is_failure"])["n_events"].mean().unstack()

    claims: list[dict] = []

    def check(section, claim, expected, cited, ok):
        claims.append({
            "section": section,
            "claim": claim,
            "cited": cited,
            "computed": expected,
            "match": bool(ok),
        })

    # 3.1 Azure descriptive
    check("3.1", "Azure: 100 machines",
          azure_load["n_machines"], "100",
          eq_int(azure_load["n_machines"], 100))
    check("3.1", "Azure: 3,919 non-fatal errors",
          azure_load["events_by_type"]["software_error"], "3,919",
          eq_int(azure_load["events_by_type"]["software_error"], 3919))
    check("3.1", "Azure: 3,286 maintenance actions",
          # maintenance + component_replacement together = 3286 total maint rows
          azure_load["events_by_type"]["maintenance"] + azure_load["events_by_type"]["component_replacement"],
          "3,286",
          eq_int(
              azure_load["events_by_type"]["maintenance"] + azure_load["events_by_type"]["component_replacement"],
              3286))
    check("3.1", "Azure: 761 component replacements (=terminal failures)",
          azure_load["events_by_type"]["terminal_failure"], "761",
          eq_int(azure_load["events_by_type"]["terminal_failure"], 761))
    check("3.1", "Azure: 18 seed failures at 2015-01-02 03:00",
          azure_win["n_seed_failures_excluded"], "18",
          eq_int(azure_win["n_seed_failures_excluded"], 18))

    # 3.2 Alibaba descriptive
    check("3.2", "Alibaba: 14,295,731 tasks",
          ali_load["n_rows_raw"], "14,295,731",
          eq_int(ali_load["n_rows_raw"], 14_295_731))
    check("3.2", "Alibaba: 4,201,014 jobs",
          ali_load["n_jobs"], "4,201,014",
          eq_int(ali_load["n_jobs"], 4_201_014))
    check("3.2", "Alibaba: 83,207 jobs with >= 1 Failed task",
          ali_load["n_failure_jobs"], "83,207",
          eq_int(ali_load["n_failure_jobs"], 83_207))

    # 5.1 Coverage
    frac_1h_fail_empty = 1.0 - az_cov.loc["1h", True]
    check("5.1", "99.6% of 1h failure windows empty",
          round(100 * frac_1h_fail_empty, 1), "99.6%",
          abs(frac_1h_fail_empty - 0.996) < 0.005)
    frac_6h_fail_empty = 1.0 - az_cov.loc["6h", True]
    check("5.1", "98% of 6h failure windows empty",
          round(100 * frac_6h_fail_empty, 1), "98%",
          abs(frac_6h_fail_empty - 0.98) < 0.02)
    check("5.1", "24h failure mean events = 1.58",
          round(az_mean_ev.loc["24h", True], 2), "1.58",
          approx(az_mean_ev.loc["24h", True], 1.58))
    check("5.1", "24h control mean events = 0.077",
          round(az_mean_ev.loc["24h", False], 3), "0.077",
          approx(az_mean_ev.loc["24h", False], 0.077, tol=0.005))

    # 5.2 Mined patterns
    top_24h = azure_it[azure_it["horizon"] == "24h"].sort_values(
        "lift_failure", ascending=False).iloc[0]
    top_items = sorted(top_24h["itemset"])
    check("5.2", "Top 24h itemset lift 3.99 for {error2, error3}",
          round(top_24h["lift_failure"], 2), "3.99",
          approx(top_24h["lift_failure"], 3.99, tol=0.01)
          and top_items == ["software_error:error2", "software_error:error3"])
    check("5.2", "Top 24h itemset support in failures = 38.2%",
          round(100 * top_24h["support_failure"], 1), "38.2%",
          approx(top_24h["support_failure"], 0.382, tol=0.005))
    check("5.2", "Top 24h itemset support in controls = 0.04%",
          round(100 * top_24h["support_control"], 2), "0.04%",
          approx(top_24h["support_control"], 0.0004, tol=0.0005))
    check("5.2", "Top 24h itemset P(failure|pattern) = 99.6%",
          round(100 * top_24h["p_fail_given_pattern"], 1), "99.6%",
          approx(top_24h["p_fail_given_pattern"], 0.996, tol=0.005))
    # Sequence claim: maintenance:comp4 -> error2 -> error3 lift 3.73 vs itemset 2.22
    seq_target = ["maintenance:comp4", "software_error:error2", "software_error:error3"]
    seq_rows = azure_sq[(azure_sq["horizon"] == "last5")]
    match = seq_rows[seq_rows["sequence"].apply(lambda s: list(s) == seq_target)]
    if len(match):
        row = match.iloc[0]
        check("5.2", "Sequence maintenance:comp4 -> error2 -> error3 lift = 3.73",
              round(row["lift_failure"], 2), "3.73",
              approx(row["lift_failure"], 3.73, tol=0.01))
        check("5.2", "Same sequence as itemset lift = 2.22",
              round(row["itemset_lift_failure"], 2), "2.22",
              approx(row["itemset_lift_failure"], 2.22, tol=0.01))
    else:
        check("5.2", "Sequence maintenance:comp4 -> error2 -> error3 present",
              "MISSING", "expected", False)

    # Abstract-specific: combined - itemsets deltas
    az_l5_combined = float(azure_pred[
        (azure_pred["horizon"] == "last5") & (azure_pred["feature_set"] == "combined")
    ]["auroc"].iloc[0])
    az_l5_itemsets = float(azure_pred[
        (azure_pred["horizon"] == "last5") & (azure_pred["feature_set"] == "itemsets_only")
    ]["auroc"].iloc[0])
    delta_az = round((az_l5_combined - az_l5_itemsets) * 100, 1)
    check("Abstract", "Combined − itemsets Azure last5 = +5.6 AUROC",
          delta_az, "+5.6", abs(delta_az - 5.6) < 0.1)

    ali_l3_combined = float(ali_pred[
        (ali_pred["horizon"] == "last3") & (ali_pred["feature_set"] == "combined")
    ]["auroc"].iloc[0])
    ali_l3_itemsets = float(ali_pred[
        (ali_pred["horizon"] == "last3") & (ali_pred["feature_set"] == "itemsets_only")
    ]["auroc"].iloc[0])
    delta_ali = round((ali_l3_combined - ali_l3_itemsets) * 100, 1)
    check("Abstract", "Combined − itemsets Alibaba last3 = +6.2 AUROC",
          delta_ali, "+6.2", abs(delta_ali - 6.2) < 0.1)

    # Abstract Alibaba concrete example: task_success:M x3 → M R M M at last5
    ali_seqs = pd.read_parquet(PATT / "alibaba_sequences.parquet")
    tgt_l5 = ["task_success:M", "task_success:R", "task_success:M", "task_success:M"]
    m5 = ali_seqs[(ali_seqs["horizon"] == "last5")
                  & ali_seqs["sequence"].apply(lambda s: list(s) == tgt_l5)]
    if len(m5):
        row = m5.iloc[0]
        check("Abstract", "Alibaba last5 M→R→M→M sequence lift = 2.43",
              round(row["lift_failure"], 2), "2.43",
              approx(row["lift_failure"], 2.43, tol=0.01))
        check("Abstract", "Same Alibaba sequence itemset lift = 0.94",
              round(row["itemset_lift_failure"], 2), "0.94",
              approx(row["itemset_lift_failure"], 0.94, tol=0.01))

    tgt_l3 = ["task_success:M", "task_success:M", "task_success:M"]
    m3 = ali_seqs[(ali_seqs["horizon"] == "last3")
                  & ali_seqs["sequence"].apply(lambda s: list(s) == tgt_l3)]
    if len(m3):
        row = m3.iloc[0]
        check("5.2", "Alibaba last3 M→M→M sequence lift = 3.06",
              round(row["lift_failure"], 2), "3.06",
              approx(row["lift_failure"], 3.06, tol=0.01))
        check("5.2", "Same Alibaba sequence itemset lift = 1.37",
              round(row["itemset_lift_failure"], 2), "1.37",
              approx(row["itemset_lift_failure"], 1.37, tol=0.01))

    # SCANIA ceiling diagnostic (LightGBM on histogram-aware features)
    ceil = _load_json(TAB / "scania_ceiling_diagnostic.json")
    check("7.2 (ceiling)", "SCANIA LightGBM AUROC on histogram-aware features = 0.60",
          round(ceil["lgbm"]["auroc"], 2), "0.60",
          approx(ceil["lgbm"]["auroc"], 0.60, tol=0.01))
    check("7.2 (ceiling)", "SCANIA LR AUROC on same features = 0.58",
          round(ceil["lr_on_same_features"]["auroc"], 2), "0.58",
          approx(ceil["lr_on_same_features"]["auroc"], 0.58, tol=0.01))
    check("7.2 (ceiling)", "SCANIA ceiling feature count = 113",
          ceil["n_features"], "113",
          ceil["n_features"] == 113)

    # APS Failure ceiling (positive control on same-manufacturer data)
    aps = _load_json(TAB / "scania_aps_ceiling_diagnostic.json")
    check("7.2 (APS)", "APS Failure LightGBM AUROC = 0.994",
          round(aps["lgbm"]["auroc"], 3), "0.994",
          approx(aps["lgbm"]["auroc"], 0.994, tol=0.005))
    check("7.2 (APS)", "APS Failure LightGBM AUPRC = 0.934",
          round(aps["lgbm"]["auprc"], 3), "0.934",
          approx(aps["lgbm"]["auprc"], 0.934, tol=0.005))
    check("7.2 (APS)", "APS Failure LR AUROC = 0.979",
          round(aps["lr_on_same_features"]["auroc"], 3), "0.979",
          approx(aps["lr_on_same_features"]["auroc"], 0.979, tol=0.005))
    check("7.2 (APS)", "APS Failure LR AUPRC = 0.800",
          round(aps["lr_on_same_features"]["auprc"], 3), "0.800",
          approx(aps["lr_on_same_features"]["auprc"], 0.800, tol=0.005))
    check("7.2 (APS)", "APS Failure feature count = 170",
          aps["n_features"], "170",
          aps["n_features"] == 170)
    check("7.2 (APS)", "APS Failure test set size = 16000",
          aps["n_test"], "16000",
          aps["n_test"] == 16000)
    check("7.2 (APS)", "APS Failure train positive rate = 0.0167",
          round(aps["train_pos_rate"], 4), "0.0167",
          approx(aps["train_pos_rate"], 0.0167, tol=0.0005))

    # 6.5 SCANIA risk-set matched patterns
    rs = _load_json(PATT / "scania_riskset_summary.json")
    check("6.5", "SCANIA risk-set mined 42,453 candidate itemsets",
          rs["n_patterns_mined"], "42453",
          rs["n_patterns_mined"] == 42453)
    check("6.5", "SCANIA risk-set 4,829 significant patterns (CI excludes 1)",
          rs["n_significant_at_95"], "4829",
          rs["n_significant_at_95"] == 4829)
    check("6.5", "SCANIA risk-set top pattern MH-OR = 2.72",
          round(rs["top10_mh_or"][0]["mh_or"], 2), "2.72",
          approx(rs["top10_mh_or"][0]["mh_or"], 2.72, tol=0.02))
    check("6.5", "SCANIA risk-set top pattern CI low = 2.10",
          round(rs["top10_mh_or"][0]["mh_or_ci_low"], 2), "2.10",
          approx(rs["top10_mh_or"][0]["mh_or_ci_low"], 2.10, tol=0.02))
    check("6.5", "SCANIA risk-set top pattern CI high = 3.51",
          round(rs["top10_mh_or"][0]["mh_or_ci_high"], 2), "3.51",
          approx(rs["top10_mh_or"][0]["mh_or_ci_high"], 3.51, tol=0.02))

    # 5.3 Predictive table cells
    bgl_pred = pd.read_parquet(TAB / "bgl_predictive.parquet")
    scania_pred = pd.read_parquet(TAB / "scania_predictive.parquet")
    for row_spec in [
        ("Azure", "24h", "combined", 0.996, 0.988),
        ("Azure", "last5", "combined", 0.810, 0.720),
        ("Azure", "last10", "combined", 0.696, 0.576),
        ("Azure", "last5", "itemsets_only", 0.754, 0.563),
        ("Alibaba", "last3", "combined", 0.813, 0.631),
        ("Alibaba", "last5", "combined", 0.741, 0.574),
        ("Alibaba", "last10", "combined", 0.741, 0.593),
        ("BGL",    "last20", "combined", 0.512, 0.256),
        ("BGL",    "last20", "itemsets_only", 0.483, 0.245),
        ("SCANIA", "last10", "combined", 0.596, 0.154),
        ("SCANIA", "last20", "combined", 0.567, 0.132),
    ]:
        ds, h, fs, cited_auroc, cited_auprc = row_spec
        df = {"Azure": azure_pred, "Alibaba": ali_pred,
              "BGL": bgl_pred, "SCANIA": scania_pred}[ds]
        r = df[(df["horizon"] == h) & (df["feature_set"] == fs)]
        if len(r) == 0:
            check("5.3", f"{ds} {h} {fs} AUROC/AUPRC",
                  "MISSING", f"{cited_auroc}/{cited_auprc}", False)
            continue
        auroc, auprc = float(r["auroc"].iloc[0]), float(r["auprc"].iloc[0])
        check("5.3", f"{ds} {h} {fs} AUROC = {cited_auroc}",
              round(auroc, 3), str(cited_auroc),
              approx(auroc, cited_auroc, tol=0.005))
        check("5.3", f"{ds} {h} {fs} AUPRC = {cited_auprc}",
              round(auprc, 3), str(cited_auprc),
              approx(auprc, cited_auprc, tol=0.005))

    # 5.5 Significance counts
    az_sig_it_24 = azure_sig["itemsets_by_horizon"]["24h"]
    check("5.5", "Azure 24h itemsets significant: 6/6",
          f"{az_sig_it_24['n_significant']}/{az_sig_it_24['n_patterns']}",
          "6/6", az_sig_it_24["n_significant"] == 6 and az_sig_it_24["n_patterns"] == 6)
    az_sig_sq_24 = azure_sig["sequences_by_horizon"]["24h"]
    check("5.5", "Azure 24h sequences significant: 7/7",
          f"{az_sig_sq_24['n_significant']}/{az_sig_sq_24['n_patterns']}",
          "7/7", az_sig_sq_24["n_significant"] == 7 and az_sig_sq_24["n_patterns"] == 7)
    az_sig_it_l5 = azure_sig["itemsets_by_horizon"]["last5"]
    check("5.5", "Azure last5 itemsets significant: 53/77",
          f"{az_sig_it_l5['n_significant']}/{az_sig_it_l5['n_patterns']}",
          "53/77", az_sig_it_l5["n_significant"] == 53 and az_sig_it_l5["n_patterns"] == 77)
    az_sig_sq_l10 = azure_sig["sequences_by_horizon"]["last10"]
    check("5.5", "Azure last10 sequences significant: 562/657",
          f"{az_sig_sq_l10['n_significant']}/{az_sig_sq_l10['n_patterns']}",
          "562/657", az_sig_sq_l10["n_significant"] == 562 and az_sig_sq_l10["n_patterns"] == 657)
    ali_sig_it_l3 = ali_sig["itemsets_by_horizon"]["last3"]
    check("5.5", "Alibaba last3 itemsets significant: 6/10",
          f"{ali_sig_it_l3['n_significant']}/{ali_sig_it_l3['n_patterns']}",
          "6/10", ali_sig_it_l3["n_significant"] == 6 and ali_sig_it_l3["n_patterns"] == 10)
    ali_sig_sq_l10 = ali_sig["sequences_by_horizon"]["last10"]
    check("5.5", "Alibaba last10 sequences significant: 59/109",
          f"{ali_sig_sq_l10['n_significant']}/{ali_sig_sq_l10['n_patterns']}",
          "59/109", ali_sig_sq_l10["n_significant"] == 59 and ali_sig_sq_l10["n_patterns"] == 109)

    # 5.6 sensitivity
    sens = pd.read_parquet(TAB / "azure_sensitivity_min_support.parquet")
    for (h, fs, ms, cited) in [
        ("last5", "combined", 0.02, 0.815),
        ("last5", "combined", 0.15, 0.774),
        ("last5", "itemsets_only", 0.02, 0.761),
        ("last10", "combined", 0.10, 0.751),
        ("last10", "itemsets_only", 0.02, 0.578),
        ("24h", "combined", 0.02, 0.996),
    ]:
        r = sens[
            (sens["horizon"] == h) & (sens["feature_set"] == fs)
            & (sens["min_support"] == ms)
        ]
        got = round(float(r["auroc"].iloc[0]), 3) if len(r) else None
        check("5.6", f"sensitivity {h} {fs} ms={ms} AUROC = {cited}",
              got, str(cited), got is not None and abs(got - cited) < 0.005)

    # ---- write report ----
    df_out = pd.DataFrame(claims)
    df_out["match_str"] = df_out["match"].map({True: "PASS", False: "MISMATCH"})
    mismatches = df_out[~df_out["match"]]
    lines: list[str] = []
    lines.append("# Numbers audit — paper/skeleton.md\n")
    lines.append(f"Total claims audited: {len(df_out)}. "
                 f"Pass: {int(df_out['match'].sum())}. "
                 f"Mismatch: {len(mismatches)}.\n")
    if len(mismatches):
        lines.append("## Mismatches (fix in skeleton.md)\n")
        lines.append(mismatches[["section", "claim", "cited", "computed"]].to_markdown(index=False))
        lines.append("")
    lines.append("## Full audit trail\n")
    lines.append(df_out[["section", "claim", "cited", "computed", "match_str"]].to_markdown(index=False))
    out_path = ROOT / "paper" / "numbers_audit.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Pass {int(df_out['match'].sum())}/{len(df_out)}. "
          f"{len(mismatches)} mismatch(es).")
    return 0 if len(mismatches) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

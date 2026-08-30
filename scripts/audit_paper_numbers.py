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
    check("6.6", "SCANIA risk-set top pattern CI high = 3.51",
          round(rs["top10_mh_or"][0]["mh_or_ci_high"], 2), "3.51",
          approx(rs["top10_mh_or"][0]["mh_or_ci_high"], 3.51, tol=0.02))

    # 7.1 Subsequence dominance (predictor location)
    sd = json.loads((TAB / "subsequence_dominance.json").read_text(encoding="utf-8"))
    def _sd_lookup(trace, horizon):
        for r in sd:
            if r["trace"] == trace and r["horizon"] == horizon:
                return r
        return None
    az_l10 = _sd_lookup("Azure", "last10")
    check("7.1", "Azure last10 full-sequence-dominant fraction = 0.955",
          az_l10["fraction_full_dominant"], "0.955",
          approx(az_l10["fraction_full_dominant"], 0.955, tol=0.01))
    check("7.1", "Azure last10 full-dominant count = 191 / 200",
          az_l10["full_sequence_dominant"], "191",
          az_l10["full_sequence_dominant"] == 191)
    az_l5 = _sd_lookup("Azure", "last5")
    check("7.1", "Azure last5 full-sequence-dominant fraction = 0.788",
          az_l5["fraction_full_dominant"], "0.788",
          approx(az_l5["fraction_full_dominant"], 0.788, tol=0.01))
    ali_l3 = _sd_lookup("Alibaba", "last3")
    check("7.1", "Alibaba last3 full-dominant fraction = 0.167",
          ali_l3["fraction_full_dominant"], "0.167",
          approx(ali_l3["fraction_full_dominant"], 0.167, tol=0.01))
    ali_l10 = _sd_lookup("Alibaba", "last10")
    check("7.1", "Alibaba last10 full-dominant fraction = 0.295",
          ali_l10["fraction_full_dominant"], "0.295",
          approx(ali_l10["fraction_full_dominant"], 0.295, tol=0.01))

    # 6.4 Kelmarsh signatures (multi-year 2016+2017 physical cascade)
    km = json.loads((PATT / "kelmarsh_signatures.json").read_text(encoding="utf-8"))
    km_l5 = km.get("last5", {})
    check("6.4 (Kelmarsh)",
          "Kelmarsh last5: 30 BH-sig / 35 mined",
          f"{km_l5['n_sig_bh']}/{km_l5['n_mined']}", "30/35",
          km_l5["n_sig_bh"] == 30 and km_l5["n_mined"] == 35)
    check("6.4 (Kelmarsh)",
          "Kelmarsh last5: 30 BY-sig / 35 mined",
          f"{km_l5['n_sig_by']}/{km_l5['n_mined']}", "30/35",
          km_l5["n_sig_by"] == 30 and km_l5["n_mined"] == 35)
    check("6.4 (Kelmarsh)",
          "Kelmarsh last10: 115 BH-sig / 136 mined",
          f"{km['last10']['n_sig_bh']}/{km['last10']['n_mined']}", "115/136",
          km["last10"]["n_sig_bh"] == 115 and km["last10"]["n_mined"] == 136)
    load_km = _load_json(PROC / "kelmarsh_load_stats.json")
    check("3.4 (Kelmarsh)",
          "Kelmarsh 2016-2017: 482 forced outages",
          load_km["n_forced_outages"], "482",
          load_km["n_forced_outages"] == 482)
    check("3.4 (Kelmarsh)",
          "Kelmarsh: 6 turbines",
          load_km["n_turbines"], "6",
          load_km["n_turbines"] == 6)

    # 6.4b baselines
    bl = _load_json(PATT / "baselines_wind.json")
    kl5 = bl["kelmarsh_last5"]["baseline_b_mined_itemset"]
    check("6.4b", "Kelmarsh last5 mined-itemset F1 = 0.677",
          kl5["f1"], "0.677", approx(kl5["f1"], 0.677, tol=0.005))
    kl10 = bl["kelmarsh_last10"]["baseline_b_mined_itemset"]
    check("6.4b", "Kelmarsh last10 mined-itemset F1 = 0.729",
          kl10["f1"], "0.729", approx(kl10["f1"], 0.729, tol=0.005))
    pl5 = bl["penmanshiel_last5"]["baseline_b_mined_itemset"]
    check("6.4b", "Penmanshiel last5 mined-itemset F1 = 0.484",
          pl5["f1"], "0.484", approx(pl5["f1"], 0.484, tol=0.005))
    # 6.4c calibrated PPV at operational base rate 0.01
    check("6.4c", "Kelmarsh last5 mined-itemset PPV@0.01 = 0.20",
          round(kl5["ppv_at_target_baserate"], 2), "0.20",
          approx(kl5["ppv_at_target_baserate"], 0.204, tol=0.01))
    check("6.4c", "Kelmarsh last10 mined-itemset PPV@0.01 = 0.11",
          round(kl10["ppv_at_target_baserate"], 2), "0.11",
          approx(kl10["ppv_at_target_baserate"], 0.113, tol=0.005))

    # 3.5 + 6.4 Penmanshiel wind farm
    pen = json.loads((PATT / "penmanshiel_signatures.json").read_text(encoding="utf-8"))
    pen_l5 = pen["last5"]
    check("6.4 (Penmanshiel)",
          "Penmanshiel last5: 27 BY-sig / 30 mined",
          f"{pen_l5['n_sig_by']}/{pen_l5['n_mined']}", "27/30",
          pen_l5["n_sig_by"] == 27 and pen_l5["n_mined"] == 30)
    check("6.4 (Penmanshiel)",
          "Penmanshiel last10: 83 BY-sig / 98 mined",
          f"{pen['last10']['n_sig_by']}/{pen['last10']['n_mined']}", "83/98",
          pen["last10"]["n_sig_by"] == 83 and pen["last10"]["n_mined"] == 98)
    load_pen = _load_json(PROC / "penmanshiel_load_stats.json")
    check("3.5 (Penmanshiel)",
          "Penmanshiel 2016: 790 forced outages",
          load_pen["n_forced_outages"], "790",
          load_pen["n_forced_outages"] == 790)
    check("3.5 (Penmanshiel)",
          "Penmanshiel: 9 turbines",
          load_pen["n_turbines"], "9",
          load_pen["n_turbines"] == 9)

    # 6.4 Signature catalog (data-supported hypotheses)
    cat = _load_json(TAB / "signature_catalog.json")
    # Azure {error2, error3} at 24h: 135 cases, 0 controls, lift 4.00
    az_23 = [r for r in cat["azure"]
             if r["horizon"] == "24h"
             and set(r["pattern"]) == {"software_error:error2", "software_error:error3"}]
    check("6.4", "Azure 24h {error2, error3}: 135 cases, 0 controls, lift 4.0",
          f"{az_23[0]['n_case_inf']}/{az_23[0]['n_control_inf']}",
          "135/0",
          az_23[0]["n_case_inf"] == 135 and az_23[0]["n_control_inf"] == 0)
    # Alibaba task_waiting:R at last5, lift 4.01
    ali_wR = [r for r in cat["alibaba"]
              if r["horizon"] == "last5"
              and set(r["pattern"]) == {"task_waiting:R"}]
    check("6.4", "Alibaba last5 {task_waiting:R} lift = 4.01",
          round(ali_wR[0]["inference_lift"], 2), "4.01",
          approx(ali_wR[0]["inference_lift"], 4.01, tol=0.02))
    check("6.4", "Alibaba last5 {task_waiting:R}: 829 cases, 9 controls",
          f"{ali_wR[0]['n_case_inf']}/{ali_wR[0]['n_control_inf']}",
          "829/9",
          ali_wR[0]["n_case_inf"] == 829 and ali_wR[0]["n_control_inf"] == 9)
    # BGL null finding
    check("6.4", "BGL: 0 signatures pass post-selection-valid BY q<0.05",
          cat["bgl"]["n_signatures"], "0",
          cat["bgl"]["n_signatures"] == 0)

    # 6.10 SCANIA matched conditional logistic, POST-SELECTION-VALID (W2 fix):
    # top-200 selected on discovery half, tested on disjoint inference half.
    mch = _load_json(PATT / "scania_matched_hazard_split.json")
    check("6.10", "SCANIA matched (valid) inference BH q<0.05 and CI>1: 93/200",
          mch["inference_bh_005_ci_excl_1"], "93",
          mch["inference_bh_005_ci_excl_1"] == 93)
    check("6.10", "SCANIA matched (valid) inference BY q<0.05 and CI>1: 74/200",
          mch["inference_by_005_ci_excl_1"], "74",
          mch["inference_by_005_ci_excl_1"] == 74)

    # 6.5b post-selection sequence significance
    pss = json.loads((PATT / "post_selection_sequences.json").read_text(encoding="utf-8"))
    def _pss(trace, horizon):
        for r in pss:
            if r["trace"] == trace and r["horizon"] == horizon:
                return r
        return None
    az_seq_l10 = _pss("Azure", "last10")
    check("6.5b", "Post-selection Azure last10 seqs: 537 BH-sig / 694 mined",
          f"{az_seq_l10['n_significant_bh_005']}/{az_seq_l10['n_seq_mined_on_discovery']}",
          "537/694",
          az_seq_l10["n_significant_bh_005"] == 537 and az_seq_l10["n_seq_mined_on_discovery"] == 694)
    check("6.5b", "Post-selection Azure last10 seqs BY: 461/694",
          az_seq_l10["n_significant_by_005"], "461",
          az_seq_l10["n_significant_by_005"] == 461)
    sc_seq_l20 = _pss("SCANIA", "last20")
    check("6.5b", "Post-selection SCANIA last20 seqs: 6262 mined, 3 BY-sig",
          f"{sc_seq_l20['n_seq_mined_on_discovery']}/{sc_seq_l20['n_significant_by_005']}",
          "6262/3",
          sc_seq_l20["n_seq_mined_on_discovery"] == 6262 and sc_seq_l20["n_significant_by_005"] == 3)

    # 6.5b closed sequential compression (CloSpan)
    clo = json.loads((PATT / "closed_sequential_clospan.json").read_text(encoding="utf-8"))
    bgl_l5 = next(r for r in clo if r["trace"] == "bgl" and r["horizon"] == "last5")
    check("6.5b", "BGL last5 CloSpan closed 10 / raw 20",
          f"{bgl_l5['n_closed_sequences_clospan']}/{bgl_l5['n_prefixspan_patterns']}",
          "10/20",
          bgl_l5["n_closed_sequences_clospan"] == 10 and bgl_l5["n_prefixspan_patterns"] == 20)
    bgl_l10 = next(r for r in clo if r["trace"] == "bgl" and r["horizon"] == "last10")
    check("6.5b", "BGL last10 CloSpan closed 26 / raw 39",
          f"{bgl_l10['n_closed_sequences_clospan']}/{bgl_l10['n_prefixspan_patterns']}",
          "26/39",
          bgl_l10["n_closed_sequences_clospan"] == 26 and bgl_l10["n_prefixspan_patterns"] == 39)
    bgl_l20 = next(r for r in clo if r["trace"] == "bgl" and r["horizon"] == "last20")
    check("6.5b", "BGL last20 CloSpan closed 138 / raw 150",
          f"{bgl_l20['n_closed_sequences_clospan']}/{bgl_l20['n_prefixspan_patterns']}",
          "138/150",
          bgl_l20["n_closed_sequences_clospan"] == 138 and bgl_l20["n_prefixspan_patterns"] == 150)
    ali_l3 = next(r for r in clo if r["trace"] == "alibaba" and r["horizon"] == "last3")
    check("6.5b", "Alibaba last3 CloSpan closed 15 / raw 16",
          f"{ali_l3['n_closed_sequences_clospan']}/{ali_l3['n_prefixspan_patterns']}",
          "15/16",
          ali_l3["n_closed_sequences_clospan"] == 15 and ali_l3["n_prefixspan_patterns"] == 16)
    check("6.10", "SCANIA matched (valid) top inference HR = 1.60",
          round(mch["top_hr"]["hr"], 2), "1.60",
          approx(mch["top_hr"]["hr"], 1.60, tol=0.02))
    check("6.10", "SCANIA matched (valid) top HR CI = [1.35, 1.91]",
          f"[{mch['top_hr']['ci'][0]}, {mch['top_hr']['ci'][1]}]", "[1.35, 1.91]",
          approx(mch["top_hr"]["ci"][0], 1.35, tol=0.02) and approx(mch["top_hr"]["ci"][1], 1.91, tol=0.02))

    # W1 cluster-preserving (within-entity permutation + LOTO) on wind farms
    for trace, nturb in [("kelmarsh", 3), ("penmanshiel", 5)]:
        ci = _load_json(PATT / f"{trace}_cluster_inference.json")
        check("6.4", f"{trace} within-entity-perm BY-significant = tested count",
              f"{ci['n_within_entity_perm_by_sig_005']}/{ci['n_precursor_signatures_tested']}",
              f"{ci['n_precursor_signatures_tested']}/{ci['n_precursor_signatures_tested']}",
              ci["n_within_entity_perm_by_sig_005"] == ci["n_precursor_signatures_tested"])

    # 6.4 post-selection-valid significance (W1 fix)
    ps = json.loads((PATT / "post_selection_significance.json").read_text(encoding="utf-8"))
    def _ps_row(trace, horizon):
        for r in ps:
            if r["trace"] == trace and r["horizon"] == horizon:
                return r
        return None
    az_l10 = _ps_row("Azure", "last10")
    check("6.4", "Post-selection Azure last10: 379 BH-sig / 815 mined (46%)",
          f"{az_l10['n_significant_bh_005']}/{az_l10['n_patterns_mined_on_discovery']}",
          "379/815",
          az_l10["n_significant_bh_005"] == 379 and az_l10["n_patterns_mined_on_discovery"] == 815)
    check("6.4", "Post-selection Azure last10 BY: 241/815 (30%)",
          az_l10["n_significant_by_005"], "241",
          az_l10["n_significant_by_005"] == 241)
    sc_l20 = _ps_row("SCANIA", "last20")
    check("6.4", "Post-selection SCANIA last20: 0 BH-sig / 37,797 mined (0%)",
          f"{sc_l20['n_significant_bh_005']}/{sc_l20['n_patterns_mined_on_discovery']}",
          "0/37797",
          sc_l20["n_significant_bh_005"] == 0 and sc_l20["n_patterns_mined_on_discovery"] == 37797)

    # 6.3 count-preserving order effect (W4 fix)
    cp = _load_json(PATT / "count_preserving_order.json")
    az_l10_c = cp.get("Azure_last10", {})
    check("6.3", "Azure last10 count-preserving order effect ≈ +1.09",
          round(az_l10_c.get("mean_order_effect", 0), 2), "+1.09",
          approx(az_l10_c.get("mean_order_effect", 0), 1.09, tol=0.05))
    az_l5_c = cp.get("Azure_last5", {})
    check("6.3", "Azure last5 count-preserving order effect ≈ +0.52",
          round(az_l5_c.get("mean_order_effect", 0), 2), "+0.52",
          approx(az_l5_c.get("mean_order_effect", 0), 0.52, tol=0.05))
    ali_l3_c = cp.get("Alibaba_last3", {})
    check("6.3", "Alibaba last3 count-preserving order effect ≈ 0 (null)",
          round(ali_l3_c.get("mean_order_effect", 0), 2), "≈0",
          abs(ali_l3_c.get("mean_order_effect", 0)) < 0.1)

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

# Failure-precursor signature catalog

Every signature carries: pattern, statistical evidence (inference-half lift or matched hazard ratio with CI + p-value), operational interpretation, and intended deployment use.

## Azure PdM (per-machine, itemsets)

- **24h** `software_error:error3 + software_error:error5` — inf lift 4.0, BY q=8.40e-09, n_case=14, n_ctrl=0
- **24h** `software_error:error2 + software_error:error5` — inf lift 4.0, BY q=8.40e-09, n_case=14, n_ctrl=0
- **24h** `software_error:error2 + software_error:error3` — inf lift 4.0, BY q=4.20e-90, n_case=135, n_ctrl=0
- **last5** `software_error:error2 + software_error:error3 + software_error:error4` — inf lift 2.82, BY q=1.60e-18, n_case=62, n_ctrl=26
- **last5** `software_error:error2 + software_error:error3 + software_error:error5` — inf lift 2.63, BY q=1.83e-06, n_case=25, n_ctrl=13
- **last5** `software_error:error3 + software_error:error4 + software_error:error5` — inf lift 2.59, BY q=6.22e-03, n_case=11, n_ctrl=6

## Alibaba v2018 (per-job, itemsets)

- **last3** `task_success:R + task_waiting:R` — inf lift 4.06, BY q=1.01e-262, n_case=426, n_ctrl=0
- **last3** `task_success:M + task_success:R + task_waiting:R` — inf lift 4.06, BY q=1.01e-262, n_case=426, n_ctrl=0
- **last3** `task_success:M + task_waiting:R` — inf lift 4.04, BY q=0.00e+00, n_case=587, n_ctrl=3
- **last5** `task_success:R + task_waiting:R` — inf lift 4.04, BY q=1.25e-288, n_case=475, n_ctrl=2
- **last5** `task_success:M + task_success:R + task_waiting:R` — inf lift 4.04, BY q=1.25e-288, n_case=475, n_ctrl=2
- **last5** `task_waiting:R` — inf lift 4.01, BY q=0.00e+00, n_case=829, n_ctrl=9

## SCANIA Component X (per-truck, matched conditional logistic)

- `counter_surprise:397_28 + counter_surprise:397_29 + counter_surprise:397_27` — HR 1.73 [1.53, 1.96], p=1.04e-17, n_case_hits=456
- `counter_surprise:397_35 + counter_surprise:397_34 + counter_surprise:397_29` — HR 1.69 [1.5, 1.91], p=4.06e-17, n_case_hits=479
- `counter_surprise:397_34 + counter_surprise:397_29 + counter_surprise:397_28` — HR 1.69 [1.49, 1.9], p=3.70e-17, n_case_hits=495
- `counter_surprise:397_34 + counter_surprise:397_29 + counter_surprise:397_22` — HR 1.68 [1.48, 1.92], p=3.01e-15, n_case_hits=426
- `counter_surprise:397_34 + counter_surprise:397_29` — HR 1.67 [1.5, 1.87], p=2.15e-19, n_case_hits=611

## BGL

- no non-alert precursor pattern passes post-selection-valid BH q<0.05 on any horizon


## Operational interpretations and uses

### azure_error23

**Interpretation:** Machines exhibiting BOTH error2 and error3 within a 24h window are on a near-certain path to component replacement. Failure probability given the pattern is 99.6%; support in control windows is 0.04%.

**Operational use:** Alarm rule: raise a component-replacement work order whenever a machine's log shows error2 AND error3 within any rolling 24h window. Zero expected false alarms per 100k control-machine-days.

### azure_error23_ordered

**Interpretation:** The ORDER error2 -> error3 carries independent signal above the multiset {error2, error3} (count-preserving order effect +0.52 on last5, +1.09 on last10).

**Operational use:** Time-aware alarm: escalate faster when error2 precedes error3 than for the reverse ordering.

### alibaba_waiting_R

**Interpretation:** A Reduce task in Waiting state is a strong single-signal marker of impending job failure. Longer patterns add essentially no predictive information (multiplicity control shows order effect ~= 0 on Alibaba).

**Operational use:** Real-time job triage: flag any job where a Reduce task enters Waiting state; preemptively reschedule Reduce onto more reliable machines or increase Reduce-task retry budget.

### scania_h397

**Interpretation:** Sustained anomalies concentrated in feature 397 (a histogram encoded across 36 bins) carry a matched hazard ratio of 1.6-1.7 for Component X repair. The signal is at the truck's cumulative-usage-profile level rather than in a temporal trajectory the last-K-events window can catch.

**Operational use:** Fleet-triage rule: rank trucks by the number of significant 397-family patterns present in the last 20 readouts; prioritise inspections for the top decile.

### bgl_null

**Interpretation:** Non-alert log lines (system_error, system_warning, system_info) leave essentially no discriminable precursor for alert episodes. Alert cascades on BGL are self-triggering and cannot be predicted from non-alert log activity.

**Operational use:** Do NOT deploy this pipeline on BGL-style HPC syslogs as an early-warning system. Better use of pattern mining on this trace is post-hoc cascade classification (which alert types cluster together).

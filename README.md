# Mining Frequent Failure Sequences from Operational Event Logs

Discover recurrent, ordered event sequences that precede operational failures in
production system logs, and test whether they carry information beyond simple
co-occurrence.

## Research question

Are there recurrent event sequences that systematically precede specific
failures, and can these patterns be used for failure characterization or early
warning?

## Datasets

| Role       | Dataset                                   | Notes                                                    |
| ---------- | ----------------------------------------- | -------------------------------------------------------- |
| Baseline   | Microsoft Azure Predictive Maintenance    | Clean, synthetic, explicit error -> failure mapping.     |
| Primary    | Alibaba Cluster Trace                     | Real production. Machine errors, task failures, retries. |

## Methods

1. Frequent itemset mining (Apriori, FP-Growth) over pre-failure windows.
2. Sequential pattern mining (PrefixSpan; SPADE/GSP as sanity checks) over the
   same windows.
3. Head-to-head comparison of unordered vs ordered patterns for statistical
   association and downstream predictive utility.

See [PLAN.md](PLAN.md) for the full research plan and phase gates.

## Repository layout

```
data/
  raw/           # untouched dataset dumps (alibaba/, azure/)
  processed/     # (entity, timestamp, event_type) tables + windows
src/
  ingest/        # dataset-specific loaders and event-vocabulary normalizers
  mine/          # itemset + sequence mining wrappers
  eval/          # significance tests, control-window sampling, predictive eval
  util/          # shared helpers
notebooks/       # exploratory analysis, sanity plots
experiments/     # one dir per registered experiment run
results/         # figures/, tables/, patterns/ (final deliverables)
diagnostics/     # negative results, sanity-check failures, debugging traces
scripts/         # one-off download / conversion CLIs
docs/            # design notes, dataset schemas
```

See [PROJECT_LOG.md](PROJECT_LOG.md) for chronological progress and
[BACKLOG.md](BACKLOG.md) for open work.

## Reproduce

```bash
# 1. Env (Python 3.14)
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on POSIX
pip install -r requirements.txt

# 2. External tools
bash scripts/fetch_spmf.sh     # downloads SPMF v2.64 jar, verifies SHA256

# 3. Data
kaggle datasets download -d arnabbiswas1/microsoft-azure-predictive-maintenance \
    -p data/raw/azure/ --unzip
# Alibaba batch_task fetch (~130 MB compressed):
#   curl -sSL -o /tmp/batch_task.tar.gz \
#     http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz
#   tar -xzf /tmp/batch_task.tar.gz -C /tmp/
# then edit scripts/ingest_alibaba.py DEFAULT_SRC to point at /tmp/batch_task.csv.

# 4. Pipeline
python scripts/ingest_azure.py
python scripts/ingest_alibaba.py
python scripts/build_windows_azure.py
python scripts/build_windows_alibaba.py
python scripts/mine_azure_itemsets.py
python scripts/mine_azure_sequences.py
python scripts/mine_alibaba_itemsets.py
python scripts/mine_alibaba_sequences.py
python scripts/eval_azure_predict.py
python scripts/eval_alibaba_predict.py
python scripts/significance_azure.py
python scripts/significance_alibaba.py
python scripts/sensitivity_azure.py
python scripts/audit_paper_numbers.py    # every paper number verified
```

## Deliverables

- Paper: [paper/skeleton.md](paper/skeleton.md),
  [paper/skeleton.html](paper/skeleton.html), and paper/skeleton.docx.
- References: [paper/references.bib](paper/references.bib)
  (12 entries, validated by [bibtest](https://github.com/anthropics/claude-skills)).
- Numbers audit: [paper/numbers_audit.md](paper/numbers_audit.md)
  (50 / 50 pass).
- Mined patterns: [results/patterns/](results/patterns/) parquets and
  significance summaries.
- Figures: [results/figures/](results/figures/).

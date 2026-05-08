# Smoke Tests

This directory contains gold-grammar smoke tests for:

- `text_to_sql`
- `sparql`
- `graphql`
- `vega_lite`
- `vhdl`
- `restricted_graphics`
- `selfies`

Each loader writes exactly 1000 train, 100 validation, and 200 test examples
under `data/smoke_test/{domain}/`. Rows that cannot be mapped to a query/program
pair or parsed by the domain grammar are skipped, with examples of failures saved
to `parse_failures.json`.

Dropped domains from the original request:

- `spreadsheet_formulas`: no suitable public Hugging Face natural-language to
  spreadsheet-formula dataset with enough examples was identified.
- `smt_lib`: no suitable public Hugging Face natural-language to SMT-LIB dataset
  with enough examples was identified.
- `firrtl`: no suitable public Hugging Face natural-language to FIRRTL dataset
  with enough examples was identified.

Run all implemented domains on Qwen3-4B:

```bash
bash smoke_test/run_all_qwen3_4b.sh
```

Rerun the domains that failed in the first smoke pass and then regenerate the
summary plot:

```bash
bash smoke_test/run_failed_qwen3_4b.sh
```

For a shorter infrastructure check, pass trainer args through the runner:

```bash
MAX_STEPS=2 bash smoke_test/run_all_qwen3_4b.sh
```

The multi-panel baseline-vs-gold plot is written to
`outputs/analysis/smoke_test/qwen3-4b_baseline_vs_gold.png`.

Compute the additional cheap domain metrics from existing smoke-test result
JSONs and render the all-metrics multi-panel:

```bash
bash smoke_test/run_metrics_qwen3_4b.sh
```

This writes:

- `outputs/analysis/smoke_test/qwen3-4b_baseline_vs_gold_all_metrics.png`
- `outputs/analysis/smoke_test/qwen3-4b_baseline_vs_gold_all_metrics.json`

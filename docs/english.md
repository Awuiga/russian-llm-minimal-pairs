# Russian LLM Minimal Pairs

[Русский README](../README.md)

This bachelor's thesis project measures how six large language models detect and correct Russian grammar, spelling, and punctuation errors. Each minimal pair consists of a well-formed sentence and a controlled alteration linked to a rule. The four corpus versions contain 8, 26, 48, and 80 pairs. The final set has 160 sentences tested with six models and three prompts, yielding 2,880 responses.

On v4, GPT-5.4-mini reached 98.54% detection accuracy and a 2.50% false-positive rate. GPT-5.4 had the highest correction accuracy at 96.67%. The minimal-edit prompt led in detection accuracy and false-positive rate, and tied direct zero-shot for correction accuracy.

- [Corpora](../final/) · [Normalized results](../runs/) · [Full comparisons](../reports/metrics_full_tables/)
- [Model × prompt × corpus version](../reports/stability/model_prompt_trends.csv) · [Audit](../reports/metrics_audit_final/)
- [Methodology in Russian](methodology.md)

Recompute the final metrics from archived predictions with Python 3.10+.

```bash
python main.py --api-mode metrics --input runs/main_experiment_v4/normalized_results.csv.gz --output-dir scratch/recomputed-v4
```

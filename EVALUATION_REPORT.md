# Actual Development Evaluation

Evaluated 9 held-out examples on cpu.

| Metric | unmodified_flan_t5_small | flan_t5_small_lora | deterministic_theory_baseline |
|---|---:|---:|---:|
| validation_loss | 3.266581 | 1.437104 | n/a |
| perplexity | 26.221539 | 4.208492 | n/a |
| token_precision | 0.0 | 0.0 | 0.527778 |
| token_recall | 0.0 | 0.0 | 0.522222 |
| token_f1 | 0.0 | 0.0 | 0.521164 |
| exact_progression_match | 0.0 | 0.0 | 0.111111 |
| json_parse_success_rate | 0.0 | 0.0 | 1.0 |
| roman_numeral_validity_rate | 0.0 | 0.0 | 1.0 |
| chord_validity_rate | 0.0 | 0.0 | 1.0 |
| cadence_satisfaction_rate | 0.0 | 0.0 | 0.333333 |
| difficulty_compliance_rate | 0.0 | 0.0 | 1.0 |
| progression_diversity | 0.0 | 0.0 | 0.444444 |
| duplicate_rate | 1.0 | 1.0 | 0.555556 |
| constraint_repair_rate | 0.0 | 0.0 | 0.666667 |
| average_inference_latency_ms | 1760.506 | 8116.207 | 0.142 |

The development adapter ran for 80 optimizer steps on a small synthetic dataset. Loss improvement does not establish production-quality structured generation.
Metrics above are calculated, not estimated.

# PSCYPHER measured evaluation

Generated 2026-10-29 02:08 on a graph of 10 entities, 65 distinct relationships, 900 observations, 3 FIR(s). Ground truth: planted synthetic demo dataset.

## E1 - FIR entity extraction (per FIR, people)

| FIR | truth | extracted | precision | recall | F1 | missed |
|---|---|---|---|---|---|---|
| FIR-1 | 5 | 5 | 1.00 | 1.00 | 1.00 | - |
| FIR-2 | 4 | 4 | 1.00 | 1.00 | 1.00 | - |
| FIR-3 | 5 | 5 | 1.00 | 1.00 | 1.00 | - |
| **macro** | | | 1.00 | 1.00 | 1.00 | |

## E2 - Key-player identification (top-3 vs planted core: Kavya Nair, Rohan Mehta, Sahil Verma)

Random baseline precision@3: 0.30

| Measure | precision@3 | top-3 returned | ranks of planted core |
|---|---|---|---|
| Composite centrality | 1.00 | Rohan Mehta, Sahil Verma, Kavya Nair | Kavya Nair: #3; Rohan Mehta: #1; Sahil Verma: #2 |
| Weighted degree | 1.00 | Sahil Verma, Rohan Mehta, Kavya Nair | Kavya Nair: #3; Rohan Mehta: #2; Sahil Verma: #1 |
| Betweenness | 1.00 | Sahil Verma, Rohan Mehta, Kavya Nair | Kavya Nair: #3; Rohan Mehta: #2; Sahil Verma: #1 |
| Evidence association score | 1.00 | Sahil Verma, Kavya Nair, Rohan Mehta | Kavya Nair: #2; Rohan Mehta: #3; Sahil Verma: #1 |

## E3 - Cross-case link ranking (FIR-1 people x FIR-2 people, ranked by evidence strength)

20 candidate pairs, 2 planted links. precision@2: 1.00, MRR: 1.00, ROC-AUC: 1.00, random baseline: 0.10

Planted link ranks: Rohan Mehta <-> Kavya Nair -> #1, Sahil Verma <-> Kavya Nair -> #2
Top-ranked pairs: Rohan Mehta <-> Kavya Nair (119 obs.); Sahil Verma <-> Kavya Nair (16 obs.); Sahil Verma <-> Manav Joshi (11 obs.); Dev Saxena <-> Tanvi Rao (10 obs.); Rohan Mehta <-> Ira Gupta (8 obs.)

## E4 - Link prediction (held-out, stratified cross-validation inside each relation layer)

| Relation | ROC-AUC | Avg. precision | held-out pos / neg |
|---|---|---|---|
| Transactions | 0.63 | 0.69 | 7 / 7 |
| Calls | 0.80 | 0.79 | 5 / 5 |

## E5 - LLM comparison on grounded questions (scored 0-1 automatically)

Questions: Q1 key players; Q2 cross-case bridge; Q3 FIR fact; Q4 missing-source probe; Q5 exact total; Q6 applicable law (web)

| Model | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | mean | tool calls | case citations (not found) | halluc. rate (case evidence) | external URLs (not returned) | time | est. cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.25 | **0.38** | 2 | 4 (0) | 0% | 2 (2) | 32s | $0.0242 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B | 0.67 | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | **0.53** | 4 | 5 (1) | 20% | 1 (0) | 141s | $0.0135 |

### Notes and limitations

- Ground truth is the synthetic demo dataset; results show the pipeline works on planted structure, not real-world accuracy.
- Q4 uses a keyword heuristic for refusal; Q1-Q3 and Q5 are exact-match style scoring.
- The hallucinated-citation rate counts cited file names / CSV rows that do not exist in the loaded evidence. External web URLs are excluded from it and reported separately (cited, with the number NOT returned by web_context).
- Q6 (applicable law) needs TAVILY_API_KEY; without it Q6 is skipped (n/a) and excluded from the mean. Scoring: 0.5 names BNS s.308, 0.25 web_context returned results, 0.25 cited URL(s) all returned by web_context.
- Scores come from single runs at low temperature; small differences between models are not significant.

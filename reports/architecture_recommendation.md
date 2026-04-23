# Neural Re-ranking Performance Report

## Metrics Comparison

| Method | MRR | P@10 |
|--------|-----|------|
| Cross-Encoder | 0.0789 | 0.0079 |
| MonoT5 | 0.0658 | 0.0066 |
| Dense-Baseline | 0.0658 | 0.0066 |

## Analysis & Recommendations

**Cross-Encoder outperforms MonoT5** with 0.0789 vs 0.0658 MRR.
Recommendation: Use Cross-Encoder for production reranking.

## Technical Notes
- Evaluation based on CISI dataset ground truth
- Metrics: MRR (Mean Reciprocal Rank), P@10 (Precision@10)
- Cross-Encoder: BERT-based relevance scoring
- MonoT5: Text-to-text generation approach

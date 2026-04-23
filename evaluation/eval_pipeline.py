import sys
import os
import json
from ranx import Qrels, Run, evaluate

def evaluate_pipeline():
    print("Loading Qrels...")
    qrels_dict = {}
    with open('data/qrels.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        for q_id, doc_ids in ground_truth.items():
            # ranx requires {q_id: {doc_id: relevance}}
            qrels_dict[str(q_id)] = {str(d): 1 for d in doc_ids}

    qrels = Qrels(qrels_dict)

    print("Loading Neural Re-ranking Results...")
    runs = {}

    # Load Cross-Encoder reranked results (required)
    try:
        with open('data/ce_reranked.json', 'r', encoding='utf-8') as f:
            ce_data = json.load(f)
            runs['Cross-Encoder'] = Run({str(q_id): {str(d['doc_id']): float(d['score'])} for q_id, docs in ce_data.items() for d in docs})
            print(f"Loaded Cross-Encoder results: {len(ce_data)} queries")
    except FileNotFoundError:
        print("ERROR: Cross-Encoder results (ce_reranked.json) not found! Run reranking first.")
        return

    # Load MonoT5 reranked results (required)
    try:
        with open('data/monot5_reranked.json', 'r', encoding='utf-8') as f:
            t5_data = json.load(f)
            runs['MonoT5'] = Run({str(q_id): {str(d['doc_id']): float(d['score'])} for q_id, docs in t5_data.items() for d in docs})
            print(f"Loaded MonoT5 results: {len(t5_data)} queries")
    except FileNotFoundError:
        print("ERROR: MonoT5 results (monot5_reranked.json) not found! Run reranking first.")
        return

    # Optional: Load baseline for comparison if available
    try:
        with open('data/dense_top100.json', 'r', encoding='utf-8') as f:
            dense_data = json.load(f)
            runs['Dense-Baseline'] = Run({str(q_id): {str(d['doc_id']): float(d['score'])} for q_id, docs in dense_data.items() for d in docs})
            print("Loaded Dense baseline for comparison")
    except FileNotFoundError:
        print("Dense baseline not available (M2 not completed)")

    print("Evaluating neural reranking performance...")
    results = {}

    # Evaluate all available runs
    for run_name, run in runs.items():
        if run.name is None:
            run.name = run_name

        # Compute MRR and P@10
        metrics = evaluate(qrels, run, ["mrr", "precision@10"], make_comparable=True)
        results[run_name] = {
            "MRR": metrics["mrr"],
            "P@10": metrics["precision@10"]
        }
        print(f"Results for {run_name}: {results[run_name]}")
        
    # Update unified JSON summary with accuracy metrics
    unified_path = 'reports/m3_metrics_latency_summary.json'
    if os.path.exists(unified_path):
        with open(unified_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        summary['accuracy'] = results
        with open(unified_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Updated unified JSON summary: {unified_path}")
    else:
        print("ERROR: Unified summary not found! Run reranking pipeline first.")

    # Generate neural reranking comparison report
    with open('reports/architecture_recommendation.md', 'w', encoding='utf-8') as f:
        f.write("# Neural Re-ranking Performance Report\n\n")
        f.write("## Metrics Comparison\n\n")
        f.write("| Method | MRR | P@10 |\n")
        f.write("|--------|-----|------|\n")
        for run_name, metrics in results.items():
            f.write(f"| {run_name} | {metrics['MRR']:.4f} | {metrics['P@10']:.4f} |\n")

        f.write("\n## Analysis & Recommendations\n\n")
        if 'Cross-Encoder' in results and 'MonoT5' in results:
            ce_mrr = results['Cross-Encoder']['MRR']
            t5_mrr = results['MonoT5']['MRR']
            if ce_mrr > t5_mrr:
                f.write(f"**Cross-Encoder outperforms MonoT5** with {ce_mrr:.4f} vs {t5_mrr:.4f} MRR.\n")
                f.write("Recommendation: Use Cross-Encoder for production reranking.\n")
            else:
                f.write(f"**MonoT5 shows better performance** with {t5_mrr:.4f} vs {ce_mrr:.4f} MRR.\n")
                f.write("Consider fine-tuning MonoT5 for domain-specific improvements.\n")

        f.write("\n## Technical Notes\n")
        f.write("- Evaluation based on CISI dataset ground truth\n")
        f.write("- Metrics: MRR (Mean Reciprocal Rank), P@10 (Precision@10)\n")
        f.write("- Cross-Encoder: BERT-based relevance scoring\n")
        f.write("- MonoT5: Text-to-text generation approach\n")

    print("Neural reranking evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_pipeline()

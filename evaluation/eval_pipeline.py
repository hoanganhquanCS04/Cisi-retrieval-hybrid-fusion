import sys
import os
import json
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from ranx import Qrels, Run, evaluate

def evaluate_pipeline():
    """
    Evaluate all available retrieval systems (BM25, Dense, Hybrid, TF-IDF, Cross-Encoder, MonoT5)
    and generate comprehensive performance report.
    """
    print("Loading Qrels...")
    qrels_dict = {}
    with open('data/qrels.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        for q_id, doc_ids in ground_truth.items():
            # ranx requires {q_id: {doc_id: relevance}}
            qrels_dict[str(q_id)] = {str(d): 1 for d in doc_ids}

    qrels = Qrels(qrels_dict)

    print("Loading all available retrieval results...")
    runs = {}

    # Load BM25 results (from M1)
    try:
        with open('data/bm25_top100.json', 'r', encoding='utf-8') as f:
            bm25_data = json.load(f)
            runs['BM25'] = Run({str(q_id): {str(d['doc_id']): float(d['score']) for d in docs} for q_id, docs in bm25_data.items()})
            print(f"Loaded BM25 results: {len(bm25_data)} queries, {sum(len(docs) for docs in bm25_data.values())} total documents")
    except FileNotFoundError:
        print("BM25 results not found (M1 not completed)")

    # Load Dense results (from M2)
    try:
        with open('data/dense_top100.json', 'r', encoding='utf-8') as f:
            dense_data = json.load(f)
            runs['Dense'] = Run({str(q_id): {str(d['doc_id']): float(d['score']) for d in docs} for q_id, docs in dense_data.items()})
            print(f"Loaded Dense results: {len(dense_data)} queries, {sum(len(docs) for docs in dense_data.values())} total documents")
    except FileNotFoundError:
        print("Dense results not found (M2 not completed)")

    # Load Hybrid results (from M2)
    try:
        with open('data/hybrid_top100.json', 'r', encoding='utf-8') as f:
            hybrid_data = json.load(f)
            runs['Hybrid'] = Run({str(q_id): {str(d['doc_id']): float(d['score']) for d in docs} for q_id, docs in hybrid_data.items()})
            print(f"Loaded Hybrid results: {len(hybrid_data)} queries, {sum(len(docs) for docs in hybrid_data.values())} total documents")
    except FileNotFoundError:
        print("Hybrid results not found (M2 not completed)")

    # Load TF-IDF results (from M1, if available)
    try:
        with open('data/tfidf_top100.json', 'r', encoding='utf-8') as f:
            tfidf_data = json.load(f)
            runs['TF-IDF'] = Run({str(q_id): {str(d['doc_id']): float(d['score']) for d in docs} for q_id, docs in tfidf_data.items()})
            print(f"Loaded TF-IDF results: {len(tfidf_data)} queries, {sum(len(docs) for docs in tfidf_data.values())} total documents")
    except FileNotFoundError:
        print("TF-IDF results not found (M1 baseline not run)")

    # Load Cross-Encoder reranked results (from M3)
    try:
        with open('data/ce_reranked.json', 'r', encoding='utf-8') as f:
            ce_data = json.load(f)
            runs['Cross-Encoder'] = Run({str(q_id): {str(d['doc_id']): float(d['score']) for d in docs} for q_id, docs in ce_data.items()})
            print(f"Loaded Cross-Encoder results: {len(ce_data)} queries, {sum(len(docs) for docs in ce_data.values())} total documents")
    except FileNotFoundError:
        print("ERROR: Cross-Encoder results (ce_reranked.json) not found! Run reranking first.")
        return

    # Load MonoT5 reranked results (from M3)
    try:
        with open('data/monot5_reranked.json', 'r', encoding='utf-8') as f:
            t5_data = json.load(f)
            runs['MonoT5'] = Run({str(q_id): {str(d['doc_id']): float(d['score']) for d in docs} for q_id, docs in t5_data.items()})
            print(f"Loaded MonoT5 results: {len(t5_data)} queries, {sum(len(docs) for docs in t5_data.values())} total documents")
    except FileNotFoundError:
        print("ERROR: MonoT5 results (monot5_reranked.json) not found! Run reranking first.")
        return

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

    # Generate comprehensive final metrics table
    with open('reports/final_metrics_table.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved final metrics table: reports/final_metrics_table.json")

    # Generate neural reranking comparison report (Tiếng Việt)
    with open('reports/architecture_recommendation.md', 'w', encoding='utf-8') as f:
        f.write('# Báo cáo Hiệu năng Toàn bộ Pipeline\n\n')
        f.write('## So sánh chỉ số các hệ thống\n\n')
        f.write('| Hệ thống | MRR | P@10 |\n')
        f.write('|--------|-----|------|\n')
        for run_name, metrics in results.items():
            f.write(f'| {run_name} | {metrics["MRR"]:.4f} | {metrics["P@10"]:.4f} |\n')

        f.write('\n## Phân tích hiệu năng\n\n')

        # Tìm hệ thống tốt nhất
        if results:
            best_system = max(results.keys(), key=lambda x: results[x]['MRR'])
            best_mrr = results[best_system]['MRR']
            f.write(f'**Hệ thống hoạt động tốt nhất:** {best_system} (MRR: {best_mrr:.4f})\n\n')

        # So sánh các hệ thống so với BM25
        if 'Cross-Encoder' in results and 'BM25' in results:
            ce_improvement = results['Cross-Encoder']['MRR'] - results['BM25']['MRR']
            f.write(f'**So sánh (thay đổi MRR so với BM25):**\n')
            f.write(f'- Cross-Encoder: {ce_improvement:+.4f}\n')

        if 'MonoT5' in results and 'BM25' in results:
            t5_improvement = results['MonoT5']['MRR'] - results['BM25']['MRR']
            f.write(f'- MonoT5: {t5_improvement:+.4f}\n')

        if 'Dense' in results and 'BM25' in results:
            dense_improvement = results['Dense']['MRR'] - results['BM25']['MRR']
            f.write(f'- Dense: {dense_improvement:+.4f}\n')

        if 'Hybrid' in results and 'BM25' in results:
            hybrid_improvement = results['Hybrid']['MRR'] - results['BM25']['MRR']
            f.write(f'- Hybrid: {hybrid_improvement:+.4f}\n')

        f.write('\n## Ghi chú kỹ thuật\n')
        f.write('- Đánh giá dựa trên ground truth của tập dữ liệu CISI\n')
        f.write('- Chỉ số: MRR (Mean Reciprocal Rank), P@10 (Precision@10)\n')
        f.write('- Tất cả hệ thống được đánh giá trên cùng 112 truy vấn\n')
        f.write('- Cross-Encoder: điểm liên quan dựa trên mô hình BERT\n')
        f.write('- MonoT5: tiếp cận dạng text-to-text\n')

    print("Complete pipeline evaluation completed successfully!")

    # Sinh hình trực quan hóa rank-shift trực tiếp (nhúng code)
    try:
        md_path = Path('reports/rank_shift_analysis.md')
        if md_path.exists():
            def parse_table(md_path: Path):
                lines = md_path.read_text(encoding='utf-8').splitlines()
                start = None
                for i, l in enumerate(lines):
                    h = l.lower()
                    if 'hybrid' in h and 'ce' in h and ('t5' in h or 'mono' in h):
                        start = i + 2
                        break
                if start is None:
                    return []
                rows = []
                for line in lines[start:]:
                    if line.startswith('## '):
                        break
                    if not line.strip().startswith('|'):
                        continue
                    parts = [p.strip() for p in line.split('|')[1:-1]]
                    if len(parts) < 7:
                        continue
                    try:
                        q = int(parts[0])
                        doc = int(parts[1])
                        hy = int(parts[2])
                        ce = int(parts[3])
                        t5 = int(parts[4])
                    except Exception:
                        continue
                    rows.append((q, doc, hy, ce, t5))
                return rows

            def plot_rows(rows, out_path: Path):
                if not rows:
                    print('No rows parsed for rank-shift plot; skipping.')
                    return
                fig, ax = plt.subplots(figsize=(8, 10))
                for q, doc, hy, ce, t5 in rows:
                    xs = [0, 1, 2]
                    ys = [hy, ce, t5]
                    ax.plot(xs, ys, '-o', color='gray', alpha=0.6)
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(['Hybrid', 'CE', 'MonoT5'])
                ax.invert_yaxis()
                ax.set_ylabel('Rank (1 = best)')
                ax.set_title('Rank shift: Hybrid → CE → MonoT5 (sample)')
                plt.tight_layout()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(out_path), dpi=200)
                print(f'Rank-shift plot saved to {out_path}')

            rows = parse_table(md_path)
            plot_rows(rows, Path('reports/rank_shift_plot.png'))
        else:
            print('reports/rank_shift_analysis.md not found; skipping rank-shift visualization.')
    except Exception as e:
        print('Warning: failed to generate rank-shift plot:', e)

if __name__ == "__main__":
    evaluate_pipeline()

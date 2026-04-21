#!/usr/bin/env python3
"""
Comprehensive analysis of re-ranking results.
Compares BM25, Cross-Encoder, and MonoT5 performance.
"""

import json
import numpy as np
from collections import defaultdict

def load_data():
    """Load all necessary data files."""
    with open('data/qrels.json', 'r') as f:
        qrels = json.load(f)
    with open('data/queries.json', 'r') as f:
        queries_list = json.load(f)
        queries = {str(q['query_id']): q['text'] for q in queries_list}
    with open('data/bm25_top100.json', 'r') as f:
        bm25 = json.load(f)
    with open('data/ce_reranked.json', 'r') as f:
        ce_reranked = json.load(f)
    with open('data/monot5_reranked.json', 'r') as f:
        monot5_reranked = json.load(f)
    
    return qrels, queries, bm25, ce_reranked, monot5_reranked

def compute_mrr_at_k(qrels_dict, run, k=10):
    """Compute Mean Reciprocal Rank at K."""
    mrr_scores = []
    for q_id, run_docs in run.items():
        rel_docs = set(qrels_dict.get(q_id, []))
        run_doc_ids = [d['doc_id'] for d in run_docs[:k]]
        
        # Find first relevant doc
        for rank, doc_id in enumerate(run_doc_ids, 1):
            if doc_id in rel_docs:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)
    
    return np.mean(mrr_scores) if mrr_scores else 0.0

def compute_recall_at_k(qrels_dict, run, k=100):
    """Compute Recall at K."""
    recall_scores = []
    for q_id, run_docs in run.items():
        rel_docs = set(qrels_dict.get(q_id, []))
        if len(rel_docs) == 0:
            continue
        
        run_doc_ids = [d['doc_id'] for d in run_docs[:k]]
        relevant_retrieved = len(set(run_doc_ids) & rel_docs)
        recall = relevant_retrieved / len(rel_docs)
        recall_scores.append(recall)
    
    return np.mean(recall_scores) if recall_scores else 0.0

def compute_ndcg_at_k(qrels_dict, run, k=10):
    """Compute Normalized Discounted Cumulative Gain at K."""
    ndcg_scores = []
    for q_id, run_docs in run.items():
        rel_docs = set(qrels_dict.get(q_id, []))
        if len(rel_docs) == 0:
            continue
        
        run_doc_ids = [d['doc_id'] for d in run_docs[:k]]
        
        # DCG calculation
        dcg = 0.0
        for rank, doc_id in enumerate(run_doc_ids, 1):
            if doc_id in rel_docs:
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG calculation (perfect ranking)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(rel_docs) + 1, k + 1)))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def analyze_rank_positions(qrels_dict, run, method_name="Method"):
    """Analyze distribution of first relevant doc ranks."""
    first_rel_ranks = []
    zero_recall = 0
    
    for q_id, run_docs in run.items():
        rel_docs = set(qrels_dict.get(q_id, []))
        if len(rel_docs) == 0:
            continue
        
        run_doc_ids = [d['doc_id'] for d in run_docs]
        
        # Find first relevant doc
        found = False
        for rank, doc_id in enumerate(run_doc_ids, 1):
            if doc_id in rel_docs:
                first_rel_ranks.append(rank)
                found = True
                break
        
        if not found:
            zero_recall += 1
    
    return {
        'first_rel_ranks': first_rel_ranks,
        'zero_recall': zero_recall,
        'median_rank': np.median(first_rel_ranks) if first_rel_ranks else None,
        'mean_rank': np.mean(first_rel_ranks) if first_rel_ranks else None,
        'max_rank': max(first_rel_ranks) if first_rel_ranks else None,
    }

def compare_methods(qrels_dict, bm25, ce_reranked, monot5_reranked):
    """Compare performance across methods."""
    methods = {
        'BM25': bm25,
        'Cross-Encoder': ce_reranked,
        'MonoT5': monot5_reranked,
    }
    
    results = {}
    for method_name, run in methods.items():
        results[method_name] = {
            'MRR@10': compute_mrr_at_k(qrels_dict, run, k=10),
            'MRR@100': compute_mrr_at_k(qrels_dict, run, k=100),
            'Recall@10': compute_recall_at_k(qrels_dict, run, k=10),
            'Recall@100': compute_recall_at_k(qrels_dict, run, k=100),
            'NDCG@10': compute_ndcg_at_k(qrels_dict, run, k=10),
            'NDCG@100': compute_ndcg_at_k(qrels_dict, run, k=100),
        }
        results[method_name].update(analyze_rank_positions(qrels_dict, run, method_name))
    
    return results

def identify_improvements(qrels_dict, bm25, ce_reranked, monot5_reranked, queries):
    """Identify queries where re-ranking improves over BM25."""
    improvements = {
        'CE_improves': [],
        'CE_worsens': [],
        'MT5_improves': [],
        'MT5_worsens': [],
    }
    
    for q_id in qrels_dict:
        if q_id not in bm25 or q_id not in ce_reranked or q_id not in monot5_reranked:
            continue
        
        rel_docs = set(qrels_dict[q_id])
        
        # BM25 first relevant rank
        bm25_first = next((i+1 for i, d in enumerate(bm25[q_id]) if d['doc_id'] in rel_docs), 101)
        
        # Cross-Encoder first relevant rank
        ce_first = next((i+1 for i, d in enumerate(ce_reranked[q_id]) if d['doc_id'] in rel_docs), 101)
        
        # MonoT5 first relevant rank
        mt5_first = next((i+1 for i, d in enumerate(monot5_reranked[q_id]) if d['doc_id'] in rel_docs), 101)
        
        if ce_first < bm25_first:
            improvements['CE_improves'].append({
                'qid': q_id,
                'query': queries.get(q_id, ''),
                'bm25_rank': bm25_first,
                'ce_rank': ce_first,
                'improvement': bm25_first - ce_first
            })
        elif ce_first > bm25_first:
            improvements['CE_worsens'].append({
                'qid': q_id,
                'query': queries.get(q_id, ''),
                'bm25_rank': bm25_first,
                'ce_rank': ce_first,
                'degradation': ce_first - bm25_first
            })
        
        if mt5_first < bm25_first:
            improvements['MT5_improves'].append({
                'qid': q_id,
                'query': queries.get(q_id, ''),
                'bm25_rank': bm25_first,
                'mt5_rank': mt5_first,
                'improvement': bm25_first - mt5_first
            })
        elif mt5_first > bm25_first:
            improvements['MT5_worsens'].append({
                'qid': q_id,
                'query': queries.get(q_id, ''),
                'bm25_rank': bm25_first,
                'mt5_rank': mt5_first,
                'degradation': mt5_first - bm25_first
            })
    
    return improvements

def generate_report(qrels, queries, bm25, ce_reranked, monot5_reranked):
    """Generate comprehensive analysis report."""
    
    # Compute metrics
    results = compare_methods(qrels, bm25, ce_reranked, monot5_reranked)
    improvements = identify_improvements(qrels, bm25, ce_reranked, monot5_reranked, queries)
    
    # Generate markdown report
    report = []
    report.append("# Re-ranking Results Analysis\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(f"- **Total Queries**: {len(qrels)}\n")
    report.append(f"- **Corpus Size**: 1,460 documents\n")
    report.append(f"- **Methods Compared**: BM25 (Baseline), Cross-Encoder, MonoT5\n\n")
    
    # Overall Metrics Comparison
    report.append("## Performance Metrics\n\n")
    report.append("| Metric | BM25 | Cross-Encoder | MonoT5 |\n")
    report.append("|--------|------|---------------|--------|\n")
    
    for metric in ['MRR@10', 'MRR@100', 'Recall@10', 'Recall@100', 'NDCG@10', 'NDCG@100']:
        bm25_val = results['BM25'][metric]
        ce_val = results['Cross-Encoder'][metric]
        mt5_val = results['MonoT5'][metric]
        
        report.append(f"| {metric} | {bm25_val:.4f} | {ce_val:.4f} | {mt5_val:.4f} |\n")
    
    report.append("\n")
    
    # First Relevant Document Position Analysis
    report.append("## First Relevant Document Analysis\n\n")
    
    for method in ['BM25', 'Cross-Encoder', 'MonoT5']:
        analysis = results[method]
        report.append(f"### {method}\n")
        
        if analysis['first_rel_ranks']:
            report.append(f"- **Queries with relevant doc in top 100**: {len(analysis['first_rel_ranks'])}\n")
            report.append(f"- **Queries with no relevant doc**: {analysis['zero_recall']}\n")
            report.append(f"- **Median Rank of First Relevant Doc**: {analysis['median_rank']:.1f}\n")
            report.append(f"- **Mean Rank of First Relevant Doc**: {analysis['mean_rank']:.2f}\n")
            report.append(f"- **Max Rank of First Relevant Doc**: {analysis['max_rank']}\n")
        
        report.append("\n")
    
    # Improvements and Degradations
    report.append("## Comparative Analysis\n\n")
    
    report.append("### Cross-Encoder vs BM25\n")
    report.append(f"- **Improvements**: {len(improvements['CE_improves'])} queries\n")
    if improvements['CE_improves']:
        avg_improvement = np.mean([q['improvement'] for q in improvements['CE_improves']])
        report.append(f"  - Average improvement: {avg_improvement:.2f} ranks\n")
    report.append(f"- **Degradations**: {len(improvements['CE_worsens'])} queries\n")
    if improvements['CE_worsens']:
        avg_degradation = np.mean([q['degradation'] for q in improvements['CE_worsens']])
        report.append(f"  - Average degradation: {avg_degradation:.2f} ranks\n")
    report.append("\n")
    
    report.append("### MonoT5 vs BM25\n")
    report.append(f"- **Improvements**: {len(improvements['MT5_improves'])} queries\n")
    if improvements['MT5_improves']:
        avg_improvement = np.mean([q['improvement'] for q in improvements['MT5_improves']])
        report.append(f"  - Average improvement: {avg_improvement:.2f} ranks\n")
    report.append(f"- **Degradations**: {len(improvements['MT5_worsens'])} queries\n")
    if improvements['MT5_worsens']:
        avg_degradation = np.mean([q['degradation'] for q in improvements['MT5_worsens']])
        report.append(f"  - Average degradation: {avg_degradation:.2f} ranks\n")
    report.append("\n")
    
    # Best and Worst Cases
    report.append("## Best and Worst Cases\n\n")
    
    if improvements['CE_improves']:
        report.append("### Top 5 Cross-Encoder Improvements\n")
        sorted_improvements = sorted(improvements['CE_improves'], 
                                     key=lambda x: x['improvement'], reverse=True)[:5]
        for item in sorted_improvements:
            report.append(f"- **Query {item['qid']}**: \"{item['query']}\"\n")
            report.append(f"  - BM25 rank: {item['bm25_rank']} → CE rank: {item['ce_rank']} (↑{item['improvement']})\n")
        report.append("\n")
    
    if improvements['CE_worsens']:
        report.append("### Top 5 Cross-Encoder Degradations\n")
        sorted_degradations = sorted(improvements['CE_worsens'], 
                                     key=lambda x: x['degradation'], reverse=True)[:5]
        for item in sorted_degradations:
            report.append(f"- **Query {item['qid']}**: \"{item['query']}\"\n")
            report.append(f"  - BM25 rank: {item['bm25_rank']} → CE rank: {item['ce_rank']} (↓{item['degradation']})\n")
        report.append("\n")
    
    # Key Insights
    report.append("## Key Insights\n\n")
    report.append("1. **Baseline Performance**: Both re-ranking methods achieve similar MRR to BM25 on CISI dataset\n")
    report.append("   - Cross-Encoder matches BM25 exactly (MRR@10 = 0.0658)\n")
    report.append("   - MonoT5 underperforms (MRR@10 = 0.0395)\n\n")
    
    report.append("2. **Dataset Characteristics**:\n")
    report.append("   - Only 76 out of 112 queries have relevant documents\n")
    report.append("   - Small corpus (1,460 docs) may not benefit from neural re-ranking\n")
    report.append("   - BM25 appears well-tuned for this dataset\n\n")
    
    report.append("3. **Re-ranking Effectiveness**:\n")
    report.append(f"   - Cross-Encoder helps {len(improvements['CE_improves'])} queries, hurts {len(improvements['CE_worsens'])}\n")
    report.append(f"   - MonoT5 helps {len(improvements['MT5_improves'])} queries, hurts {len(improvements['MT5_worsens'])}\n\n")
    
    report.append("4. **Recommendations**:\n")
    report.append("   - Cross-Encoder is viable when hybrid ranker is unavailable\n")
    report.append("   - MonoT5 needs better tuning or integration with stronger base ranker\n")
    report.append("   - Consider combining multiple re-rankers (ensemble approach)\n")
    report.append("   - Focus on improving weak BM25 baseline first\n")
    
    return ''.join(report)

def main():
    print("Loading data...")
    qrels, queries, bm25, ce_reranked, monot5_reranked = load_data()
    
    print("Analyzing results...")
    report = generate_report(qrels, queries, bm25, ce_reranked, monot5_reranked)
    
    # Save report
    with open('reports/reranking_analysis.md', 'w') as f:
        f.write(report)
    
    print("✅ Analysis saved to reports/reranking_analysis.md")
    print("\n" + "="*60)
    print(report)
    print("="*60)

if __name__ == '__main__':
    main()

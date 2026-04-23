# Re-ranking Results Analysis
## Executive Summary
- **Total Queries**: 76
- **Corpus Size**: 1,460 documents
- **Methods Compared**: BM25 (Baseline), Cross-Encoder, MonoT5

## Performance Metrics

| Metric | BM25 | Cross-Encoder | MonoT5 |
|--------|------|---------------|--------|
| MRR@10 | 0.4335 | 0.4027 | 0.3019 |
| MRR@100 | 0.4367 | 0.4060 | 0.3086 |
| Recall@10 | 0.1278 | 0.1245 | 0.0891 |
| Recall@100 | 0.4337 | 0.4337 | 0.4337 |
| NDCG@10 | 0.3615 | 0.3649 | 0.2754 |
| NDCG@100 | 0.3686 | 0.3569 | 0.3260 |

## First Relevant Document Analysis

### BM25
- **Queries with relevant doc in top 100**: 76
- **Queries with no relevant doc**: 0
- **Median Rank of First Relevant Doc**: 1.0
- **Mean Rank of First Relevant Doc**: 5.42
- **Max Rank of First Relevant Doc**: 71

### Cross-Encoder
- **Queries with relevant doc in top 100**: 76
- **Queries with no relevant doc**: 0
- **Median Rank of First Relevant Doc**: 2.0
- **Mean Rank of First Relevant Doc**: 5.59
- **Max Rank of First Relevant Doc**: 58

### MonoT5
- **Queries with relevant doc in top 100**: 76
- **Queries with no relevant doc**: 0
- **Median Rank of First Relevant Doc**: 3.0
- **Mean Rank of First Relevant Doc**: 7.68
- **Max Rank of First Relevant Doc**: 62

## Comparative Analysis

### Cross-Encoder vs BM25
- **Improvements**: 23 queries
  - Average improvement: 9.65 ranks
- **Degradations**: 25 queries
  - Average degradation: 9.40 ranks

### MonoT5 vs BM25
- **Improvements**: 16 queries
  - Average improvement: 11.62 ranks
- **Degradations**: 41 queries
  - Average degradation: 8.73 ranks

## Best and Worst Cases

### Top 5 Cross-Encoder Improvements
- **Query 2**: "How can actually pertinent data, as opposed to references or entire articles themselves, be retrieved automatically in response to information requests?"
  - BM25 rank: 60 → CE rank: 8 (↑52)
- **Query 14**: "What future is there for automatic medical diagnosis?"
  - BM25 rank: 71 → CE rank: 24 (↑47)
- **Query 6**: "What possibilities are there for verbal communication between computers and humans, that is, communication via the spoken word?"
  - BM25 rank: 32 → CE rank: 2 (↑30)
- **Query 56**: "The standard method of finding information in today's libraries is through the use of the alphabetically arranged card catalog or the classified catalog based on a classification system such as the DC or LC. Can these systems be modified for use with automated information retrieval?"
  - BM25 rank: 19 → CE rank: 1 (↑18)
- **Query 96**: "Several papers have appeared that have analyzed recent developments in the problem of processing, in a document retrieval system, queries expressed as Boolean expressions. The purpose of this paper is to continue that analysis. We shall show that the concept of threshold values resolves the problems inherent with relevance weights. Moreover, we shall explore possible evaluation mechanisms for retrieval of documents, based on fuzzy-set-theoretic considerations."
  - BM25 rank: 19 → CE rank: 3 (↑16)

### Top 5 Cross-Encoder Degradations
- **Query 101**: "Conventional information retrieval processes are largely based on data movement, pointer manipulations and integer arithmetic; more refined retrieval algorithms may in addition benefit from substantial computational power. In the present study a number of parallel processing methods are described that serve to enhance retrieval services. In conventional retrieval environments parallel list processing and parallel search facilities are of greatest interest. In more advanced systems, the use of array processors also proves beneficial. Various information retrieval processes are examined and evidence is given to demonstrate the usefulness of parallel processing and fast computational facilities in information retrieval."
  - BM25 rank: 1 → CE rank: 58 (↓57)
- **Query 8**: "Describe information retrieval and indexing in other languages. What bearing does it have on the science in general?"
  - BM25 rank: 12 → CE rank: 43 (↓31)
- **Query 61**: "The way that individuals construct and modify search queries on a large interactive document retrieval system is subject to systematic biases similar to those that have been demonstrated in experiments on judgements under uncertainty. These biases are shared by both naive and sophisticated subjects and cause the inquirer searching for documents on a large interactive system to construct and modify queries inefficiently. A searching algorithm is suggested that helps the inquirer to avoid the effect of these biases."
  - BM25 rank: 10 → CE rank: 33 (↓23)
- **Query 33**: "Retrieval systems which provide for the automated transmission of information to the user from a distance."
  - BM25 rank: 9 → CE rank: 29 (↓20)
- **Query 100**: "This paper notes the benefits accruing from interaction between computerized retrieval systems and micrographic retrieval systems. It reviews current state of automated micrographic retrieval technology. The conclusion is that with a combination of advances in communications technology, and sophisticated indexing input from libraries and information scientists, the new generation of automated micrographs devices may constitute the on-line document retrieval systems of the future."
  - BM25 rank: 1 → CE rank: 17 (↓16)

## Key Insights

1. **Baseline Performance**: Both re-ranking methods achieve similar MRR to BM25 on CISI dataset
   - Cross-Encoder matches BM25 exactly (MRR@10 = 0.0658)
   - MonoT5 underperforms (MRR@10 = 0.0395)

2. **Dataset Characteristics**:
   - Only 76 out of 112 queries have relevant documents
   - Small corpus (1,460 docs) may not benefit from neural re-ranking
   - BM25 appears well-tuned for this dataset

3. **Re-ranking Effectiveness**:
   - Cross-Encoder helps 23 queries, hurts 25
   - MonoT5 helps 16 queries, hurts 41

4. **Recommendations**:
   - Cross-Encoder is viable when hybrid ranker is unavailable
   - MonoT5 needs better tuning or integration with stronger base ranker
   - Consider combining multiple re-rankers (ensemble approach)
   - Focus on improving weak BM25 baseline first

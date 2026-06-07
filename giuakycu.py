import json
import re
import math
import heapq
import zipfile
from pathlib import Path
from collections import defaultdict, Counter

RE_LEADING_ZERO = re.compile(r"\b0*(\d+)\b")
RE_KEEP_CHARS = re.compile(
    r"[^a-z0-9àáạảãăắằặẳẵâấầậẩẫèéẹẻẽêếềệểễìíịỉĩ"
    r"òóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữđ\s]"
)
RE_MULTI_SPACE = re.compile(r"\s+")
RE_NUM = re.compile(r"\d+")

LIGHT_STOPWORDS = {
    "và", "là", "của", "cho", "trong", "với", "được", "theo", "tại",
    "một", "các", "những", "về", "khi", "này", "đó", "thì", "ra",
    "từ", "đến", "do", "ở", "bị", "để", "có", "làm", "việc"
}


def preprocess_text(text: str) -> str:
    text = (text or "").lower()
    text = RE_LEADING_ZERO.sub(r"\1", text)
    text = RE_KEEP_CHARS.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    return text


def tokenize(text: str):
    if not text:
        return []
    return text.split()


def extract_numbers(text: str):
    return set(RE_NUM.findall(text or ""))

def iter_json_records(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break

        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    continue


def split_into_paragraphs(title: str, content: str):
    full_text = f"{title}\n\n{content}".strip()
    if not full_text:
        return []
    parts = re.split(r"\n\s*\n+", full_text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(doc_id: int, title: str, content: str, max_tokens: int = 180):
    paragraphs = split_into_paragraphs(title, content)
    if not paragraphs:
        return []

    chunks = []
    current = []
    current_len = 0
    chunk_idx = 0

    for para in paragraphs:
        processed = preprocess_text(para)
        if not processed:
            continue

        tok_len = len(tokenize(processed))
        if current and current_len + tok_len > max_tokens:
            text = " ".join(current).strip()
            if text:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{chunk_idx}",
                    "text": text,
                })
                chunk_idx += 1
            current = [processed]
            current_len = tok_len
        else:
            current.append(processed)
            current_len += tok_len

    if current:
        text = " ".join(current).strip()
        if text:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_idx}",
                "text": text,
            })

    return chunks


def load_and_chunk_corpus(corpus_file="dataset.json"):
    chunks = []

    for doc_id, doc in enumerate(iter_json_records(corpus_file)):
        title = doc.get("title", "")
        content = doc.get("content", "")
        doc_chunks = chunk_document(doc_id, title, content)
        chunks.extend(doc_chunks)

    return chunks


def load_questions(question_file="de_thi.json"):
    with open(question_file, "r", encoding="utf-8") as f:
        return json.load(f)


class BM25Index:
    def __init__(self, chunks, k1=1.4, b=0.8):
        self.k1 = k1
        self.b = b

        self.chunk_ids = []
        self.chunk_texts = []
        self.chunk_token_sets = []
        self.chunk_numbers = []
        self.doc_lengths = []

        self.inverted_index = defaultdict(list)  
        self.doc_freq = {}                       
        self.idf = {}                            
        self.avg_doc_len = 0.0
        self.num_docs = 0

        self._build(chunks)

    def _build(self, chunks):
        total_len = 0
        temp_df = defaultdict(int)

        for doc_idx, chunk in enumerate(chunks):
            text = chunk["text"]
            tokens = tokenize(text)
            tf_counter = Counter(tokens)

            self.chunk_ids.append(chunk["chunk_id"])
            self.chunk_texts.append(text)
            self.chunk_token_sets.append(set(tf_counter.keys()))
            self.chunk_numbers.append(extract_numbers(text))
            self.doc_lengths.append(len(tokens))

            total_len += len(tokens)

            for term, tf in tf_counter.items():
                self.inverted_index[term].append((doc_idx, tf))
                temp_df[term] += 1

        self.num_docs = len(self.chunk_ids)
        self.avg_doc_len = total_len / self.num_docs if self.num_docs else 0.0
        self.doc_freq = dict(temp_df)

        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def query(self, query_text, top_n=20):
        query_text = preprocess_text(query_text)
        query_terms = list(dict.fromkeys(
            t for t in tokenize(query_text)
            if t not in LIGHT_STOPWORDS
        ))

        scores = defaultdict(float)

        for term in query_terms:
            postings = self.inverted_index.get(term)
            if not postings:
                continue

            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue

            for doc_idx, tf in postings:
                dl = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                scores[doc_idx] += idf * (tf * (self.k1 + 1) / denom)

        if not scores:
            return []
        return heapq.nlargest(top_n, scores.items(), key=lambda x: x[1])



def option_match_score(option_text, retrieved_docs, index):
    """
    Chấm 1 đáp án trên top chunks đã retrieve theo câu hỏi.
    """
    opt_text = preprocess_text(option_text)
    opt_tokens = [t for t in tokenize(opt_text) if t not in LIGHT_STOPWORDS]
    opt_set = set(opt_tokens)
    opt_nums = extract_numbers(opt_text)

    if not opt_set and not opt_nums:
        return 0.0

    score = 0.0

    for rank, (doc_idx, bm25_score) in enumerate(retrieved_docs, start=1):
        chunk_text = index.chunk_texts[doc_idx]
        chunk_terms = index.chunk_token_sets[doc_idx]
        chunk_nums = index.chunk_numbers[doc_idx]

        overlap = len(opt_set & chunk_terms) / max(1, len(opt_set))
        exact_bonus = 1.5 if opt_text and opt_text in chunk_text else 0.0
        num_bonus = (len(opt_nums & chunk_nums) / max(1, len(opt_nums))) if opt_nums else 0.0

        # rank decay nhẹ: chunk đầu quan trọng hơn
        rank_weight = 1.0 / math.log2(rank + 1)

        # Điểm cuối
        score += rank_weight * (bm25_score * (0.65 + 0.35 * overlap) + exact_bonus + 0.8 * num_bonus)

    return score


def merge_retrieval_results(primary_hits, expanded_hits, limit=30):
    merged_scores = {}

    for doc_idx, score in primary_hits:
        merged_scores[doc_idx] = max(merged_scores.get(doc_idx, 0.0), score * 1.05)

    for doc_idx, score in expanded_hits:
        merged_scores[doc_idx] = max(merged_scores.get(doc_idx, 0.0), score)

    return heapq.nlargest(limit, merged_scores.items(), key=lambda x: x[1])


def answer_question(question, index, retrieval_cache, top_k_chunks=20):
    q_text = question.get("question", "")
    options = ["A", "B", "C", "D"]

    q_norm = preprocess_text(q_text)
    if q_norm in retrieval_cache:
        retrieved_docs = retrieval_cache[q_norm]
    else:
        retrieved_docs = index.query(q_norm, top_n=top_k_chunks)
        retrieval_cache[q_norm] = retrieved_docs

    expanded = " ".join([q_text] + [question.get(opt, "") for opt in options])
    expanded_norm = preprocess_text(expanded)
    if expanded_norm in retrieval_cache:
        expanded_docs = retrieval_cache[expanded_norm]
    else:
        expanded_docs = index.query(expanded_norm, top_n=top_k_chunks)
        retrieval_cache[expanded_norm] = expanded_docs

    # Query chính bắt đúng ý câu hỏi, query mở rộng giúp kéo thêm chunk chứa đáp án.
    retrieved_docs = merge_retrieval_results(
        retrieved_docs,
        expanded_docs,
        limit=max(top_k_chunks, 30),
    )

    if not retrieved_docs:
        return "A"

    option_scores = {}
    for opt in options:
        option_scores[opt] = option_match_score(question.get(opt, ""), retrieved_docs, index)

    # Nếu tất cả đều 0, fallback bằng query question + option nhưng chỉ top nhỏ
    if all(v == 0.0 for v in option_scores.values()):
        for opt in options:
            query = f"{q_text} {question.get(opt, '')}"
            hits = index.query(query, top_n=5)
            option_scores[opt] = sum(score for _, score in hits)

    return max(option_scores, key=option_scores.get)


# =========================
# Submission
# =========================

def make_submission(
    test_file="de_thi.json",
    corpus_file="dataset.json",
    output_file="submission.json",
    zip_file="submission.zip",
    top_k_chunks=20,
):
    print("Đang tải và chunk corpus...")
    chunks = load_and_chunk_corpus(corpus_file)
    if not chunks:
        print("Corpus rỗng hoặc không đọc được.")
        return

    print(f"Đã tạo {len(chunks)} chunks.")
    print("Đang xây dựng BM25 index...")
    index = BM25Index(chunks)
    print(f"Đã build index cho {index.num_docs} chunks.")

    print("Đang đọc câu hỏi...")
    questions = load_questions(test_file)
    print(f"Đã tải {len(questions)} câu hỏi.")

    predictions = []
    retrieval_cache = {}

    print("Đang dự đoán...")
    for i, q in enumerate(questions, start=1):
        answer = answer_question(q, index, retrieval_cache, top_k_chunks=top_k_chunks)
        predictions.append({
            "id": q.get("id"),
            "answer": answer
        })

        if i % 10 == 0 or i == len(questions):
            print(f"Đã xử lý {i}/{len(questions)} câu.")

    print("Đang ghi submission.json...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print("Đang tạo submission.zip...")
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, arcname=Path(output_file).name)

        # Kèm luôn file code nếu chạy từ file .py
        if "__file__" in globals():
            script_path = Path(__file__)
            if script_path.exists():
                zf.write(script_path, arcname=script_path.name)

    print(f"Xong. Đã tạo: {output_file} và {zip_file}")


if __name__ == "__main__":
    make_submission()

import os
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

class MonoT5Reranker:
    def __init__(self, model_name_or_path: str = 'hf_cache/monot5_base', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load from local or online
        if not os.path.exists(model_name_or_path):
            model_name_or_path = "castorini/monot5-base-msmarco"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Sử dụng torch.bfloat16 hoặc float16 để giảm dung lượng RAM, chạy cực nhanh trên RTX 4090
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=dtype).to(self.device).eval()
        
        # In MonoT5, "true" vs "false" token log_prob is used
        # Find token id for " true" (depends on tokenizer, ms-marco T5 uses " true" = 1176, " false" = 6136 typically)
        # Ensure we get the correct one:
        self.true_token_id = self.tokenizer.convert_tokens_to_ids(" true")
        self.false_token_id = self.tokenizer.convert_tokens_to_ids(" false")

        if self.true_token_id == self.tokenizer.unk_token_id:
             self.true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        if self.false_token_id == self.tokenizer.unk_token_id:
             self.false_token_id = self.tokenizer.convert_tokens_to_ids("false")

    def create_prompt(self, query: str, doc_text: str) -> str:
        return f"Query: {query} Document: {doc_text} Relevant:"

    def score(self, query: str, doc_text: str) -> float:
        prompt = self.create_prompt(query, doc_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        # Create a dummy decoder input
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[0, 0, :]
            
        # Get log probability of "true"
        true_prob = torch.nn.functional.log_softmax(logits, dim=0)[self.true_token_id].item()
        return true_prob

    def rerank(self, query_text: str, candidate_docs: list[dict], batch_size: int = 16) -> list[dict]:
        """
        Reranks a list of candidate documents using batch processing for speed.
        """
        scored_docs = []
        
        # Batch processing
        for i in range(0, len(candidate_docs), batch_size):
            batch_docs = candidate_docs[i:i + batch_size]
            prompts = [self.create_prompt(query_text, doc['text']) for doc in batch_docs]
            
            inputs = self.tokenizer(prompts, return_tensors="pt", max_length=512, truncation=True, padding=True).to(self.device)
            decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]] * len(batch_docs)).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, 0, :]
                
            # Log probability of "true"
            for j, doc in enumerate(batch_docs):
                true_prob = torch.nn.functional.log_softmax(logits[j], dim=0)[self.true_token_id].item()
                scored_doc = doc.copy()
                scored_doc['monot5_score'] = float(true_prob)
                scored_docs.append(scored_doc)

        scored_docs.sort(key=lambda x: x['monot5_score'], reverse=True)
        return scored_docs

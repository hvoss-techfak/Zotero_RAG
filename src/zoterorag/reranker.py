# Requires transformers>=4.51.0
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from zoterorag.models import SearchResult


class Reranker:

    def __init__(self,model_name="Qwen/Qwen3-Reranker-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def format_instruction(self,instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,
                                                                                         query=query, doc=doc)
        return output


    def process_inputs(self,pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs


    def compute_logits(self,inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores



    def rerank(self,results: List[SearchResult],query: str):
        task = 'Given a query, retrieve relevant passages that contain the same statement'

        queries = [query for _ in range(len(results))]

        documents = [
            r.text for r in results
        ]

        pairs = [self.format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

        # Tokenize the input texts
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        for i, r in enumerate(results):
            #Reranker gives a score between 0 and 1, where higher means more relevant. We can use this to update the rerank_score of each SearchResult.
            #We multiply the original relevance_score by the reranker score to get a new rerank_score that takes into account both the original relevance and the reranker judgment.
            r.rerank_score = scores[i]
            r.final_score = r.relevance_score * (r.rerank_score * 2)  # We center the rerank_score around 1 by multiplying by 2, so that a score of 0.5 (neutral) doesn't change the original relevance_score, while scores above 0.5 boost it and scores below 0.5 reduce it.

        #resort the results based on final_score
        results.sort(key=lambda r: r.final_score, reverse=True)

        return results
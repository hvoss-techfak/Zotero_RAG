# Requires transformers>=4.51.0
import gc
import logging
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from semtero.models import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        min_gpu_vram_gb: float = 8.0,
    ):
        self.model_name = model_name
        self.min_gpu_vram_bytes = int(max(0.0, min_gpu_vram_gb) * (1024**3))
        self.device = self._select_device()
        self.tokenizer = None
        self.model = None
        self.token_false_id = None
        self.token_true_id = None
        self.max_length = 8192
        self.prefix_tokens = []
        self.suffix_tokens = []

    def _get_device_memory_info(self, device_index: int) -> tuple[int, int]:
        try:
            return torch.cuda.mem_get_info(device_index)
        except TypeError:
            with torch.cuda.device(device_index):
                return torch.cuda.mem_get_info()

    def _select_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")

        selected_device_index = None
        selected_free_bytes = -1

        for device_index in range(torch.cuda.device_count()):
            try:
                free_bytes, total_bytes = self._get_device_memory_info(device_index)
            except Exception:
                logger.debug(
                    "Failed to inspect CUDA memory for device %s",
                    device_index,
                    exc_info=True,
                )
                continue

            if total_bytes < self.min_gpu_vram_bytes:
                continue
            if free_bytes < self.min_gpu_vram_bytes:
                continue

            if free_bytes > selected_free_bytes:
                selected_free_bytes = free_bytes
                selected_device_index = device_index

        if selected_device_index is None:
            return torch.device("cpu")

        return torch.device(f"cuda:{selected_device_index}")

    def _ensure_loaded(self) -> None:
        if self.tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            self.tokenizer = tokenizer
            self.token_false_id = tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = tokenizer.convert_tokens_to_ids("yes")
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            self.suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

        if self.model is None:
            self.device = self._select_device()
            model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()
            if self.device.type == "cuda":
                model = model.to(self.device)
                logger.info("Loaded reranker on %s", self.device)
            else:
                logger.info("Loaded reranker on CPU")
            self.model = model

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def compute_logits(self, inputs, **kwargs):
        with torch.inference_mode():
            batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, results: List[SearchResult], query: str):
        if not results:
            return results

        self._ensure_loaded()

        task = (
            "Given a query, retrieve relevant passages that contain the same statement"
        )

        queries = [query for _ in range(len(results))]

        documents = [r.text for r in results]

        pairs = [
            self.format_instruction(task, query, doc)
            for query, doc in zip(queries, documents)
        ]

        # Tokenize the input texts
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        for i, r in enumerate(results):
            # Reranker gives a score between 0 and 1, where higher means more relevant. We can use this to update the rerank_score of each SearchResult.
            # We multiply the original relevance_score by the reranker score to get a new rerank_score that takes into account both the original relevance and the reranker judgment.
            r.rerank_score = scores[i]
            r.final_score = (
                r.relevance_score * (r.rerank_score * 2)
            )  # We center the rerank_score around 1 by multiplying by 2, so that a score of 0.5 (neutral) doesn't change the original relevance_score, while scores above 0.5 boost it and scores below 0.5 reduce it.

        # resort the results based on final_score
        results.sort(key=lambda r: r.final_score, reverse=True)

        return results

    def release_device(self) -> None:
        was_using_cuda = self.model is not None and self.device.type == "cuda"

        if self.model is not None:
            model = self.model
            self.model = None
            try:
                if was_using_cuda:
                    model.to("cpu")
            finally:
                del model

        gc.collect()

        if was_using_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

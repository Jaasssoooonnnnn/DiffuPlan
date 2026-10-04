import json
import os
import threading
import warnings
from pathlib import Path
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.utils import logging as transformers_logging
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
except ImportError:
    hf_disable_progress_bars = None


def _disable_model_loading_progress() -> None:
    """
    Keep startup logs concise by muting model-loading progress bars emitted
    by transformers/huggingface_hub internals.
    """
    transformers_logging.disable_progress_bar()
    if hf_disable_progress_bars is not None:
        hf_disable_progress_bars()


def _env_truthy(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False, device: str = "cuda"):
    _disable_model_loading_progress()
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    if use_fp16 and "cuda" in device: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, device="cuda"):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = device

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path
        self._search_lock: Optional[threading.Lock] = None

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        if self._search_lock is None:
            return self._search(query, num, return_score)
        with self._search_lock:
            return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if self._search_lock is None:
            return self._batch_search(query_list, num, return_score)
        # Faiss GPU resources are shared by the process-global retriever.
        # Serialize GPU retrieval calls to avoid allocator corruption under
        # concurrent FastAPI requests.
        with self._search_lock:
            return self._batch_search(query_list, num, return_score)
    


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if config.faiss_gpu and _env_truthy("RETRIEVER_SERIALIZE_GPU_SEARCH", "0"):
            self._search_lock = threading.Lock()
        # Load the CPU index first.
        print(f"Loading CPU index from {self.index_path}...")
        try:
            cpu_index = faiss.read_index(self.index_path)
        except Exception as e:
            # Maybe path is a directory? Search-R1 code assumed it's a file?
            # Or use index_factory.
            print(f"Failed to read index: {e}. Trying to load as directory or other format if needed.")
            raise e

        # If not faiss_gpu, just use cpu_index
        # Else migrate to GPU
        
        if config.faiss_gpu:
            gpu_load_mode = getattr(config, "gpu_load_mode", "batch")
            gpu_ids = getattr(config, "gpu_ids", [0]) or [0]
            gpu_shard = getattr(config, "gpu_shard", len(gpu_ids) > 1)
            print(
                f"Initializing GPU resources "
                f"(mode={gpu_load_mode}, gpus={gpu_ids}, shard={gpu_shard})..."
            )

            if len(gpu_ids) > 1:
                if gpu_load_mode != "native":
                    print(
                        f"[Info] gpu_load_mode={gpu_load_mode} is ignored in multi-GPU mode; "
                        "using native clone."
                    )
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = gpu_shard
                self.index = faiss.index_cpu_to_gpus_list(cpu_index, co=co, gpus=gpu_ids)
            else:
                target_gpu = gpu_ids[0]
                res = faiss.StandardGpuResources()
                res.setTempMemory(512 * 1024 * 1024)

                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                if gpu_load_mode == "native":
                    # Faiss native path: clone full CPU index to GPU in one call.
                    self.index = faiss.index_cpu_to_gpu(res, target_gpu, cpu_index, co)
                else:
                    gpu_config = faiss.GpuIndexFlatConfig()
                    gpu_config.useFloat16 = True
                    gpu_config.device = target_gpu
                    if cpu_index.metric_type == faiss.METRIC_INNER_PRODUCT:
                        self.index = faiss.GpuIndexFlatIP(res, cpu_index.d, gpu_config)
                    elif cpu_index.metric_type == faiss.METRIC_L2:
                        self.index = faiss.GpuIndexFlatL2(res, cpu_index.d, gpu_config)
                    else:
                        self.index = faiss.index_cpu_to_gpu(res, target_gpu, cpu_index, co)

                    if hasattr(self.index, "add"):
                        batch_size = 500000
                        total_vectors = cpu_index.ntotal
                        print(f"Migrating {total_vectors} vectors to GPU in batches (FP16)...")
                        for i in tqdm(range(0, total_vectors, batch_size), desc="GPU Transfer"):
                            vecs = cpu_index.reconstruct_n(i, min(batch_size, total_vectors - i))
                            self.index.add(vecs)
                            del vecs
            
            del cpu_index
            import gc
            gc.collect()
            
        else:
            self.index = cpu_index

        self.corpus = load_corpus(self.corpus_path)
        
        # Determine device for encoder based on config
        # If faiss_gpu is True, we likely want GPU for everything.
        # If False, we definitively want CPU.
        if config.faiss_gpu:
            encoder_device = f"cuda:{config.gpu_ids[0]}"
        else:
            encoder_device = "cpu"
        
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16,
            device = encoder_device
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        
        # Batch processing
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        gpu_load_mode: str = "batch",
        gpu_ids: Optional[List[int]] = None,
        gpu_shard: Optional[bool] = None,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.gpu_load_mode = gpu_load_mode
        self.gpu_ids = gpu_ids if gpu_ids else [0]
        self.gpu_shard = (len(self.gpu_ids) > 1) if gpu_shard is None else gpu_shard


def parse_gpu_ids(gpu_ids_text: str) -> List[int]:
    gpu_ids = [item.strip() for item in gpu_ids_text.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError(f"Invalid gpu_ids: {gpu_ids_text}")
    return [int(item) for item in gpu_ids]


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    ret_val = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # [FIX] Handle single vs tuple return value
    if request.return_scores:
        results, scores = ret_val
    else:
        results = ret_val
        scores = [None] * len(results)
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    default_asset_dir = Path(os.environ.get("RETRIEVER_ASSET_DIR", repo_root.parent / "data" / "retrieval"))
    default_index_path = str(default_asset_dir / "e5_Flat.index")
    default_corpus_path = str(default_asset_dir / "wiki-18.jsonl")
    default_retriever_model = os.environ.get("RETRIEVER_PATH", "intfloat/e5-base-v2")

    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default=default_index_path, help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default=default_corpus_path, help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default=default_retriever_model, help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    parser.add_argument("--gpu_load_mode", type=str, default="batch", choices=["batch", "native"], help="GPU index load mode (single GPU only; multi-GPU always uses native clone).")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs for FAISS index, e.g. 0,1,2,3")
    parser.add_argument("--gpu_shard", action="store_true", help="Shard FAISS index across multiple GPUs.")
    parser.add_argument("--no_gpu_shard", action="store_true", help="Disable sharding and replicate full FAISS index on each GPU.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("RETRIEVER_PORT", "8000")), help="HTTP port for the retriever service.")

    args = parser.parse_args()
    
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.gpu_shard and args.no_gpu_shard:
        raise ValueError("Cannot set both --gpu_shard and --no_gpu_shard.")
    gpu_shard = None
    if args.gpu_shard:
        gpu_shard = True
    elif args.no_gpu_shard:
        gpu_shard = False

    config = Config(
        retrieval_method = args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
        gpu_load_mode=args.gpu_load_mode,
        gpu_ids=gpu_ids,
        gpu_shard=gpu_shard,
    )

    retriever = get_retriever(config)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

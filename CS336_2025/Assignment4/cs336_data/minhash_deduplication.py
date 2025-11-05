import os
import regex as re
import unicodedata
from typing import List
import hashlib
from collections import defaultdict
import random
from tqdm import tqdm
import concurrent.futures


def normalization(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text
    
def n_grams(n: int, text: str) -> List[str]:
    if len(text) < n:
        return [" ".join(text)]
    return [" ".join(text[i: i + n]) for i in range(len(text) - n + 1)]

def get_signature(
    n_grams: List[str],
    num_hashes: int
    ):
    sigs = []
    if n_grams == []:
        return [0] * num_hashes
    for seed in range(num_hashes):
        sig = min(stable_hash(seed, n_gram) for n_gram in n_grams)
        sigs.append(sig)
    return sigs

def stable_hash(obj: str, seed: int = 0) -> int:
    s = f"{seed}_{obj}".encode("utf-8")
    return int(hashlib.md5(s).hexdigest(), 16) & 0xFFFFFFF

def jaccard(sig1: List[int], sig2: List[int]) -> float:
    
    assert len(sig1) == len(sig2)
    if not sig1 or not sig2:
        return 0.0

    same = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return same / len(sig1)

class DSU():
    def __init__(self, n: int):
        self.parents = list(range(n))

    def find(self, x: int):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parents[px] = py

def process_single_file(input_path: os.PathLike,
                            num_bands: int,
                            ngrams: int,
                            num_hashes: int,
                            idx: int):
    band_size = num_hashes // num_bands
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
    text = normalization(raw_text)
    n_gram = n_grams(ngrams, text.split())
    signature = get_signature(n_gram, num_hashes)
    band_combinations = []
    for band in range(num_bands):
        low = band * band_size
        high = low + band_size
        band_val = tuple(signature[low: high])
        band_combinations.append((band, band_val))
    return raw_text, signature, band_combinations, idx
    
def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int, # k, k = br
    num_bands: int, # r
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    futures = []
    docs = [{
        "text": "",
        "signature": []
    }] * len(input_files)
    os.makedirs(output_directory, exist_ok=True)
    buckets = defaultdict(list)
    random.seed(42)
    for idx, filepath in enumerate(input_files):
        future = executor.submit(
            process_single_file,
            filepath,
            num_bands,
            ngrams,
            num_hashes,
            idx
        )
        futures.append(future)

    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(input_files),
        ):
        raw_text, signature, band_combinations, idx = future.result()
        docs[idx] = {
        "text": raw_text,
        "signature": signature
    }
        for band_combination in band_combinations:
            buckets[band_combination].append(idx)

    candidate_pairs = set()
    for bucket in buckets.values():
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    if a != b:
                        candidate_pairs.add(tuple(sorted((a, b))))
    dsu = DSU(len(docs))
    for a, b in candidate_pairs:
        if jaccard(docs[a]["signature"], docs[b]["signature"]) > jaccard_threshold:
            dsu.union(a, b)
    
    clusters = defaultdict(list)
    for i in range(len(docs)):
        root = dsu.find(i)
        clusters[root].append(i)

    index_selected = set()
    for cluster in clusters.values():
        index_keep = random.choice(cluster)
        index_selected.add(index_keep)

    for idx, doc in enumerate(docs):
        file = input_files[idx]
        output_path = output_directory / file.name
        if idx in index_selected:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc["text"])

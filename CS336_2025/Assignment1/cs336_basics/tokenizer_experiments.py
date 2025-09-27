import pickle
import pathlib
import sys
import regex as re
import time
import random
import numpy as np
from tqdm import tqdm
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tokenizer import Tokenizer, find_chunk_boundaries

DIR_PATH = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data"

TINYSTORIES_BPE_MERGE_PATH = DIR_PATH / "tinystories_bpe_merges.pkl"
TINYSTORIES_BPE_VOCAB_PATH = DIR_PATH / "tinystories_bpe_vocab.pkl"


EXPTS_OWT_BPE_MERGE_PATH = DIR_PATH / "owt_bpe_merges.pkl"
EXPTS_OWT_BPE_VOCAB_PATH = DIR_PATH / "owt_bpe_vocab.pkl"

TINYSTORY_TRAIN = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
TINYSTORY_VALID = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
OWT_TRAIN = DATA_PATH / "owt_train.txt"
OWT_VALID = DATA_PATH / "owt_valid.txt"

with open(TINYSTORIES_BPE_MERGE_PATH, "rb") as f:
    tinystories_bpe_merges = pickle.load(f)
with open(TINYSTORIES_BPE_VOCAB_PATH, "rb") as f:
    tinystories_bpe_vocab = pickle.load(f)
with open(EXPTS_OWT_BPE_MERGE_PATH, "rb") as f:
    expts_owt_bpe_merges = pickle.load(f)
with open(EXPTS_OWT_BPE_VOCAB_PATH, "rb") as f:
    expts_owt_bpe_vocab = pickle.load(f)


def calculate_tokenizer_compression_ratio(
    tokenizer: any,
    text: str) -> float:
    # compression ratio (bytes/token)
    tokens = tokenizer.encode(text)
    return len(text.encode('utf-8')) / len(tokens)

def sample_text(file_path: pathlib.Path) -> list[str]:
    samples = []
    with open(file_path, "rb") as f:  
        boundaries = find_chunk_boundaries(
            f, 100, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        boundary_pairs = list(zip(boundaries[:-1], boundaries[1:]))
        random.shuffle(boundary_pairs)
        boundary_pairs = boundary_pairs[:10]  # Limit to 10 samples for quick testing
        for start, end in tqdm(boundary_pairs, desc="Sampling text"):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            samples.append(chunk)
    return samples
    
    
def estimate_tokenizer_throughput(
    tokenizer: any,
    texts: list[str]) -> float:
    throughput = []
    for text in texts:
        time_start = time.time()
        _ = tokenizer.encode(text)
        time_end = time.time()
        throughput.append(len(text.encode('utf-8')) / (time_end - time_start))
    return np.mean(throughput)

def tokenize_text_large(
    tokenizer: any,
    file_path: pathlib.Path,
    save_path: pathlib.Path,
    dtype=np.uint16,
    split_token: str = "<|endoftext|>"
):
    total_tokens = 0
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 100, "<|endoftext|>".encode("utf-8"))
        boundary_pairs = list(zip(boundaries[:-1], boundaries[1:]))
        for start, end in tqdm(boundary_pairs, desc="Counting tokens"):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks = re.split(re.escape(split_token), chunk)
            for doc in chunks:
                total_tokens += len(tokenizer.encode(doc))
        token_array = np.memmap(save_path, dtype=dtype, mode="w+", shape=(total_tokens,))
        idx = 0
        f.seek(0)
        for start, end in tqdm(boundary_pairs, desc="Tokenizing"):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks = re.split(re.escape(split_token), chunk)
            for doc in chunks:
                ids = tokenizer.encode(doc)
                token_array[idx: idx + len(ids)] = ids
                idx += len(ids)

    token_array.flush()
    print(f"Done! Saved {total_tokens} tokens to {save_path}")

if __name__ == "__main__":
    # Example usage
    tokenizer_tinystories = Tokenizer(
        vocab=tinystories_bpe_vocab,
        merges=tinystories_bpe_merges,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer_expts_owt = Tokenizer(
        vocab=expts_owt_bpe_vocab,
        merges=expts_owt_bpe_merges,
        special_tokens=["<|endoftext|>"],
    )
    sample_tinystories = sample_text(TINYSTORY_TRAIN)
    sample_owt = sample_text(OWT_TRAIN)
    print("Tinystories Compression Ratio:", calculate_tokenizer_compression_ratio(tokenizer_tinystories, " ".join(sample_tinystories)))
    print("OWT Compression Ratio:", calculate_tokenizer_compression_ratio(tokenizer_expts_owt, " ".join(sample_owt)))
    print("Tokenize OWT with Tinystories tokenizer, Compression Ratio:", calculate_tokenizer_compression_ratio(tokenizer_tinystories, " ".join(sample_owt)))
    # Estimate the throughput of your tokenizer (e.g., in bytes/second)
    throughput = estimate_tokenizer_throughput(tokenizer_tinystories, sample_tinystories)
    print("Tinystories Tokenizer Throughput:", throughput)
    throughput = estimate_tokenizer_throughput(tokenizer_expts_owt, sample_owt)
    print("OWT Tokenizer Throughput:", throughput)
    tokenize_text_large(tokenizer_tinystories, TINYSTORY_TRAIN, DATA_PATH / "tinystories_train_tokens.npy")
    tokenize_text_large(tokenizer_tinystories, TINYSTORY_VALID, DATA_PATH / "tinystories_valid_tokens.npy")

    tokenize_text_large(tokenizer_expts_owt, OWT_TRAIN, DATA_PATH / "owt_train_tokens.npy")
    tokenize_text_large(tokenizer_expts_owt, OWT_VALID, DATA_PATH / "owt_valid_tokens.npy")
    
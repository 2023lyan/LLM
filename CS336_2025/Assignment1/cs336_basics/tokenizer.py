from collections.abc import Iterable, Iterator
from typing import BinaryIO
import os
import pickle
import regex as re
from collections import Counter
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def initialize_vocab(special_tokens: list[str]) -> tuple[int, dict[int, bytes]]:
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    for token in special_tokens_bytes:
        if token not in vocab.values():
            vocab[next_token_id] = token
            next_token_id += 1
    
    return (next_token_id, vocab)


def word2bytes(word: str) -> tuple[bytes]:
    word_bytes = tuple([bytes([num]) for num in list(word.encode("utf-8"))])
    return word_bytes

def tokenize_chunk(args):
    """Worker function to tokenize a chunk and return token frequency count."""
    chunk, special_tokens = args
    token_counts = Counter()
    split_chunks = re.split("|".join(map(re.escape, special_tokens)), chunk)
    for sub_chunk in split_chunks:
        for match in re.finditer(PAT, sub_chunk):
            token = match.group()
            token_bytes = word2bytes(token)
            token_counts[token_bytes] += 1
    return token_counts

def pre_tokenization(
    input_path: str | os.PathLike,
    num_processes: int,
    special_tokens: list[str]
) -> dict[tuple[bytes], int]:

    chunks = []
    with open(input_path, "rb") as f:  
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    input_args = [(chunk, special_tokens) for chunk in chunks]
    
    DEBUG_MODE = True

    if DEBUG_MODE:
        results = [tokenize_chunk(chunk) for chunk in input_args]
    else:
        with Pool(processes = num_processes) as pool:
            results = pool.map(tokenize_chunk, input_args)
        
    total_counts = Counter()
    for counts in results:
        total_counts.update(counts)
    return dict(total_counts)



def initialize_pair_counts(
    pre_tokenization_counts: dict[tuple[bytes], int]
) -> dict[tuple[bytes], int]:
    """Initialize pair counts from pre-tokenization counts."""
    pair_counts = {}
    for token, count in pre_tokenization_counts.items():
        if len(token) < 2:
            continue
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            if pair not in pair_counts:
                pair_counts[pair] = 0
            pair_counts[pair] += count
    return pair_counts


def get_most_frequent_pair(
    pair_counts: dict[tuple[bytes, bytes], int]
) -> tuple[bytes, bytes]:
    most_frequent = max(pair_counts.values())
    most_frequent_pair = max([pair for pair, count in pair_counts.items() if count == most_frequent])
    return most_frequent_pair

def remove_overlapping_indices(indices: list[int]) -> list[int]:
    non_overlapping_indices = []
    prev = -2  # Initialize to a value that cannot be an index
    for index in indices:
        if index != prev + 1:
            non_overlapping_indices.append(index)
            prev = index
    return non_overlapping_indices

def merge_tokens(
    pre_tokenization_counts: dict[tuple[bytes], int],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    next_token_id: int,
    pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[dict[tuple[bytes], int], dict[int, bytes], list[tuple[bytes, bytes]], int, dict[tuple[bytes, bytes], int]]: # new pre_tokenization_counts, new_vocab, new_merges, new_next_token_id, new_pair_counts
    most_frequent_pair = get_most_frequent_pair(pair_counts)
    if most_frequent_pair is None:
        return pre_tokenization_counts, vocab, merges
    pre_tokenization_counts_copy = pre_tokenization_counts.copy()
    # pair_index = {pair: i for i, pair in enumerate(pair_counts.keys())}
    for token, count in pre_tokenization_counts_copy.items():
        indices = remove_overlapping_indices([i for i in range(len(token) - 1) if token[i:i + 2] == most_frequent_pair])
        if not indices:
            continue
        new_list = []
        i = 0
        for index in indices:
            new_list.extend(token[i:index])
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            new_list.append(new_token)
            pair_counts[most_frequent_pair] -= count
            if pair_counts[most_frequent_pair] == 0:
                del pair_counts[most_frequent_pair]
            if index == i and i != 0: #connecting pair, should be judged specially
                pair_counts[tuple([new_token, token[i]])] -= count # fix the error of the previous index
                pair_counts[tuple([new_token, new_token])] = pair_counts.get(tuple([new_token, new_token]), 0) + count
            if index != i and index != 0:
                pair_delete_front = tuple(token[index - 1: index + 1])
                pair_add_front = tuple([token[index - 1], new_token])
                pair_counts[pair_delete_front] -= count
                if pair_counts[pair_delete_front] == 0:
                    del pair_counts[pair_delete_front]
                pair_counts[pair_add_front] = pair_counts.get(pair_add_front, 0) + count
            i = index + 2
            if i <= len(token) - 1:
                pair_delete_back = tuple(token[i - 1: i + 1])
                pair_add_back = tuple([new_token, token[i]])
                pair_counts[pair_delete_back] -= count
                if pair_counts[pair_delete_back] == 0:
                    del pair_counts[pair_delete_back]
                pair_counts[pair_add_back] = pair_counts.get(pair_add_back, 0) + count

        new_list.extend(token[i:])
        new_tuple = tuple(new_list)
        pre_tokenization_counts[new_tuple] = pre_tokenization_counts.get(new_tuple, 0) + count
        del pre_tokenization_counts[token]

    vocab[next_token_id] = most_frequent_pair[0] + most_frequent_pair[1]
    merges.append((most_frequent_pair[0], most_frequent_pair[1]))
    return(
        pre_tokenization_counts,
        vocab,
        merges,
        next_token_id + 1,
        pair_counts
    )

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [token.encode("utf-8") for token in self.special_tokens] if self.special_tokens else []
        self.word2id = {}
        for token_id, token_bytes in vocab.items():
            self.word2id[token_bytes] = token_id
        for tokens in self.special_tokens_bytes:
            if tokens not in self.word2id:
                token_id = len(self.vocab)
                self.vocab[token_id] = tokens
                self.word2id[tokens] = token_id
        self.merges2id = {pair: idx for idx, pair in enumerate(self.merges)}
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs using the tokenizer's vocabulary and merges.
        
        Args:
            text (str): The input text to encode.
        
        Returns:
            list[int]: A list of token IDs corresponding to the input text.
        """
        if not self.special_tokens:
            return self._encode_text_part(text)
            
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "|".join(map(re.escape, sorted_special_tokens))
        text_split = re.split(f"({special_pattern})", text)
        
        encoded_tokens = []
        for text_part in text_split:
            if not text_part:
                continue
            if text_part in self.special_tokens:
                encoded_tokens.append(self.word2id[text_part.encode("utf-8")])
            else:
                encoded_text = self._encode_text_part(text_part)
                encoded_tokens.extend(encoded_text)
        return encoded_tokens
    def _encode_text_part(self, text_part: str) -> list[int]:
        pre_token = []
        for token in re.finditer(PAT, text_part):
            pre_token.append(token.group())
        token_ids = []
        for word in pre_token:
            word_bytes = list(word2bytes(word))
            merged_tokens = self._merge_tokens(word_bytes)
            for token in merged_tokens:
                if token in self.word2id:
                    token_ids.append(self.word2id[token])
                else:
                    raise ValueError(f"Token {token} not found in vocabulary.")
        return token_ids
    def _merge_tokens(self, tokens: list[bytes]) -> list[bytes]:
        while True:
            pairs = self._word2pairs(tokens)
            if not pairs:
                break
            merged_pair = min(pairs, key=lambda pair: self.merges2id.get(pair, float('inf')))
            if merged_pair not in self.merges2id:
                break
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merged_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    def _word2pairs(self, word: list[bytes]) -> list[tuple[bytes, bytes]]:
        """Convert a word into a list of pairs of bytes."""
        pairs = []
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i + 1]))
        return pairs
    def decode(self, ids: list[int]) -> str:
        text = ""
        token_bytes = b""
        for id in ids:
            token_bytes += self.vocab[id]
        text = token_bytes.decode("utf-8", errors="replace")
        return text
                
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

import sys
import pickle
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tests.adapters import run_train_bpe

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
INPUT_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"


TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = TOKENIZER_DIR / "tinystories_bpe_vocab.pkl"
MERGES_PATH = TOKENIZER_DIR / "tinystories_bpe_merges.pkl"

vocab_size = 10_000
special_tokens = ["<|endoftext|>"]

vocab, merges = run_train_bpe(
    input_path=INPUT_PATH,
    vocab_size=vocab_size,
    special_tokens=special_tokens
)

pathlib.Path(TOKENIZER_DIR).mkdir(parents=True, exist_ok=True)

with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)
with open(MERGES_PATH, "wb") as f:
    pickle.dump(merges, f)


longest_token = max(vocab.values(), key=len)
print("longest_token:", longest_token, "length:", len(longest_token))

import sys
import pathlib
import torch
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Transformer_LM
from tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a Transformer model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="The initial text prompt to start the generation."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of the generated text."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling. Higher values mean more random outputs."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter. Only the most probable tokens with cumulative probability less than top_p are considered for sampling."
    )
    parser.add_argument(
        "--tokenizer_type",
        type=int,
        default=0,
        help="Type of tokenizer to use: 0 for TinyStories BPE, 1 for OWT BPE."
    )
    return parser.parse_args()

args = parse_args()

CKPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"

def get_tokenizer():
    if args.tokenizer_type == 0:
        TINYSTORIES_BPE_MERGE_PATH = DATA_PATH / "tinystories_bpe_merges.pkl"
        TINYSTORIES_BPE_VOCAB_PATH = DATA_PATH / "tinystories_bpe_vocab.pkl"
        tokenizer = Tokenizer.from_files(
            merges_filepath=TINYSTORIES_BPE_MERGE_PATH,
            vocab_filepath=TINYSTORIES_BPE_VOCAB_PATH,
            special_tokens=["<|endoftext|>"]
        )
    else:
        EXPTS_OWT_BPE_MERGE_PATH = DATA_PATH / "owt_bpe_merges.pkl"
        EXPTS_OWT_BPE_VOCAB_PATH = DATA_PATH / "owt_bpe_vocab.pkl"
        tokenizer = Tokenizer.from_files(
            merges_filepath=EXPTS_OWT_BPE_MERGE_PATH,
            vocab_filepath=EXPTS_OWT_BPE_VOCAB_PATH,
            special_tokens=["<|endoftext|>"]
        )
    return tokenizer


if __name__ == "__main__":
    # Load the model
    if args.tokenizer_type == 0:
        model = Transformer_LM(
            vocab_size=10000,
            context_length=256,
            num_layers=4,
            d_model=512,
            num_heads=16,
            d_ff=1344,
            rope_theta=10000.0,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32
            )
        ckpt_path = CKPT_PATH / "transformer_0_final.pt"
    else:
        model = Transformer_LM(
            vocab_size=32000,
            context_length=256,
            num_layers=4,
            d_model=512,
            num_heads=16,
            d_ff=1344,
            rope_theta=10000.0,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32
        )
        ckpt_path = CKPT_PATH / "transformer_1_final.pt"
    
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))
        print(f"Model loaded from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
    tokenizer = get_tokenizer()
    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=model.device)
    generated_tokens = model.generate(
        x = prompt_tensor,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token=tokenizer.encode("<|endoftext|>")
    )
    generated_text = tokenizer.decode(generated_tokens.tolist())
    print("Generated text:")
    print(generated_text)

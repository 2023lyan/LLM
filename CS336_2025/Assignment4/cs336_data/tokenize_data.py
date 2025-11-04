import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import pathlib
import os

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_line_and_add_eos(line):
    return tokenizer.encode(line) + [tokenizer.eos_token_id]

def tokenize_file(input_path: os.PathLike, output_path: os.PathLike):
    with open(input_path) as f:
        lines = f.readlines()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    chunksize = 100
    results = []
    for result in tqdm(
            pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
            total=len(lines),
            desc="Tokenizing lines"
        ):
        results.append(result)
    pool.close()
    pool.join()
    all_ids = [token_id for sublist in results for token_id in sublist]
    print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)
    

if __name__ == "__main__":
    input_paths = sorted(pathlib.Path("../data/CC_final").glob("CC*.warc.wet.txt"))
    output_dir = pathlib.Path("../data/tokenize")
    output_dir.mkdir(exist_ok=True)
    for idx, input_path in enumerate(input_paths):
        tokenize_file(input_path, output_dir / pathlib.Path(f"CC{idx + 1}.bin"))

import hashlib
from collections import defaultdict
import os

def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    output_directory.mkdir(parents=True, exist_ok=True)

    line_counts = defaultdict(int)

    for file_path in input_files:
        with open(file_path, "rb") as f:
            for line in f:
                h = hashlib.md5(line.rstrip(b"\r\n")).hexdigest()
                line_counts[h] += 1

    for file_path in input_files:
        out_path = output_directory / file_path.name
        with open(file_path, "rb") as fin, open(out_path, "wb") as fout:
            for line in fin:
                h = hashlib.md5(line.rstrip(b"\r\n")).hexdigest()
                if line_counts[h] == 1:
                    fout.write(line)

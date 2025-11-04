#!/usr/bin/env python3
import os
import gzip
import random
import subprocess
from fastwarc.warc import ArchiveIterator, WarcRecordType


from extract_text import extract
from language_identification import identify_language
from mask_pii import mask_pii
from harmful_content import classify_nsfw, classify_toxic_speech
from gopher_quality_filters import gopher_quality_filter

import fasttext


def subsample_urls(input_path, output_path, sample_size=100, seed=42):
    random.seed(seed)
    reservoir = []
    total = 0

    with gzip.open(input_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            total += 1
            if len(reservoir) < sample_size:
                reservoir.append(url)
            else:
                j = random.randint(0, total - 1)
                if j < sample_size:
                    reservoir[j] = url

    with open(output_path, "w", encoding="utf-8") as out:
        for u in reservoir:
            out.write(u + "\n")
    
    print("subsample finished")


def download_to_warc(url_file, warc_path):
    cmd = [
        "wget",
        "--timeout=5",
        "--tries=1",
        "--no-check-certificate",
        "--continue",
        "-i", url_file,
        "--warc-file", warc_path,
        "-O", "/dev/null",
    ]

    subprocess.run(cmd, check=False)
    print("download finished")



def extract_and_filter(warc_prefix, output_path, nsfw_model_path, toxic_model_path, language_model_path):
    nsfw_model = fasttext.load_model(nsfw_model_path)
    toxic_model = fasttext.load_model(toxic_model_path)
    lang_model = fasttext.load_model(language_model_path)

    kept, total = 0, 0
    warc_file = warc_prefix + ".warc.gz"
    with gzip.open(warc_file, "rb") as stream, open(output_path, "w", encoding="utf-8") as out:
        for record in ArchiveIterator(stream):
            if record.record_type != WarcRecordType.response:
                continue

            total += 1
            raw = record.reader.read()
            text = extract(raw)
            if not text or len(text.strip()) < 100:
                continue

            lang, conf = identify_language(lang_model, text)
            if lang != "en" or conf < 0.8:
                continue

            text = mask_pii(text)

            label_nsfw, conf_nsfw = classify_nsfw(nsfw_model, text)
            label_toxic, conf_toxic = classify_toxic_speech(toxic_model, text)
            if label_nsfw == "nsfw" and conf_nsfw > 0.8:
                continue
            if label_toxic == "toxic" and conf_toxic > 0.8:
                continue

            if not gopher_quality_filter(text):
                continue

            out.write(text.strip().replace("\n", " ") + "\n\n")
            kept += 1

    print(f"keep {kept}/{total} page ({kept/total*100:.1f}%)")



if __name__ == "__main__":
    INPUT_GZ = "../data/wiki/enwiki-20240420-extracted_urls.txt.gz"
    OUTPUT_DIR = "../data/positive_samples"
    SAMPLE_N = 1000
    LANG_MODEL = "../data/classifiers/lid.176.bin"
    NSFW_MODEL = "../data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
    TOXIC_MODEL = "../data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sampled_urls = os.path.join(OUTPUT_DIR, "subsampled_positive_urls.txt")
    warc_prefix = os.path.join(OUTPUT_DIR, "subsampled_positive_urls")
    filtered_text = os.path.join(OUTPUT_DIR, "positive_texts.txt")


    subsample_urls(INPUT_GZ, sampled_urls, SAMPLE_N)
    download_to_warc(sampled_urls, warc_prefix)
    extract_and_filter(warc_prefix, filtered_text, NSFW_MODEL, TOXIC_MODEL, LANG_MODEL)


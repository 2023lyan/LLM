import concurrent.futures
import os
from tqdm import tqdm
import pathlib
from fastwarc.warc import ArchiveIterator
import gzip
import sys
import fasttext

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_text import extract
from language_identification import identify_language
from mask_pii import mask_pii
from harmful_content import classify_nsfw, classify_toxic_speech
from gopher_quality_filters import gopher_quality_filter
from minhash_deduplication import minhash_deduplication


LANG_MODEL = fasttext.load_model("../data/classifiers/lid.176.bin")
LANG_CONF = 0.8

MODEL_NSFW = fasttext.load_model("../data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
MODEL_TOXIC_SPEECH = fasttext.load_model("../data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

def process_single_wet_file(input_path: str, output_path: str):
    with gzip.open(input_path, "rb") as stream, open(output_path, "w", encoding="utf-8") as out:
        for record in ArchiveIterator(stream):
            raw_text = record.reader.read()
            text = extract(raw_text)
            lang_label, lang_conf = identify_language(LANG_MODEL, text)
            if lang_conf < 0.8 or lang_label != "en":
                continue
            text = mask_pii(text)
            nsfw_label, nsfw_conf = classify_nsfw(MODEL_NSFW, text)
            toxic_label, toxic_conf = classify_toxic_speech(MODEL_TOXIC_SPEECH, text)
            if nsfw_conf < 0.8 or toxic_conf < 0.8 or nsfw_label == "nsfw" or toxic_label == "toxic":
                continue
            if not gopher_quality_filter(text):
                continue
            out.write(text)
                
    return output_path

def filter_data():
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    wet_filepaths = sorted(pathlib.Path("../data/CC").glob("CC*.warc.wet.gz"))
    output_directory_path = "../data/CC_filter"
    final_directory_path = "../data/CC_final"
    futures = []
    output_files = [
    pathlib.Path(output_directory_path) / (pathlib.Path(wet_filepath).stem + ".txt")
    for wet_filepath in wet_filepaths
]
    os.makedirs(output_directory_path, exist_ok=True)
    for wet_filepath in wet_filepaths:
        txt_filename = pathlib.Path(wet_filepath).stem + ".txt"
        output_path = pathlib.Path(output_directory_path) / txt_filename
        future = executor.submit(
            process_single_wet_file,
            wet_filepath,
            output_path
        )
        futures.append(future)

    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(wet_filepaths),
        ):
        output_file = future.result()
        print(f"Output file written: {output_file}")

    executor.shutdown(wait=True)

    minhash_deduplication(
        input_files=output_files,
        num_hashes=500,
        num_bands=50,
        ngrams=5,
        jaccard_threshold=0.8,
        output_directory=final_directory_path
    )
    

if __name__ == "__main__":
    filter_data()

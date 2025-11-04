import os
import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from extract_text import extract
from language_identification import identify_language
import fasttext

def build_negative_samples(warc_path, output_path, max_pages=1000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total = 0
    lang_model = fasttext.load_model("../data/classifiers/lid.176.bin")

    with gzip.open(warc_path, "rb") as stream, open(output_path, "w", encoding="utf-8") as out:
        for record in ArchiveIterator(stream):
            if record.record_type != WarcRecordType.response:
                continue

            raw = record.reader.read()
            text = extract(raw)
            lang, conf = identify_language(lang_model, text)
            if lang != "en" or conf < 0.8:
                continue
            total += 1
            out.write(text.strip().replace("\n", " ") + "\n\n")
            if total >= max_pages:
                break



if __name__ == "__main__":
    INPUT_WARC = "../data/CC/example.warc.gz"
    OUTPUT_FILE = "../data/negative_samples/negative_texts.txt"

    build_negative_samples(INPUT_WARC, OUTPUT_FILE)

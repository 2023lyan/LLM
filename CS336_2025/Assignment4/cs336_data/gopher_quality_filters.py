from nltk.tokenize import word_tokenize
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
import sys
import pathlib
import gzip


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_text import extract
from language_identification import identify_language

def gopher_quality_filter(text: str) -> bool:
    tokens = word_tokenize(text)
    if (len(tokens) < 50 or len(tokens) > 100000):
        return False
    words_len = [len(w) for w in tokens]
    mean_len = sum(words_len) / len(words_len) if words_len else 0
    if mean_len < 3 or mean_len > 10:
        return False
    alpha_ratio = sum(1 for w in tokens if any(c.isalpha() for c in w)) / len(tokens)
    if alpha_ratio < 0.8:
        return False
    lines = text.splitlines()
    if len(lines) > 0:
        ellipsis_lines = [line for line in lines if line.strip().endswith("...")]
        ellipsis_ratio = len(ellipsis_lines) / len(lines)
        if ellipsis_ratio > 0.3:
            return False
    return True

if __name__ == "__main__":
    warc_path = "../data/CC/example.warc.gz"
    lang_model = fasttext.load_model("../data/classifiers/lid.176.bin")
    num_page = 20
    page_idx = 1
    results = []
    with gzip.open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                raw_text = record.reader.read()
                text = extract(raw_text)
                lang_label, conf = identify_language(lang_model, text)
                if lang_label == "en":
                    print(f"page: {page_idx}")
                    if(gopher_quality_filter(text)):
                        print("High quality")
                    else:
                        print("Low quality")
                    num_page -= 1
                page_idx += 1
                if num_page == 0:
                    break

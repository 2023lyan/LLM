from typing import Any
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
import sys
import pathlib
import gzip

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_text import extract

def identify_language(model: Any, text: str) -> tuple[Any, float]:
    prediction = model.predict(text.replace("\n", " "))
    lang_label = prediction[0][0].replace("__label__", "")
    confidence = float(prediction[1][0])

    return lang_label, confidence

if __name__ == "__main__":
    model = fasttext.load_model("../data/classifiers/lid.176.bin")
    warc_path = "../data/CC/example.warc.gz"
    num_page = 20
    results = []
    with gzip.open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                raw_text = record.reader.read()
                text = extract(raw_text)
                label, conf = identify_language(model, text)
                results.append({
                    "label": label,
                    "conf": conf
                })
                num_page -= 1
                if num_page == 0:
                    break
    for i, result in enumerate(results):
        print(f"{i + 1}: label: {result["label"]}, conf: {result["conf"]}\n")
    
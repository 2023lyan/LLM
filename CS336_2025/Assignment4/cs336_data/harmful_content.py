from typing import Any
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
import sys
import pathlib
import gzip

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_text import extract

def classify_nsfw(model: Any, text: str) -> tuple[Any, float]:
    prediction = model.predict(text.replace("\n", " "))
    label = prediction[0][0].replace("__label__", "")
    confidence = float(prediction[1][0])

    return label, confidence


def classify_toxic_speech(model: Any, text: str) -> tuple[Any, float]:
    prediction = model.predict(text.replace("\n", " "))
    label = prediction[0][0].replace("__label__", "")
    confidence = float(prediction[1][0])

    return label, confidence

if __name__ == "__main__":
    model_nsfw = fasttext.load_model("../data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
    model_toxic_speech = fasttext.load_model("../data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
    warc_path = "../data/CC/example.warc.gz"
    num_page = 20
    results = []
    with gzip.open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                raw_text = record.reader.read()
                text = extract(raw_text)
                label_nsfw, conf_nsfw = classify_nsfw(model_nsfw, text)
                label_toxic_speech, conf_toxic_speech = classify_toxic_speech(model_toxic_speech, text)
                results.append({
                    "label_nsfw": label_nsfw,
                    "conf_nsfw": conf_nsfw,
                    "label_toxic_speech": label_toxic_speech,
                    "conf_toxic_speech": conf_toxic_speech,
                    
                })
                num_page -= 1
                if num_page == 0:
                    break
    for i, result in enumerate(results):
        print(
            f"{i + 1}: "
            f"NSFW={result['label_nsfw']} ({result['conf_nsfw']:.3f}), "
            f"Toxic={result['label_toxic_speech']} ({result['conf_toxic_speech']:.3f})"
        )


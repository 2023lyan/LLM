import fasttext
from typing import Any


def train_classifier():
    pos = "../data/positive_samples/positive_texts.txt"
    neg = "../data/negative_samples/negative_texts.txt"
    train_file = "../data/train_quality.txt"
    with open(train_file, "w", encoding="utf-8") as out:
        len_pos = 0
        with open(pos, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write("__label__wiki " + line.strip() + "\n")
                    len_pos += 1
        with open(neg, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write("__label__cc " + line.strip() + "\n")
                    len_pos -= 1
                    if len_pos == 0:
                        break
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,
        epoch=150,
        wordNgrams=2,
        dim=100,
        verbose=2
    )
    model.save_model("../data/classifiers/quality_classifier.bin")
    print("quality_classifier.bin saved successfully.")

def classify_quality(model: Any, text: str) -> tuple[Any, float]:
    prediction = model.predict(text.replace("\n", " "))
    label = prediction[0][0].replace("__label__", "")
    confidence = float(prediction[1][0])
    return label, confidence

if __name__ == "__main__":
    train_classifier()
    

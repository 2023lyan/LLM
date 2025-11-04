from __future__ import annotations

import os
from typing import Any

import pathlib
import sys

import fasttext
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from cs336_data.extract_text import extract
from cs336_data.language_identification import identify_language
from cs336_data.mask_pii import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.quality_classifier import classify_quality
from cs336_data.exact_deduplication import exact_line_deduplication
from cs336_data.minhash_deduplication import minhash_deduplication

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract(html_bytes=html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("./data/classifiers/lid.176.bin")
    return identify_language(model=model, text=text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text=text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text=text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text=text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("./data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
    return classify_nsfw(model=model, text=text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("./data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
    return classify_toxic_speech(model=model, text=text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("./data/classifiers/quality_classifier.bin")
    return classify_quality(model=model, text=text)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_deduplication(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory
    )
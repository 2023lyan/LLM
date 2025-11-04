import regex as re
from fastwarc.warc import ArchiveIterator, WarcRecordType
import sys
import pathlib
import gzip

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_text import extract


def mask_emails(text: str) -> tuple[str, int]:
    PAT = r"[A-Za-z0-9._%+-]+@[a-z]+\.[a-z]+"
    new_text, count = re.subn(PAT, "|||EMAIL_ADDRESS|||", text)
    return new_text, count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    PAT = r"(\+[0-9]+[\s\-\.]*)?\(?\d{2,4}\)?[\s\-\.]*\d{3,4}[\s\-\.]*\d{3,4}"
    new_text, count = re.subn(PAT, "|||PHONE_NUMBER|||", text)
    return new_text, count

def mask_ips(text: str) -> tuple[str, int]:
    PAT = r"(\d{1,3}\.){3}\d{1,3}"
    new_text, count = re.subn(PAT, "|||IP_ADDRESS|||", text)
    return new_text, count

def mask_pii(text: str) -> str:
    text, _ = mask_emails(text)
    text, _ = mask_phone_numbers(text)
    text, _ = mask_ips(text)
    return text

if __name__ == "__main__":
    warc_path = "../data/CC/example.warc.gz"
    num_page = 20
    with gzip.open(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                raw_text = record.reader.read()
                text = extract(raw_text)
                text = mask_pii(text)
                print("=" * 80)
                print(f"URL: {record.headers.get('WARC-Target-URI')}")
                print(f"{text}\n")
                num_page -= 1
                if num_page == 0:
                    break
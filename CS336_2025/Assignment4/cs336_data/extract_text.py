from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import gzip

def extract(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    if encoding is None:
        return None
    html_str = html_bytes.decode(encoding, errors="ignore")
    return extract_plain_text(html_str)

def extract_from_warc(path_to_warc: str, limit: int = 1):
    with gzip.open(path_to_warc, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                raw_text = record.reader.read()
                text = extract(raw_text)
                print("=" * 80)
                print(f"URL: {record.headers.get('WARC-Target-URI')}")
                print(f"{text}\n")
                limit -= 1
                if (limit == 0):
                    break

if __name__ == "__main__":
    warc_path = "../data/CC/example.warc.gz"
    extract_from_warc(path_to_warc = warc_path, limit = 1)
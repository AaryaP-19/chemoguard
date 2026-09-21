"""Pull the visible text out of a generated PDF.

Deliberately crude: it reads the string literals out of the content streams
rather than pulling in a PDF parsing dependency for one assertion. Enough to
check which unit string was printed, which is all the export test needs.
"""

import base64
import re
import zlib

_LITERAL = re.compile(rb"\((?:[^()\\]|\\.)*\)", re.S)
_STREAM = re.compile(rb"stream\r?\n(.*?)endstream", re.S)


def _decode(stream):
    """Undo reportlab's default /ASCII85Decode then /FlateDecode."""
    body = stream.strip(b"\r\n")

    if body.rstrip().endswith(b"~>"):
        try:
            body = base64.a85decode(body, adobe=True)
        except ValueError:
            return stream

    try:
        return zlib.decompress(body)
    except zlib.error:
        return body


def extract_text(pdf_bytes):
    """Every string literal drawn in the document, concatenated."""
    chunks = []

    for stream in _STREAM.findall(pdf_bytes):
        for literal in _LITERAL.findall(_decode(stream)):
            text = literal[1:-1]
            text = text.replace(b"\\(", b"(").replace(b"\\)", b")")
            chunks.append(text.decode("latin-1"))

    return "".join(chunks)

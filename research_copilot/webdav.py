"""Read-only Zotero WebDAV attachment adapter. Credentials never persisted."""
import argparse
import base64
import getpass
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError

import rag

MAX_BYTES = 100 * 1024 * 1024

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def root(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Supply an HTTPS WebDAV base URL without credentials, query or fragment.")
    url = url.rstrip("/")
    return url + "/" if url.endswith("/zotero") else url + "/zotero/"

def read_archive(url, key, auth):
    if not re.fullmatch(r"[A-Z0-9]{8}", key):
        raise ValueError("Invalid Zotero attachment key")
    req = Request(root(url) + key + ".zip", headers={"Authorization": auth}, method="GET")
    with build_opener(NoRedirect()).open(req, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Compressed attachment exceeds size limit")
    return data

def extract_pdf(data, filename):
    # Never extract archive paths to disk. Refuse ambiguous or oversized PDFs.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        candidates = [i for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith(".pdf")]
        matches = [i for i in candidates if Path(i.filename).name == filename]
        selected = matches if matches else candidates
        if len(selected) != 1:
            raise ValueError("Expected exactly one unambiguous PDF")
        info = selected[0]
        if info.file_size > MAX_BYTES:
            raise ValueError("Uncompressed PDF exceeds size limit")
        with archive.open(info) as stream:
            content = stream.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES or not content.startswith(b"%PDF-"):
            raise ValueError("Invalid or oversized PDF")
        return content

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--url", required=True, help="WebDAV base HTTPS URL, without credentials")
    parser.add_argument("--local", action="store_true", help="Use local Zotero metadata API")
    parser.add_argument("--user-id", help="Numeric Zotero user ID for remote metadata")
    parser.add_argument("--limit", type=int, default=10, help="Maximum PDF downloads per run")
    parser.add_argument("--collection-key", help="Only index parent items directly in this collection")
    parser.add_argument("--refresh", action="store_true", help="Re-index existing attachments")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    root(args.url)
    user = os.environ.get("WEBDAV_USER") or input("WebDAV username: ")
    password = os.environ.get("WEBDAV_PASSWORD") or getpass.getpass("WebDAV app password: ")
    auth = "Basic " + base64.b64encode((user + ":" + password).encode()).decode()
    items = list(rag.zotero_items(args.local, args.user_id))
    parents = {x["key"]: x["data"] for x in items if x["data"].get("itemType") not in ("attachment", "note", "annotation")}
    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = rag.database(args.db)
    report = {"downloaded": 0, "attempted": 0, "skipped_indexed": 0, "errors": []}
    try:
        for item in items:
            data = item["data"]
            key = item["key"]
            parent = parents.get(data.get("parentItem"), {})
            if data.get("contentType") != "application/pdf" or data.get("linkMode") not in ("imported_file", "imported_url"):
                continue
            if args.collection_key and args.collection_key not in parent.get("collections", []):
                continue
            source = "zotero:attachment:" + key
            if not args.refresh and db.execute("SELECT 1 FROM evidence WHERE source=? AND kind='pdf' LIMIT 1", (source,)).fetchone():
                report["skipped_indexed"] += 1
                continue
            if report["attempted"] >= args.limit:
                break
            report["attempted"] += 1
            try:
                content = extract_pdf(read_archive(args.url, key, auth), data.get("filename", ""))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "attachment.pdf"
                    path.write_bytes(content)
                    rag.pdf(db, path, source, parent.get("title") or data.get("title") or key, parent.get("DOI", ""))
                report["downloaded"] += 1
            except HTTPError as exc:
                report["errors"].append({"key": key, "http_status": exc.code})
                if exc.code in (401, 403):
                    break
            except Exception as exc:
                report["errors"].append({"key": key, "error_type": type(exc).__name__})
        print(json.dumps(report, indent=2))
    finally:
        db.close()

if __name__ == "__main__":
    main()

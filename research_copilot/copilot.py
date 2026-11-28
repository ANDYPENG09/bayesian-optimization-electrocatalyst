"""Read-only ima OpenAPI backend and unified literature/BO evidence CLI."""
import argparse
import getpass
import json
import os
import subprocess
import tempfile
import shutil
from urllib.parse import urlsplit
from html.parser import HTMLParser
from pathlib import Path
import rag

BASE = "https://ima.qq.com/openapi/wiki/v1/"
ALLOWED = {"search_knowledge_base", "get_knowledge_base", "get_knowledge_list", "search_knowledge", "get_addable_knowledge_base_list", "get_media_info"}

class ImaError(RuntimeError):
    pass

class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)

def clean(value):
    if not isinstance(value, str):
        return ""
    parser = TextOnly()
    parser.feed(value)
    return " ".join("".join(parser.parts).split())

def response_data(result):
    if not isinstance(result,dict):
        raise ImaError("Unexpected ima response.")
    status = result.get("retcode",result.get("code"))
    if status != 0 or not isinstance(result.get("data"),dict):
        raise ImaError("ima request failed; API code: " + str(status if isinstance(status,int) else "unknown"))
    return result["data"]

class ImaClient:
    def __init__(self, client_id=None, api_key=None):
        self.client_id = client_id or os.getenv("IMA_CLIENT_ID") or os.getenv("IMA_OPENAPI_CLIENTID")
        self.api_key = api_key or os.getenv("IMA_API_KEY") or os.getenv("IMA_OPENAPI_APIKEY")
        if not self.client_id or not self.api_key:
            raise ImaError("Set IMA_CLIENT_ID and IMA_API_KEY in your private runtime environment.")
    def post(self, action, payload):
        if action not in ALLOWED:
            raise ImaError("Only allowlisted read operations are permitted.")
        # Credentials are passed via stdin, never command-line arguments.
        if any(ch in self.client_id + self.api_key for ch in "\r\n\x00"):
            raise ImaError("Invalid newline or control character in credentials.")
        headers = ("Content-Type: application/json\n"
                   + "ima-openapi-clientid: " + self.client_id + "\n"
                   + "ima-openapi-apikey: " + self.api_key + "\n")
        try:
            proc = subprocess.run(
                ["curl", "-q", "-4", "--http2", "-sS", "--connect-timeout", "15",
                 "--max-time", "45", "--max-filesize", str(8*1024*1024),
                 "-X", "POST", "-H", "@-", "--data-binary",
                 json.dumps(payload, ensure_ascii=True),
                 "-w", "\\nHTTP_STATUS:%{http_code}", BASE + action],
                input=headers, text=True, encoding="utf-8", errors="replace",
                capture_output=True, timeout=50, check=False)
        except FileNotFoundError:
            raise ImaError("curl is required; install an HTTP/2-enabled curl.") from None
        except subprocess.TimeoutExpired:
            raise ImaError("ima HTTP/2 request timed out.") from None
        if proc.returncode:
            raise ImaError(f"ima curl failed (exit {proc.returncode}); check curl HTTP/2 support and connectivity.") 
        body, sep, status = proc.stdout.rpartition("\nHTTP_STATUS:")
        if not sep or not status.isdigit():
            raise ImaError("ima response missing HTTP status.")
        if len(body.encode("utf-8")) > 8*1024*1024:
            raise ImaError("ima response exceeded size cap.")
        try:
            result=json.loads(body)
        except ValueError:
            raise ImaError(f"ima HTTP {status}: non-JSON response.") from None
        if status != "200":
            code=result.get("code",result.get("retcode")) if isinstance(result,dict) else None
            raise ImaError(f"ima HTTP {status}; API code: {code if isinstance(code,int) else 'unknown'}")
        return response_data(result)
    def pages(self, action, payload, field, max_pages=3):
        cursor=""
        seen={cursor}
        for _ in range(max_pages):
            data=self.post(action, dict(payload,cursor=cursor))
            items=data.get(field)
            if not isinstance(items,list):
                raise ImaError("Unexpected ima response: missing list " + field)
            yield from items
            cursor=data.get("next_cursor")
            if data.get("is_end") is True or not cursor:
                return
            if not isinstance(cursor,str) or cursor in seen:
                raise ImaError("Unexpected repeated or invalid pagination cursor.")
            seen.add(cursor)
    def libraries(self, query="", max_pages=3):
        def normalize(items):
            found = {}
            for x in items:
                if not isinstance(x,dict):
                    continue
                key = x.get("id") or x.get("knowledge_base_id") or x.get("kb_id")
                basic = x.get("basic_info")
                basic = basic if isinstance(basic,dict) else {}
                name = clean(x.get("name") or basic.get("name") or x.get("title") or x.get("knowledge_base_name") or "")
                if isinstance(key,str) and key:
                    found[key] = {"id":key,"name":name or "(unnamed)"}
            return list(found.values())
        found = normalize(self.pages("search_knowledge_base",{"query":query,"limit":20},"info_list",max_pages))
        if not found and not query:
            found = normalize(self.pages("get_addable_knowledge_base_list",{"limit":50},"addable_knowledge_base_list",max_pages))
        # Resolve missing display names using the dedicated metadata endpoint.
        missing = [x for x in found if x["name"] == "(unnamed)"]
        for start in range(0,len(missing),20):
            batch = missing[start:start+20]
            try:
                details = self.post("get_knowledge_base",{"ids":[x["id"] for x in batch]})
                infos = details.get("infos",{})
                if isinstance(infos,dict):
                    for x in batch:
                        info = infos.get(x["id"],{})
                        if isinstance(info,dict):
                            basic = info.get("basic_info")
                            basic = basic if isinstance(basic,dict) else {}
                            name = clean(info.get("name") or basic.get("name") or info.get("title") or info.get("knowledge_base_name") or "")
                            if name:
                                x["name"] = name
            except ImaError as exc:
                print("Library name lookup:",str(exc))
        return found
    def media_info(self, media_id):
        return self.post("get_media_info",{"media_id":media_id})
    def browse(self, kb, folder=None, max_pages=3):
        payload={"knowledge_base_id":kb,"limit":50}
        if folder:
            payload["folder_id"]=folder
        return [{k:x[k] for k in ("media_id","title","folder_id","name","parent_folder_id") if k in x}
                for x in self.pages("get_knowledge_list",payload,"knowledge_list",max_pages) if isinstance(x,dict)]
    def search(self, kb, query, k=5, max_pages=3):
        hits=[]
        seen=set()
        for item in self.pages("search_knowledge",{"knowledge_base_id":kb,"query":query},"info_list",max_pages):
            if not isinstance(item,dict):
                continue
            mid=item.get("media_id")
            if item.get("folder_id") and not mid:
                continue
            if not isinstance(mid,str) or not mid or mid.startswith("folder_") or mid in seen:
                continue
            seen.add(mid)
            body=clean(item.get("highlight_content"))
            hits.append({"source":"ima:"+mid,"media_id":mid,"title":clean(item.get("title")),
                         "doi":None,"page":None,"kind":"ima_snippet" if body else "metadata_only",
                         "body":body,"knowledge_base_id":kb})
            if len(hits)>=k:
                break
        return hits

def packet(query, evidence, backend, bo_report=None):
    # A title-only match is useful for discovery, but not factual evidence.
    output={"question":query,"backend":backend,"evidence":[],"discovery_only":[]}
    for item in evidence:
        item=dict(item)
        if not item.get("body","").strip():
            output["discovery_only"].append(item)
            continue
        item["citation"]=f"E{len(output['evidence'])+1}"
        output["evidence"].append(item)
    if bo_report is not None:
        output["bo_report"]=bo_report
        output["bo_warning"]="Independent existing model output, not a literature-derived prediction or experimental validation."
    return output

def probe_pdf(info):
    """Verify one PDF locally; do not print signed URLs, text or retain files."""
    url_info = info.get("url_info")
    url = url_info.get("url") if isinstance(url_info,dict) else None
    if not isinstance(url,str) or not url or any(ord(c)<33 or ord(c)>126 for c in url):
        return {"error":"Missing or invalid media URL."}
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        permitted = (parsed.scheme == "https" and parsed.port in (None,443)
                     and not parsed.username and not parsed.password
                     and (host == "ima.qq.com" or host.endswith(".ima.qq.com")
                          or host.endswith(".myqcloud.com")))
    except ValueError:
        return {"error":"Malformed media URL."}
    if not permitted:
        return {"error":"Media host not on download allowlist.", "host":host}
    if not shutil.which("pdftotext"):
        return {"error":"Install poppler-utils to extract PDF text."}
    cap = 25*1024*1024
    with tempfile.TemporaryDirectory(prefix="ima-probe-") as tmp:
        path = Path(tmp)/"probe.pdf"
        try:
            # URL travels through private stdin, not command line; no ima headers,
            # redirects, proxy or curl config are used for the media download.
            with subprocess.Popen(["curl","-q","-4","--http2","-sS",
                                   "--noproxy","*","--proto","=https",
                                   "--connect-timeout","15","--max-time","90",
                                   "--max-filesize",str(cap),"--fail","--config","-"],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL) as proc:
                proc.stdin.write(("url = "+json.dumps(url)+"\n").encode("ascii"))
                proc.stdin.close()
                count = 0
                with path.open("wb") as dest:
                    while True:
                        block = proc.stdout.read(65536)
                        if not block:
                            break
                        count += len(block)
                        if count > cap:
                            proc.kill()
                            return {"error":"PDF download exceeded 25 MiB cap."}
                        dest.write(block)
                if proc.wait(timeout=5):
                    return {"error":"Media download failed.", "curl_exit":proc.returncode,"host":host}
            with path.open("rb") as handle:
                if b"%PDF-" not in handle.read(1024):
                    return {"error":"Downloaded media is not a PDF.", "download_bytes":count,"host":host}
            extracted = subprocess.run(["pdftotext","-f","1","-l","3","-enc","UTF-8",str(path),"-"],
                                       capture_output=True,timeout=30,check=False)
            chars = len(extracted.stdout.decode("utf-8",errors="replace").strip())
            return {"pdf_verified":True,"download_bytes":count,
                    "sample_pages_requested":3,"sample_text_chars":chars,
                    "sample_text_verified":extracted.returncode==0 and chars>0,
                    "extract_exit":extracted.returncode}
        except (OSError,subprocess.TimeoutExpired):
            return {"error":"Media download or extraction failed locally."}

def doctor(query, library_query="", kb=None, probe_media=False, test_pdf=False):
    """Interactive read-only smoke check; never prints credentials or excerpts."""
    client_id = os.getenv("IMA_CLIENT_ID") or os.getenv("IMA_OPENAPI_CLIENTID") or input("ima Client ID: ").strip()
    api_key = os.getenv("IMA_API_KEY") or os.getenv("IMA_OPENAPI_APIKEY") or getpass.getpass("ima API Key: ").strip()
    client = ImaClient(client_id,api_key)
    libraries = [{"id":kb,"name":"Explicit API knowledge base"}] if kb else client.libraries(library_query)
    if not libraries:
        return {"authentication":"request accepted","libraries_found":0,"warning":"No libraries found using discovery; try --library-query with the exact library name or --kb with its encoded OpenAPI ID."}
    print("Discovered libraries (read access not yet verified):")
    for idx, lib in enumerate(libraries,1):
        print(f"{idx}. {lib['name']}")
    selected = "1" if kb else input("Select library number: ").strip()
    if not selected.isdigit() or not 1 <= int(selected) <= len(libraries):
        raise ImaError("Invalid library selection.")
    library = libraries[int(selected)-1]
    items = client.browse(library["id"],max_pages=1)
    hits = client.search(library["id"],query,10,max_pages=2)
    evidence = packet(query,hits,"ima")
    result = {"authentication":"request accepted","libraries_found":len(libraries),
            "root_items_in_first_page":len(items),"search_hits":len(hits),
            "snippet_hits":len(evidence["evidence"]),"title_only_hits":len(evidence["discovery_only"]),
            "full_text_verified":False,
            "next_step":"Probe media access or use local Zotero PDF retrieval when snippets are absent or insufficient."}
    if probe_media or test_pdf:
        result["media_probes"] = []
        for hit in hits[:3]:
            try:
                info = client.media_info(hit["media_id"])
                url_info = info.get("url_info")
                url_info = url_info if isinstance(url_info,dict) else {}
                if test_pdf and "pdf_probe" not in result and info.get("media_type") in (1,"1") and url_info.get("url"):
                    result["pdf_probe"] = probe_pdf(info)
                result["media_probes"].append({"media_type":info.get("media_type"),
                    "has_access_url":bool(url_info.get("url")),
                    "has_note_reference":bool(info.get("notebook_ext_info")),
                    "response_fields":sorted(info.keys())})
            except ImaError as exc:
                result["media_probes"].append({"error":str(exc)})
        result["note"]="A returned access URL is not yet proof of successful PDF download or extraction."
    if test_pdf and "pdf_probe" not in result:
        result["pdf_probe"] = {"error":"No file candidate among first three hits."}
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("doctor"); a.add_argument("--query",default="IrOx"); a.add_argument("--library-query",default=""); a.add_argument("--kb"); a.add_argument("--probe-media",action="store_true"); a.add_argument("--test-pdf",action="store_true")
    a=sub.add_parser("libraries"); a.add_argument("--query",default="")
    a=sub.add_parser("browse"); a.add_argument("--kb",required=True); a.add_argument("--folder")
    a=sub.add_parser("ask")
    a.add_argument("query")
    a.add_argument("--backend",choices=["ima","local"],default="ima")
    a.add_argument("--kb",default=os.getenv("IMA_KB_ID"))
    a.add_argument("--db",type=Path)
    a.add_argument("--k",type=int,default=5)
    a.add_argument("--semantic",action="store_true",help="Local dense retrieval only")
    a.add_argument("--generate",action="store_true",help="Explicitly send excerpts to configured LLM")
    a.add_argument("--bo-report",type=Path)
    args=parser.parse_args()
    try:
        if args.command=="doctor":
            result=doctor(args.query,args.library_query,args.kb,args.probe_media,args.test_pdf)
        elif args.command=="libraries":
            result={"libraries":ImaClient().libraries(args.query),"max_pages":3}
        elif args.command=="browse":
            result={"items":ImaClient().browse(args.kb,args.folder),"max_pages":3}
        else:
            if not 1 <= args.k <= 50 or not args.query.strip():
                parser.error("Nonempty query and k between 1 and 50 required.")
            if args.backend=="ima":
                if not args.kb:
                    parser.error("Set IMA_KB_ID or use --kb; discover IDs with libraries.")
                if args.semantic:
                    parser.error("--semantic is only for the local backend.")
                evidence=ImaClient().search(args.kb,args.query,args.k)
            else:
                if not args.db or not args.db.is_file():
                    parser.error("Local backend requires an existing --db.")
                db=rag.database(args.db)
                try:
                    evidence=rag.search(db,args.query,args.k,args.semantic)
                finally:
                    db.close()
            result=packet(args.query,evidence,args.backend,
                          json.loads(args.bo_report.read_text(encoding="utf-8")) if args.bo_report else None)
            if args.generate:
                result["answer"]=rag.generate(args.query,result["evidence"])
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except ImaError as exc:
        parser.exit(1,str(exc)+"\n")

if __name__=="__main__":
    main()

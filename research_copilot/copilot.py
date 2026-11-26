"""Read-only ima OpenAPI backend and unified literature/BO evidence CLI."""
import argparse
import getpass
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
import rag

BASE = "https://ima.qq.com/openapi/wiki/v1/"
ALLOWED = {"search_knowledge_base", "get_knowledge_base", "get_knowledge_list", "search_knowledge"}

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

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
        req = Request(BASE + action, data=json.dumps(payload).encode("utf-8"), headers={
            "Content-Type":"application/json",
            "ima-openapi-clientid":self.client_id,
            "ima-openapi-apikey":self.api_key,
        }, method="POST")
        try:
            with build_opener(NoRedirect()).open(req, timeout=30) as response:
                raw = response.read(8*1024*1024+1)
        except HTTPError as exc:
            raise ImaError(f"ima HTTP {exc.code}; check permission, credentials or rate limits.") from None
        except URLError:
            raise ImaError("ima connection failed; check network.") from None
        if len(raw) > 8*1024*1024:
            raise ImaError("ima response exceeded size cap.")
        try:
            result=json.loads(raw)
        except ValueError:
            raise ImaError("ima returned non-JSON data.") from None
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
        return [{"id":x["id"],"name":clean(x.get("name"))} for x in self.pages(
            "search_knowledge_base",{"query":query,"limit":20},"info_list",max_pages)
            if isinstance(x,dict) and isinstance(x.get("id"),str)]
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

def doctor(query):
    """Interactive read-only smoke check; never prints credentials or excerpts."""
    client_id = os.getenv("IMA_CLIENT_ID") or os.getenv("IMA_OPENAPI_CLIENTID") or input("ima Client ID: ").strip()
    api_key = os.getenv("IMA_API_KEY") or os.getenv("IMA_OPENAPI_APIKEY") or getpass.getpass("ima API Key: ").strip()
    client = ImaClient(client_id,api_key)
    libraries = client.libraries()
    if not libraries:
        return {"authentication":"request accepted","libraries_found":0,"warning":"No accessible libraries found in first three pages."}
    print("Accessible libraries (first three pages):")
    for idx, lib in enumerate(libraries,1):
        print(f"{idx}. {lib['name']}")
    selected = input("Select library number: ").strip()
    if not selected.isdigit() or not 1 <= int(selected) <= len(libraries):
        raise ImaError("Invalid library selection.")
    library = libraries[int(selected)-1]
    items = client.browse(library["id"],max_pages=1)
    hits = client.search(library["id"],query,10,max_pages=2)
    evidence = packet(query,hits,"ima")
    return {"authentication":"request accepted","libraries_found":len(libraries),
            "root_items_in_first_page":len(items),"search_hits":len(hits),
            "snippet_hits":len(evidence["evidence"]),"title_only_hits":len(evidence["discovery_only"]),
            "full_text_verified":False,
            "next_step":"Use local Zotero PDF retrieval when snippets are absent or insufficient."}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("doctor"); a.add_argument("--query",default="IrOx")
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
            result=doctor(args.query)
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

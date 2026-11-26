# Materials Research Copilot — integration prototype

The existing AI4M/S planner remains unchanged. This directory is an opt-in
literature retrieval layer, not a validated autonomous scientist.

## Modular architecture
- [Zotero Scholar Bridge](https://github.com/ANDYPENG09/zotero-scholar-bridge):
  reading notes, summaries and evidence exports (existing, separate AGPL project).
- [Zotero IMA Sync](https://github.com/ANDYPENG09/zotero-ima-sync): optional hosted knowledge base; no automatic upload by this prototype.
- `rag.py`: read-only Zotero metadata/abstract ingestion, local PDF extraction,
  page-level evidence records, character TF-IDF or optional multilingual dense retrieval,
  optional OpenAI-compatible evidence-grounded generation.
- [Electrochemistry analyzer](https://github.com/ANDYPENG09/PEMFC-Electrocatalyst-Activity-Analyzer):
  export QC and measurements; join with synthesis design by sample_id using the
  existing scripts/import_analyzer_results.py.
- [XRD Toolkit](https://github.com/ANDYPENG09/xrd-toolkit):
  extract structural descriptors with method/units retained.
- Existing BO planner: validate on independent synthesis batches before recommending
  experiments. The retrieval CLI can attach a local BO JSON report for joint human review;
  literature retrieval does not automatically alter acquisition or material predictions.

## Quick start
Run from repository root. Keep databases and PDFs outside the git working tree.
Install Python 3.10+, then:

```bash
pip install pymupdf scikit-learn
python research_copilot/rag.py --db ~/materials-private/evidence.sqlite ingest-pdf ~/materials-private/papers
python research_copilot/rag.py --db ~/materials-private/evidence.sqlite ask "IrOx calcination temperature and OER activity"
```

For multilingual semantic retrieval, install `sentence-transformers` and add
`--semantic` (downloads BAAI/bge-m3 on first use). This prototype recomputes
vectors at query time; a persistent vector index is future work.

### Zotero local (recommended)
Enable Settings > Advanced > Allow other applications on this computer to
communicate with Zotero. Keep Zotero open. Download attachments locally first.
Use the actual Zotero data directory, which is configurable:

```bash
python research_copilot/rag.py --db ~/materials-private/evidence.sqlite sync-zotero --local --storage ~/Zotero/storage
```

Local API is on 127.0.0.1:23119. Metadata/abstracts are read, and PDFs found
under storage/<attachment key>/<filename> are parsed with actual PDF page indices.
Linked files, notes, annotations, scanned PDFs and missing attachments are not
fully supported. Abstracts have null page. An index is a private local snapshot;
rebuild it to reflect remote deletions.

### Zotero Web API
Create a dedicated read-only key at https://www.zotero.org/settings/keys,
store it locally as ZOTERO_API_KEY, and run:
```bash
python research_copilot/rag.py --db ~/materials-private/evidence.sqlite sync-zotero --user-id YOUR_NUMERIC_ID
```
This reads metadata/abstracts only unless local storage is supplied. WebDAV
normally stores attachment archives, not complete bibliographic metadata.
Use Zotero's built-in WebDAV sync to populate local storage first.
Direct read-only WebDAV ingestion is available as an experimental adapter.
Use a private local database outside the repository. Zotero appends /zotero/
to the configured WebDAV URL; the adapter does likewise. No PUT, DELETE or
remote modification is performed.

```bash
python research_copilot/webdav.py --db ~/materials-private/evidence.sqlite --url https://YOUR-WEBDAV-HOST/dav/ --local --limit 10
```

The adapter prompts for WebDAV credentials, or reads WEBDAV_USER and
WEBDAV_PASSWORD from the local environment. It pairs attachment keys with Zotero
metadata and downloads the corresponding ZIP archives, parses a single
unambiguous PDF, and retains title, DOI and physical PDF page. Use
`--user-id YOUR_NUMERIC_ID` instead of `--local` for server mode with the
read-only ZOTERO_API_KEY environment variable. Optionally use
`--collection-key KEY` to select one direct collection or `--refresh` to
re-index existing attachments. Authentication and live server compatibility
must be verified locally. Existing indexed files are skipped by default; refresh
does not automatically detect changed or deleted remote attachments. No
credentials, PDF data or database files should be committed. Do not expose
credentials or localhost publicly.

### Optional generation
Install `openai`, set OPENAI_API_KEY, RAG_MODEL and optionally OPENAI_BASE_URL
in the local environment, then add `--generate` to ask. This explicitly sends
the retrieved excerpts to the configured provider. Use a trusted local provider
for restricted material. Without this flag retrieval runs locally.
Output evidence IDs, DOI and PDF page are provided to verify citations manually.
Generated answers may still contain errors; prompt instructions are not a guarantee.

### Existing experiment workflow
```bash
python scripts/import_analyzer_results.py --design design.csv --results results.csv --out joined.csv
python scripts/bo_pipeline.py --data joined.csv --config campaign.yaml --out next_experiment.json
python research_copilot/rag.py --db ~/materials-private/evidence.sqlite ask "IrOx structure–activity relationship" --bo-report next_experiment.json
```
Run with synthetic or authorized inputs only. The last command bundles a report,
not an autonomous agent or validated closed-loop experiment.

## Validation and roadmap
- Run `python -m unittest discover -s research_copilot/tests -v`.
- Next: persistent vector index, hybrid retrieval/reranking, Zotero notes/Scholar Bridge
  Markdown ingestion, Notion read-only adapter, structured materials SQL schema,
  citation verification and benchmark questions, agent tool calling with review.
- Evaluate retrieval recall@k, citation accuracy, evidence faithfulness,
  abstention on unanswerable questions, and independent batch prediction error separately.
- Do not publish private PDFs, proprietary datasets, credentials, indexed databases,
  or exports. No real literature corpus is included.

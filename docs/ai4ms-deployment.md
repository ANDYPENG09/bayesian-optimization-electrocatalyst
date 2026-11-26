# AI4M/S deployment checkpoint — 2026-11-26

## Demo environment (verified by operator)

Ubuntu 24.04 LTS x86_64, 2 vCPU, 4 GB RAM, 60 GB disk, 2 GB swap. Docker Engine 29.1.3 and Compose 2.40.3. AnythingLLM runs as a single container with SQLite and LanceDB, 1.5 CPU and 2 GiB memory limits. Idle measurement: 361.5 MiB container RAM, 0.01% CPU; not a load benchmark.

External SiliconFlow OpenAI-compatible API: DeepSeek-V3.2 chat tested successfully; BAAI/bge-m3 embedding configured but not yet validated with documents. No local model inference. Telemetry disabled. Independent Cloudflare Tunnel established and HTTPS browser access verified. Application password authentication enabled per operator; loopback-only port 3001 verified. Cloud firewall rule removal and API key rotation still require confirmation. Access identity layer deferred.

## Reproduction template

Use a trusted, verified AnythingLLM image (pin digest after verification). The original deployment used a third-party registry proxy because Docker Hub was unreachable; upstream digest equivalence was not independently verified.

```yaml
services:
  anythingllm:
    image: mintplexlabs/anythingllm:latest # replace with verified digest
    restart: unless-stopped
    ports:
      - "127.0.0.1:3001:3001"
    volumes:
      - ./storage:/app/server/storage
      - ./.env:/app/server/.env
    environment:
      STORAGE_DIR: /app/server/storage
      DISABLE_TELEMETRY: "true"
    cpus: 1.5
    mem_limit: 2g
    memswap_limit: 3g
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

Create storage and an empty .env before startup, grant container UID/GID 1000 write access, restrict .env permissions, and keep both out of version control. Run cloudflared as a separate host service with a dedicated tunnel routing to http://127.0.0.1:3001. Require application authentication; do not expose the origin port. Never publish credentials or live infrastructure identifiers.

## Production (planned)

Separate NAS deployment of WeKnora with ima incremental synchronization, independent from the public demo. ima API authentication, knowledge-base enumeration (nine libraries), keyword retrieval and original-media URL retrieval verified by operator; PDF download and parsing remain unverified.

## Resume checklist

- [ ] Confirm cloud firewall port 3001 rule removed; test login in a private browser window.
- [ ] Rotate previously HTTP-transmitted API credentials; apply OS security updates.
- [ ] Retry public PDF upload over HTTPS; if unsuccessful inspect `docker logs --since 5m anythingllm` (redact secrets).
- [ ] Validate BGE-M3 embedding, retrieval and source citations with one open-access paper, then expand to 5–10.
- [ ] Connect Materials Research Copilot and structured material data; integrate BO and characterization tools with explicit units, provenance and scientist review.
- [ ] Evaluate Hermes Agent integration and optional Cloudflare Access.
- [ ] Validate NAS production deployment separately.

No proprietary data, private papers, live hostnames, IP addresses, API keys or tunnel tokens are included. RAG and tool orchestration are not yet end-to-end validated.

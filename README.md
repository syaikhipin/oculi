# Oculi

**Oculi** is a single-node video analytics MVP: face recognition and watchlists,
person/vehicle classification, license-plate recognition, passive liveness, tracking, a web
dashboard, a REST API, Redpanda events, and a Milestone Generic Events bridge.

![Oculi demo](docs/assets/oculi-demo.gif)

Repository: <https://github.com/syaikhipin/oculi>

It is a quality-first rebuild of the SentiVeillance-style pipeline. **The default face engine is
InsightFace buffalo_l** (best open-model accuracy); its code is MIT but its pretrained weights
are non-commercial. For commercial deployment, switch to the bundled SFace engine or license
the InsightFace weights.

## Licensing and readiness

| Concern | Status |
|---|---|
| Default face models | **InsightFace buffalo_l** — SCRFD + ArcFace 512-D; **weights non-commercial** |
| Commercial face alternative | `OCULI_FACE_ENGINE=sface` — YuNet (MIT) + SFace (Apache-2.0), 128-D |
| Object / ALPR / liveness models | YOLOX (Apache-2.0), fast-alpr (MIT + MIT upstream), MiniFASNet (Apache-2.0) |
| Auth | `OCULI_API_KEY` bearer/x-api-key on `/v1/*`, `/health`, `/metrics` |
| TLS | `oculi serve --ssl-certfile/--ssl-keyfile`, or terminate at a reverse proxy |
| Security headers | CSP, `nosniff`, `DENY`, `no-referrer` |
| Metrics | Prometheus `/metrics` (gated by API key) |
| Event retention | `OCULI_EVENT_RETENTION_DAYS` (default 30) + hourly prune |
| Event evidence | thumbnails enabled by default (`OCULI_THUMBNAILS`), embedded in event records |
| Backups | `oculi backup --output db.backup` (SQLite online backup) |
| Upload limits | `OCULI_MAX_IMAGE_BYTES` (default 15 MB) and image decode validation |
| Data protection | thumbnails/templates are biometric data; retention + backup guidance below |

## Features

| Capability | Implementation |
|---|---|
| Face detection | InsightFace SCRFD (buffalo_l) |
| Face templates / 1:N | InsightFace ArcFace `w600k_r50`, 512-D; SFace fallback (128-D) |
| Liveness | MiniFASNetV2 + MiniFASNetV1SE ONNX (Apache-2.0) |
| Person / vehicle | YOLOX-s COCO (Apache-2.0) |
| ALPR | fast-alpr: YOLO-v9 plate detector + CCT OCR |
| Tracking | deterministic IoU tracker with per-source track memory |
| Event evidence | JPEG snapshot thumbnail (base64) attached to each event |
| Events | SQLite + Redpanda (Kafka wire protocol) |
| VMS | RTSP ingest + Milestone Generic Events (TCP/UDP/webhook) |

InsightFace templates are 512 × float32 = **2048 bytes** (SFace fallback: 128 × float32 =
512 bytes) plus a checksummed JSON envelope. Templates are engine-specific, so switching engines
requires re-enrollment.

## Quick start

```bash
cd oculi
python3 -m venv .venv
.venv/bin/pip install -e ".[faiss,test]"

export OCULI_API_KEY="$(openssl rand -hex 16)"
.venv/bin/oculi serve --host 0.0.0.0 --port 8080
```

- Dashboard: `http://localhost:8080/` (enter the API key when prompted)
- API docs: `http://localhost:8080/docs`
- Models auto-download into `./models` on first use (buffalo_l ~326 MB, YOLOX 35 MB,
  ALPR ~40 MB, liveness 3.4 MB; optional SFace 37 MB + YuNet 228 KB).

TLS directly from Oculi:

```bash
oculi serve --port 8443 --ssl-certfile cert.pem --ssl-keyfile key.pem
```

or put Caddy/Nginx in front and keep Oculi on loopback.

### CLI

```bash
oculi enroll --subject alice --name "Alice" --image a.jpg
oculi search --image query.jpg --top-k 5
oculi detect --image street.jpg
oculi alpr --image car.jpg
oculi scan-video --source clip.mp4 --every 2
oculi events --limit 50
oculi sources-add --name Lobby --url rtsp://user:pass@camera/stream
oculi worker
oculi backup --output backups/oculi-$(date +%F).db
oculi prune-events --days 30
```

### API (selected)

| Method | Path | Purpose |
|---|---|---|
| GET | `/livez`, `/readyz` | probes (public) |
| GET | `/health` | engine/bus/worker status (API key) |
| GET | `/metrics` | Prometheus text |
| POST | `/v1/watchlists/{id}/subjects` | enroll (`multipart`: name, images) |
| POST | `/v1/watchlists/{id}/search` | 1:N face search |
| POST | `/v1/verify` | 1:1 face verification |
| POST | `/v1/detect` | faces + liveness + objects + plates |
| POST | `/v1/alpr` | plates only |
| GET/POST/PATCH/DELETE | `/v1/sources` | RTSP/file source registry |
| GET | `/v1/events?after_id=&type=&source_id=&thumbnails=` | event log with snapshots |
| GET | `/v1/workers` | worker counters |
| POST | `/v1/milestone/test` | test Generic Event delivery |

```bash
curl -H "Authorization: Bearer $OCULI_API_KEY" \
  -X POST localhost:8080/v1/detect -F image=@street.jpg
```

## Architecture

```
RTSP/file ──▶ SourceWorker ──▶ FramePipeline
                               ├─ InsightFace SCRFD + ArcFace (faces, 512-D)
                               ├─ MiniFASNet liveness
                               ├─ YOLOX person/vehicle
                               ├─ fast-alpr plates + OCR
                               └─ IoU tracking per source
                                     │
                                     ▼
                            EventService ──▶ SQLite
                                     ├─▶ Redpanda topic `oculi.events`
                                     ├─▶ Milestone Generic Events
                                     └─▶ Web UI polling
```

Single-node by design: run API and workers in one process (`oculi serve`). The split
`oculi worker` mode is available for development, but the API's FAISS index is rebuilt from
SQLite at startup and does not observe enrollments made by another process.

## Redpanda

```bash
export OCULI_REDPANDA_BROKERS=127.0.0.1:9092
export OCULI_REDPANDA_TOPIC=oculi.events
oculi worker
```

`docker-compose.yml` starts Redpanda and Oculi. Redpanda is **BSL 1.1** (source-available, not
OSI); its BSL terms permit this application's commercial use but prohibit offering a competing
streaming service. If your policy requires OSI licences only, use the Apache Kafka 3.9 compose
file (Apache-2.0):

```bash
export OCULI_API_KEY="$(openssl rand -hex 16)"
docker compose -f docker-compose.kafka.yml up --build
```

The application only needs `OCULI_REDPANDA_BROKERS`; aiokafka speaks the same protocol.

Events are JSON, keyed by `source_id`, with an idempotent producer. With no broker configured,
events are persisted to SQLite and the bus is a no-op.

## Milestone

XProtect cameras are RTSP sources. Events are mirrored as Generic Events:

```bash
export OCULI_MILESTONE=1
export OCULI_MILESTONE_PROTOCOL=tcp     # tcp | udp
export OCULI_MILESTONE_HOST=192.168.1.10
export OCULI_MILESTONE_PORT=9999
# and/or
export OCULI_MILESTONE_WEBHOOK=https://vms-bridge/events
```

Mark a source with `milestone: true` (UI checkbox or `--milestone`). Wire format:
`OCULI|<type>|<source>|<subject|->|<plate|->|<timestamp>`.
A native MIP plugin is Windows-only and intentionally not bundled; this bridge requires no
Milestone SDK.

## Face engine selection

Default (highest open-model accuracy):

```bash
OCULI_FACE_ENGINE=insightface oculi serve   # default; buffalo_l weights non-commercial
```

Permissive/commercial path:

```bash
OCULI_FACE_ENGINE=sface oculi serve         # YuNet MIT + SFace Apache-2.0
```

The default threshold is 0.40 for InsightFace and 0.363 for SFace when
`OCULI_MATCH_THRESHOLD` is not set. Switching engines requires a fresh watchlist database or
re-enrollment.

**Licence warning:** InsightFace's code is MIT, but the buffalo_l weights are distributed for
non-commercial research. Shipping the default engine in a commercial product requires a licence
from the model provider, or use the SFace path.

## Operations

- **Auth:** `OCULI_API_KEY`; send `Authorization: Bearer <key>` or `X-API-Key`.
  `/livez` and `/readyz` remain public for probes; `/health` is key-protected.
- **TLS:** native uvicorn certs or a reverse proxy. Never expose port 8080 directly.
- **Metrics:** Prometheus scrapes `/metrics` with the API key:

  ```yaml
  scrape_configs:
    - job_name: oculi
      authorization:
        credentials: "YOUR_OCULI_API_KEY"
      static_configs:
        - targets: ["oculi:8080"]
  ```

- **Retention:** hourly prune deletes events older than `OCULI_EVENT_RETENTION_DAYS`.
- **Backups:** `oculi backup` uses SQLite's online backup API; schedule it externally.
- **Scaling:** SQLite/FAISS are single-host. For multi-node, move events to PostgreSQL, index to
  a shared vector DB, and run GPU workers separately.
- **Biometric data:** templates and event crops are personal data; apply access control,
  retention, and GDPR/BIPA obligations.

## GPU benchmark (Modal T4)

Full results: [`docs/benchmarks.md`](docs/benchmarks.md) · raw JSON: [`docs/gpu_results.json`](docs/gpu_results.json).
Run it with `pip install modal && modal run benchmarks/modal_gpu_bench.py` (GPU credentials are
read from the Modal environment; none are stored in the repo).

| Face detect + 512-D embedding | CPU | CUDA (T4) | Speed-up | CUDA img/s |
|---|---:|---:|---:|---:|
| 640×480 | 130.1 ms | 36.7 ms | 3.5× | 27.2 |
| 1280×720 | 141.9 ms | 32.7 ms | **4.3×** | 30.6 |
| 1920×1080 | 147.8 ms | 32.9 ms | 4.5× | 30.4 |

On the same T4 host: liveness 5.7 ms (174 FPS), ALPR 26.2 ms (38 FPS), YOLOX objects 32.5 ms
(30.8 FPS, ONNX Runtime CUDA), full `analyze()` 130.1 ms (**7.7 FPS**), face search 60.0 ms
(16.7 FPS). YOLOX was moved from OpenCV DNN to ONNX Runtime for this run; the remaining CPU
stages are the next optimisation target. Inference dominates HTTP/event routing, so Go/Rust
would not materially reduce end-to-end latency; the process split already allows a Go/Rust
ingest gateway later while the CV layer stays Python/ONNX Runtime/OpenCV.

## Licence audit

See `THIRD_PARTY_LICENSES.md`. The default face weights (InsightFace buffalo_l) are
**non-commercial**; switch to SFace for a permissive path. Redpanda is BSL (source-available);
the SFace/YOLOX/MiniFASNet/fast-alpr stack is MIT/Apache-2.0.

## Tests

```bash
.venv/bin/pytest -q

# Real-model integration: same person, same person, different person
OCULI_TEST_IMAGES=/path/same1.jpg:/path/same2.jpg:/path/other.jpg \
  .venv/bin/pytest -q tests/test_integration_faces.py
```

## Scope / limitations

- Single-node MVP; no k3s/Helm yet.
- Default face weights are non-commercial; see the licence section.
- InsightFace default weights are non-commercial; the SFace fallback is less accurate.
  Calibrate thresholds on your own data.
- Passive liveness is not certified PAD and will not stop all presentation attacks.
- ALPR/vehicle accuracy depends on region and cameras; validate before commitments.
- No native Milestone MIP plugin.
- Redpanda compose is provided but not run in this environment (Docker socket unavailable);
  the bus was verified against the Kafka wire protocol with a local Kafka 4.4 broker.

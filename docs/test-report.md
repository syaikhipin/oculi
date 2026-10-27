# Oculi test report

Version **0.4.0** · InsightFace buffalo_l default · captured 2026-10-27.

## Automated suite

```text
$ .venv/bin/pytest -q
40 passed, 1 skipped in 1.2s
```

- Raw output: `test-output.txt` · integration: `integration-output.txt`
- JUnit XML: `test-results.xml`
- Coverage: **56%** overall (`coverage/index.html`, `coverage.xml`)
- Skipped test: network-dependent real-image integration unless `OCULI_TEST_IMAGES` is set.

Real-model integration:

```text
$ OCULI_TEST_IMAGES=/tmp/obama.jpg:/tmp/obama_flip.jpg:/tmp/biden.jpg \
    .venv/bin/pytest -q tests/test_integration_faces.py
1 passed
```

## Docker

```text
$ docker build -t oculi:mvp .
Successfully tagged oculi:mvp
$ docker run ... -e OCULI_API_KEY=mvpkey oculi:mvp
/livez -> 200
/health without key -> 401
/health with key -> 200 (engine insightface-buffalo_l-v1, 512-D)
/v1/detect (bus.jpg) -> 1 face, 5 objects, liveness 0.9999
/v1/alpr (plate.png) -> 5AU5341, score 0.8428, OCR 0.9998
container restart -> models served from volume, 0 downloads
```

## Redpanda

Redpanda `v24.3.1` and Oculi ran in Docker on a shared network. A file source was added
through the API; the worker produced `face`, `object`, and `face_match` events.

```text
$ rpk topic consume oculi.events -n 3
{"type": "face", "source_id": "dockercam", "track_id": 1, ...}
```

Metrics inside the container:

```text
oculi_frames_total{source_id="dockercam"} 30.0
oculi_events_total{source_id="dockercam",type="face_match"} 1.0
```

## Security

- TLS server (`--ssl-certfile/--ssl-keyfile`) with API key:
  `/health` without key **401**, with key **200**.
- `/metrics` protected by the same key.
- CSP, `nosniff`, `DENY`, `no-referrer` present.
- Upload size limit and API-key middleware covered by `tests/test_security.py`.

## User interface

Captured from the running app at 1440×900:

- `assets/dashboard.png`
- `assets/events.png`
- `assets/watchlists.png`
- `assets/detect.png`
- `assets/sources.png`
- Animated tour: `assets/oculi-demo.gif`

The UI test exercised: authenticated dashboard, event table with snapshot thumbnails,
1:N face search (InsightFace 512-D), ALPR overlay, and source management.

## Notes

- Docker Compose plugin was not installed in the test host; the equivalent `docker run`,
  network, broker, and volume flows were exercised manually.
- GPU benchmark: see `benchmarks.md`. Modal T4: face detection+embedding is 3.5–4.5× faster on
  CUDA (27–31 img/s); YOLOX runs on ONNX Runtime CUDA at 30.8 FPS, and the full pipeline is
  7.7 FPS.

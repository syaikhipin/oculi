# Security

## Deployment checklist

1. Set a random `OCULI_API_KEY` (at least 32 hex characters) and never commit it.
2. Terminate TLS at Oculi (`--ssl-certfile/--ssl-keyfile`) or a reverse proxy; bind Oculi to
   loopback when a proxy is used.
3. Keep `/v1/*`, `/health`, and `/metrics` behind the API key. `/livez`/`/readyz` are public probes.
4. Restrict outbound access; model downloads should be pre-provisioned in offline deployments.
5. Run as an unprivileged user; the provided image runs as UID 10001.
6. Back up SQLite with `oculi backup`; protect backups as biometric data.
7. Set `OCULI_EVENT_RETENTION_DAYS` and prune event records/thumbnails according to policy.
   Thumbnails are stored inline in the events table and are copied by backups.
8. Validate camera credentials; prefer RTSP URLs with dedicated least-privilege accounts.

## Known limitations

- API key auth only (no per-user accounts/OAuth). Use an identity-aware proxy for multi-user
  deployments.
- No rate limiting. Put a reverse proxy/WAF in front for internet-facing use.
- SQLite and FAISS are single-node.
- Liveness is a passive classifier, not a certified PAD solution.
- No encryption at rest; use full-disk/volume encryption.

Report security issues privately to the deployment owner; this is a self-hosted MVP.

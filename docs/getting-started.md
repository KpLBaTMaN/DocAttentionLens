# Running and configuration

Run all commands below from the repository root. For a first launch, see the
[quick start](../README.md#get-started).

## Run

Install Docker Desktop with Linux containers. From this directory:

    docker compose -f infra/compose.yaml config --quiet
    docker compose -f infra/compose.yaml up --build -d

Open http://localhost:8080. API documentation is at http://localhost:8000/docs.
For NVIDIA GPU execution, use:

    docker compose -f infra/compose.yaml --profile gpu up --build -d

The GPU profile requires a compatible NVIDIA device exposed to Docker. Models have
pinned checkpoints and download on first use into the persistent model-cache volume.
Setup reports each model's memory requirements and current availability.

Upload PDFs or images in Investigate or Library. Mark elements to add boxes, types,
and reading order to uploaded pages, then freeze the marked version before creating
a case. Benchmarks can be registered from Library when their exports are present
under data/ (see data/README.md). A clone already includes two small examples, an
annual report with layout elements and three of its tables with cell annotations; see
[examples.md](examples.md). Retrieval, generation, saved investigations, table
grounding, document studies, interventions, and evaluation remain application features.

## Configuration and data

The supplied Compose configuration is for trusted local operation: all host ports
bind to loopback. Its sample database password is for local development. Copy
infra/env.example to infra/.env, set deployment credentials, and pass
--env-file infra/.env to Compose when provisioning. Changing an environment variable
does not change the password of an already initialized PostgreSQL volume.

Public access requires an authenticated TLS gateway and deployment-specific access
controls. The application does not provide its own authentication.

Native processes accept DOCATTENTIONLENS_DATABASE_URL, DOCATTENTIONLENS_DATA_ROOT, DOCATTENTIONLENS_ARTIFACT_ROOT,
DOCATTENTIONLENS_MODEL_CACHE, DOCATTENTIONLENS_WORKER_RESOURCE_CLASS, DOCATTENTIONLENS_WORKER_CONCURRENCY, and
DOCATTENTIONLENS_LEASE_SECONDS. Compose supplies container paths. Workers run one job at a time,
renew durable leases, and publish checksum-verified results.
Long-running services restart automatically after failures. The migration service
runs once and must complete successfully before application startup.

The persistent volumes are metadata-store, artifact-store, and model-cache.
Dataset mounts are read-only. Keep the database and artifact store together in
backups; preserve uploaded checkpoints as well as captures and investigations.
The downloaded model cache can be reconstructed from pinned model registrations.

For a consistent offline backup, stop the API and workers, stop PostgreSQL cleanly,
then archive the metadata and artifact volumes with file ownership and permissions.
Restore both while services are stopped. The migration service applies pending
additive migrations before the API or workers start. Never alter applied SQL files.

    docker compose -f infra/compose.yaml logs --tail 100 api worker-cpu
    docker compose -f infra/compose.yaml stop

Stop and ordinary down retain saved volumes. Do not use down --volumes for normal
shutdown: it destroys saved application data.

## Troubleshooting

Check service status and recent logs from the repository root:

```sh
docker compose -f infra/compose.yaml ps
docker compose -f infra/compose.yaml logs --tail 100 api worker-cpu
```

- **The app has not opened yet:** the database must become healthy and migrations
  must finish before the API and frontend start. Inspect the `metadata`, `migrate`
  and `api` service logs if startup fails.
- **A model is unavailable:** check the application's setup view for the selected
  model's resource requirements. For GPU execution, start the `gpu` profile and
  confirm that Docker can access the NVIDIA GPU. Model weights download on first
  use; the saved cache is reused afterwards.
- **No examples are listed:** examples depend on captures already stored in this
  deployment. A fresh installation does not include the captures shown in the
  README demo. Upload a document or register a benchmark, then acquire a capture.
- **A port is already in use:** stop the conflicting native process or adjust the
  loopback port mapping in `infra/compose.yaml`. The supplied frontend uses 8080,
  the API 8000, and PostgreSQL 5432.

The running API's [interactive reference](http://localhost:8000/docs) lists the
available endpoints. For native development and verification, see
[development.md](development.md).


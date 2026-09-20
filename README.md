# O-HIVE Monocular RGB Sparse SLAM

Assignment 2 is a compact, explainable monocular visual-SLAM application. A user uploads a short RGB video; the backend validates and samples it, estimates a relative camera trajectory, triangulates a sparse 3D map, reduces local reprojection error, and returns a browser-explorable scene.

The implementation does not fabricate geometry and does not describe monocular coordinates as metres. Failed initialization is reported as a failure; late tracking loss can return a visibly labelled partial reconstruction.

## Live applications

- **Official AWS application:** pending the authenticated manual deployment described below.
- **Optional Render mirror:** pending the manual mirror deployment. Render is not the official assignment host.
- **Source:** <https://github.com/Ansh701/o-hive-monocular-slam>
- **Deployment mode:** manual only. GitHub Actions, Dependabot, workers, and automatic Render deploys are deliberately disabled/absent.

The live URLs and immutable deployed commit are recorded here only after public acceptance testing succeeds.

## Architecture

```mermaid
flowchart TD
    U[Browser] -->|HTTPS| CF[Amazon CloudFront]
    CF -->|CloudFront origin network only| EC2[One Amazon EC2 instance]

    subgraph APP[One Docker application]
      UI[React + TypeScript + Three.js]
      API[FastAPI]
      SLAM[OpenCV + NumPy + SciPy sparse SLAM]
      TMP[(Ephemeral tmpfs video)]
      UI -->|same origin /api| API
      API --> SLAM
      API --> TMP
      SLAM --> API
    end

    EC2 --> APP
    API -->|bounded run metadata only| DB[(Isolated PostgreSQL database)]
    TMP -->|deleted in every terminal path| DEL[Deletion]
```

CloudFront supplies the public HTTPS endpoint. Its origin is a single CPU EC2 instance whose port 80 security-group rule accepts only the AWS-managed CloudFront origin-facing prefix list. Session Manager replaces SSH. The stack has no NAT Gateway, load balancer, Elastic IP, GPU, RDS instance, worker, or deployment pipeline.

In production FastAPI serves both the Vite build at `/` and the API at `/api/*` from one origin. PostgreSQL stores small run metadata; the result geometry remains in bounded process memory and uploaded video bytes live only in a temporary directory.

## Technology stack

| Layer | Components | Reason |
|---|---|---|
| Browser | React 19, TypeScript, Vite, Three.js, Lucide | Typed interaction state and direct GPU scene rendering without a UI-framework dependency tree |
| API | Python 3.12, FastAPI, Pydantic Settings | Multipart upload, typed responses, polling, and async I/O |
| Geometry | OpenCV Headless, NumPy, SciPy | Explainable classical vision primitives and bounded nonlinear least squares |
| Persistence | SQLAlchemy 2 async, asyncpg, Alembic, PostgreSQL | Isolated status/diagnostic metadata and clean migrations |
| Packaging | One multistage Dockerfile | Same frontend/API image on AWS, Render, and local Docker |
| AWS | EC2, CloudFront, Systems Manager, SSM Parameter Store, Budgets | Small manually operated public deployment with HTTPS and cost controls |

No pretrained model or external reconstruction API is used.

## Algorithm overview

The active hot path is intentionally classical and bounded:

1. Validate the real container, codec, duration, dimensions, frame count, and byte size with OpenCV.
2. Sample at an effective 8 FPS and resize the long edge to 640 pixels.
3. Approximate or accept provided pinhole intrinsics.
4. Detect Shi–Tomasi corners and track them with pyramidal Lucas–Kanade optical flow.
5. Reject tracks whose forward/backward error is too large.
6. Estimate an Essential Matrix with RANSAC and recover the initial relative pose.
7. Triangulate and filter the initial sparse landmarks.
8. Estimate subsequent world-to-camera poses with `solvePnPRansac`, anchored to persistent 3D landmarks.
9. Select keyframes from translation, rotation, parallax, tracking quality, or time-gap thresholds.
10. Triangulate new landmarks only from useful geometry.
11. Run bounded local bundle adjustment over recent poses, landmarks, and observations.
12. Serialize true counts plus a separately bounded visualization sample.

ORB descriptor/keyframe utilities were implemented and tested while evaluating loop-closure/relocalization support. They are not in the current performance-critical tracking path: LK was measurably simpler and faster for adjacent frames, and the submission does not claim loop closure.

### Video preprocessing

The public limits are centralized in `Settings`:

| Setting | Value |
|---|---:|
| File size | 50 MiB |
| Duration | 30 seconds |
| Input dimensions | 3840 × 2160 maximum |
| Declared frame count | 1,800 maximum |
| Processing long edge | 640 pixels |
| Effective processing rate | 8 FPS |
| Processed-frame cap | 240 |
| Feature cap | 900 |
| Concurrent reconstructions | 1 |
| Processing timeout | 45 seconds |

MP4, MOV, and WebM are accepted only when the underlying container/codec actually decodes. MP4 with H.264 is the recommended public-demo input. Seeking to selected source frames avoids retaining intermediate images.

### Camera intrinsics

Default mode approximates a pinhole matrix with the principal point at the image centre and `fx = fy = 0.9 × max(width, height)`. This is a pragmatic assumption, not camera calibration. The upload workspace exposes optional `fx`, `fy`, `cx`, and `cy`; supplied values are validated and rescaled when frames are resized. Accurate calibration improves pose and triangulation quality.

### Features and tracking

Shi–Tomasi supplies well-localized corners. Pyramidal LK tracks adjacent-frame positions efficiently. A backward track is also computed; tracks are accepted only when the round-trip displacement is bounded. This rejects many blur, occlusion, and dynamic-object errors before geometric estimation.

### Initialization and the Essential Matrix

The Essential Matrix relates corresponding normalized image points in two calibrated views. `cv2.findEssentialMat` uses RANSAC to reject inconsistent matches; `cv2.recoverPose` recovers relative rotation and a translation **direction**. Translation magnitude is not observable from one moving monocular camera, which is why the result scale is arbitrary.

Initialization scans a bounded early window and requires displacement, inlier support, and valid triangulated landmarks. Pure rotation or insufficient parallax produces `INITIALIZATION_FAILED`; the code never substitutes a fake map.

### Triangulation

Triangulation estimates a 3D point from the rays observed at multiple camera positions. `cv2.triangulatePoints` is followed by finite-value, positive-depth (cheirality), minimum-parallax, maximum-depth, and reprojection-error filters. Invalid geometry is discarded before mapping or visualization.

### PnP and persistent landmarks

Perspective-n-Point estimates a camera pose from known 3D landmarks and their current 2D observations. After initialization, `solvePnPRansac` plus iterative refinement anchors new poses to the existing sparse map. This is less drift-prone than blindly chaining pairwise translation estimates.

All matrices use the world-to-camera convention. Camera centres rendered by the browser are `-Rᵀt`.

### Keyframes and local map

A keyframe is a selected frame kept as an important geometric reference. The policy inserts one when any of these occurs:

- relative translation ≥ 0.18 arbitrary units;
- rotation ≥ 5 degrees;
- median parallax ≥ 12 pixels;
- tracked mapped landmarks ≤ 50; or
- 10 pose frames have elapsed.

The reusable local-map selector is bounded to five keyframes and 600 unique landmarks. The active pipeline triangulates only on keyframe insertion, limiting duplicate landmarks and CPU work.

### Bundle adjustment and drift reduction

Drift is the accumulation of many small pose errors. Local bundle adjustment jointly changes recent camera poses and landmarks so their projected positions better match the measured pixels.

This implementation uses `scipy.optimize.least_squares` with `soft_l1` loss and a sparse Jacobian. A problem is capped at five poses, 120 well-observed landmarks, 600 observations, and 25 evaluations in the pipeline. The oldest selected pose stays fixed to remove gauge freedom. Step sizes are bounded, and a non-finite, worse, or insufficiently improved candidate is rejected transactionally.

The deterministic synthetic unit problem reduces median reprojection error from approximately **7.4389 px to 0.0000 px**, while preserving the fixed anchor pose. Tests also prove that non-finite and worse solutions leave the original window unchanged.

This reduces local drift; it does not provide full global loop closure or Sim(3) pose-graph optimization.

### Scale ambiguity

A monocular camera cannot determine absolute global scale without an external reference. Returned coordinates are **relative/arbitrary units**, not metres. The UI discloses this in the explorer and accessible text summary. Display normalization affects only the Three.js view; raw relative matrices/points remain in the API result.

## API and application flow

```text
POST /api/slam-runs           validate, persist metadata, return 202, start bounded task
GET  /api/slam-runs/{uuid}    poll stage and obtain result/error
GET  /health                  process liveness only
GET  /ready                   verifies the slam_runs table is queryable
```

The browser uses real `XMLHttpRequest.upload` progress for upload bytes and polls persisted run state for reconstruction stages. A semaphore bounds CPU work to one run by default. A restart does not resume in-flight work; completed geometry is intentionally not stored in PostgreSQL, so only its run metadata survives a process restart.

## Three.js explorer and UI

The result workspace renders:

- sparse points (true point count retained; render count separately disclosed);
- camera-centre trajectory;
- keyframe markers;
- reference grid;
- orbit, pan, zoom, layer toggles, and view reset;
- poses, landmark/keyframe counts, processing time, diagnostics, and arbitrary-scale text.

The interface has purposeful empty, selected, upload-progress, processing, success, partial, and error states. Light is the deliberate first-visit default; the dark preference is stored locally. Keyboard controls, labels, visible focus, `aria-live`, a text result summary, and reduced-motion CSS cover non-pointer and assistive use.

Automated and visual checks cover 320, 375, 390, 430, 768, 1024, 1280, 1440, and 1920 pixel widths. No checked width has horizontal document overflow.

## Database and migration isolation

PostgreSQL is not required for the geometry algorithm. It records only `slam_runs` metadata:

- UUID, safe display filename, status, and current stage;
- input dimensions/frame count/duration;
- processing time and pose/point/keyframe counts;
- algorithm version and whether intrinsics were approximate;
- bounded error category/message and timestamps.

Video bytes and point arrays are never stored in PostgreSQL. Assignment 2 uses the isolated logical database `o_hive_slam_production` on the existing Render PostgreSQL service where permissions allow. It never uses Assignment 1’s logical database. Its unique Alembic revision is:

```text
20260920_slam_01_initial
```

If the service credential cannot create a logical database, the safe fallback is a dedicated role/schema plus an independently configured Alembic version table; that fallback must be documented before use rather than silently sharing Assignment 1 tables.

## Privacy and security

- Upload bytes are streamed in 1 MiB chunks to a generated UUID path and hard-stopped at 50 MiB.
- Original filenames are sanitized and used only as bounded display metadata.
- Declared MIME, extension, codec, real decode, timing, dimensions, and frame count are validated.
- The temporary file is unlinked on validation error, success, modelled failure, timeout, and unexpected exception. AWS mounts `/app/uploads` as a 64 MiB `noexec,nosuid` tmpfs.
- Production responses contain safe error categories, not tracebacks.
- Per-client sliding-window limiting defaults to four new runs/minute and returns 429 with `Retry-After`.
- CPU concurrency is one and all algorithmic dimensions are capped.
- CSP, `nosniff`, frame denial, strict referrer policy, and restrictive permissions policy are applied to every response.
- Production is same-origin; CORS is enabled only for the explicit local Vite origin.
- CloudFront is HTTPS-only to viewers; the EC2 origin accepts port 80 only from the AWS CloudFront managed prefix list.
- EC2 requires IMDSv2, uses encrypted gp3 EBS, has no SSH ingress, and reads exactly one SSM SecureString with its instance role.
- Secrets are environment variables or SSM SecureString values and never enter Git, frontend code, logs, screenshots, or CloudFormation parameters.

See [SECURITY.md](SECURITY.md) for the threat boundary and reporting guidance.

## Local setup

Prerequisites: Python 3.12+, Node.js 22+, and optionally Docker Desktop.

```powershell
git clone https://github.com/Ansh701/o-hive-monocular-slam.git
cd o-hive-monocular-slam

py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

Copy-Item .env.example .env
docker compose up -d postgres
$env:DATABASE_URL = "postgresql+asyncpg://slam_app:local-slam-only@localhost:5433/o_hive_slam"
.venv\Scripts\alembic.exe upgrade head
.venv\Scripts\uvicorn.exe backend.app.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api`, `/health`, and `/ready`; the production app uses one origin.

To generate the privacy-safe synthetic fixtures:

```powershell
.venv\Scripts\python.exe scripts\generate_test_videos.py --output-dir test-videos
```

To run the exact local benchmark:

```powershell
.venv\Scripts\python.exe scripts\benchmark_slam.py `
  --video test-videos\benchmark-10s.mp4 `
  --environment "local Windows x86_64" `
  --output benchmark-results\local-10s.json
```

### One-container local runtime

```powershell
docker compose up --build
```

The root image builds the Vite frontend, installs only the Python runtime package and headless OpenCV libraries, applies `alembic upgrade head`, then starts Uvicorn on `$PORT`.

## Environment variables

`.env.example` contains non-secret placeholders and every common tuning limit. Production requires:

| Variable | Purpose |
|---|---|
| `APP_ENV=production` | Disables API docs and development CORS |
| `DATABASE_URL` | SQLAlchemy async URL for isolated Assignment 2 PostgreSQL |
| `ALLOWED_HOSTS` | Comma-separated origin hostnames |
| `TEMP_UPLOAD_DIR` | Ephemeral upload directory |
| `MAX_CONCURRENT_RUNS` | CPU cost bound, default 1 |
| `RATE_LIMIT_RUNS_PER_MINUTE` | Public abuse bound, default 4 |
| `PROCESSING_TIMEOUT_SECONDS` | Per-run upper bound, default 45 |

Never commit a populated `.env`.

## Tests and verification

Backend:

```powershell
.venv\Scripts\python.exe -m ruff format --check backend scripts
.venv\Scripts\python.exe -m ruff check backend scripts
.venv\Scripts\python.exe -m mypy backend scripts\benchmark_slam.py
.venv\Scripts\python.exe -m pytest backend\tests -q
```

Frontend:

```powershell
cd frontend
npm ci
npm run lint
npx tsc -b
npm test -- --reporter=dot
npm run build
npm audit --audit-level=moderate
```

Infrastructure:

```powershell
cfn-lint infra\aws\app-ec2.yaml
cfn-guard validate --rules infra\aws\guard.rules --data infra\aws\app-ec2.yaml --show-summary all
```

Latest verified local result: **59 backend tests and 8 frontend tests pass**; Ruff, strict mypy, ESLint, TypeScript, Vite, cfn-lint, and CloudFormation Guard pass; npm audit reports zero vulnerabilities. A clean database upgrade creates `slam_runs` at revision `20260920_slam_01_initial`. The production-style same-origin runtime returns React at `/`, liveness at `/health`, and table-aware readiness at `/ready`.

Docker source and policy tests pass, but the final local image build must be rerun on a host with Docker because Docker is not installed on the authoring workstation. AWS user data performs the real image build during deployment.

## Benchmark methodology and current result

`scripts/generate_test_videos.py` creates a deterministic 10.0-second, 960×540, 30 FPS synthetic textured translation sequence (300 source frames). It contains only generated points/shapes and no personal footage. The measured interval begins immediately before `SlamPipeline.process` and ends when the final `SlamResult` exists; upload/network time is excluded. Wall-clock wrapper time is reported separately.

The current **local engineering benchmark** is committed at [docs/evaluation/local-10s-benchmark.json](docs/evaluation/local-10s-benchmark.json):

| Metric | Local result |
|---|---:|
| Input | `benchmark-10s.mp4` |
| Duration / source | 10.0 s / 960×540 / 30 FPS / 300 frames |
| Processing profile | 640×360 / 8 effective FPS / 75 decoded samples |
| Frames with retained diagnostics | 72 |
| Poses / keyframes / sparse points | 72 / 71 / 640 |
| Median tracking reprojection error | 1.472901 px |
| Pure processing | **3.162586 s** |
| Wall-clock wrapper | 3.174697 s |
| Ratio | 0.316259× video duration |
| Environment | Windows 11 x86_64, Intel Family 6 Model 158 |
| Local ≤10 s gate | PASS |

This local result is not substituted for the assignment’s AWS acceptance result. The AWS row will be added only after the exact same video and script run on the deployed EC2 environment.

## AWS cost strategy and manual deployment

The default is `c7i-flex.large` (2 vCPU, 4 GiB) because AWS currently lists it as Free-Tier-credit eligible for qualifying accounts created on/after 15 July 2025. Eligibility and remaining credits are account-specific; “Free Tier eligible” does not mean universally free. This account's 2026-09-20 Free Tier API response contained no EC2 allocation, so the deployment is conservatively treated as paid. The stack includes a configurable $10 monthly budget with an 80% forecast alert.

Other cost choices:

- 16 GiB gp3 root volume;
- one public IPv4 address, no Elastic IP;
- CloudFront PriceClass 100 and disabled application caching;
- no NAT Gateway, load balancer, GPU, RDS, ECR, long-retention logging, or idle worker;
- `c7i.xlarge` is an explicit measured fallback only if the 10-second AWS gate fails.

The AWS Price List API returned **$0.08479/hour** for on-demand Linux `c7i-flex.large` in `us-east-1` on 2026-09-20. AWS documents one public IPv4 at **$0.005/hour** when not covered by applicable Free Tier usage; 16 GiB gp3 is approximately **$1.28/month** at the published `$0.08/GB-month` rate. A continuously running month is therefore approximately **$66.83 before CloudFront requests/data transfer and external PostgreSQL egress**. The cost-minimizing operating plan is to keep it up only for the review window and delete the Assignment 2 stack afterward. Spot was measured around $0.0285–$0.0336/hour that day, but rejected for the official public URL because interruption plus a changing origin would reduce reviewer reliability.

### Manual AWS deployment

1. Push the reviewed commit to the public repository.
2. Create the isolated `o_hive_slam_production` database/role on the existing PostgreSQL service.
3. Store its async SQLAlchemy URL as a SecureString (do not put it on the command line if shell history is shared):

   ```powershell
   aws ssm put-parameter --region us-east-1 --name /o-hive/slam/database-url `
     --type SecureString --value '<postgresql+asyncpg URL>' --overwrite
   ```

4. Resolve the region’s AWS-managed CloudFront origin prefix list:

   ```powershell
   $prefixList = aws ec2 describe-managed-prefix-lists --region us-east-1 `
     --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing `
     --query 'PrefixLists[0].PrefixListId' --output text
   ```

5. Validate and deploy manually:

   ```powershell
   cfn-lint infra\aws\app-ec2.yaml
   cfn-guard validate --rules infra\aws\guard.rules --data infra\aws\app-ec2.yaml
   aws cloudformation deploy --region us-east-1 --stack-name o-hive-slam `
     --template-file infra\aws\app-ec2.yaml --capabilities CAPABILITY_IAM `
     --parameter-overrides GitRef='<reviewed commit SHA>' `
       CloudFrontOriginPrefixListId=$prefixList `
       BillingAlertEmail='<notification address>'
   aws cloudformation describe-stacks --region us-east-1 --stack-name o-hive-slam `
     --query 'Stacks[0].Outputs'
   ```

6. Wait for EC2 user data to build/start the container. Use Session Manager—not SSH—for `docker logs o-hive-slam` and the benchmark command.
7. Run the synthetic 10-second acceptance upload and `scripts/benchmark_slam.py` on that exact instance. If pure processing exceeds 10.0 seconds, profile first; change to `c7i.xlarge` only if measured tuning is insufficient.
8. For later code changes, update manually through Session Manager or a reviewed stack update. No webhook or pipeline watches GitHub.

Stack deletion removes the disposable app instance/volume. PostgreSQL is external and is not deleted by this template.

## Manual Render mirror

`render.yaml` defines exactly one Docker web service, `autoDeploy: false`, no worker, cron, or static frontend.

1. In the existing Render PostgreSQL service, create the separate logical database `o_hive_slam_production` and a least-privilege Assignment 2 role when the service plan permits it.
2. Create a web service from this public repository and select Docker.
3. Set `DATABASE_URL` as a secret and ensure it points to the isolated database. Do not reuse Assignment 1’s database URL.
4. Apply the remaining values from `render.yaml` and keep automatic deploys off.
5. Manually deploy the reviewed commit, verify `/health` and `/ready`, then run one benchmark upload and one invalid/low-texture case.

Render’s free-service sleep/cold-start and CPU variability make it a convenience mirror, not the official benchmark environment.

## Error and partial-result policy

Safe categories include invalid/oversized/too-long/decode failures, insufficient geometry, tracking loss, optimization failure, timeout/resource pressure, and internal failure. Messages explain what happened and what to try. One malformed video cannot create geometry or persist video bytes. If enough valid geometry exists before tracking degrades, a partial result is retained and labelled; otherwise the run fails honestly.

## Known limitations

- Global metric scale is unavailable from monocular geometry alone.
- Default intrinsics are approximate; distortion coefficients are not currently accepted.
- Low-texture scenes, pure rotation, fast motion, blur, rolling shutter, and heavy occlusion can fail.
- Dynamic objects can survive RANSAC and bias pose/map estimates.
- The map is sparse, not a dense surface reconstruction.
- There is no global loop closure, relocalization, Sim(3) pose graph, or map reuse.
- The active pipeline’s landmark propagation is LK-based; ORB keyframe descriptors are not yet wired into recovery.
- Local BA runs over a capped recent window and cannot correct old global drift.
- Results are process-memory bounded and not recoverable after application restart; only run metadata persists.
- A timed-out Python worker thread cannot be forcibly killed by `asyncio`; resource bounds minimize impact, and the temp path is unlinked, but process isolation would be stronger for hostile workloads.
- In-process jobs are suitable for this one-instance assessment, not multi-instance production scheduling.
- The public demo targets short offline uploads, not real-time live-camera SLAM.

## Future improvements

1. Calibrated distortion support and EXIF/container calibration import.
2. ORB-based relocalization and geometrically verified loop candidates.
3. Sim(3) pose-graph correction after loop closure.
4. Separate cancellable process workers without introducing a distributed queue.
5. Persist an encrypted, short-TTL result object when restart recovery is required.
6. Better dynamic-object rejection and rolling-shutter handling.
7. WebGL level-of-detail/octree rendering for larger sparse maps.
8. Arm/Graviton profiling after confirming compatible numerical wheels and equivalent accuracy.

## AI Usage

Codex and ChatGPT were used as development assistants for architecture discussion, current AWS/Free-Tier research, implementation support, deterministic test generation, debugging, security review, UI critique, documentation, and quality-gate orchestration.

Significant adopted recommendations included:

- a small same-origin React/FastAPI application instead of separate deployments;
- a classical Shi–Tomasi/LK + Essential Matrix + triangulation + map-anchored PnP pipeline;
- bounded local bundle adjustment with a robust loss and transactional acceptance;
- deterministic synthetic geometry/video fixtures rather than private footage or live paid services;
- isolated PostgreSQL run metadata with a project-specific Alembic revision;
- ephemeral raw-video handling, strict resource limits, rate limiting, and CloudFront-only origin access;
- one CPU EC2 deployment with CloudFront HTTPS, SSM operations, an AWS Budget, and manual releases.

Recommendations rejected or modified after review:

- Kubernetes, microservices, Redis/Celery, Kafka, a GPU, dense neural reconstruction, and cloud reconstruction APIs were rejected as unnecessary and less explainable.
- Global BA and full loop closure were rejected for this bounded take-home; a capped local BA window was chosen.
- ORB was evaluated and retained in tested keyframe utilities, but adjacent-frame tracking uses faster LK and the app does not claim ORB recovery or loop closure.
- Persisting videos or large point arrays was rejected; only small run metadata is stored.
- Separate frontend/backend hosting and automatic GitHub/Render/AWS deployments were rejected; one container and manual deployment meet the requested operational model.
- An Elastic IP, NAT Gateway, load balancer, RDS, and broad IAM policies were rejected to reduce fixed cost and attack surface.

The candidate reviewed the architecture, code, tests, measurements, security controls, and documentation and remains responsible for the submission. AI output was treated as reviewable engineering input, not an independent author or proof of correctness.

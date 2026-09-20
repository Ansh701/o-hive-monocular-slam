# O-HIVE Monocular Sparse SLAM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, benchmark, secure, document, and manually deploy a real monocular RGB sparse SLAM web application to AWS, with an optional Render mirror.

**Architecture:** One multistage Docker image serves a React/Three.js frontend and FastAPI API. FastAPI validates an ephemeral video, runs a bounded OpenCV/NumPy/SciPy sparse SLAM pipeline, persists only small run metadata to an isolated PostgreSQL database, and returns relative geometry for browser visualization.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLAlchemy async, asyncpg, Alembic, OpenCV headless, NumPy, SciPy, React, TypeScript, Vite, Three.js, Vitest, pytest, Docker, CloudFormation, EC2.

**Spec:** `docs/superpowers/specs/2026-09-20-monocular-slam-design.md`

## Global Constraints

- All source and artifacts stay under `C:\Users\user\Downloads\o-hive\ass2`; Assignment 1 is read-only and its AWS stack/secrets remain untouched.
- GitHub is source hosting only: no `.github/workflows`, Dependabot, Actions, CodePipeline, CodeBuild, or automatic deploys.
- AWS is the official public deployment; Render is an optional manual mirror and never substitutes for AWS.
- Monocular coordinates are relative/arbitrary scale, never meters.
- Videos are ephemeral and deleted in every terminal path; PostgreSQL stores only bounded run metadata.
- Pure SLAM processing for the exact 10-second benchmark must be at most 10.0 seconds on the documented AWS environment without fabricating or trivializing geometry.
- Migration revision is `20260920_slam_01_initial`; readiness checks the `slam_runs` table.

## Review Focus

1. A valid container with an unsupported codec must return `VIDEO_INVALID`, not crash or hang; covered in Task 2 validation tests.
2. Pure rotation or insufficient parallax must return `INITIALIZATION_FAILED`, not fake a map; covered in Task 4 geometry tests.
3. Temporary video cleanup must occur after timeout and exception as well as success; covered in Task 7 API tests.
4. Bundle adjustment must reject non-finite/worse solutions and preserve the pre-optimization state; covered in Task 6 optimizer tests.
5. Very large returned maps must retain true point count while separately bounding rendered points; covered in Task 8 serialization/frontend tests.

---

### Task 1: Repository foundation and typed domain contracts

**Files:** Create `pyproject.toml`, `.gitignore`, `.env.example`, `backend/slam/types.py`, `backend/app/config.py`, `backend/tests/test_types.py`.

**Interfaces:** Produce `CameraIntrinsics`, `Pose`, `Landmark`, `FrameDiagnostics`, `SlamResult`, `SlamFailure`, and centralized `Settings` limits consumed by all later tasks.

- [ ] Write failing tests proving validated intrinsics, 4×4 pose shape, bounded serialization, safe error categories, and configuration limits.
- [ ] Run `pytest backend/tests/test_types.py -q`; verify missing imports/classes fail.
- [ ] Implement minimal frozen dataclasses/Pydantic response schemas and settings.
- [ ] Run focused tests and the full backend suite; keep all green.
- [ ] Commit `feat: establish typed SLAM domain`.

### Task 2: Deterministic video generation, validation, and decode

**Files:** Create `backend/slam/video.py`, `backend/tests/test_video.py`, `scripts/generate_test_videos.py`.

**Interfaces:** Produce `validate_and_probe(path, settings) -> VideoMetadata`, `iter_sampled_frames(path, profile)`, and reproducible benchmark/failure fixtures.

- [ ] Write failing tests for valid MP4, wrong extension, fake video, unsupported codec, oversized bytes, excess duration/dimensions/frames, safe filename, and decode failure.
- [ ] Run the focused tests and confirm behavior—not fixture setup—fails.
- [ ] Implement OpenCV probing, bounded sampling, resizing, typed failures, and deterministic synthetic video generation.
- [ ] Run focused and full tests; generate the four required fixtures and verify metadata.
- [ ] Commit `feat: validate and decode bounded videos`.

### Task 3: Intrinsics, features, tracking, and pose utilities

**Files:** Create `backend/slam/intrinsics.py`, `backend/slam/features.py`, `backend/slam/geometry.py`, and matching tests.

**Interfaces:** Produce `build_intrinsics`, `detect_corners`, `track_lk_forward_backward`, `estimate_relative_pose`, `compose_world_to_camera`, and `camera_center`.

- [ ] Write deterministic failing tests for approximate/provided intrinsics, feature bounds, forward/backward rejection, Essential Matrix weak-support failure, pose composition, and camera-center convention.
- [ ] Run focused tests to observe expected missing behavior.
- [ ] Implement Shi-Tomasi, LK, RANSAC Essential Matrix/recoverPose, and pose utilities with explicit thresholds.
- [ ] Run focused/full suites and benchmark the feature hot path.
- [ ] Commit `feat: add geometric tracking foundation`.

### Task 4: Initialization and filtered triangulation

**Files:** Create `backend/slam/map.py`, `backend/tests/test_triangulation.py`.

**Interfaces:** Produce `triangulate_filtered(K, pose_a, pose_b, points_a, points_b, config) -> TriangulationResult` and initialization result/failure.

- [ ] Write failing synthetic tests for positive depth, finite coordinates, parallax, reprojection error, depth bounds, low texture, and pure rotation.
- [ ] Run focused tests and confirm each filter catches its intended invalid point.
- [ ] Implement calibrated projection, OpenCV triangulation, vectorized filters, and typed initialization failure.
- [ ] Run focused/full suites.
- [ ] Commit `feat: triangulate a filtered sparse map`.

### Task 5: Landmark association, PnP, and keyframes

**Files:** Create `backend/slam/tracker.py`, `backend/slam/keyframes.py`, and matching tests.

**Interfaces:** Produce `estimate_pose_pnp`, `associate_landmarks`, `should_insert_keyframe`, `Keyframe`, and bounded local-map selection.

- [ ] Write failing tests for PnP recovery/inlier minimum/failure, association bounds, motion/parallax/tracking/temporal keyframe triggers, and map-size caps.
- [ ] Verify failures, then implement ORB keyframe descriptors, LK landmark propagation, `solvePnPRansac`, refinement, and keyframe policy.
- [ ] Run focused/full suites and a synthetic sequence smoke test.
- [ ] Commit `feat: anchor poses to sparse landmarks`.

### Task 6: Sliding-window local bundle adjustment

**Files:** Create `backend/slam/bundle_adjustment.py`, `backend/tests/test_bundle_adjustment.py`.

**Interfaces:** Produce `bundle_adjust(window, observations, K, config) -> BundleAdjustmentResult` with before/after error and accepted flag.

- [ ] Write failing synthetic tests where controlled pose/landmark noise yields lower reprojection error, the oldest pose remains fixed, bounds are respected, and non-finite/worse optimizer output is rejected.
- [ ] Run focused tests and confirm the improvement assertion fails without optimization.
- [ ] Implement Rodrigues pose parameterization, capped observations, vectorized residuals, `least_squares(loss="soft_l1")`, and transactional acceptance.
- [ ] Run focused/full suites and record deterministic before/after error.
- [ ] Commit `feat: reduce local drift with bundle adjustment`.

### Task 7: End-to-end pipeline, FastAPI, persistence, and cleanup

**Files:** Create `backend/slam/pipeline.py`, `backend/app/main.py`, `backend/app/api.py`, `backend/app/models.py`, `backend/app/db.py`, `backend/app/security.py`, Alembic config/revision, and API tests.

**Interfaces:** `SlamPipeline.process(path, intrinsics, progress) -> SlamResult`; API routes from the spec; `slam_runs` metadata schema.

- [ ] Write failing tests for a good video, low texture, invalid video, lost tracking partial policy, temporary cleanup on success/error/timeout, rate/concurrency limits, persistence, actual-table readiness, headers, safe errors, and malformed IDs.
- [ ] Run tests to verify the missing API/pipeline behavior.
- [ ] Implement the bounded pipeline and synchronous processing in a worker thread behind a semaphore, structured safe logging, isolated metadata persistence, and `20260920_slam_01_initial`.
- [ ] Run focused/full suites and a clean migration.
- [ ] Commit `feat: expose persistent bounded SLAM runs`.

### Task 8: React upload workspace and Three.js explorer

**Files:** Create `frontend/package.json`, Vite/TypeScript config, `frontend/src` components, styles, API client, and tests.

**Interfaces:** Consume `/api/slam-runs`; render upload metadata, calibration controls, processing stages, errors, accessible textual summary, point cloud, trajectory, frustums, toggles, reset, and diagnostics.

- [ ] Write failing Vitest tests for empty/upload/error/processing/success/partial states, metadata, advanced intrinsics, light default/dark persistence, layer toggles, reset, and visualization downsampling count disclosure.
- [ ] Run tests and confirm the intended components are absent.
- [ ] Implement a responsive editorial glass workspace with object-URL cleanup, real upload progress, ARIA status, reduced motion, and a lazy Three.js scene.
- [ ] Run tests, ESLint, TypeScript, and Vite build; verify all required widths/light/dark using automated browser checks.
- [ ] Commit `feat: add interactive sparse map workspace`.

### Task 9: Benchmarking, Docker, and infrastructure

**Files:** Create `scripts/benchmark_slam.py`, root `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `render.yaml`, and `infra/aws/app-ec2.yaml`.

**Interfaces:** Benchmark emits machine-readable and human-readable required fields; one container serves frontend/API; CloudFormation creates one hardened manually updated EC2 application.

- [ ] Write failing tests for benchmark metric completeness, PASS/FAIL calculation, Docker/Render one-service policy, `autoDeploy: false`, no CI files, IMDSv2, encrypted EBS, no SSH, and least-privilege instance role.
- [ ] Implement benchmark reporting, multistage image, local PostgreSQL compose, manual Render manifest, and CloudFormation.
- [ ] Run local benchmark; profile and tune resize/frame/feature/BA caps without weakening geometric validity.
- [ ] Run lint/tests/build/audits, CloudFormation lint/guard, container build when a Docker engine is available, and runtime health checks.
- [ ] Commit `build: package manual AWS deployment`.

### Task 10: Documentation, public source, deployments, and acceptance

**Files:** Create/update `README.md`, `SECURITY.md`, benchmark/evaluation artifacts containing no secrets or private videos.

**Interfaces:** Public repo `Ansh701/o-hive-monocular-slam`; official AWS URL; optional Render mirror; final 44-item report.

- [ ] Write README from real code and measurements, including Mermaid architecture, freshman-friendly theory, exact `## AI Usage`, setup/deploy instructions, arbitrary-scale disclosure, limitations, and benchmark environment.
- [ ] Run a secret/PII scan and confirm `.github/workflows`/Dependabot are absent.
- [ ] Create or safely reuse the public GitHub repo, disable Actions, push verified `main`, and confirm clean equality.
- [ ] Manually deploy the reviewed commit to AWS, run the exact 10-second benchmark, and tune instance/profile until the honest result passes or record the blocker.
- [ ] Manually create the isolated Assignment 2 database/role on the existing Render PostgreSQL service and deploy the one-service mirror with `autoDeploy: false`.
- [ ] Execute the full AWS E2E, Render smoke test, workbook-free privacy cleanup check, responsive/accessibility QA, and final fresh quality gates.
- [ ] Commit documentation/measurements, redeploy the immutable final SHA manually where required, and deliver the final report with no secrets.

# Security policy

## Supported version

Only the current `main` branch and the immutable commit identified in the README are supported for this take-home application.

## Reporting

Do not open a public issue containing a secret, private video, database URL, AWS account identifier, or exploitable proof against the live demo. Contact the repository owner privately through their GitHub profile and include only the minimum reproduction information.

## Data boundary

Uploaded video is untrusted and may contain personal information. The application writes it to a generated temporary path, validates/decodes it within hard limits, processes it offline, and unlinks it in every modelled terminal path. On AWS the upload directory is an ephemeral, size-bounded `noexec,nosuid` tmpfs. The application does not store video bytes or point arrays in PostgreSQL.

PostgreSQL contains bounded run metadata: safe display filename, timing/input dimensions, status, result counts, algorithm version, error category/message, and timestamps. Production operators are responsible for deleting stale metadata according to their review-window needs.

## Implemented controls

- streamed byte-size enforcement before complete upload buffering;
- allowlisted MIME/extension plus real codec/decode/timing/dimension validation;
- generated internal filenames and path-traversal-safe display names;
- frame/duration/resolution/feature/map/concurrency/time bounds;
- per-client processing rate limit;
- safe production errors and no PII/image logging;
- same-origin production, development-only explicit CORS;
- CSP, frame denial, `nosniff`, referrer, and permissions headers;
- least-privilege EC2 role for one SSM SecureString plus Session Manager;
- IMDSv2, encrypted root volume, no SSH ingress, CloudFront-prefix-list origin rule;
- secrets excluded from source/build context; no GitHub Actions or deployment credentials.

## Operational notes

Keep `DATABASE_URL` in Render secrets or SSM Parameter Store SecureString. Never paste it into source, issues, screenshots, or logs. Rotate a credential immediately if it ever enters Git history; deleting a later commit is not sufficient.

Keep `MAX_CONCURRENT_RUNS=1` and the upload/rate limits for the public demo unless a measured capacity review supports a change. Use AWS Budget alerts and stop/delete the Assignment 2 stack when the review window ends. Do not modify the separate Assignment 1 database or `o-hive-qwen` stack.

The current async timeout cannot terminate an already-running Python thread. Inputs are tightly bounded, but a production hardening path should run geometry in a cancellable subprocess with OS CPU/memory limits.

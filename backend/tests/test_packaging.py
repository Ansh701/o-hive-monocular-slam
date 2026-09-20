from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_slam import BenchmarkMetrics, evaluate_benchmark

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_report_is_complete_and_uses_hard_ten_second_gate() -> None:
    metrics = BenchmarkMetrics(
        source="synthetic-benchmark-10s.mp4",
        video_duration_seconds=10.0,
        original_resolution="960x540",
        original_fps=30.0,
        original_frame_count=300,
        processing_resolution="640x360",
        effective_processing_fps=8.0,
        frames_decoded=75,
        frames_processed=72,
        processing_seconds=9.9,
        total_end_to_end_seconds=9.95,
        processing_ratio=0.99,
        pose_count=14,
        point_count=1_200,
        keyframe_count=5,
        median_reprojection_error_px=0.42,
        status="COMPLETED",
        environment="test",
        generated_at="2026-09-20T00:00:00Z",
    )

    report = evaluate_benchmark(metrics, threshold_seconds=10.0)

    assert report["passed"] is True
    assert report["threshold_seconds"] == 10.0
    assert set(json.loads(json.dumps(report))) >= {
        "source",
        "video_duration_seconds",
        "original_resolution",
        "original_fps",
        "original_frame_count",
        "processing_resolution",
        "effective_processing_fps",
        "frames_decoded",
        "frames_processed",
        "processing_seconds",
        "total_end_to_end_seconds",
        "processing_ratio",
        "pose_count",
        "point_count",
        "keyframe_count",
        "median_reprojection_error_px",
        "status",
        "environment",
        "generated_at",
        "threshold_seconds",
        "passed",
    }

    failed = evaluate_benchmark(
        BenchmarkMetrics(**{**metrics.__dict__, "processing_seconds": 10.01}),
        threshold_seconds=10.0,
    )
    assert failed["passed"] is False


@pytest.mark.parametrize("forbidden", [".env", "credentials", "id_rsa", ".git"])
def test_dockerignore_excludes_secret_and_build_context_noise(forbidden: str) -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").lower()
    assert forbidden in ignored


def test_dockerfile_is_one_multistage_application_image() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lowered = dockerfile.lower()
    assert lowered.count("\nfrom ") + lowered.startswith("from ") >= 2
    assert "npm ci" in dockerfile
    assert "npm run build" in dockerfile
    assert "python:3.12" in lowered
    assert "alembic upgrade head" in lowered
    assert "uvicorn" in lowered
    assert "--host 0.0.0.0" in dockerfile


def test_render_manifest_has_one_manual_web_service_and_no_worker() -> None:
    manifest = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert manifest.count("- type: web") == 1
    assert "type: worker" not in manifest
    assert "autoDeploy: false" in manifest
    assert "runtime: docker" in manifest
    assert "o_hive_slam_production" in manifest


def test_cloudformation_hardens_single_instance_without_fixed_cost_networking() -> None:
    template = (ROOT / "infra" / "aws" / "app-ec2.yaml").read_text(encoding="utf-8")
    assert "AWSAgentToolkit: aws-cloudformation@2" in template
    assert "HttpTokens: required" in template
    assert "Encrypted: true" in template
    assert "DeleteOnTermination: true" in template
    assert "AmazonSSMManagedInstanceCore" in template
    assert "CidrIp: 0.0.0.0/0" not in template
    assert "FromPort: 22" not in template
    assert "AWS::EC2::NatGateway" not in template
    assert "AWS::ElasticLoadBalancing" not in template
    assert "AWS::EC2::EIP" not in template
    assert "AWS::RDS::" not in template


def test_repository_contains_no_ci_configuration() -> None:
    assert not (ROOT / ".github" / "workflows").exists()
    assert not (ROOT / ".github" / "dependabot.yml").exists()

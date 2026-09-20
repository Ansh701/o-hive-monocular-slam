"""Create isolated SLAM run metadata.

Revision ID: 20260920_slam_01_initial
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "20260920_slam_01_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "slam_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("source_filename", sa.String(length=120), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("processing_seconds", sa.Float(), nullable=True),
        sa.Column("pose_count", sa.Integer(), nullable=False),
        sa.Column("point_count", sa.Integer(), nullable=False),
        sa.Column("keyframe_count", sa.Integer(), nullable=False),
        sa.Column("algorithm_version", sa.String(length=40), nullable=False),
        sa.Column("intrinsics_approximate", sa.Boolean(), nullable=False),
        sa.Column("error_code", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_slam_runs_status", "slam_runs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_slam_runs_status", table_name="slam_runs")
    op.drop_table("slam_runs")

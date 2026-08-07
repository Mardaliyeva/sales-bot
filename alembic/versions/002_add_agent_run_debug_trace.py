"""Add agent run debug trace.

Revision ID: 002_agent_run_debug_trace
Revises: 001_vertical_slice
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "002_agent_run_debug_trace"
down_revision: str | None = "001_vertical_slice"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "debug_trace",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "debug_trace")

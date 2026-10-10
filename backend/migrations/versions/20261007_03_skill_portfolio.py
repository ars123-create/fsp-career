"""Keep private provenance for the current fixed assessment family set."""
from alembic import op
import sqlalchemy as sa

revision = "20261007_03"
down_revision = "20261007_02"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("candidate", sa.Column("skill_portfolio", sa.JSON(), nullable=False, server_default="{}"))


def downgrade():
    op.drop_column("candidate", "skill_portfolio")

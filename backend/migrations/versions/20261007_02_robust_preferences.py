"""Save employer priorities alongside the existing search snapshot."""
from alembic import op
import sqlalchemy as sa

revision = "20261007_02"
down_revision = "20261007_01"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("need", sa.Column("ranking", sa.JSON(), nullable=False, server_default="{}"))


def downgrade():
    op.drop_column("need", "ranking")

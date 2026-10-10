"""Create local configuration without overwriting any existing value or secret."""
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1]
path = root / ".env"
existing = path.read_text() if path.exists() else ""
values = {"FSP_SECRET_KEY": secrets.token_urlsafe(48), "FSP_POSTGRES_PASSWORD": secrets.token_urlsafe(32),
          "FSP_DEMO_MODE": "true", "FSP_FRONTEND_URL": "http://127.0.0.1:8018"}
keys = {line.split("=", 1)[0].strip() for line in existing.splitlines() if "=" in line and not line.startswith("#")}
addition = "\n".join(f"{k}={v}" for k, v in values.items() if k not in keys)
if addition:
    path.write_text(existing.rstrip() + "\n" + addition + "\n")
    path.chmod(0o600)
(root / "data").mkdir(exist_ok=True)
print("Local configuration ready; existing values preserved. Secrets are not printed.")

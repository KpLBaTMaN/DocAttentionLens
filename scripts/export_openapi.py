"""Export transport contracts without connecting to metadata or a worker."""
import json
from pathlib import Path

from docattentionlens_api.main import create_app

root = Path(__file__).resolve().parents[1]
destination = root / "contracts" / "openapi.json"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(create_app().openapi(), indent=2) + "\n", encoding="utf-8")
print("Exported contracts/openapi.json")

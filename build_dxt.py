"""Build pinpoint.dxt for Claude Desktop."""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "pinpoint_mcp"
DATA = ROOT / "Data_Set" / "clean_data"
OUT = Path.home() / "Library/Application Support/Claude/Claude Extensions/pinpoint.dxt"

manifest = {
    "manifest_version": "0.3",
    "name": "pinpoint",
    "display_name": "Pinpoint",
    "version": "1.0.0",
    "description": "Business location recommendation engine — ranked county recommendations, trend analysis, and side-by-side comparisons across US counties.",
    "author": {"name": "Baron Kim"},
    "server": {
        "type": "python",
        "entry_point": "server/main.py",
        "mcp_config": {
            "command": "/Library/Frameworks/Python.framework/Versions/3.13/bin/uv",
            "args": [
                "run",
                "--with", "mcp[cli]>=1.0.0",
                "--with", "pandas>=2.0.0",
                "--with", "numpy>=1.24.0",
                "${__dirname}/server/main.py"
            ],
            "env": {
                "PYTHONPATH": "${__dirname}/server"
            }
        }
    },
    "tools": [
        {"name": "recommend_locations",  "description": "Ranked county recommendations for a business type"},
        {"name": "analyze_county",       "description": "Deep-dive profile of a specific county"},
        {"name": "get_industry_trends",  "description": "GDP time-series for an industry in a state/county"},
        {"name": "compare_locations",    "description": "Side-by-side score comparison of specific counties"},
        {"name": "list_supported_regions","description": "Enumerate available regions and states"}
    ]
}

MAIN_PY = """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinpoint_mcp.server import mcp
mcp.run()
"""

def should_skip(path: Path) -> bool:
    return any(p in ("__pycache__", ".DS_Store") for p in path.parts)

OUT.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    zf.writestr("server/main.py", MAIN_PY)

    # pinpoint_mcp package
    for f in SRC.rglob("*"):
        if f.is_file() and not should_skip(f):
            zf.write(f, f"server/pinpoint_mcp/{f.relative_to(SRC)}")

    # BEA data files
    for f in DATA.glob("*.csv"):
        zf.write(f, f"server/Data_Set/clean_data/{f.name}")

print(f"Built {OUT}")
print(f"Size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")

#!/usr/bin/env python3
"""Pre-push checks for the PSCYPHER repository. Run from the repository root: python scripts/check_repo.py
Exits non-zero if anything that must not be published (or must be present) is wrong."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
errors, warnings = [], []
script = ROOT / "Pscypher_nebius.py"
text = script.read_text(encoding="utf-8")

for name in ("NEBIUS_API_KEY_DEFAULT", "TAVILY_API_KEY_DEFAULT"):
    m = re.search(rf"^{name}\s*=\s*(.*?)\s*(#.*)?$", text, re.M)
    if not m or m.group(1).strip() not in ('""', "''"):
        errors.append(f"{name} must be an empty string in committed code")

for pat, label in ((r"tvly-[A-Za-z0-9]{8,}", "Tavily-style key"), (r"sk-[A-Za-z0-9]{16,}", "API-key-like string")):
    for p in (script, ROOT / "README.md"):
        if re.search(pat, p.read_text(encoding="utf-8")):
            errors.append(f"{label} found in {p.name}")

for pat in ("python-Levenshtein", "gemini", "google.generativeai"):
    if re.search(pat, text, re.I):
        errors.append(f"'{pat}' still appears in Pscypher_nebius.py")

for junk in ("pscypher_town.sqlite3", "pscypher_saved_answers.json"):
    if (ROOT / junk).exists():
        errors.append(f"remove local state file before pushing: {junk}")
for p in ROOT.rglob("*"):
    if p.is_file() and (p.name.endswith("_transcription.txt") or "_fir_cache_" in p.name):
        errors.append(f"remove generated FIR file (contains local paths): {p.relative_to(ROOT)}")

data = ROOT / "pscypher_demo_dataset"
for f in ("town_10000_citizens.csv", "town_relationships.csv", "cdr_500_demo.csv", "financial_200_demo.csv",
          "README_DEMO.txt", "ground_truth/ground_truth_DEMO_ONLY.csv", "ground_truth/ground_truth_links_DEMO_ONLY.csv"):
    if not (data / f).is_file():
        errors.append(f"missing demo file: pscypher_demo_dataset/{f}")
firs = [p for p in data.glob("*.pdf") if p.name.upper().startswith("FIR_")]
if len(firs) < 3:
    errors.append(f"demo needs 3 FIR_*.pdf files in pscypher_demo_dataset/, found {len(firs)}")

# docs: screenshots and evaluation referenced by the README
for f in ("docs/screenshots/graph_hierarchy.png", "docs/screenshots/graph_network.png",
          "docs/evaluation/pscypher_evaluation.md", "docs/evaluation/pscypher_evaluation.json"):
    if not (ROOT / f).is_file():
        errors.append(f"missing {f}")
readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
for ref in re.findall(r"\]\((docs/[^)\s]+)\)", readme_text):
    if not (ROOT / ref.rstrip("/")).exists():
        errors.append(f"README links to a missing path: {ref}")

# the evaluation must have been run on a clean demo load: 500 call rows + 200 transaction rows = 700 observations
ev = ROOT / "docs/evaluation/pscypher_evaluation.json"
if ev.is_file():
    import json
    try:
        obs = json.loads(ev.read_text(encoding="utf-8")).get("observations")
        if obs != 700:
            errors.append(f"evaluation reports {obs} observations; a clean demo load has 700 "
                          "(a CSV was probably imported twice). Reload with Load Demo Case, re-run the evaluation, "
                          "replace docs/evaluation/ and update README section 6")
    except (ValueError, OSError) as exc:
        errors.append(f"cannot read evaluation JSON: {exc}")

for f in ("LICENSE", "THIRD_PARTY.md", "README.md", "requirements.txt"):
    if not (ROOT / f).is_file():
        errors.append(f"missing {f}")
readme = readme_text
n_fill = len(re.findall(r"\[FILL", readme))
if n_fill:
    warnings.append(f"README.md still has {n_fill} [FILL] placeholders")

for w in warnings:
    print("WARN :", w)
for e in errors:
    print("ERROR:", e)
print("OK" if not errors else f"{len(errors)} error(s)")
sys.exit(1 if errors else 0)

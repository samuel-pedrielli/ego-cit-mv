import json
import collections
from pathlib import Path

p = Path("prompts/critic_calib_v0.jsonl")
assert p.exists(), f"Missing: {p}"

rows = []
bad = 0

with p.open(encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            rows.append(r)
        except Exception as e:
            bad += 1
            print(f"BAD LINE {i}: {e}")

print(f"ROWS: {len(rows)}  BAD: {bad}")

keys = set()
for r in rows:
    keys.update(r.keys())
print("KEYS:", sorted(keys))

print("h_C counts:", collections.Counter(r.get("h_C") for r in rows))

neg_rule_counts = collections.Counter()
for r in rows:
    if r.get("h_C") == 0:
        for rr in r.get("violated_rules", []):
            neg_rule_counts[rr] += 1
print("neg per rule:", dict(neg_rule_counts))
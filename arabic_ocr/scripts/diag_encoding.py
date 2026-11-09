"""Determine the TRUE encoding of labels.json."""
import json
from pathlib import Path

p = Path("data/hflines/val/labels.json")
raw = p.read_bytes()
print("file size:", len(raw))
print("first 120 bytes:", raw[:120])

# Locate the first "text" value's raw bytes
import re
m = re.search(rb'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
print("\nraw text bytes (first match):", m.group(1)[:60] if m else None)

if m:
    b = m.group(1)
    print("hex:", b[:40].hex())
    for enc in ["utf-8", "cp1256", "utf-16", "latin-1"]:
        try:
            print(f"  {enc:9s} -> {b.decode(enc)[:50]}")
        except Exception as e:
            print(f"  {enc:9s} -> FAIL {type(e).__name__}")

print()
for enc in ["utf-8", "cp1256", "utf-8-sig"]:
    try:
        data = json.loads(raw.decode(enc))
        print(f"json.loads({enc!r}) OK -> {data[0]['text'][:50]!r}")
    except Exception as e:
        print(f"json.loads({enc!r}) FAIL {type(e).__name__}: {str(e)[:80]}")

# Also check the train split
p2 = Path("data/hflines/train/labels.json")
raw2 = p2.read_bytes()
m2 = re.search(rb'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw2)
print("\ntrain raw text bytes:", m2.group(1)[:60] if m2 else None)
if m2:
    b2 = m2.group(1)
    for enc in ["utf-8", "cp1256"]:
        try:
            print(f"  {enc:9s} -> {b2.decode(enc)[:50]}")
        except Exception as e:
            print(f"  {enc:9s} -> FAIL {type(e).__name__}")

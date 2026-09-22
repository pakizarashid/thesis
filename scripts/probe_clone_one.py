"""
probe_clone_one.py -- diagnostic only, writes NOTHING.

apply_gentext_patch.py has now aborted twice on the same anchor (edit 1, the
clone_one() function). That means my cached copy of clone_one() doesn't match
your live src/eval/cloner_watermark_eval.py -- guessing a third anchor blind
isn't worth another round trip. This prints the ACTUAL live function (repr'd, so
whitespace/indentation/line-endings are visible) plus a few sanity checks, so the
patch can be built from truth instead of assumption.

Run in a fresh cell:
    python probe_clone_one.py
or from repo root implicitly (same convention as every other script here).
"""
import re

TARGET = "src/eval/cloner_watermark_eval.py"
content = open(TARGET, "r").read()

print("=== file stats ===")
print(f"length: {len(content)} chars")
print(f"contains CRLF (\\r\\n): {chr(13)+chr(10) in content}")
print(f"already has --gen_text_from_transcript: {'--gen_text_from_transcript' in content}")
print()

m = re.search(r"def clone_one\(.*?\n(?:.*\n)*?        return clone_fn\([^\n]*\)\n", content)
if m:
    print("=== clone_one(), exact repr ===")
    print(repr(m.group(0)))
else:
    print("!!! 'def clone_one(' block not found by regex -- trying a looser search.")
    idx = content.find("def clone_one(")
    if idx == -1:
        print("!!! 'def clone_one(' does not appear in the file AT ALL.")
    else:
        print("=== 400 chars from 'def clone_one(' onward, exact repr ===")
        print(repr(content[idx:idx + 400]))

print()
print("=== the two other anchors this patch also needs (diagnostic/main-loop call sites) ===")
for label, needle in [
    ("diagnostic call site", 'cloned = clone_one(recon_wm, "diag")'),
    ("main-loop call site", 'cloned = clone_one(recon_wm, f"u{i}")'),
]:
    print(f"{label}: present={needle in content}")
    if needle not in content:
        # show what's actually around 'clone_one(recon_wm,' near this label's rough area
        for mm in re.finditer(r".{0,60}clone_one\(recon_wm,.{0,60}", content):
            print(f"    nearby: {mm.group(0)!r}")

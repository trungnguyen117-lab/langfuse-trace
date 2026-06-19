import json
import sys
import argparse

from dotenv import load_dotenv

import langfuse_latency as L

load_dotenv()
ns = argparse.Namespace(base_url=None, public_key=None, secret_key=None)
client = L.build_client(ns)

trace_id = sys.argv[1]

# --- Trace level: name + latency ---
trace = client.fetch_trace(trace_id)
print("=== TRACE LEVEL ===")
print("  name   :", trace.get("name"))
print("  latency:", trace.get("latency"), f"({type(trace.get('latency')).__name__})")
print()

# --- Generations: dump every key + time-related fields + metadata ---
gens = client.fetch_generations(trace_id=trace_id)
print(f"=== {len(gens)} GENERATION(S) ===")
for i, g in enumerate(gens):
    print(f"--- #{i} name={g.get('name')!r} ---")
    print("  all keys:", sorted(g.keys()))
    for k in ("startTime", "completionStartTime", "endTime", "latency", "timeToFirstToken"):
        print(f"    {k}: {g.get(k)!r}")
    md = g.get("metadata")
    print("    metadata type:", type(md).__name__)
    if isinstance(md, dict):
        # surface anything that smells like first-token / timing
        hits = {k: v for k, v in md.items()
                if any(s in k.lower() for s in ("token", "chunk", "first", "latency", "ms", "time"))}
        print("    metadata timing-ish:", json.dumps(hits, ensure_ascii=False)[:500])
        print("    metadata all keys:", sorted(md.keys())[:40])
    print()

import json
import sys
import argparse

from dotenv import load_dotenv

import langfuse_latency as L

load_dotenv()
ns = argparse.Namespace(base_url=None, public_key=None, secret_key=None)
client = L.build_client(ns)

trace_id = sys.argv[1]

# --- Trace level ---
trace = client.fetch_trace(trace_id)
print("=== TRACE LEVEL ===")
print("trace.input  type:", type(trace.get("input")).__name__)
print(json.dumps(trace.get("input"), ensure_ascii=False, indent=2)[:1200])
print("trace.output type:", type(trace.get("output")).__name__)
print(json.dumps(trace.get("output"), ensure_ascii=False, indent=2)[:600])
print()

# --- Generation level ---
gens = client.fetch_generations(trace_id=trace_id)
print(f"=== {len(gens)} GENERATION(S) ===")
for i, g in enumerate(gens):
    print(f"--- #{i} name={g.get('name')!r} ---")
    print("  input  type:", type(g.get("input")).__name__,
          "| output type:", type(g.get("output")).__name__)

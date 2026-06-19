import json, sys
import langfuse_latency as L
import argparse

ns = argparse.Namespace(base_url=None, public_key=None, secret_key=None)
# load .env like main() does
from dotenv import load_dotenv; load_dotenv()
client = L.build_client(ns)
trace_id = sys.argv[1]
gens = client.fetch_generations(trace_id=trace_id)
print(f"{len(gens)} generation(s)\n")
for i, g in enumerate(gens):
    print(f"=== generation #{i} name={g.get('name')!r} ===")
    print("INPUT type:", type(g.get("input")).__name__)
    print(json.dumps(g.get("input"), ensure_ascii=False, indent=2)[:1200])
    print()

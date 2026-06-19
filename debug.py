import sys
import argparse

from dotenv import load_dotenv

import langfuse_latency as L

load_dotenv()
ns = argparse.Namespace(base_url=None, public_key=None, secret_key=None)
client = L.build_client(ns)

trace_id = sys.argv[1]

# Pull ALL observations (every type), so we can see SPAN -> GENERATION nesting.
obs = client._get(
    "/api/public/observations",
    {"traceId": trace_id, "limit": 100},
).get("data", [])

print(f"{len(obs)} observation(s) in trace {trace_id}\n")

by_id = {o["id"]: o for o in obs}
children: dict = {}
for o in obs:
    children.setdefault(o.get("parentObservationId"), []).append(o)


def short(o):
    return (
        f"[{o.get('type')}] {o.get('name')!r} "
        f"start={o.get('startTime')} "
        f"ttft={o.get('timeToFirstToken')} latency={o.get('latency')}"
    )


def walk(parent_id, depth):
    kids = sorted(
        children.get(parent_id, []),
        key=lambda o: o.get("startTime") or "",
    )
    for o in kids:
        print("  " * depth + "- " + short(o))
        walk(o["id"], depth + 1)


# Roots = observations whose parent is missing or not in this trace.
roots = [o for o in obs if o.get("parentObservationId") not in by_id]
roots.sort(key=lambda o: o.get("startTime") or "")
for r in roots:
    print(short(r))
    walk(r["id"], 1)

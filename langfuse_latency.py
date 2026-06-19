#!/usr/bin/env python3
"""Query Langfuse for LLM latency metrics.

Pulls GENERATION observations straight from the Langfuse public REST API and
reports, per generation:

  - Time to first token (TTFT) = completionStartTime - startTime
  - Total generation time      = endTime - startTime

A "question" is one Langfuse trace, which may contain several model calls
(generations) when the agent loops through tool steps. Metrics are reported
per question:

  TTFT of the question  = TTFT of its earliest generation (first token seen).
  Total of the question = max(endTime) - min(startTime) across the trace.

Two modes:

  trace   Look up a single question by its trace id, show every generation,
          plus the question's TTFT and total time.

  range   Group generations by trace into questions over a time window and
          report mean + p95 for both TTFT and total time (export to CSV too).

TTFT is only available when the model response was streamed and Langfuse
recorded `completionStartTime`. When it is missing the script reports the
generation with TTFT = None and counts it separately instead of treating it
as zero (which would skew the statistics).

Config is read from the same env vars the backend uses:

  LANGFUSE_BASE_URL    e.g. https://cloud.langfuse.com  (default)
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY

Examples:

  python scripts/langfuse_latency.py trace --trace-id abc123
  python scripts/langfuse_latency.py range \\
      --from 2026-06-18T00:00:00Z --to 2026-06-19T00:00:00Z --csv out.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://cloud.langfuse.com"
PAGE_LIMIT = 100
REQUEST_TIMEOUT = 30


class LangfuseClient:
    """Thin wrapper over the Langfuse public REST API (Basic Auth)."""

    def __init__(self, base_url: str, public_key: str, secret_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        token = base64.b64encode(
            f"{public_key}:{secret_key}".encode("utf-8")
        ).decode("ascii")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Basic {token}"})

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def fetch_generations(
        self,
        *,
        trace_id: str | None = None,
        from_start_time: str | None = None,
        to_start_time: str | None = None,
        name: str | None = None,
    ) -> list[dict]:
        """Fetch all GENERATION observations matching the filters (paginated).

        Uses the stable v1 `GET /api/public/observations` endpoint, which works
        on self-hosted Langfuse and returns full observation objects (including
        `completionStartTime`, `input` and `output`) by default. The v2
        endpoint with its `fields` selector is Beta and currently Cloud-only.
        """
        observations: list[dict] = []
        page = 1
        while True:
            params = {
                "type": "GENERATION",
                "limit": PAGE_LIMIT,
                "page": page,
            }
            if trace_id is not None:
                params["traceId"] = trace_id
            if from_start_time is not None:
                params["fromStartTime"] = from_start_time
            if to_start_time is not None:
                params["toStartTime"] = to_start_time
            if name is not None:
                params["name"] = name

            payload = self._get("/api/public/observations", params)
            observations.extend(payload.get("data", []))

            meta = payload.get("meta", {})
            total_pages = meta.get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1

        return observations

    def fetch_trace(self, trace_id: str) -> dict:
        return self._get(f"/api/public/traces/{trace_id}")


def parse_ts(value: str | None) -> datetime | None:
    """Parse a Langfuse ISO-8601 timestamp into an aware datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_timings(obs: dict) -> tuple[float | None, float | None]:
    """Return (ttft_seconds, total_seconds), each None when not derivable."""
    start = parse_ts(obs.get("startTime"))
    end = parse_ts(obs.get("endTime"))
    completion_start = parse_ts(obs.get("completionStartTime"))

    ttft = (
        (completion_start - start).total_seconds()
        if start is not None and completion_start is not None
        else None
    )
    total = (
        (end - start).total_seconds()
        if start is not None and end is not None
        else None
    )
    return ttft, total


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolation percentile. `fraction` in [0, 1]. None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(seconds: float | None) -> str:
    return f"{seconds:.3f}s" if seconds is not None else "—"


def content_to_text(content: object) -> str:
    """Flatten a message `content` (string / parts list / message dict) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    if isinstance(content, dict):
        return content_to_text(content.get("content"))
    return str(content)


def extract_question(input_value: object) -> str:
    """The user's question = the last `user` message in the generation input."""
    if isinstance(input_value, list):
        for message in reversed(input_value):
            if isinstance(message, dict) and message.get("role") == "user":
                return content_to_text(message.get("content"))
        if input_value:
            return content_to_text(input_value[-1])
        return ""
    return content_to_text(input_value)


def extract_answer(output_value: object) -> str:
    """The answer = the assistant text in the generation output."""
    return content_to_text(output_value)


def truncate(text: str, length: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 1] + "…"


def earliest_start(obs: dict) -> datetime:
    """Sort key: startTime, with missing timestamps pushed to the end."""
    ts = parse_ts(obs.get("startTime"))
    return ts if ts is not None else datetime.max.replace(tzinfo=timezone.utc)


def group_into_questions(generations: list[dict]) -> list[dict]:
    """Aggregate generations by trace into one row per question.

    A question = a trace (which may contain several model calls). Per the
    user-facing definition:

      TTFT of the question  = TTFT of its earliest generation (the first token
                              the user actually sees).
      Total of the question = max(endTime) - min(startTime) across the trace's
                              generations (covers the tool-calling steps in
                              between).
    """
    by_trace: dict[str, list[dict]] = {}
    for obs in generations:
        by_trace.setdefault(obs.get("traceId") or "", []).append(obs)

    questions: list[dict] = []
    for trace_id, gens in by_trace.items():
        gens_sorted = sorted(gens, key=earliest_start)
        first = gens_sorted[0]
        last = gens_sorted[-1]
        ttft, _ = compute_timings(first)

        starts = [parse_ts(o.get("startTime")) for o in gens]
        ends = [parse_ts(o.get("endTime")) for o in gens]
        starts = [s for s in starts if s is not None]
        ends = [e for e in ends if e is not None]
        total = (
            (max(ends) - min(starts)).total_seconds()
            if starts and ends
            else None
        )

        questions.append(
            {
                "trace_id": trace_id,
                "name": first.get("name") or "",
                "start_time": min(starts).isoformat() if starts else "",
                "generations": len(gens),
                "ttft_seconds": ttft,
                "total_seconds": total,
                # Question = first call's user message; answer = last call's
                # output (after any tool-calling steps in between).
                "question": extract_question(first.get("input")),
                "answer": extract_answer(last.get("output")),
            }
        )

    questions.sort(key=lambda q: q["start_time"])
    return questions


def build_client(args: argparse.Namespace) -> LangfuseClient:
    base_url = args.base_url or os.environ.get(
        "LANGFUSE_BASE_URL"
    ) or DEFAULT_BASE_URL
    public_key = args.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = args.secret_key or os.environ.get("LANGFUSE_SECRET_KEY")

    missing = [
        name
        for name, value in (
            ("LANGFUSE_PUBLIC_KEY", public_key),
            ("LANGFUSE_SECRET_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        sys.exit(
            f"Missing required credentials: {', '.join(missing)}. "
            "Set them as env vars or pass --public-key/--secret-key."
        )

    return LangfuseClient(base_url, public_key, secret_key)


def cmd_trace(args: argparse.Namespace) -> None:
    client = build_client(args)
    generations = client.fetch_generations(trace_id=args.trace_id)

    if not generations:
        print(f"No generations found for trace {args.trace_id}.")
        return

    print(f"Trace {args.trace_id} — {len(generations)} generation(s)\n")
    header = f"{'name':<32} {'model':<24} {'TTFT':>10} {'total':>10}"
    print(header)
    print("-" * len(header))

    starts: list[datetime] = []
    ends: list[datetime] = []
    for obs in generations:
        ttft, total = compute_timings(obs)
        name = (obs.get("name") or "")[:32]
        model = (obs.get("model") or obs.get("providedModelName") or "")[:24]
        print(f"{name:<32} {model:<24} {fmt(ttft):>10} {fmt(total):>10}")

        start = parse_ts(obs.get("startTime"))
        end = parse_ts(obs.get("endTime"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)

    # Total time of the whole question: prefer the trace's own latency, else
    # span from the earliest generation start to the latest generation end.
    total_question = None
    trace: dict = {}
    try:
        trace = client.fetch_trace(args.trace_id)
        latency = trace.get("latency")
        if isinstance(latency, (int, float)):
            total_question = float(latency)
    except requests.HTTPError:
        pass
    if total_question is None and starts and ends:
        total_question = (max(ends) - min(starts)).total_seconds()

    gens_sorted = sorted(generations, key=earliest_start)
    # TTFT of the question = TTFT of the earliest generation (first token the
    # user actually sees). Q&A: prefer the trace's own input/output, else fall
    # back to the first call's input and the last call's output.
    question_ttft, _ = compute_timings(gens_sorted[0])
    question = trace.get("input") if trace.get("input") is not None else None
    question = (
        content_to_text(question)
        if question is not None
        else extract_question(gens_sorted[0].get("input"))
    )
    answer = trace.get("output") if trace.get("output") is not None else None
    answer = (
        content_to_text(answer)
        if answer is not None
        else extract_answer(gens_sorted[-1].get("output"))
    )

    limit = None if args.full else 500
    print("\n" + "=" * len(header))
    print("Question:")
    print(f"  {truncate(question, limit) if limit else question or '—'}")
    print("Answer:")
    print(f"  {truncate(answer, limit) if limit else answer or '—'}")
    print()
    print(f"Time to first token (question): {fmt(question_ttft)}")
    print(f"Total time for the question:    {fmt(total_question)}")


def cmd_range(args: argparse.Namespace) -> None:
    client = build_client(args)
    generations = client.fetch_generations(
        from_start_time=getattr(args, "from"),
        to_start_time=args.to,
        name=args.name,
    )

    if not generations:
        print("No generations found in the given window.")
        return

    questions = group_into_questions(generations)

    ttft_values = [
        q["ttft_seconds"] for q in questions if q["ttft_seconds"] is not None
    ]
    total_values = [
        q["total_seconds"] for q in questions if q["total_seconds"] is not None
    ]
    missing_ttft = sum(1 for q in questions if q["ttft_seconds"] is None)

    print(
        f"{len(questions)} question(s) / {len(generations)} generation(s) "
        f"[{getattr(args, 'from')} → {args.to}]"
        + (f" name={args.name}" if args.name else "")
        + "\n"
    )

    header = (
        f"{'start_time':<26} {'gens':>5} {'TTFT':>10} {'total':>10}  question"
    )
    print(header)
    print("-" * len(header))
    for q in questions:
        print(
            f"{q['start_time']:<26} {q['generations']:>5} "
            f"{fmt(q['ttft_seconds']):>10} {fmt(q['total_seconds']):>10}  "
            f"{truncate(q['question'], 60)}"
        )

    print("\n" + "=" * len(header))
    print("Statistics per question (seconds)")
    print(f"{'metric':<16} {'count':>8} {'mean':>10} {'p95':>10}")
    print(
        f"{'TTFT':<16} {len(ttft_values):>8} "
        f"{fmt(mean(ttft_values)):>10} {fmt(percentile(ttft_values, 0.95)):>10}"
    )
    print(
        f"{'total':<16} {len(total_values):>8} "
        f"{fmt(mean(total_values)):>10} {fmt(percentile(total_values, 0.95)):>10}"
    )
    if missing_ttft:
        print(
            f"\nNote: {missing_ttft}/{len(questions)} question(s) had no "
            "completionStartTime on the first generation (non-streamed) — "
            "excluded from TTFT stats."
        )

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "trace_id",
                    "name",
                    "start_time",
                    "generations",
                    "ttft_seconds",
                    "total_seconds",
                    "question",
                    "answer",
                ],
            )
            writer.writeheader()
            writer.writerows(questions)
        print(f"\nWrote {len(questions)} rows to {args.csv}")


def main() -> None:
    # Load LANGFUSE_* vars from a local .env (real env vars still take
    # precedence — load_dotenv does not override what's already set).
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Query Langfuse for TTFT and total generation time.",
    )
    parser.add_argument("--base-url", help="Override LANGFUSE_BASE_URL")
    parser.add_argument("--public-key", help="Override LANGFUSE_PUBLIC_KEY")
    parser.add_argument("--secret-key", help="Override LANGFUSE_SECRET_KEY")

    sub = parser.add_subparsers(dest="command", required=True)

    p_trace = sub.add_parser("trace", help="Inspect a single trace (question)")
    p_trace.add_argument("--trace-id", required=True)
    p_trace.add_argument(
        "--full",
        action="store_true",
        help="Print the full question/answer text instead of truncating",
    )
    p_trace.set_defaults(func=cmd_trace)

    p_range = sub.add_parser("range", help="List + summarize a time window")
    p_range.add_argument(
        "--from",
        required=True,
        help="ISO-8601 start, e.g. 2026-06-18T00:00:00Z",
    )
    p_range.add_argument(
        "--to",
        required=True,
        help="ISO-8601 end, e.g. 2026-06-19T00:00:00Z",
    )
    p_range.add_argument("--name", help="Filter by generation name")
    p_range.add_argument("--csv", help="Write per-question rows to this path")
    p_range.set_defaults(func=cmd_range)

    args = parser.parse_args()
    try:
        args.func(args)
    except requests.HTTPError as error:
        response = error.response
        sys.exit(
            f"Langfuse API error: {response.status_code} "
            f"{response.reason} — {response.text[:300]}"
        )
    except requests.RequestException as error:
        sys.exit(f"Request failed: {error}")


if __name__ == "__main__":
    main()

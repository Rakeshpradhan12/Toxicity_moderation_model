#!/usr/bin/env python3
"""Unified Content Moderation Benchmark & Stress Testing Tool.

Measures throughput (RPS), latency percentiles (P50/P90/P95/P99), error rates,
and monitors server CPU/RAM utilization during load.

Works with zero external dependencies (pure Python standard library).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import statistics
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Sample test comments across English, Hindi, Hinglish, Odia
SAMPLE_TEXTS = [
    "This is a wonderful and informative post, thank you!",
    "Great work team! Keep it up.",
    "This is completely stupid and useless, you idiot.",
    "Bohut badiya post hai bhai, shukriya!",
    "Tu bilkul pagal hai kya? Bakwas mat kar.",
    "Dhanyabad apananka sundar lekha pai.",
    "Can you please review the attached document?",
    "Worst service ever, totally disgusted with this.",
    "Kya baat hai, bohot khoob!",
    "Please share your phone number and email for verification.",
]


def create_minimal_jpeg(width: int = 128, height: int = 128) -> bytes:
    """Creates a minimal valid JPEG image in pure Python (no PIL dependency required)."""
    # 1x1 minimal valid JPEG binary fallback
    return bytes.fromhex(
        "ffd8ffe000104a46494600010101004800480000ffdb00430008060607060508070707090908"
        "0a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c303134"
        "34341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101"
        "010100000000000000000102030405060708090a0bffda0008010100003f007f00ffd9"
    )


def fetch_system_metrics(base_url: str) -> Optional[Dict[str, Any]]:
    """Polls /metrics endpoint for server CPU and RAM statistics."""
    url = f"{base_url.rstrip('/')}/metrics"
    req = urllib.request.Request(url, headers={"User-Agent": "ModerationBenchmark/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None


class CPUMonitor(threading.Thread):
    """Background thread that samples server CPU & Memory every interval."""

    def __init__(self, base_url: str, interval: float = 0.5):
        super().__init__(daemon=True)
        self.base_url = base_url
        self.interval = interval
        self.running = True
        self.samples: List[Dict[str, Any]] = []

    def run(self):
        while self.running:
            metrics = fetch_system_metrics(self.base_url)
            if metrics:
                self.samples.append(metrics)
            time.sleep(self.interval)

    def stop(self):
        self.running = False


def send_http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
    timeout: float = 15.0,
) -> Tuple[int, float, Optional[str]]:
    """Sends an HTTP request and measures latency in milliseconds."""
    req_headers = headers or {}
    if "User-Agent" not in req_headers:
        req_headers["User-Agent"] = "ModerationBenchmark/1.0"

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read()
            latency_ms = (time.perf_counter() - start) * 1000.0
            return resp.status, latency_ms, None
    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return e.code, latency_ms, str(e.reason)
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return 0, latency_ms, str(e)


def build_multipart_body(file_bytes: bytes, filename: str = "test.jpg") -> Tuple[bytes, str]:
    """Constructs multipart/form-data payload in pure Python."""
    boundary = "----BenchmarkBoundary" + str(int(time.time() * 1000))
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(file_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    content_type = f"multipart/form-data; boundary={boundary}"
    return body.getvalue(), content_type


def run_benchmark_batch(
    base_url: str,
    endpoint: str,
    total_requests: int,
    concurrency: int,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes a load test against the designated endpoint with concurrency."""
    base = base_url.rstrip("/")
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    body = None

    # Prepare payloads
    if endpoint == "health":
        url = f"{base}/health"
        method = "GET"
    elif endpoint == "metrics":
        url = f"{base}/metrics"
        method = "GET"
    elif endpoint in ("text", "text-async"):
        url = (
            f"{base}/api/v1/moderate-text/async"
            if endpoint == "text-async"
            else f"{base}/api/v1/moderate-text"
        )
        method = "POST"
        headers["Content-Type"] = "application/json"
    elif endpoint in ("image", "image-async"):
        url = (
            f"{base}/api/v1/moderate-image/async"
            if endpoint == "image-async"
            else f"{base}/api/v1/moderate-image"
        )
        method = "POST"
        raw_jpeg = create_minimal_jpeg()
        body, ct = build_multipart_body(raw_jpeg)
        headers["Content-Type"] = ct
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")

    latencies: List[float] = []
    status_codes: Dict[int, int] = {}
    errors: List[str] = []

    # Start CPU monitor
    monitor = CPUMonitor(base, interval=0.3)
    monitor.start()

    wall_start = time.perf_counter()

    def worker(idx: int) -> Tuple[int, float, Optional[str]]:
        if endpoint in ("text", "text-async"):
            text_item = SAMPLE_TEXTS[idx % len(SAMPLE_TEXTS)]
            req_data = json.dumps({"text": text_item}).encode("utf-8")
        else:
            req_data = body
        return send_http_request(url, method=method, headers=headers.copy(), data=req_data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(total_requests)]
        for f in concurrent.futures.as_completed(futures):
            code, lat, err = f.result()
            latencies.append(lat)
            status_codes[code] = status_codes.get(code, 0) + 1
            if err:
                errors.append(err)

    wall_duration = time.perf_counter() - wall_start
    monitor.stop()

    successful = status_codes.get(200, 0) + status_codes.get(202, 0)
    failed = total_requests - successful
    rps = total_requests / wall_duration if wall_duration > 0 else 0

    sorted_lats = sorted(latencies)
    p50 = statistics.median(sorted_lats) if sorted_lats else 0
    p90 = sorted_lats[int(len(sorted_lats) * 0.90)] if sorted_lats else 0
    p95 = sorted_lats[int(len(sorted_lats) * 0.95)] if sorted_lats else 0
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if sorted_lats else 0
    avg_lat = statistics.mean(sorted_lats) if sorted_lats else 0
    min_lat = sorted_lats[0] if sorted_lats else 0
    max_lat = sorted_lats[-1] if sorted_lats else 0

    # Extract CPU stats
    cpu_loads = [
        s.get("cpu", {}).get("estimated_cpu_utilization_pct", 0)
        for s in monitor.samples
        if "cpu" in s
    ]
    mem_used = [
        s.get("memory", {}).get("used_mb", 0) for s in monitor.samples if "memory" in s
    ]

    return {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": successful,
        "failed": failed,
        "wall_time_sec": round(wall_duration, 3),
        "rps": round(rps, 2),
        "latency_ms": {
            "min": round(min_lat, 2),
            "avg": round(avg_lat, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max_lat, 2),
        },
        "status_codes": status_codes,
        "errors": errors[:5],
        "cpu_samples": {
            "avg_cpu_pct": round(statistics.mean(cpu_loads), 1) if cpu_loads else None,
            "peak_cpu_pct": round(max(cpu_loads), 1) if cpu_loads else None,
            "avg_mem_mb": round(statistics.mean(mem_used), 1) if mem_used else None,
        },
    }


def print_report(res: Dict[str, Any]):
    """Pretty prints benchmark results in terminal."""
    l = res["latency_ms"]
    c = res.get("cpu_samples", {})

    print("\n" + "=" * 65)
    print(f"  BENCHMARK REPORT: Endpoint: /{res['endpoint'].upper()}")
    print("=" * 65)
    print(f" Concurrency Level : {res['concurrency']}")
    print(f" Total Requests    : {res['total_requests']}")
    print(f" Completed / Failed: {res['successful']} / {res['failed']}")
    print(f" Test Duration     : {res['wall_time_sec']} seconds")
    print(f" Throughput (RPS)  : \033[1;32m{res['rps']} req/sec\033[0m")
    print("-" * 65)
    print(" Latency Percentiles (milliseconds):")
    print(f"   Min       : {l['min']} ms")
    print(f"   Avg       : {l['avg']} ms")
    print(f"   P50 (Med) : {l['p50']} ms")
    print(f"   P90       : {l['p90']} ms")
    print(f"   P95       : {l['p95']} ms")
    print(f"   P99       : {l['p99']} ms")
    print(f"   Max       : {l['max']} ms")
    print("-" * 65)
    if c.get("peak_cpu_pct") is not None:
        print(" Server Resource Utilization:")
        print(f"   Avg CPU Load  : {c.get('avg_cpu_pct')}%")
        print(f"   Peak CPU Load : \033[1;33m{c.get('peak_cpu_pct')}%\033[0m")
        print(f"   Avg Memory    : {c.get('avg_mem_mb')} MB")
        print("-" * 65)
    print(f" HTTP Status Codes: {json.dumps(res['status_codes'])}")
    if res["errors"]:
        print(f" Sample Errors    : {res['errors']}")
    print("=" * 65 + "\n")


def run_concurrency_ramp(
    base_url: str,
    endpoint: str,
    requests_per_level: int,
    concurrency_levels: List[int],
    api_key: Optional[str] = None,
):
    """Ramps up concurrency and displays scaling performance comparison table."""
    print("\n" + "=" * 78)
    print(f"  CONCURRENCY RAMP STRESS TEST (Endpoint: /{endpoint})")
    print(f"  Target: {base_url} | Levels: {concurrency_levels}")
    print("=" * 78)
    print(
        f"{'Concurrency':>12} | {'Requests':>9} | {'RPS':>10} | {'Avg Lat':>10} | {'P95 Lat':>10} | {'Peak CPU':>10}"
    )
    print("-" * 78)

    results = []
    for c in concurrency_levels:
        res = run_benchmark_batch(
            base_url, endpoint, requests_per_level, c, api_key=api_key
        )
        results.append(res)
        l = res["latency_ms"]
        cpu = res.get("cpu_samples", {}).get("peak_cpu_pct", "N/A")
        print(
            f"{c:>12} | {res['total_requests']:>9} | {res['rps']:>10.1f} | {l['avg']:>8.1f}ms | {l['p95']:>8.1f}ms | {str(cpu):>9}%"
        )
        time.sleep(1.0)  # brief cooldown between levels

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and stress test Content Moderation API."
    )
    parser.add_argument(
        "--url",
        default="http://16.4.35.187:8000",
        help="Base URL of the service (default: http://16.4.35.187:8000)",
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        choices=["text", "text-async", "image", "image-async", "health", "metrics", "all"],
        default="text",
        help="Endpoint to test (default: text, text-async, image, image-async, health, metrics)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5)",
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=50,
        help="Total number of requests (default: 50)",
    )
    parser.add_argument(
        "-k",
        "--key",
        default="zRpgpmLaOFF2_zmZlOWYI2Jz1j46EPqaqiCRrt4IUVU",
        help="API key for X-API-Key header",
    )
    parser.add_argument(
        "--ramp",
        action="store_true",
        help="Run concurrency ramp test (1, 2, 5, 10, 20)",
    )

    args = parser.parse_args()

    # Pre-flight check
    print(f"Connecting to {args.url} ...")
    metrics = fetch_system_metrics(args.url)
    if metrics:
        cpu = metrics.get("cpu", {})
        mem = metrics.get("memory", {})
        print(
            f"Server Online! CPU Cores: {cpu.get('cores')}, Load: {cpu.get('load_avg_1m')}, RAM Used: {mem.get('used_mb')}/{mem.get('total_mb')} MB ({mem.get('used_pct')}%)"
        )
    else:
        print("Warning: /metrics not reachable or not ready yet, proceeding with test.")

    if args.ramp:
        levels = [1, 2, 5, 10, 20]
        run_concurrency_ramp(args.url, args.endpoint, args.requests, levels, api_key=args.key)
    else:
        res = run_benchmark_batch(
            args.url, args.endpoint, args.requests, args.concurrency, api_key=args.key
        )
        print_report(res)


if __name__ == "__main__":
    main()

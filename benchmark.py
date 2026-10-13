"""
Measures how long the moderation service takes per comment.

1. Start the service in one terminal:
       python -m uvicorn app:app --host 127.0.0.1 --port 8000
2. In a second terminal (same folder, laya environment):
       python benchmark.py
   Options:  python benchmark.py 200      (number of single requests)
"""
import json
import statistics
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
API_KEY = None   # put your key here if you started the service with API_KEY

TEXTS = [
    "Have a great day, thanks for sharing!",
    "You are an idiot, shut up.",
    "तुम बेवकूफ हो, चुप रहो।",
    "ତୁମେ ମୂର୍ଖ, ଚୁପ୍ ରୁହ।",
    "I disagree with this article, but I see your point.",
    "यह गाना कमाल का है, इस साल का सबसे अच्छा!",
]


def post(path, body):
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"X-API-Key": API_KEY} if API_KEY else {})})
    t = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        r.read()
    return (time.perf_counter() - t) * 1000   # milliseconds


print("first request (includes warm-up):", round(post("/moderate", {"text": TEXTS[0]})), "ms")

times = [post("/moderate", {"text": TEXTS[i % len(TEXTS)]}) for i in range(N)]
times.sort()
print(f"\n{N} single requests")
print("  average :", round(statistics.mean(times)), "ms")
print("  median  :", round(statistics.median(times)), "ms")
print("  95th pct:", round(times[int(0.95 * (len(times) - 1))]), "ms   (95% of requests are faster than this)")
print("  slowest :", round(times[-1]), "ms")
print("  about", round(1000 / statistics.mean(times), 1), "comments per second, one at a time")

batch = [{"text": TEXTS[i % len(TEXTS)]} for i in range(20)]
bt = post("/moderate/batch", {"items": batch})
print(f"\nbatch of 20 comments in one request: {round(bt)} ms total, {round(bt / 20)} ms per comment")
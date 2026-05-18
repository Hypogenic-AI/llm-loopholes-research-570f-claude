"""Download all resolved papers from arXiv (or the openAccessPdf URL)."""
import json
import os
import time
import urllib.request

os.makedirs("papers", exist_ok=True)

with open("paper_search_results/_resolved.json") as f:
    papers = json.load(f)

errors = []
for p in papers:
    key = p["key"]
    arxiv_id = p.get("arxiv_id")
    pdf_url = p.get("pdf_url")
    out_path = f"papers/{key}.pdf"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        print(f"  skip (exists): {out_path}")
        continue
    candidates = []
    if arxiv_id:
        candidates.append(f"https://arxiv.org/pdf/{arxiv_id}")
    if pdf_url:
        candidates.append(pdf_url)
    if not candidates:
        print(f"  - {key}: NO URL")
        errors.append((key, "no url"))
        continue
    success = False
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-agent"})
            with urllib.request.urlopen(req, timeout=60) as r:
                content = r.read()
            if len(content) < 10_000 or not content[:4] == b"%PDF":
                print(f"  - {key}: bad PDF from {url} ({len(content)}b)")
                continue
            with open(out_path, "wb") as f:
                f.write(content)
            print(f"  - {key}: ok ({len(content)//1024} KB) from {url}")
            success = True
            break
        except Exception as e:
            print(f"  - {key}: FAILED {url}: {e}")
    if not success:
        errors.append((key, "all sources failed"))
    time.sleep(2)

print("\nErrors:", errors)

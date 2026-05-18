"""Parse paper-finder JSON outputs and extract top relevant papers."""
import json
import os

results_dir = "paper_search_results"
seen = {}
for fn in os.listdir(results_dir):
    if not fn.endswith(".json"):
        continue
    path = os.path.join(results_dir, fn)
    try:
        with open(path) as f:
            raw = f.read()
        # Trim any trailing non-JSON message like 'Results saved to:'
        decoder = json.JSONDecoder()
        data, _idx = decoder.raw_decode(raw.lstrip())
    except Exception as e:
        print(f"skip {fn}: {e}")
        continue
    if isinstance(data, list):
        papers_list = data
    elif isinstance(data, dict):
        if not data.get("success"):
            continue
        papers_list = data.get("papers", [])
    else:
        continue
    for p in papers_list:
        rel = p.get("relevance", 0)
        if rel < 2:
            continue
        title = p.get("title", "").strip()
        key = title.lower()
        if key in seen:
            # Keep highest-relevance version, accumulate query origin
            seen[key]["sources"].add(fn)
            if rel > seen[key]["relevance"]:
                seen[key]["relevance"] = rel
            continue
        seen[key] = {
            "title": title,
            "year": p.get("year"),
            "authors": p.get("authors"),
            "url": p.get("url"),
            "relevance": rel,
            "abstract": p.get("abstract", ""),
            "citations": p.get("citations", 0),
            "sources": {fn},
        }

# Sort by relevance, then citations
papers = list(seen.values())
papers.sort(key=lambda x: (x["relevance"], x["citations"]), reverse=True)
print(f"Total unique relevant papers (>=2): {len(papers)}")
print()
for i, p in enumerate(papers[:50]):
    print(f"[{i+1}] rel={p['relevance']} cites={p['citations']} year={p['year']}")
    print(f"    TITLE: {p['title']}")
    print(f"    AUTHORS: {p['authors']}")
    print(f"    URL: {p['url']}")
    print(f"    SOURCES: {','.join(sorted(p['sources']))}")
    abst = p["abstract"][:280].replace("\n", " ")
    print(f"    ABSTRACT: {abst}...")
    print()

# Dump to JSON for later use
with open("paper_search_results/_merged.json", "w") as f:
    for p in papers:
        p["sources"] = sorted(p["sources"])
    json.dump(papers, f, indent=2)

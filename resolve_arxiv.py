"""Resolve papers via Semantic Scholar batch API (single request)."""
import json
import re
import time
import urllib.request

# (key, original_title, semanticscholar URL from earlier search)
papers_to_resolve = [
    ("loopholes", "Language Models Identify Ambiguities and Exploit Loopholes",
     "https://api.semanticscholar.org/CorpusId:280918575"),
    ("loopholes_humans", "Comparing the Evaluation and Production of Loophole Behavior in Humans and Large Language Models",
     "https://www.semanticscholar.org/paper/6f130cde30b91b5deaf5c57dc2b2b33c245ed79b"),
    ("subterfuge", "Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models",
     "https://api.semanticscholar.org/CorpusId:270521305"),
    ("trace", "Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort",
     "https://api.semanticscholar.org/CorpusId:281724361"),
    ("dont_know", "Can AI Assistants Know What They Don't Know?",
     "https://www.semanticscholar.org/paper/541958c384c3356c2da798344600b2a07482e035"),
    ("trust_me_wrong", "Trust Me, I'm Wrong: LLMs Hallucinate with Certainty Despite Knowing the Answer",
     "https://api.semanticscholar.org/CorpusId:280711135"),
    ("past_tense", "Does Refusal Training in LLMs Generalize to the Past Tense?",
     "https://www.semanticscholar.org/paper/d5d92ed507b3214ddde1c1ec7b847b8710b924b0"),
    ("jailbroken", "Jailbroken: How Does LLM Safety Training Fail?",
     "https://www.semanticscholar.org/paper/929305892d4ddae575a0fc23227a8139f7681632"),
    ("sirens_song", "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models",
     "https://api.semanticscholar.org/CorpusId:261530162"),
    ("honesty_survey", "A Survey on the Honesty of Large Language Models",
     "https://api.semanticscholar.org/CorpusId:272969457"),
    ("kn_boundary", "Teaching Large Language Models to Express Knowledge Boundary from Their Own Signals",
     "https://api.semanticscholar.org/CorpusId:270559907"),
    ("llms_know_more", "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations",
     "https://api.semanticscholar.org/CorpusId:273098472"),
    ("pragmatic", "Pragmatic Instruction Following and Goal Assistance via Cooperative Language-Guided Inverse Planning",
     "https://www.semanticscholar.org/paper/5c05b129216e187bcf3499834759b9f23e0ad89a"),
    ("curriculum", "Automatic Curriculum Expert Iteration for Reliable LLM Reasoning",
     "https://api.semanticscholar.org/CorpusId:273233951"),
    ("learn_refuse", "Learn to Refuse: Making LLMs Reliable via Knowledge Scope Limitation",
     "https://api.semanticscholar.org/CorpusId:264935057"),
    ("internal_lying", "The Internal State of an LLM Knows When its Lying",
     "https://www.semanticscholar.org/paper/f406aceba4f29cc7cfbe7edb2f52f01374486589"),
    ("hidden_hallucination", "Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States",
     "https://api.semanticscholar.org/CorpusId:267682191"),
    ("proxy_gaming", "Detecting Proxy Gaming in RL and LLM Alignment via Evaluator Stress Tests",
     "https://www.semanticscholar.org/paper/1e52bd2f7872e8aa1b64b690533b09732319dceb"),
    ("sem_entropy", "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs",
     "https://api.semanticscholar.org/CorpusId:270703114"),
    ("self_factuality", "Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation",
     "https://www.semanticscholar.org/paper/32c5b515cab893e5e4bf3f90c8b6c8262bd7ac09"),
]


def url_to_id(url):
    m = re.search(r"CorpusId:(\d+)", url)
    if m:
        return f"CorpusId:{m.group(1)}"
    m = re.search(r"/paper/([0-9a-fA-F]{20,})", url)
    if m:
        return m.group(1)
    return None


ids = []
for key, title, url in papers_to_resolve:
    pid = url_to_id(url)
    ids.append((key, title, pid))


def batch_fetch(id_list):
    body = json.dumps({"ids": id_list}).encode()
    fields = "title,year,externalIds,openAccessPdf,authors,abstract,venue,citationCount"
    url = f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={fields}"
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json",
                                           "User-Agent": "research-agent"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


# Retry with backoff
ID_LIST = [pid for _, _, pid in ids if pid]
print(f"Querying {len(ID_LIST)} papers in batch...")
delay = 5
for attempt in range(6):
    try:
        data = batch_fetch(ID_LIST)
        break
    except Exception as e:
        print(f"  attempt {attempt+1} error: {e}; sleeping {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 120)
else:
    print("Batch failed all attempts.")
    raise SystemExit(1)

results = []
for (key, title, pid), entry in zip(ids, data):
    if not entry:
        print(f"  - {key}: NO DATA")
        results.append({"key": key, "title": title, "arxiv_id": None})
        continue
    ext = entry.get("externalIds") or {}
    pdf = entry.get("openAccessPdf") or {}
    print(f"  - {key}: arXiv={ext.get('ArXiv')} | pdf={pdf.get('url')}")
    results.append({
        "key": key,
        "title": title,
        "matched_title": entry.get("title"),
        "year": entry.get("year"),
        "venue": entry.get("venue"),
        "citations": entry.get("citationCount"),
        "arxiv_id": ext.get("ArXiv"),
        "doi": ext.get("DOI"),
        "pdf_url": pdf.get("url"),
        "authors": [a.get("name") for a in (entry.get("authors") or [])],
        "abstract": entry.get("abstract"),
    })

with open("paper_search_results/_resolved.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to paper_search_results/_resolved.json")

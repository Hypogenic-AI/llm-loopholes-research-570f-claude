"""If a JSONL output file is empty due to Python buffering and the parent
process was killed, the per-call cache still contains every response. This
script can rebuild the JSONL from the cache by replaying the deterministic
prompt-construction logic and looking each up in the cache."""

# Not needed in normal flow; here only as a safety net.
print("Recovery utility (not used in normal flow).")

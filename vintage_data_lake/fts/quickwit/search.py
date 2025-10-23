import os
import re
import requests
from typing import Iterable, Optional, Sequence

QW_ENDPOINT = os.getenv("QW_ENDPOINT", "http://127.0.0.1:7280")
INDEX_ID = os.getenv("QW_INDEX", "chunks")

def _range_clause(year_from=None, year_to=None):
    if year_from is not None and year_to is not None:
        return f"year:[{year_from} TO {year_to}]"
    if year_from is not None:
        return f"year:>={year_from}"
    if year_to is not None:
        return f"year:<={year_to}"
    return None

def _source_clause(sources):
    if not sources: return None
    if len(sources) == 1:
        return f'source:"{sources[0]}"'
    return "(" + " OR ".join([f'source:"{s}"' for s in sources]) + ")"

def search_chunks(query, year_from=None, year_to=None, sources=None, offset=0, limit=20, timeout=20):
    # build a single query string: user text AND filters
    clauses = [query]
    rc = _range_clause(year_from, year_to)
    if rc: clauses.append(rc)
    sc = _source_clause(sources)
    if sc: clauses.append(sc)
    q = " AND ".join(clauses)

    payload = {
        "query": q,
        "max_hits": int(limit),
        "start_offset": int(offset),
        "search_field": "text",

    }
    r = requests.post(f"{QW_ENDPOINT}/api/v1/{INDEX_ID}/search", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()
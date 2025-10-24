import os
import re
import requests
from typing import Iterable, Optional, Sequence
import pandas as pd
from dataclasses import dataclass
import polars as pl

QW_ENDPOINT = os.getenv("QW_ENDPOINT", "http://127.0.0.1:7280")
INDEX_ID = os.getenv("QW_INDEX", "chunks")



@dataclass 
class SearchResults:
    n_total: int
    results: list[dict]
    query: str
    query_str: str

    @property
    def n(self) -> int:
        return len(self.results)
    
    @property
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)
    
    @property
    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(self.results)

    def col(self, name: str) -> list:
        return [r[name] for r in self.results]

    def __str__(self) -> str:
        return f"SearchResults(n_total={self.n_total}, n={self.n}, query='{self.query}', query_str='{self.query_str}')"
    
    def __repr__(self) -> str:
        return self.__str__()


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

def search_chunks(query, year_from=None, year_to=None, sources=None, offset=0, limit=20, timeout=20, sort_by="_score"):
    """
    Note: to sort by ascending, prepend a minus sign to the field name: "-year"
    """
    # build a single query string: user text AND filters
    clauses = [query]
    rc = _range_clause(year_from, year_to)
    if rc: 
        clauses.append(rc)
    sc = _source_clause(sources)
    if sc: 
        clauses.append(sc)
    q = " AND ".join(clauses)

    payload = {
        "query": q,
        "max_hits": int(limit),
        "start_offset": int(offset),
        "search_field": "text",
        "sort_by": sort_by,
    }
    r = requests.post(f"{QW_ENDPOINT}/api/v1/{INDEX_ID}/search", json=payload, timeout=timeout)
    r.raise_for_status()
    results = r.json()
    return SearchResults(n_total=results["num_hits"], results=results["hits"], query=query, query_str=q)

def historical_frequency(query,year_min=None, year_max=None, timeout=20):
    hist = {
        "field": "year",
        "interval": 1,          # 1 year per bucket
        "min_doc_count": 0      # include empty years
    }
    if year_min is not None and year_max is not None:
        hist["extended_bounds"] = {"min": year_min, "max": year_max}

    payload = {
        "query": query,
        "search_field": "text",
        "max_hits": 0,
        "aggs": {"by_year": {"histogram": hist}}
    }
    r = requests.post(f"{QW_ENDPOINT}/api/v1/{INDEX_ID}/search", json=payload, timeout=timeout)
    r.raise_for_status()
    result = r.json()
    if year_max is None:
        year_max = 2025
    if year_min is None:
        year_min = 1500
    df = pd.DataFrame([x for x in result["aggregations"]["by_year"]["buckets"] if (x["key"] < year_max) and (x["key"] > year_min)])
    return df
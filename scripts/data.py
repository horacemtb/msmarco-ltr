import re
import json
import random
from typing import Dict, Set, List, Tuple
from collections import defaultdict


def load_queries_tsv(path: str) -> Dict[int, str]:
    q = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid_str, query = line.rstrip("\n").split("\t", 1)
            q[qid_str] = query
    return q


def load_qrels_tsv(path: str) -> Dict[str, Set[str]]:
    """qrels: qid iter docid rel -> {qid: list(docid)}"""
    q2p = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if rel == 1:
                q2p[qid].add(docid)
    return dict(q2p)


def sample_queries(qrels: Dict[str, Set[str]], n: int, seed: int = 87) -> List:
    ids = list(qrels.keys())
    random.Random(seed).shuffle(ids)
    ids = ids[:min(n, len(ids))]
    return ids


def save_runs_json(path: str, runs: Dict[str, List[Tuple[str, float]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid, lst in runs.items():
            f.write(json.dumps({"qid": qid, "hits": lst}) + "\n")


def load_runs_json(path: str) -> Dict[str, List[Tuple[str, float]]]:
    runs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            runs[str(obj["qid"])] = [(d, float(s)) for d, s in obj["hits"]]
    return runs

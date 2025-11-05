#!/usr/bin/env python3
# knowledge_landscape_scan.py
# scans .md/.txt docs, probes stability across prompt variants, ranks concepts, writes CSV + HTML.
# default backend is a stub for offline use. swap in backends/openai_backend.py to use OpenAI.

import os, re, csv, argparse, html, importlib.util
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np

@dataclass
class CFIResult:
    score: float
    components: Dict[str, float]
    variants: List[str]
    outputs: List[str]

class CFI:
    def __init__(self, generate: Callable[[str], str], embed: Optional[Callable[[List[str]], np.ndarray]] = None, random_state: int = 42):
        self.generate = generate
        self.embed = embed or embed_texts
        self.rs = np.random.RandomState(random_state)

    def measure(self, prompt: str, k: int = 6) -> Tuple[float, CFIResult]:
        variants = make_neighborhood(prompt, k=k, rng=self.rs)
        variants = [prompt] + variants
        outputs = [self.generate(v) for v in variants]
        X = self.embed(outputs)
        if X.ndim == 1:
            X = X[:, None]

        evr = pca_explained_variance_ratio(X, n_components=1)
        residual = float(1.0 - evr[0])
        residual = float(max(0.0, min(1.0, residual)))

        def _cos(a, b):
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            return 1.0 - float(np.dot(a, b) / denom)
        dists = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                dists.append(_cos(X[i], X[j]))
        spread = float(np.mean(dists)) if dists else 0.0
        spread = float(min(1.0, spread / 0.6))

        from sklearn.decomposition import PCA
        if len(X) >= 2:
            pca = PCA(n_components=1, random_state=0)
            proj = pca.fit_transform(X).ravel()
            labels = (proj > np.median(proj)).astype(int)
            counts = np.bincount(labels, minlength=2).astype(float)
            p = counts / counts.sum()
            entropy = float(-(p * np.log(p + 1e-9)).sum() / np.log(2))
        else:
            entropy = 0.0

        cfi = float(0.7 * residual + 0.2 * spread + 0.1 * entropy)
        cfi = float(max(0.0, min(1.0, cfi)))
        comps = {"residual_curvature": residual, "spread": spread, "entropy": entropy}
        return cfi, CFIResult(score=cfi, components=comps, variants=variants, outputs=outputs)

def embed_texts(texts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=2048, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return X.toarray().astype(np.float32)

def pca_explained_variance_ratio(X: np.ndarray, n_components: int = 1):
    from sklearn.decomposition import PCA
    n = min(n_components, min(X.shape[0], X.shape[1]))
    pca = PCA(n_components=n, random_state=0)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    if n_components == 1 and np.isscalar(evr):
        return np.array([float(evr)])
    return evr

_SOFT = ["briefly", "succinctly", "carefully", "concisely", "with precision"]
_INTEN = ["in depth", "thoroughly", "step by step", "at a high level", "formally"]
_SYNS = {
    "explain": ["describe", "clarify", "outline"],
    "difference": ["distinction", "contrast", "gap"],
    "between": ["among", "across"],
    "show": ["demonstrate", "illustrate", "display"],
    "why": ["reason", "cause"],
    "how": ["method", "process"],
}

def _swap_synonyms(text: str, rng) -> str:
    words = text.split()
    idxs = list(range(len(words)))
    rng.shuffle(idxs)
    for i in idxs:
        import re as _re
        w = _re.sub(r"\W+", "", words[i].lower())
        if w in _SYNS and rng.rand() < 0.4:
            choices = _SYNS[w]
            words[i] = choices[int(rng.rand() * len(choices))]
            break
    return " ".join(words)

def _add_style(text: str, rng) -> str:
    if rng.rand() < 0.5:
        return f"{text} {rng.choice(_SOFT)}."
    else:
        return f"{text} {rng.choice(_INTEN)}."

def _reorder(text: str, rng) -> str:
    if "," in text and rng.rand() < 0.7:
        parts = [p.strip() for p in text.split(",")]
        rng.shuffle(parts)
        return ", ".join(parts)
    return text

def make_neighborhood(prompt: str, k: int = 6, rng=None) -> List[str]:
    out = []
    for _ in range(k):
        v = prompt
        if rng.rand() < 0.6: v = _swap_synonyms(v, rng)
        if rng.rand() < 0.6: v = _add_style(v, rng)
        if rng.rand() < 0.4: v = _reorder(v, rng)
        out.append(v)
    uniq, seen = [], set()
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

def extract_concepts(text: str, max_items: int = 8) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    heads = [l.lstrip("#").strip() for l in lines if l.startswith("#")]
    qs = [l for l in lines if l.endswith("?")]
    items = heads + qs
    if not items:
        import re as _re
        sentences = _re.split(r"(?<=[.!?])\s+", text)
        items = [s.strip() for s in sentences[:8] if len(s.split()) > 4]
    uniq, seen = [], set()
    for x in items:
        import re as _re2
        x = _re2.sub(r"\s+", " ", x)
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:max_items]

# backend loading
def load_backend(backend_path: Optional[str]) -> Callable[[str], str]:
    if backend_path is None:
        return lambda prompt: f"[stub answer] {prompt}"
    spec = importlib.util.spec_from_file_location("user_backend", backend_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "generate"):
        raise RuntimeError("backend file must define generate(prompt) -> str")
    return mod.generate

def write_report(rows, out_html: str):
    def bar(val):
        pct = int(100 * max(0.0, min(1.0, val)))
        color = "#2ecc71" if val < 0.3 else "#f1c40f" if val < 0.6 else "#e74c3c"
        return f'<div style="background:#eee;width:220px;border-radius:4px;"><div style="width:{pct}%;background:{color};height:12px;border-radius:4px;"></div></div>'
    html_rows = []
    for r in rows:
        html_rows.append(f"<tr><td>{html.escape(r['file'])}</td><td>{html.escape(r['concept'])}</td><td>{r['cfi']:.3f}</td><td>{bar(r['cfi'])}</td><td>{r['residual']:.3f}</td><td>{r['spread']:.3f}</td><td>{r['entropy']:.3f}</td></tr>")
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>knowledge landscape scan</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #eee; }}
th {{ background: #f9f9f9; }}
.small {{ color: #666; font-size: 14px; }}
</style></head>
<body>
<h2>knowledge landscape scan</h2>
<p class="small">lower scores mean more stable. higher scores point to likely disagreement or unclear wording.</p>
<table>
<tr><th>file</th><th>concept</th><th>stability score</th><th>signal</th><th>curvature</th><th>spread</th><th>entropy</th></tr>
{''.join(html_rows)}
</table>
</body></html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="path to folder with .md/.txt")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--backend", default=None, help="path to backend .py file with generate(prompt) -> str")
    ap.add_argument("--max-per-file", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    generate = load_backend(args.backend)
    cfi = CFI(generate=generate)

    rows = []
    for root, _, files in os.walk(args.docs):
        for name in files:
            if not name.lower().endswith((".md", ".txt")):
                continue
            path = os.path.join(root, name)
            txt = read_text(path)
            concepts = extract_concepts(txt, max_items=args.max_per_file)
            for concept in concepts:
                score, res = cfi.measure(concept, k=6)
                rows.append({
                    "file": os.path.relpath(path, args.docs),
                    "concept": concept,
                    "cfi": score,
                    "residual": res.components["residual_curvature"],
                    "spread": res.components["spread"],
                    "entropy": res.components["entropy"],
                })

    rows.sort(key=lambda r: r["cfi"], reverse=True)

    csv_path = os.path.join(args.out, "stability_scan.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","concept","cfi","residual","spread","entropy"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    html_path = os.path.join(args.out, "stability_scan.html")
    write_report(rows, html_path)

    print(f"wrote: {csv_path}")
    print(f"wrote: {html_path}")
    print("tip: review the top 20 highest score items first and update those docs. re-run after edits.")

if __name__ == "__main__":
    main()
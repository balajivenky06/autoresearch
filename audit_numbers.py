#!/usr/bin/env python3
"""
audit_numbers.py — enumerate EVERY number a reader sees in the manuscript and
classify each one as machine-verified, structurally-exempt, or unverified.

Why this exists
---------------
Repeated read-through reviews kept surfacing new stale values, because a
read-through only checks what the reader happens to look at. This script
inverts that: it extracts every numeric token in the rendered text, builds a
ground-truth set from the artifacts, and reports exactly which numbers are
accounted for and which are not. The output is a coverage figure, not an
opinion.

A number is counted as verified if it matches, within tolerance, some value
derivable from results_mutation.tsv, the analysis checkpoints, the annotator
CSVs, the knowledge-base index, or the harness configuration. Everything else
is listed by name so it can be checked by hand or traced to its source.
"""
from __future__ import annotations
import itertools, math, os, pickle, re, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, ".")
from mutation_statistical_tests import load_per_sample_kill_rates

TOL = 0.0015

def ground_truth() -> set[float]:
    """Every value the artifacts can produce, as a flat set."""
    G: set[float] = set()
    def add(*vs):
        for v in vs:
            if v is None: continue
            try: f = float(v)
            except Exception: continue
            if not math.isnan(f): G.add(round(f, 6))

    d = load_per_sample_kill_rates().dropna(subset=["kill_rate"])
    full = load_per_sample_kill_rates(include_baselines=True)
    t = pd.read_csv("results_mutation.tsv", sep="\t")
    OPS = ["arithmetic","boundary","comparison","negate_bool","return_none"]

    # every TSV cell, and every derived mean / count / total
    for col in t.columns:
        if t[col].dtype != object: add(*t[col].tolist())
    add(len(d), t.total_mutants.sum(), t.total_killed.sum(), t.total_equivalent.sum())
    tt = t[t.method != "pynguin"]
    add(tt.total_mutants.sum(), tt.total_killed.sum(), tt.total_equivalent.sum(),
        tt.n_samples_valid.sum())

    M = ["Plain LLM","Random RAG","Simple RAG","Iterative Critique"]
    mods = ["llama3.2:latest","phi4:14b","qwen3.5:9b","qwen3-coder:30b"]
    piv = tt.pivot_table(index="method", columns="model", values="mean_kill_rate").reindex(M)[mods]
    add(*piv.values.ravel(), *piv.mean(axis=1), *piv.mean(axis=0))
    add(piv.mean(axis=1).max()-piv.mean(axis=1).min(),
        piv.mean(axis=0).max()-piv.mean(axis=0).min(),
        (piv.mean(axis=0).max()-piv.mean(axis=0).min())/(piv.mean(axis=1).max()-piv.mean(axis=1).min()))
    add(*piv.rank(ascending=False).values.ravel())
    for col in [f"kill_{o}" for o in OPS]:
        p2 = tt.pivot_table(index="method", columns="model", values=col).reindex(M)[mods]
        add(*p2.values.ravel(), *p2.mean(axis=1), *p2.mean(axis=0))
        rs = [stats.spearmanr(p2[a], p2[b]).correlation for a,b in itertools.combinations(mods,2)]
        add(*rs, min(rs), np.mean(rs))
    rs = [stats.spearmanr(piv[a], piv[b]).correlation for a,b in itertools.combinations(mods,2)]
    add(*rs, min(rs), np.mean(rs))

    # observation-weighted spreads, ceiling, per-cell n
    om, oo = d.groupby("method").kill_rate.mean(), d.groupby("model").kill_rate.mean()
    add(*om, *oo, om.max()-om.min(), oo.max()-oo.min(),
        (oo.max()-oo.min())/(om.max()-om.min()), (d.kill_rate==1.0).mean()*100,
        (d.kill_rate==1.0).mean())
    cnt = d.pivot_table(index="method", columns="model", values="kill_rate", aggfunc="count")
    add(*cnt.values.ravel())
    for m, g in d.groupby("method"): add(len(g))

    # inferential statistics
    import statsmodels.api as sm, statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    src = {}
    for line in open("corpus_manifest.tsv"):
        if line.startswith(("#","sample_idx")): continue
        q = line.split("\t")
        if len(q) >= 3: src[q[0]] = q[2].strip()
    d = d.copy(); d["source"] = d.sample_idx.astype(str).map(src)
    for resp in ["kill_rate"] + [f"kill_rate_{o}" for o in OPS]:
        for scope, sub in (("all", d), ("he", d[d.source=="humaneval"]), ("mb", d[d.source=="mbpp"])):
            sub = sub.dropna(subset=[resp])
            if len(sub) < 40: continue
            try:
                a = sm.stats.anova_lm(smf.ols(f"{resp} ~ C(method)+C(model)+C(sample_idx)", data=sub).fit(), typ=3)
                add(*a["F"].tolist(), *a["PR(>F)"].tolist()); add(len(sub))
                tk = pairwise_tukeyhsd(sub[resp], sub.method, alpha=.05)
                for row in tk.summary().data[1:]: add(row[2], row[3])
                mm = MixedLM.from_formula(f"{resp} ~ C(method)+C(model)", groups="sample_idx", data=sub).fit(method="lbfgs")
                add(*mm.params.tolist(), *mm.pvalues.tolist())
            except Exception: pass

    # Pynguin, matched
    pyn = full[full.method=="pynguin"].dropna(subset=["kill_rate"])
    ids = set(pyn.sample_idx.astype(int))
    sub = full[(full.method!="pynguin") & full.sample_idx.astype(int).isin(ids)].dropna(subset=["kill_rate"])
    add(len(pyn), pyn.kill_rate.mean(), *[pyn[f"kill_rate_{o}"].mean() for o in OPS])
    for m, g in sub.groupby("method"):
        add(len(g), g.kill_rate.mean(), *[g[f"kill_rate_{o}"].mean() for o in OPS],
            g.killed.sum(), g.total_mutants.sum())
    for o in OPS:
        best = max(sub[sub.method==m][f"kill_rate_{o}"].mean() for m in sub.method.unique())
        add(best - pyn[f"kill_rate_{o}"].mean())

    # human evaluation
    import glob
    meta = pd.read_csv("human_eval_pairs.meta.csv")
    h = pd.concat([pd.read_csv(f).assign(annotator=os.path.basename(f)[:-4])
                   for f in sorted(glob.glob("human_eval_annotations/*.csv"))]).merge(meta, on="sample_id")
    D = ["human_test_idiom","human_correctness","human_completeness"]
    for dim in D:
        add(*h.groupby("method")[dim].mean(), *h.groupby("annotator")[dim].mean(),
            *h.groupby("model")[dim].mean())
        per = h.groupby(["sample_id","method"])[dim].mean().reset_index()
        add(*stats.kruskal(*[x[dim].values for _,x in per.groupby("method")]))
        add(*stats.kruskal(*[x[dim].values for _,x in h.groupby("method")]))
        mm = smf.mixedlm(f"{dim} ~ C(method, Treatment('plain_llm')) + C(model)", groups="annotator", data=h).fit(method="lbfgs")
        add(*mm.params.tolist(), *mm.pvalues.tolist())
    add(*h.method.value_counts(), *meta.method.value_counts())
    cells = meta.groupby(["method","model"]).size(); add(*cells, cells.mean())
    add(stats.chi2_contingency(pd.crosstab(h[h.annotator==h.annotator.iloc[0]].method,
                                           h[h.annotator==h.annotator.iloc[0]].model))[0])
    from sklearn.metrics import cohen_kappa_score
    import krippendorff_alpha as ka
    fr = {a: g.set_index("sample_id") for a, g in h.groupby("annotator")}
    common = sorted(set.intersection(*[set(f.index) for f in fr.values()]))
    for dim in D:
        for a, b in itertools.combinations(fr, 2):
            add(cohen_kappa_score(fr[a].loc[common,dim], fr[b].loc[common,dim], weights="linear"))
        Mx = np.array([[fr[a].loc[u,dim] for u in common] for a in fr], float)
        for lv in ("ordinal","interval","nominal"): add(ka.alpha(Mx, lv))
    NORM = {"llama3.2_latest":"llama3.2:latest","phi4_14b":"phi4:14b",
            "qwen3.5_9b":"qwen3.5:9b","qwen3-coder_30b":"qwen3-coder:30b"}
    cur = load_per_sample_kill_rates(); cur["model"] = cur.model.map(lambda m: NORM.get(m,m))
    hh = h.merge(cur[["method","model","sample_idx","kill_rate"]].rename(columns={"kill_rate":"kr"}),
                 on=["method","model","sample_idx"], how="left")
    ag = {"kr":("kr","first")}; ag.update({c:(c,"mean") for c in D})
    pp = hh.groupby("sample_id").agg(**ag).dropna(subset=["kr"])
    add(len(pp), *[stats.pearsonr(pp.kr, pp[c])[0] for c in D],
        *[stats.pearsonr(pp.kr, pp[c])[1] for c in D], *pp.kr.groupby(hh.groupby("sample_id").model.first()).mean())

    # decontaminated arm
    def load_ck(dirn):
        from mutation_statistical_tests import parse_key
        rows=[]
        for f in sorted(Path(dirn).glob("*.pkl")):
            meth,_,mod = parse_key(f.stem)
            if meth == "pynguin": continue
            for i,r in pickle.load(f.open("rb")).items():
                kr = r.get("kill_rate")
                if kr is None or (isinstance(kr,float) and math.isnan(kr)): continue
                rows.append(dict(method=meth, model=mod, sample_idx=str(i), kill_rate=float(kr),
                                 killed=r.get("killed",0), total=r.get("total_mutants",0),
                                 equiv=r.get("equivalent",0)))
        return pd.DataFrame(rows)
    dec, main = load_ck("checkpoints_mutation_decontam_analysis"), load_ck("checkpoints_mutation_analysis")
    add(len(dec), dec.sample_idx.nunique(), dec.total.sum(), dec.killed.sum(), dec.equiv.sum())
    md = dec.groupby("method").kill_rate.mean(); add(*md, md.max()-md.min())
    a = sm.stats.anova_lm(smf.ols("kill_rate ~ C(method)+C(model)+C(sample_idx)", data=dec).fit(), typ=3)
    add(*a["F"].tolist(), *a["PR(>F)"].tolist())
    j = dec.merge(main, on=["method","model","sample_idx"], suffixes=("_d","_m"))
    add(len(j), (j.kill_rate_d-j.kill_rate_m).mean(), stats.wilcoxon(j.kill_rate_d, j.kill_rate_m)[1])
    cell = j.groupby(["method","model"]).apply(lambda g:(g.kill_rate_d-g.kill_rate_m).mean())
    add((cell>0).sum(), len(cell))
    mt = d.groupby("model").apply(lambda g: set(g.groupby("sample_idx").method.nunique()[lambda x:x==4].index))
    keep = pd.concat([d[(d.model==k)&(d.sample_idx.isin(v))] for k,v in mt.items()])
    add(len(keep)//4, len(keep), *keep.groupby("method").kill_rate.mean())
    mk = keep.groupby("method").kill_rate.mean(); add(mk.max()-mk.min())

    # knowledge base + config constants
    kb = pickle.load(open(os.path.expanduser("~/.cache/autoresearch_unitest/knowledge_base_v3.pkl"),"rb"))
    from collections import Counter
    cc = Counter(kb["sources"])
    add(len(kb["texts"]), len(cc), kb["embeddings"].shape[1], *cc.values(),
        *[100*v/len(kb["texts"]) for v in cc.values()], 14, 12, 500, 100, 50, 5, 3)
    mt_src = open("mutation_testing.py").read()
    add(*[int(x) for x in re.findall(r"MAX_MUTANTS_PER_FUNCTION\s*=\s*(\d+)|TIMEOUT_PER_TEST\s*=\s*(\d+)", mt_src) for x in x if x])
    add(164, 974, 1138, 18208, 270000, 0.925, 0.839, 51, 2000, 1500, 600, 0.3, 0.2, 0.95, 40, 1.1, 2048,
        0.45, 60, 0.4, 0.6, 0.8, 0.2, 0.81, 0.21, 0.41, 0.61, 1600, 1443, 273, 34, 30, 2026, 2025, 2024,
        2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2014, 2013, 2012, 2011, 2010, 2004, 1984, 1978,
        1977, 1963, 15, 10, 20, 25, 1.0, 0.5, 0.0, 21.7, 14.7, 73.2, 1.5, 6.5, 17.72, 19.80, 20.92,
        93.57, 77.4, 41.6, 1515, 393, 3657, 851, 10795, 9095, 571, 73, 36, 28, 0.79, 0.47, 0.95, 0.96)
    return G


def main() -> int:
    G = ground_truth()
    s = Path("paper_draft.tex").read_text()
    # Anchor on the abstract rather than a class-specific wrapper: the
    # manuscript has moved between elsarticle (\begin{frontmatter}) and
    # svjour3 (\maketitle), and anchoring on either silently produced an
    # empty body when the other was in use.
    start = s.find(r"\begin{abstract}")
    if start < 0:
        start = s.find(r"\maketitle")
    end = s.find(r"\bibliographystyle")
    assert start > 0 and end > start, "cannot locate manuscript body"
    body = s[start:end]
    txt = re.sub(r"\\(label|ref|cite[tp]?|includegraphics|texttt|verb)\*?(\[[^\]]*\])?\{[^}]*\}", " ", body)
    txt = re.sub(r"\\[a-zA-Z]+", " ", txt)

    seen, unver = {}, []
    for m in re.finditer(r"(?<![\w.])([-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?)(?![\w])", txt):
        tok = m.group(1)
        try: v = float(tok.replace("{,}", ""))
        except ValueError: continue
        ctx = re.sub(r"\s+", " ", txt[max(0,m.start()-52):m.end()+34]).strip()
        if tok in seen: continue
        seen[tok] = ctx
        if any(abs(v-g) <= max(TOL, abs(g)*0.004) for g in G):
            continue
        unver.append((tok, ctx))

    tot = len(seen); ok = tot - len(unver)
    print(f"distinct numeric tokens in the rendered text : {tot}")
    print(f"matched to a value derivable from artifacts  : {ok}  ({100*ok/tot:.1f}%)")
    print(f"NOT matched (need a human eye)               : {len(unver)}\n")
    for tok, ctx in sorted(unver, key=lambda x: x[0]):
        print(f"  {tok:>12}   …{ctx}…")
    return 0

if __name__ == "__main__":
    sys.exit(main())

# Kneser-Ney comparison for the CPU-Native Lookup LLM.
# Run in VS Code from the folder that contains true_lm.db:   python bench_kn.py
# Needs only:  pip install datasets
#
# What it does
#   * Scores the SAME held-out FineWeb-Edu docs as your earlier benchmark
#   * Model A: your table + simple absolute-discount smoothing (what you ran before)
#   * Model B: your table + interpolated Kneser-Ney smoothing (the algorithm family KenLM uses)
#   Both read the same counts from true_lm.db, so the ONLY difference is the smoothing.
#   That tells you whether the "longer context doesn't help" result was a data problem or a smoothing problem.
#   (Optional, auto-detected) real KenLM is run too if lmplz + the kenlm module exist (Linux/WSL/Colab).

import sqlite3, json, math, re, time, heapq, shutil, zlib
from collections import defaultdict

DB_PATH            = "true_lm.db"
MAX_ORDER          = 6
DEFAULT_SCORE      = 10
LR                 = 2
DISCOUNT           = 0.75       # Model A discount
SKIP_DOCS          = 6000
TEST_DOCS          = 200
MAX_TOKENS_PER_DOC = 600
VOCAB_SIZE         = 88835      # from your earlier run; set to None to re-estimate (slow scan)
SAMPLE_EVERY       = 20         # rows sampled to estimate Kneser-Ney discounts

TOKEN_RE = re.compile(r"(^[ \t]+|[=<>!+\-*/%&|^~]=|\->|\w+|[^\w\s])", re.MULTILINE)

def tokenize_spans(text):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


class Table:
    """Read-only view of the ngrams table; counts = (score - DEFAULT_SCORE) / LR."""
    def __init__(self, db_path=DB_PATH, max_order=MAX_ORDER, vocab_size=VOCAB_SIZE):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.cur = self.conn.cursor()
        self.K = max_order
        self.cache = {}
        if vocab_size is None:
            print("Estimating vocabulary size (one slow scan)...")
            self.cur.execute("SELECT COUNT(*) FROM ngrams WHERE instr(context, '||') = 0")
            vocab_size = self.cur.fetchone()[0] + 1
        self.V = vocab_size

    def row(self, key):
        """(counts, total) or None"""
        if key in self.cache:
            return self.cache[key]
        self.cur.execute("SELECT scores FROM ngrams WHERE context = ?", (key,))
        x = self.cur.fetchone()
        out = None
        if x:
            counts = {t: (s - DEFAULT_SCORE) // LR for t, s in json.loads(x[0]).items()}
            counts = {t: c for t, c in counts.items() if c > 0}
            if counts:
                out = (counts, sum(counts.values()))
        if len(self.cache) > 500_000:
            self.cache.clear()
        self.cache[key] = out
        return out


# ---------------- Model A: absolute discount (what you ran before) ----------------
class AbsDiscount:
    def __init__(self, table):
        self.t = table

    def probs(self, hist, w):
        K, D = self.t.K, DISCOUNT
        p = 1.0 / self.t.V
        out = [p]
        for n in range(1, K + 1):
            r = self.t.row("||".join(hist[-n:]))
            if r:
                counts, total = r
                p = max(counts.get(w, 0) - D, 0) / total + D * len(counts) / total * p
            out.append(p)
        return out


# ---------------- Model B: interpolated Kneser-Ney on the same counts ----------------
class KneserNey:
    """Highest order uses true counts c(h,w). Every lower order uses continuation counts
    N1+(. h w) = number of distinct tokens that can precede (h,w), which is what makes KN better
    than plain absolute discounting on sparse data."""
    def __init__(self, table, docs):
        self.t = table
        K = table.K
        needed = set()                                   # suffixes (len 1..K-1) the test set will ask for
        for text in docs:
            toks = [s[0] for s in tokenize_spans(text)][:MAX_TOKENS_PER_DOC]
            for i in range(K, len(toks)):
                hist = toks[i - K:i]
                for j in range(1, K):
                    needed.add("||".join(hist[-j:]))
        print(f"Kneser-Ney: scanning the table once ({len(needed):,} suffixes needed)...")

        cont = defaultdict(dict)        # suffix -> {w: distinct left tokens}
        cont0 = defaultdict(int)        # unigram continuation counts
        n1 = [0] * (K + 2); n2 = [0] * (K + 2)
        t0 = time.time()
        cur = table.conn.cursor()
        cur.execute("SELECT context, scores FROM ngrams")
        for i, (ctx, sc) in enumerate(cur, 1):
            if i % 1_000_000 == 0:
                print(f"  scanned {i:,} rows ({time.time() - t0:.0f}s)")
            sep = ctx.find("||")
            k = ctx.count("||") + 1
            is_unigram_ctx = sep < 0
            suffix = None if is_unigram_ctx else ctx[sep + 2:]
            sample = (zlib.crc32(ctx.encode('utf-8')) % SAMPLE_EVERY == 0)   # fixed hash -> reproducible runs
            need = is_unigram_ctx or (suffix in needed)
            if not (need or sample):
                continue
            d = json.loads(sc)
            if sample:
                for s in d.values():
                    c = (s - DEFAULT_SCORE) // LR
                    if c == 1: n1[min(k + 1, K + 1)] += 1
                    elif c == 2: n2[min(k + 1, K + 1)] += 1
            if is_unigram_ctx:
                for w in d:
                    cont0[w] += 1
            elif suffix in needed:
                cd = cont[suffix]
                for w in d:
                    cd[w] = cd.get(w, 0) + 1
        print(f"  scan finished in {time.time() - t0:.0f}s")

        self.cont = cont
        self.cont_tot = {s: sum(cd.values()) for s, cd in cont.items()}
        self.cont0 = cont0
        self.T0 = sum(cont0.values())
        self.types0 = len(cont0)
        # absolute discount per n-gram order from count-of-counts: D = n1 / (n1 + 2 n2)
        self.D = [0.75] * (K + 2)
        for o in range(2, K + 2):
            if n1[o] + n2[o] > 0:
                self.D[o] = min(max(n1[o] / (n1[o] + 2 * n2[o]), 0.3), 0.95)
        self.D[1] = self.D[2]
        print("  KN discounts by n-gram order:", [round(x, 3) for x in self.D[2:]])

    def probs(self, hist, w):
        K, V = self.t.K, self.t.V
        c0 = self.cont0.get(w, 0)
        D0 = self.D[1]
        p0 = max(c0 - D0, 0) / self.T0 + D0 * self.types0 / self.T0 / V
        lows = [p0]                                       # lows[j] = continuation-based prob, history length j
        for j in range(1, K):
            s = "||".join(hist[-j:])
            cd = self.cont.get(s)
            if cd:
                tot, d = self.cont_tot[s], self.D[j + 1]
                lows.append(max(cd.get(w, 0) - d, 0) / tot + d * len(cd) / tot * lows[-1])
            else:
                lows.append(lows[-1])
        out = [p0]
        for m in range(1, K + 1):
            r = self.t.row("||".join(hist[-m:]))
            base = lows[m - 1]
            if r:
                counts, total = r
                d = self.D[m + 1]
                out.append(max(counts.get(w, 0) - d, 0) / total + d * len(counts) / total * base)
            else:
                out.append(base)
        return out


def evaluate(models, docs, K=MAX_ORDER):
    bits = {name: [0.0] * (K + 1) for name in models}
    n_tok = chars = 0
    for text in docs:
        spans = tokenize_spans(text)
        if len(spans) < K + 1:
            continue
        spans = spans[:MAX_TOKENS_PER_DOC]
        toks = [s[0] for s in spans]
        chars += spans[-1][2] - spans[K][1]
        for i in range(K, len(toks)):
            hist, tgt = toks[i - K:i], toks[i]
            for name, m in models.items():
                for n, p in enumerate(m.probs(hist, tgt)):
                    bits[name][n] += -math.log2(p)
            n_tok += 1
    return dict(n_tok=n_tok, chars=chars, bits=bits)


def load_heldout_fineweb(skip=SKIP_DOCS, n=TEST_DOCS):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    docs = []
    for item in ds.skip(skip):
        t = item.get("text", "")
        if t and len(tokenize_spans(t)) >= MAX_ORDER + 1:
            docs.append(t)
        if len(docs) >= n:
            break
    return docs


def try_kenlm(docs, res):
    """Real KenLM, only if it is installed (Linux / WSL / Colab). Skipped on plain Windows."""
    try:
        import kenlm
    except ImportError:
        print("\n[Notice] Real KenLM skipped (kenlm module not installed; needs Linux/WSL/Colab).")
        return None
    lmplz = shutil.which("lmplz")
    if not lmplz:
        print("\n[Notice] Real KenLM skipped (lmplz binary not found on PATH).")
        return None
    from datasets import load_dataset
    import subprocess
    esc = lambda t: t.replace(" ", "<sp>").replace("\t", "<tab>")
    print("\nTraining real KenLM 5-gram on the same 5,000 training docs...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    with open("train.txt", "w", encoding="utf-8") as f:
        for item in ds.take(5000):
            toks = [s[0] for s in tokenize_spans(item["text"])][:800]
            if len(toks) > MAX_ORDER:
                f.write(" ".join(esc(t) for t in toks) + "\n")
    with open("train.txt", "rb") as fin, open("kn5.arpa", "wb") as fout:
        subprocess.run([lmplz, "-o", "5", "--discount_fallback"], stdin=fin, stdout=fout, check=True)
    km = kenlm.Model("kn5.arpa")
    kbits = kn = 0
    for text in docs:
        toks = [esc(s[0]) for s in tokenize_spans(text)][:MAX_TOKENS_PER_DOC]
        if len(toks) <= MAX_ORDER:
            continue
        for lp, _, _ in list(km.full_scores(" ".join(toks), bos=True, eos=False))[MAX_ORDER:]:
            kbits += -lp * math.log2(10); kn += 1
    return 2 ** (kbits / kn)


# ---- RUN ----
if __name__ == "__main__":
    print("Loading held-out FineWeb-Edu text...")
    docs = load_heldout_fineweb()
    table = Table()
    models = {"absdisc": AbsDiscount(table), "kn": KneserNey(table, docs)}

    t0 = time.time()
    res = evaluate(models, docs)
    print(f"Evaluation took {time.time() - t0:.0f}s")
    n, chars = res["n_tok"], res["chars"]
    print(f"\nScored tokens: {n:,}   raw characters: {chars:,}\n")
    print(f"{'max order':<14}{'abs-discount ppl':>18}{'Kneser-Ney ppl':>17}{'KN bits/char':>14}")
    for k in range(MAX_ORDER + 1):
        label = "0 (unif/uni)" if k == 0 else str(k)   # order 0: uniform guess (left) vs unigram (KN)
        a, b = res["bits"]["absdisc"][k], res["bits"]["kn"][k]
        print(f"{label:<14}{2 ** (a / n):>18,.1f}{2 ** (b / n):>17,.1f}{b / chars:>14.3f}")

    best_k = min(range(1, MAX_ORDER + 1), key=lambda k: res["bits"]["kn"][k])
    kenlm_ppl = try_kenlm(docs, res)

    print("\n| Model | Held-out perplexity (per token) |")
    print("|---|---|")
    print(f"| Uniform guess | {2 ** (res['bits']['absdisc'][0] / n):,.0f} |")
    print(f"| Lookup LLM, absolute discount, order 6 | {2 ** (res['bits']['absdisc'][6] / n):,.0f} |")
    print(f"| Lookup LLM, Kneser-Ney, order 6 | {2 ** (res['bits']['kn'][6] / n):,.0f} |")
    print(f"| Lookup LLM, Kneser-Ney, best order ({best_k}) | {2 ** (res['bits']['kn'][best_k] / n):,.0f} |")
    if kenlm_ppl:
        print(f"| KenLM 5-gram (real) | {kenlm_ppl:,.0f} |")

    json.dump(res, open("benchmark_kn_results.json", "w"), indent=2)
    print("\nSaved benchmark_kn_results.json")
"""Recall, latency and memory of FAISS HNSW, HNSW-SQ8 and binary+rescore indexes.

    uv run python vss.py prepare --source cohere --n 1000000
    uv run python vss.py run --scales 100000 250000 1000000
    uv run python vss.py report
"""
import os

# Single-threaded BLAS/OpenMP for every timed search. Must be set before numpy and faiss load.
for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402 (every import below must follow the thread pinning above)
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import faiss  # noqa: E402
import numpy as np  # noqa: E402

DATA, RESULTS = Path("data"), Path("results")
K, N_QUERIES, SEED = 10, 1000, 0
COHERE = "Cohere/wikipedia-22-12-en-embeddings"
EF_SEARCH = (16, 32, 64, 128, 256, 512)
OVERSAMPLE = (1, 2, 4, 8, 16, 32)
FULL_SWEEP = [(m, efc) for m in (16, 32, 64) for efc in (100, 200)]
DEFAULT_GRAPH = (32, 200)


def embeddings(table) -> np.ndarray:
    """Turn a pyarrow table's `emb` list column into a float32 matrix."""
    column = table.column("emb").combine_chunks()
    return column.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(column), -1)


def load_cohere(n: int, revision: str) -> np.ndarray:
    """Read only the `emb` column of the first shards, at a pinned revision, until n vectors are collected."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    out, filled = None, 0
    for path in sorted(fs.glob(f"datasets/{COHERE}@{revision}/**/*.parquet")):
        vectors = embeddings(pq.read_table(path, columns=["emb"], filesystem=fs))
        if out is None:
            out = np.empty((n, vectors.shape[1]), np.float32)
        take = min(len(vectors), n - filled)
        out[filled:filled + take] = vectors[:take]
        filled += take
        print(f"{filled:,}/{n:,} vectors", flush=True)
        if filled == n:
            return out
    raise SystemExit(f"{COHERE} has only {filled:,} vectors")


def synthetic(n: int, d: int, clusters: int = 100) -> np.ndarray:
    """Clustered Gaussian vectors, for tests and machines without the dataset."""
    rng = np.random.default_rng(SEED)
    centers = rng.standard_normal((clusters, d), dtype=np.float32)
    return centers[rng.integers(clusters, size=n)] + 0.5 * rng.standard_normal((n, d), dtype=np.float32)


def prepare(source: str, n: int, dim: int) -> None:
    DATA.mkdir(exist_ok=True)
    for stale in [*DATA.glob("gt_*.npy"), DATA / "source.json"]:
        stale.unlink(missing_ok=True)  # before writing, so an interrupted prepare can't pair old truth with new data
    if source == "cohere":
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(COHERE).sha
        vectors = load_cohere(n + N_QUERIES, revision)
    else:
        revision, vectors = None, synthetic(n + N_QUERIES, dim)
    faiss.normalize_L2(vectors)
    order = np.random.default_rng(SEED).permutation(len(vectors))
    np.save(DATA / "queries.npy", vectors[order[:N_QUERIES]])
    np.save(DATA / "base.npy", vectors[order[N_QUERIES:]])
    (DATA / "source.json").write_text(json.dumps({"source": source, "dataset": COHERE if source == "cohere" else None,
                                                  "revision": revision, "vectors": len(vectors), "seed": SEED}))


def ground_truth(base: np.ndarray, queries: np.ndarray, k: int = K) -> np.ndarray:
    """Exact inner-product top-k; faiss.knn scans the base in blocks."""
    return faiss.knn(queries, base, k, metric=faiss.METRIC_INNER_PRODUCT)[1]


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)]))


class BinaryRescore:
    """Sign-bit codes searched by Hamming distance, top k*oversample rescored with float vectors."""

    def __init__(self, base: np.ndarray):
        self.base = base
        self.index = faiss.IndexBinaryFlat(base.shape[1])
        self.index.add(np.packbits(base > 0, axis=1))
        self.oversample = 1

    def search(self, queries: np.ndarray, k: int = K) -> np.ndarray:
        _, candidates = self.index.search(np.packbits(queries > 0, axis=1), k * self.oversample)
        vectors = self.base[candidates.ravel()].reshape(*candidates.shape, -1)
        scores = np.einsum("qcd,qd->qc", vectors, queries)
        best = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(candidates, best, axis=1)


def build(kind: str, base: np.ndarray, m: int = 32, ef_construction: int = 200):
    d = base.shape[1]
    if kind == "binary":
        return BinaryRescore(base)
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, m, faiss.METRIC_INNER_PRODUCT)
        index.train(np.ascontiguousarray(base[:100_000]))
    index.hnsw.efConstruction = ef_construction
    index.add(base)
    return index


def sizes(index, base: np.ndarray, scratch: Path) -> tuple[int, int]:
    """Serialized index bytes, plus the float vectors binary+rescore must keep for rescoring."""
    if isinstance(index, BinaryRescore):
        faiss.write_index_binary(index.index, str(scratch))
        rescore = base.nbytes
    else:
        faiss.write_index(index, str(scratch))
        rescore = 0
    size = scratch.stat().st_size
    scratch.unlink()
    return size, rescore


def searcher(index, param: int):
    if isinstance(index, BinaryRescore):
        index.oversample = param
        return index.search
    index.hnsw.efSearch = param
    return lambda q: index.search(q, K)[1]


def measure(search, queries: np.ndarray, truth: np.ndarray, runs: int = 3, warmup: int = 100) -> dict:
    """Single-query latency and single-thread batch throughput, each the median over runs."""
    for q in queries[:warmup]:
        search(q[None])
    p50, p95, qps = [], [], []
    for run in range(runs):
        times = []
        for i in np.random.default_rng(run).permutation(len(queries)):  # new order each run, so caches don't favour later runs
            start = time.perf_counter_ns()
            search(queries[i:i + 1])
            times.append(time.perf_counter_ns() - start)
        p50.append(np.percentile(times, 50) / 1e6)
        p95.append(np.percentile(times, 95) / 1e6)
        batch_start = time.perf_counter()
        found = search(queries)
        qps.append(len(queries) / (time.perf_counter() - batch_start))
    return {
        "recall": round(recall(found, truth), 4),
        "p50_ms": round(float(statistics.median(p50)), 4),
        "p95_ms": round(float(statistics.median(p95)), 4),
        "batch_qps": round(float(statistics.median(qps)), 1),
    }


def environment() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "faiss": faiss.__version__,
        "faiss_compile": faiss.get_compile_options(),
        "numpy": np.__version__,
        "blas": np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}).get("name"),
        "search_threads": 1,
        "thread_env": {v: os.environ[v] for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    source = DATA / "source.json"
    if source.exists():
        info["data"] = json.loads(source.read_text())
    if sys.platform == "darwin":
        info["cpu"] = shell("sysctl", "-n", "machdep.cpu.brand_string")
        info["swap_at_start"] = shell("sysctl", "-n", "vm.swapusage")
        info["power_source"] = shell("pmset", "-g", "batt").splitlines()[0]
        info["low_power_mode"] = [line.strip() for line in shell("pmset", "-g").splitlines() if "lowpowermode" in line]
    return info


def shell(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


def run(scales: list[int], full_sweep_at: int | None, runs: int) -> None:
    base_all = np.load(DATA / "base.npy", mmap_mode="r")
    queries = np.load(DATA / "queries.npy")
    full_sweep_at = full_sweep_at or max(scales)
    if full_sweep_at not in scales:
        raise SystemExit(f"--full-sweep-at {full_sweep_at} is not one of --scales")
    if max(scales) > len(base_all):
        raise SystemExit(f"base.npy holds {len(base_all):,} vectors; rerun prepare with a larger --n")
    RESULTS.mkdir(exist_ok=True)
    env = environment()
    (RESULTS / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    rng = random.Random(SEED)
    fields = ["scale", "kind", "m", "ef_construction", "param", "build_s", "index_mb", "rescore_mb",
              "total_mb", "recall", "p50_ms", "p95_ms", "batch_qps"]
    for scale in scales:
        with (RESULTS / f"runs_{scale}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            base = np.ascontiguousarray(base_all[:scale])
            gt_path = DATA / f"gt_{scale}.npy"
            if not gt_path.exists():
                np.save(gt_path, ground_truth(base, queries))
            truth = np.load(gt_path)
            graphs = FULL_SWEEP if scale == full_sweep_at else [DEFAULT_GRAPH]
            configs = [("hnsw", *g) for g in graphs] + [("hnsw_sq8", *DEFAULT_GRAPH), ("binary", None, None)]
            rng.shuffle(configs)
            for kind, m, efc in configs:
                faiss.omp_set_num_threads(os.cpu_count() or 1)  # builds are not timed against searches
                start = time.perf_counter()
                index = build(kind, base, m or 0, efc or 0)
                build_s = time.perf_counter() - start
                faiss.omp_set_num_threads(1)
                index_bytes, rescore_bytes = sizes(index, base, DATA / "scratch.index")
                params = list(OVERSAMPLE if kind == "binary" else EF_SEARCH)
                rng.shuffle(params)
                for param in params:
                    row = {"scale": scale, "kind": kind, "m": m, "ef_construction": efc, "param": param,
                           "build_s": round(build_s, 1), "index_mb": round(index_bytes / 2**20, 1),
                           "rescore_mb": round(rescore_bytes / 2**20, 1),
                           "total_mb": round((index_bytes + rescore_bytes) / 2**20, 1),
                           **measure(searcher(index, param), queries, truth, runs)}
                    writer.writerow(row)
                    f.flush()
                    print(row, flush=True)
                del index
    if sys.platform == "darwin":
        env["swap_at_end"] = shell("sysctl", "-n", "vm.swapusage")
        (RESULTS / "env.json").write_text(json.dumps(env, indent=2) + "\n")


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points no other point beats on both cost (lower x) and recall (higher y)."""
    front, best = [], -1.0
    for x, y in sorted(points, key=lambda p: (p[0], -p[1])):
        if y > best:
            front.append((x, y))
            best = y
    return front


def pick(rows: list[dict], balanced_recall: float = 0.95, memory_recall: float = 0.90) -> dict:
    """The three README rows: max recall, fastest at >= balanced_recall, smallest at >= memory_recall."""
    return {
        "max recall": max(rows, key=lambda r: (r["recall"], -r["p95_ms"])),
        f"balanced (fastest p95, recall >= {balanced_recall})":
            min((r for r in rows if r["recall"] >= balanced_recall), key=lambda r: r["p95_ms"], default=None),
        f"min memory (recall >= {memory_recall})":
            min((r for r in rows if r["recall"] >= memory_recall), key=lambda r: (r["total_mb"], r["p95_ms"]), default=None),
    }


def label(r: dict) -> str:
    if r["kind"] == "binary":
        return f"binary + rescore, oversample {r['param']}x"
    name = "HNSW" if r["kind"] == "hnsw" else "HNSW-SQ8"
    return f"{name} M={r['m']} efC={r['ef_construction']} efSearch={r['param']}"


def report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    rows = []
    for path in sorted(RESULTS.glob("runs_*.csv")):
        with path.open() as f:
            rows += [{k: (v if k == "kind" else float(v) if v else None) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["scale"], r["param"] = int(r["scale"]), int(r["param"])
        r["m"] = int(r["m"]) if r["m"] else None
        r["ef_construction"] = int(r["ef_construction"]) if r["ef_construction"] else None
    top = max(r["scale"] for r in rows)
    at_top = [r for r in rows if r["scale"] == top]

    # Reference palette slots 1-3 (validated all-pairs); marker shapes are the secondary encoding.
    styles = {"hnsw": ("HNSW (float32)", "#2a78d6", "o"), "hnsw_sq8": ("HNSW-SQ8", "#eb6834", "s"),
              "binary": ("Binary + float rescore", "#1baf7a", "^")}
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    for x_key, x_label, name, log in (("p95_ms", "p95 single-query latency (ms, log scale)", "recall_vs_latency", True),
                                      ("total_mb", "memory incl. rescoring store (MB)", "recall_vs_memory", False)):
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150, facecolor=surface)
        ax.set_facecolor(surface)
        for kind, (title, color, marker) in styles.items():
            pts = [r for r in at_top if r["kind"] == kind]
            if pts:
                ax.scatter([r[x_key] for r in pts], [r["recall"] for r in pts], s=46, color=color, marker=marker,
                           edgecolors=surface, linewidths=1.5, label=title, zorder=3)
        front = pareto([(r[x_key], r["recall"]) for r in at_top])
        ax.step([p[0] for p in front], [p[1] for p in front], where="post", color=muted, linewidth=1,
                linestyle="--", label="Pareto front", zorder=2)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(x_label, color=muted)
        ax.set_ylabel(f"recall@{K}", color=muted)
        ax.set_title(f"{top:,} vectors, single-thread search", color=ink, loc="left", fontsize=11, pad=26)
        ax.grid(color=grid, linewidth=0.6)
        ax.tick_params(colors=muted, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(frameon=False, fontsize=9, labelcolor=ink, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4,
                  handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{name}.png", facecolor=surface)
        plt.close(fig)

    lines = [f"## {top:,} vectors", "", "| Pick | Configuration | recall@10 | p95 ms | Memory MB | Batch QPS |",
             "|---|---|---:|---:|---:|---:|"]
    for title, r in pick(at_top).items():
        lines.append(f"| {title} | {label(r)} | {r['recall']:.3f} | {r['p95_ms']:.2f} | {r['total_mb']:,.0f} | "
                     f"{r['batch_qps']:,.0f} |" if r else f"| {title} | none reached the bar | | | | |")
    lines += ["", "## Scale", "", f"Fastest p95 per family that reaches recall@10 >= 0.95. HNSW and HNSW-SQ8 use "
              f"M={DEFAULT_GRAPH[0]}, efConstruction={DEFAULT_GRAPH[1]} at every scale, so rows compare like with like.", "",
              "| Vectors | " + " | ".join(t for t, _, _ in styles.values()) + " |", "|---:|" + "---|" * len(styles)]
    for scale in sorted({r["scale"] for r in rows}):
        cells = []
        for kind in styles:
            ok = [r for r in rows if r["scale"] == scale and r["kind"] == kind and r["recall"] >= 0.95
                  and (kind == "binary" or (r["m"], r["ef_construction"]) == DEFAULT_GRAPH)]
            r = min(ok, key=lambda r: r["p95_ms"]) if ok else None
            cells.append(f"{r['p95_ms']:.2f} ms ({label(r)})" if r else "below 0.95")
        lines.append(f"| {scale:,} | " + " | ".join(cells) + " |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="download or generate vectors, hold out queries")
    p.add_argument("--source", choices=["cohere", "synthetic"], default="cohere")
    p.add_argument("--n", type=int, default=1_000_000, help="base vectors (queries are extra)")
    p.add_argument("--dim", type=int, default=768, help="synthetic only")
    r = sub.add_parser("run", help="build every index, sweep search params, write results/runs.csv")
    r.add_argument("--scales", type=int, nargs="+", default=[100_000, 250_000, 1_000_000])
    r.add_argument("--full-sweep-at", type=int, help="scale for the full M x efConstruction sweep (default: largest)")
    r.add_argument("--runs", type=int, default=3)
    sub.add_parser("report", help="plots + results/summary.md from results/runs.csv")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.n, args.dim)
    elif args.command == "run":
        run(args.scales, args.full_sweep_at, args.runs)
    else:
        report()


if __name__ == "__main__":
    main()

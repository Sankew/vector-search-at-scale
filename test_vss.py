import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import vss


def unit(n, d, seed=1):
    x = vss.synthetic(n, d) if seed is None else np.random.default_rng(seed).standard_normal((n, d), dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def test_ground_truth_and_recall_match_brute_force():
    base, queries = unit(2000, 32), unit(50, 32, seed=2)
    truth = vss.ground_truth(base, queries)
    expected = np.argsort(-(queries @ base.T), axis=1)[:, :vss.K]
    assert vss.recall(truth, expected) == 1.0
    assert vss.recall(expected[:, ::-1], expected) == 1.0  # order-insensitive
    assert vss.recall(expected[:, :5].repeat(2, axis=1), expected) == 0.5


def test_indexes_reach_high_recall_and_binary_counts_its_rescoring_store(tmp_path):
    base, queries = unit(5000, 64, seed=None), unit(5100, 64, seed=None)[5000:]
    truth = vss.ground_truth(base, queries)
    for kind in ("hnsw", "hnsw_sq8"):
        index = vss.build(kind, base, 16, 100)
        assert vss.recall(vss.searcher(index, 256)(queries), truth) >= 0.95, kind
    binary = vss.build("binary", base)
    low = vss.recall(vss.searcher(binary, 1)(queries), truth)
    high = vss.recall(vss.searcher(binary, 32)(queries), truth)
    assert high > low
    index_bytes, rescore_bytes = vss.sizes(binary, base, tmp_path / "b.index")
    assert rescore_bytes == base.nbytes and index_bytes < base.nbytes / 16


def test_binary_rescore_is_exact_when_every_vector_is_a_candidate():
    base, queries = unit(320, 32), unit(10, 32, seed=4)
    binary = vss.build("binary", base)
    assert vss.recall(vss.searcher(binary, 32)(queries), vss.ground_truth(base, queries)) == 1.0


def test_measure_reports_latency_and_throughput():
    base, queries = unit(1000, 16), unit(20, 16, seed=3)
    index = vss.build("hnsw", base, 16, 100)
    result = vss.measure(vss.searcher(index, 64), queries, vss.ground_truth(base, queries), runs=2, warmup=5)
    assert result["recall"] > 0.9 and 0 < result["p50_ms"] <= result["p95_ms"] and result["batch_qps"] > 0


def test_embeddings_read_a_list_column(tmp_path):
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"id": [1, 2], "emb": pa.array([[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]], pa.list_(pa.float32()))}), path)
    out = vss.embeddings(pq.read_table(path, columns=["emb"]))
    assert out.dtype == np.float32 and out.tolist() == [[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_pareto_and_picks():
    assert vss.pareto([(1, 0.5), (2, 0.4), (3, 0.9), (3, 0.8)]) == [(1, 0.5), (3, 0.9)]
    rows = [
        {"recall": 0.99, "p95_ms": 5.0, "total_mb": 3000},
        {"recall": 0.96, "p95_ms": 1.0, "total_mb": 3100},
        {"recall": 0.92, "p95_ms": 2.0, "total_mb": 800},
        {"recall": 0.50, "p95_ms": 0.1, "total_mb": 100},
    ]
    picks = list(vss.pick(rows).values())
    assert picks == [rows[0], rows[1], rows[2]]


def test_binary_candidate_search_does_not_pad_with_invalid_ids():
    base, queries = unit(20, 32), unit(5, 32, seed=4)
    index = vss.build("binary", base)
    assert vss.recall(vss.searcher(index, 32)(queries), vss.ground_truth(base, queries)) == 1.0


def test_report_rejects_stale_or_incomplete_runs(tmp_path, monkeypatch):
    import json
    import pytest
    monkeypatch.setattr(vss, "DATA", tmp_path / "data")
    monkeypatch.setattr(vss, "RESULTS", tmp_path / "results")
    vss.DATA.mkdir();vss.RESULTS.mkdir()
    for name in ("base.npy", "queries.npy", "source.json"):
        (vss.DATA/name).write_bytes(b"test")
    manifest = {"scales": [100], "data_fingerprint": vss.data_fingerprint(), "complete": True}
    path = vss.RESULTS/"run_manifest.json"
    path.write_text(json.dumps(manifest))
    (vss.RESULTS/"runs_1000000.csv").write_text("obsolete")
    assert vss.report_paths() == [vss.RESULTS/"runs_100.csv"]
    (vss.DATA/"queries.npy").write_bytes(b"new queries")
    with pytest.raises(SystemExit, match="Prepared data changed"):
        vss.report_paths()
    manifest["complete"]=False;path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="incomplete"):
        vss.report_paths()

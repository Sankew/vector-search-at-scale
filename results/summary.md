## 1,000,000 vectors

| Pick | Configuration | recall@10 | p95 ms | Memory MiB | Batch QPS |
|---|---|---:|---:|---:|---:|
| max recall | HNSW M=64 efC=100 efSearch=512 | 0.997 | 13.21 | 4,410 | 116 |
| balanced (fastest p95, recall >= 0.95) | HNSW-SQ8 M=32 efC=200 efSearch=32 | 0.951 | 0.81 | 1,236 | 1,791 |
| min memory (recall >= 0.9) | HNSW-SQ8 M=32 efC=200 efSearch=16 | 0.904 | 0.57 | 1,236 | 2,862 |

## Scale

Fastest p95 per family that reaches recall@10 >= 0.95. HNSW and HNSW-SQ8 use M=32, efConstruction=200 at every scale, so rows compare like with like.

| Vectors | HNSW (float32) | HNSW-SQ8 | Binary + float rescore |
|---:|---|---|---|
| 100,000 | 0.80 ms (HNSW M=32 efC=200 efSearch=32) | 0.63 ms (HNSW-SQ8 M=32 efC=200 efSearch=32) | 3.87 ms (binary + rescore, oversample 16x) |
| 250,000 | 0.87 ms (HNSW M=32 efC=200 efSearch=32) | 1.17 ms (HNSW-SQ8 M=32 efC=200 efSearch=64) | 9.02 ms (binary + rescore, oversample 16x) |
| 1,000,000 | 1.03 ms (HNSW M=32 efC=200 efSearch=32) | 0.81 ms (HNSW-SQ8 M=32 efC=200 efSearch=32) | 31.70 ms (binary + rescore, oversample 8x) |

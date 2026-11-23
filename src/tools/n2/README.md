# Native search proposals

This is an opt-in A-track engineering entrance, separate from both the public
N1 product passes and `legacy-heir`. It takes original native-linalg bytes and
their input-spec bytes, not a selected HEIR graph or a `seed_profile` basis.
Build on the project server with its pinned SDK:

```sh
cmake -S src/tools/n2 -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$EXT/highs-install" -DSNABF_MLIR_ROOT="$EXT/mlir24/usr/lib/llvm-24" -DHEIR="$EXT/heir"
cmake --build "$BUILD" --target snabf-n2-propose --parallel 1
python3 -B src/tools/n2/search.py --proposer "$BUILD/snabf-n2-propose" \
  --mlir "$SOURCE" --input-spec "$PORTS" --log-n 15 --slots 4096 --q-count 8 \
  --min-scale 42 --max-scale 49 --out "$NEW_OUT" --cache "$PRIVATE_CACHE" --estimator "$EXT/lattice-estimator"
python3 -B src/tools/n2/replay.py --out "$NEW_OUT" --mlir "$SOURCE" --input-spec "$PORTS" \
  --proposer "$BUILD/snabf-n2-propose" --cache "$PRIVATE_CACHE" --rebuild-proposals
```

All cells use the same measured `(N, 128-bit classical)` power-of-two
envelope. The bounded grid is a search policy, not a sharp security bound.
The pinned estimator must actually pass at the returned power of two. The
matrix driver subsequently measures the exact ordered Q, QP and envelope
requests for each proposal, using an owner-private cache. The C++ worker
alone produces **unscreened** proposals even when given a large cap.
Rows pin `alpha=1/2`; columns use narrow or unanchored `[24,59]` non-base
search widths. Other policy, source/spec, and graph comparisons are explicit.
The reported spread is `log2(max(q[1:])/min(q[1:]))` on actual primes.
The initial cut here is `max-k=0`. A screened matrix does not imply a product
MLIR seal, a physical multi-P lowering, a ciphertext run, or a speedup.

## Full-slot runtime fixture

`test/compiler/a2b_fullslot_graph.py` consumes a locally built ringswitch Go
test executable (`go test -mod=readonly -c -o "$GO_TEST" ./src/backend/ringswitch`)
and `snabf-pass1`. Run with `--go-test "$GO_TEST" --pass1 "$PASS1" --out
"$NEW_GRAPH_DIR" --cache "$PRIVATE_CACHE" --estimator "$EXT/lattice-estimator"`;
use `--replay "$NEW_GRAPH_DIR" --cache "$PRIVATE_CACHE"` to replay the slots
and match the exact six estimator requests against the trusted private cache.
Without that cache replay is observation-only. Raw slot data,
measurements, build products and keys do not belong in Git. Both inputs run
on every arm. Adaptive minimizes marginal serialized key bytes after the
domain screens, not latency. Per-output two-half margin witnesses are
observations on the tested inputs; they are not a Pass 3 budget or a seal.

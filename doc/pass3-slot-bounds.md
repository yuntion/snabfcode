# Pass 3 ideal-magnitude replay

`--snabf-pass3='magnitude-mode=co-slot-v1 ...'` opts a **legacy Pass 3** plan into per-encoded-slot ideal magnitudes. Omitting the option or using `legacy` reproduces the pre-existing scalar recurrence and writes no new option into the plan. The checker takes its mode from the concrete `options` record (including the profile copy); an unknown mode fails. Do not retrofit a marker or reseal an archived module.

For each bound prepared diagonal `D_d` and selected physical decomposition `d = b+g (mod N/2)`, baby slot `s` uses `B[(s+b) mod P]` and the mask at that slot is `abs(D_d[(s-g) mod P])`, with `P` the **encoded slot count**, not the auxiliary prime basis. The coset sum is pointwise, the giant rotates it left by `g`, and the merge is pointwise. Thus the completed row `i` is bounded by `sum_d abs(D_d[i])*B[(i+d) mod P]`. Every product and addition of ideal bounds uses binary64 rounding toward positive infinity. The scalar fed to Pass 3 at each expanded edge is the maximum **after** this transfer. A decoded ideal rescale preserves the slot values; the physical prime divisor is already accounted for by the scale recurrence.

The mode requires one verified periodic source ring, finite binary32/64 literal tables, a matching prepared slot count and row order, and a valid physical decomposition. It does not assume that a logical-width prefix represents the rest of a table. A fresh Pass 2 affine proof, when enabled, checks the replicas and encloses the selected real polynomial including its approximation contract. Only the matching completed SSA result may intersect that proof, lane by lane. Polynomial intermediate products use their actual parents. Unknown packing, mixed-ring conversion and native-only IR are outside this mode; a failed proof is not permission to use a tighter bound.

This changes the *ideal message* envelope, not the error transfer model. A mask still has noise gain `max(abs(D_d))`, sums/rotations still have unit input gains, and a ciphertext product still has the two opposite global parent magnitudes as gains plus the existing error-product term. Arbitrary encoded `mul_plain` vectors use their uniform maximum unless a lane interpretation is proved. No security screen, exact-scale gate, or noise certificate is bypassed.

## Server gates

Run only in an authorized **isolated** server checkout. Set `SNABF_EXTERN` to the server dependency prefix and use its normal SDK/shared-library environment. Configure and build in a fresh directory:

```bash
B=$(mktemp -d)
SDK=${SNABF_MLIR_ROOT:-$SNABF_EXTERN/mlir24/usr/lib/llvm-24}
export LD_LIBRARY_PATH="$SDK/lib:$SDK/../x86_64-linux-gnu:$SNABF_EXTERN/highs-install/lib:${LD_LIBRARY_PATH:-}"
cmake -S src/compiler -B "$B" -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release \
  -DSNABF_MLIR_ROOT="$SDK" -DHEIR="$SNABF_EXTERN/heir" -DCMAKE_PREFIX_PATH="$SNABF_EXTERN/highs-install"
cmake --build "$B" -j4
ctest --test-dir "$B" --output-on-failure -R '^Pass3SlotBounds(Test|IR)$'
ctest --test-dir "$B" --output-on-failure
```

`Pass3SlotBoundsTest` checks the unsplit signed-row oracle, BSGS frame changes, nonuniform parents, directed rounding, fail-closed transfers and affine projection. `Pass3SlotBoundsIR` constructs a small post-import module and drives actual Pass 2/3, verifies the historical and new fixed plans up to the missing security screen, and rejects a changed replay policy. It does not fabricate security evidence. For a **known base-compatible** sealed plan already on the server, compare fresh canonical checker output from the base and candidate binaries; leave the input untouched:

```bash
"$BASE_OPT" --allow-unregistered-dialect --verify-each --snabf-check-plan "$SEALED" -o "$B/base.mlir"
"$B/ir/snabf-opt" --allow-unregistered-dialect --verify-each --snabf-check-plan "$SEALED" -o "$B/candidate.mlir"
cmp "$B/base.mlir" "$B/candidate.mlir"
```

For the 17Q replay, `A2_CAPTURE` is the existing **server-resident** `capture` directory with each arm's own `selected.mlir`. No historical archive is needed in the transfer. The optional `magnitude-audit` is written before parameter solving; `base_width_term_bits` uses the actual MILP guard implementation for each default scale, not an inferred Q width.

```bash
for a in 1 2; do for k in 0 1; do
  python3 -B test/compiler/pass3_slot_bounds_17q.py --opt "$B/ir/snabf-opt" \
    --selected "$A2_CAPTURE/a$a-k$k/selected.mlir" --alpha "$a" --max-k "$k" \
    --out "$B/row-bound-a$a-k$k" || exit 1
done; done
```

The gate requires the documented old first-layer `16.37841796875` and `lin3.rescale` guard `292.75..302.75` bits; the new first-layer bound must be `0.5`, and the full base-width lower bound must be below the 59-bit builder cap at all 11 exponents. It records subsequent solver failure separately from a parameter proposal. Passing this diagnostic is **not** evidence of an accepted parameter set, a ciphertext run, or native N1 execution. The same comparison accepts independently replanned, already-resident affine selected inputs with the same 17Q source.

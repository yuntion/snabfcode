# Phase 2 decisions (2026-11-20)

Baseline: `ce6a47fe21fbb0bd5988c46f4867d0bf2334c2ff`. Keep Pass 1-4 numbering. DEV-1 and DEV-2 are the first deliverables; WP1-WP6 are a roadmap. Append new decisions here.

1. **Security.** Rounded log QP is a search hint only. Bind measurements to the actual integer modulus, ordered Q/P, ring, key basis/role, estimator revision, distributions, sample and attack model. Cache measurements inside a trusted boundary and replay current thresholds; a checksum is not a signature. A miss without the pinned estimator rejects. Reject unsupported ordinary-domain targets (including 192) rather than pretending a 128-bit screen suffices.
2. **Execution vs certification.** Engineering execution requires independently checked parameters, security, layout and emission. It does not issue a failure probability. A certificate request must fail closed until primitive contracts, dependency handling, actual scales, every sink's margin and the common-event allocation `delta_valid + sum(delta_sink) <= 2^-40` all pass. The observed `2^-12` error threshold is not that probability.
3. **One boundary, two values.** Convert both distinct live ciphertexts; one physical switching key is counted only when the domain pair and concrete key basis match. Require `max(required) <= L_cut <= min(current)`; dropping levels does not rescale. Check physical scale and level again at the actual downstream addition. Force-down rejects infeasibility; adaptive may retain source. Pass 3 extraction/replay, Pass 4 suffix, and the emitter/runtime must all agree before advertising this capability.
4. **Fixed gadget blocks.** For prefix `0 <= k <= ell`, block `b` is active iff its original last Q-limb index is `>= k`. Active count is `ceil((ell+1)/alpha) - floor(k/alpha)`, not the repacked-suffix shorthand in paper II-C / Pass 3. The counts coincide on block-aligned cuts but not in general: `[q0,q1]|[q2]`, `k=1` still needs both blocks. Use those indicators in the solver, checker and kernel. Keep zero limbs, skipped full external products and measured net limb-NTTs separate. Paper correction pending.
5. **R3 compatibility.** Preserve the frozen seven scenarios per entry (`source`, `pass3`) and the one-live-value backend profile. Its crossing residual must retain the source ring. The two-value profile has separate keep-source / force-down / adaptive expectations on `W1*SiLU(h)+h`, not `W1*(SiLU(h)+h)`. Only compare ciphertext copies under identical source parameters and actual keys.

Review WP scheduling **the week all three DEV-2 arms have checked encrypted results**: full-slot repacking, multiple boundaries, then deep bootstrapping. No calendar or performance commitment before those results.

6. **DEV-2 execution policy.** `dual-live-v1` admits at most two distinct ciphertexts on one periodic boundary before the terminal linear layer, including its target-domain residual addition. Charge each conversion, but generate one physical key for the checked domain pair and exact cut Q/P. Pass 3 can reserve part of the residual arithmetic allowance during source-basis search; this is not a conversion contract or a relaxed final tolerance. Replay both real conversions and the join. `force-down` chooses the cheapest verified down proposal (even if unprofitable); `adaptive` still requires strict net benefit. A verified source-only fallback is not a successful forced execution.

7. **Arithmetic reserves.** The default one-half share in the dual-live profile is a search heuristic for one boundary. It is not a transfer theorem or the allocation policy for multiple boundaries. Before enabling cascades, feed concrete conversion requirements back from Pass 4 to Pass 3 and replay the entire graph under the original sink budget.
8. **Price and depth claims.** Count fixed-block external products separately from limb transforms. A partly zero digit remains a digit over its ORIGINAL block product even when its first limb is zero. Use the product of all actual P primes for ordinary key noise; do not divide the projection residual by P. Abstract work units and serialized bytes are not latency or peak memory. Calibrate new kernels and measure paired programs before claiming a speedup. P count alone does not determine attainable depth or replace the security/scale checks.
9. **HEIR scale capability.** The pinned materializer cannot implement a negative `adjust_scale`. For heterogeneous source compilation, bind the integer scale anchor in the search options and replay against actual Q before emission. Report the measured spread of non-base Q separately from the allowed search range. Feasibility of the unanchored native MILP is not evidence that its program can be emitted.

10. **ABC2 opt-in contracts.** Narrow scales remain the production default. `--scale-mode heir-anchor` is the pinned-HEIR ablation. Never ignore a conflicting explicit mode. A measured-cost route must identify its hardware, exact ordered basis and covered call shapes; missing data is not zero cost. Domain/level/security checks and the legacy objective remain available independently.
11. **Representation boundaries.** A source/encoded plaintext scale receipt describes an actual encoder call; store the effective encoder scale separately from rounded CKKS metadata. A missing receipt cannot authorize a probability certificate. Affine interval symbols describe deterministic message domains, not probabilistically independent errors. Only explicitly opted-in affine plans may use new intervals, and independent plan checking must reconstruct them. These interfaces do not enable full-slot conversion, automatic bootstrapping or certificate issuance by themselves.

## Batch 2 (2026-11-21)

12. **Common base and ownership.** Start `feat/track-a2-batch2`, `feat/track-b2-batch2` and `feat/track-c2-batch2` from released `main` (`902b103805479de28ba097b50ec865a1101faa4f`, PRs #1 and #2). Land a small contract PR and rebase all affected tracks before changing shared IR types/ops, plan-checker inputs, or the direct emitter/runtime ABI. Keep generated programs, measurements and audit archives out of Git.
13. **A: feasibility and pricing.** Attribute a refusal to HEIR depth, the concrete QP security cap, the auxiliary-prime block bound or the sink margin only with a replayable witness at that stage. Allow several causes, or an explicitly unknown cause; a solver status alone proves none of them. Mark unexecuted arms `not_run`. Compare fixed-block prices only on the profiled hardware, ordered Q/P and call shape; check claims against paired ciphertext runs.
14. **A: split and feedback.** Preserve `single-live-v1` and `dual-live-v1`. Opt-in `full-slot-split-v1` initially covers only a 1-to-2 split at `N = 2*N'`: complementary source masks, one consumed level per branch, two mask multiplies/rescales, two half-period folds with one rotation key, two conversions with one concrete switching key, and all four target matrix blocks. Check the decoded block order independently. For cut-margin feedback, cache fingerprints, tighten monotonically, and cap the search at four rounds including the initial solve; no feedback may relax the original security or sink threshold. A verified source fallback is valid for adaptive, not force-down.
15. **B: conditional certification.** A lower count of missing contracts is not a certificate. Bind actual encoder scales, concrete parameter/distribution receipts and primitive contracts; account for shared events without assuming independence from distinct SSA or source IDs, rounding or key reuse. Use exact rationals and directed bounds, allocate one common validity budget, and independently replay every sink's Eq. (15) against checked IR and the bound program before issuance. Missing DownConv contracts refuse certification, not otherwise verified engineering execution.
16. **C: nonlinear and frontend evidence.** Certify the emitted binary64 ReLU reconstruction over the full closed input interval, including zero. Require the original SSA witness, independent structural/tamper checks and encrypted execution with independent slot-wise truth. For generalized linear maps record logical dimensions, masks and output order; multiple input IDs track provenance, not independence. Affine message ranges and probabilistic noise symbols stay separate. A kernel test or a single graph's improvement does not establish whole-graph support or general speedup.

17. **Delete merged PR heads.** Keep GitHub's automatic head-branch deletion enabled. Before manually retiring any branch, fetch current refs, exclude active tracks and open PRs, verify that its exact tip is an ancestor of `origin/main`, and delete against the audited tip. If code was projected, squashed or cherry-picked, verify the content and preserve the original history first; a merged PR number alone is insufficient.
18. **Four active branches; five temporarily.** Keep `main` and one working branch per track, plus at most one short-lived contract or integration branch. Retire it after merge. Starting with the next batch, reuse `track/a`, `track/b` and `track/c`: fast-forward a clean, fully landed track, or recreate its auto-deleted ref from the checked `main`. Never discard unpublished commits or dirty worktrees to make a quota.
19. **Milestones are annotated tags.** Tag the exact verified commit for releases, regression baselines and paper snapshots. Publish only the intended tags, verify their peeled commit IDs, and do not move or reuse them. A branch denotes live development, not an archive.

Implementation note for decision 13 (Batch 2): the 17-Q replay distinguishes a
pre-solver base-width exclusion, an independently contradictory solver row,
and a solver status with no such witness. A relaxed diagnostic row cannot
authorize a production plan. A successful refusal replay is reported
separately from a measured mid-depth matrix; see the README for the probe.

## RFC-004 native frontend (N0)

20. **Dual frontend (`legacy-heir` default, `native-linalg` explicit).** Keep `legacy-heir` as the default CLI path and never rewrite or backfill attributes onto frozen R3/DEV-2/A1/C1 archives. `--frontend native-linalg` must be explicitly selected, bind provenance to original source bytes, execute zero `heir-opt` / `heir-translate` subprocesses, emit Go packages with zero `heirckks` or static-helper dependency, and fail closed on unsupported ops rather than silently falling back to HEIR. Function-argument `secret.secret` annotations are permitted strictly as a whitelisted input-classification marker when consistent with explicit input specs; no HEIR `secret`/`mgmt` passes may run.

21. **Exact scale and independent replay.** Model ideal/requested scale, runtime metadata scale, and per-call effective encoding scale (`EncodingCallReceipt`, bound to call/site/operand, batched flag, argument type/precision, `enc.Prec()`, quantization rule, payload digest, and actual `M`/`E`) separately. Use GMP rationals to represent actual step-by-step rounded runtime states—never substitute a single product division for multi-prime step rounding (`PREC128`). Unknown encoding branches, missing ordered `Q`, or missing operand magnitude bounds on either side of `add`/`sub` forbid certificate issuance. The independent Python verifier must recompute from source witnesses, checked SSA, ordered `Q`/`P`, and public constant bitstrings without linking or trusting C++ derived receipts.

22. **Four-tier scale coordination.** Retain the Tier 1–4 priority order under tightened soundness rules. Tier 1 (`fold_into_plain`) may only re-encode existing pre-join plaintext multiplications without altering logical model weights; `rho in [1/2, 2]` is a search window only, multi-use SSA values require explicit per-use verification before rewriting, and folding must report `requested`, `realized`, and `residual` separately. Tier 2 requires exact concrete match on ordered `Q`/`P` and both scales. Tier 3 keeps relative mismatch `<= 1e-8` as a pre-filter only, never an acceptance gate; local mismatch bias must account for both operands' actual metadata-derived integer multipliers, magnitude bounds, consistent units, and explicitly allocated residual budget (unallocated budget defaults to `0`). Tier 4 (`snabf.align_scale`) encodes plaintext `1` (never scalar `Mul(ct, 1.0)`) and records `mul_depth_delta`, `levels_consumed`, and ordered `dropped_q[]` separately; reject multi-prime rescale modes until explicitly implemented. Report wide-heterogeneous spread as actual non-base `log2(max(Q)/min(Q))` and report gains only from same-graph controlled measurements.

23. **Whole-package native emission and verification.** Native Lattigo emission builds, tests, and publishes the complete `planned` Go package (`model.go`, `native_runtime.go`, `program.json`, etc.) in an isolated directory—never single-file `go build model.go`. Compile-time `program.json` is immutable; runtime observations (actual levels, scales, timings) write to separate run logs linked by the same plan fingerprint. Keep frozen and native test profiles separate; tampered `ScalePlan`, swapped encoding branches producing identical scale values, 1-ULP edits, or reversed `Q` chains must be rejected independently by both `snabf-check-plan` and lowering without emitting Go.

Implementation status of 20 and 23: `--native-import-only` is a source inspection
on the pinned MLIR stack. Its receipt must not be fed to a legacy emitter or
used as a release certificate. For inspection see [native probes](native-probes.md);
the public N1 path requires both MLIR entries, a standalone generated package
and independent slot replay.

Implementation note for decisions 16 and 20: native composite ReLU binds
the source SSA before fusion. On closed cells covering the input domain,
bound the emitted reconstruction with the shared input and all local
perturbations present together. The legacy frontend continues to refuse ReLU;
a ciphertext engineering check is not an Eq. (15) certificate.

Implementation boundary for native N2 and full-slot Pass 4: the fixed-block
N2 profile binds the measured search envelope, concrete ordered Q/P, block
width (alpha 1 or 2) and narrow/wide scale policy back to the original
source and input spec. It cannot be combined with the affine ReLU profile.
The separate bounded full-slot graph supports keep-source, force-down and
adaptive arms; a split carries two ordered halves and preserves the logical
extent of the padded tail. Both plan checking and lowering replay the source
before accepting the carrier. This does not extend support to arbitrary
graphs, certify a failure probability, or establish a latency improvement.

### Wave 1 B: production refusal is required

Residual and single GeLU each still lack `node/input/encode`,
`node/input/encrypt`, `common/validated_joint_law_and_domain`, and
`sink/sink0/reference_approx_and_decode`. Local arithmetic proofs and dry-run
budgets do not discharge these four obligations. Both graphs issue zero
certificates. Accept the production refusal only at rc=1 with the expected
denial, independently replayed missing terms, no valid event allocation and
no certified sink margin.

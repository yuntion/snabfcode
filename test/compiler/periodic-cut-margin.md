# Periodic cut-margin diagnostics

On a screened single-domain checkpoint, `--snabf-pass4=cut-profile=dual-live-feedback-v1`
adds a bounded diagnostic for a refused periodic cut. The public driver's
single- and dual-live defaults are unchanged. The diagnostic projects the
actual boundary MILP rows onto the producer/use bindings. It passes a joint
error budget in decoded units to the source search; it does not split a
guessed reserve between values. A numerical deficit is a model bound, not
measured ciphertext error, a security estimate or permission to convert.

The initial profile counts towards the four-profile limit. A repeated
fingerprint, unavailable switching-key estimate or infeasible source search
stops the retry. A different physical basis requires fresh specialization and
security evidence before it could become an executable candidate. The normal
checker and lowering reject edited feedback receipts.

Build the baseline `5c301ff019cbd4ce8513acd9f43b391034375145` and the candidate
with the same pinned compiler dependencies, then use an empty output directory:

```sh
python3 test/compiler/phase2_cut_margin.py --baseline-opt "$BASE_OPT" \
  --opt "$CANDIDATE_OPT" --fixtures "$R3_FROZEN" --out "$EMPTY_DIR"
```

The script validates frozen input bytes, tests both gates and writes all
observations under the output directory. Do not commit that directory.

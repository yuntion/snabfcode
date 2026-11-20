#pragma once
#include "GadgetNoiseContract.h"
namespace snabf::pass3 {
Solution optimize(const Graph &, const Options &);
// A live ciphertext fixes Q, P and its scale. Re-solve the remaining integer
// gadget decisions using those actual primes; no change of basis is implicit.
Solution optimizeAtBoundary(const Graph &,const Options &,Solution physical);
}

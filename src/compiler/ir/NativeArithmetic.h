#pragma once

#include "NativeLinearRoute.h"
#include "pass2/CompositePolyDP.h"
#include "pass3/GadgetNoiseContract.h"

namespace snabf::native {
// One periodic ciphertext; Pass 1 is priced in work/key counts, not time.
struct ArithmeticOptions {
  PeriodicSingleDomain domain;
  int qCount = 8;
  double activationTolerance = 0x1p-16;
  pass2::DPOptions polynomial;
};

struct NativeArithmeticProgram {
  NativeLinearProgram source;
  std::vector<size_t> routeIndices;
  std::vector<pass2::PolyCandidate> polynomials;
  pass3::Graph graph;
  // Masks contain one inverse-giant-adjusted period. Coefficient/AddPlain
  // carry one signed binary64 operand; Pass 3 magnitude is not its value.
  std::map<int, std::vector<double>> publicOperands;
  std::map<std::string, int> sourceValues;
};

// No disk receipts as inputs; failure leaves the destination unchanged.
mlir::LogicalResult deriveNativeArithmetic(llvm::StringRef bytes, llvm::StringRef sourceName,
    llvm::StringRef inputSpec, const ArithmeticOptions &, NativeArithmeticProgram &,
    llvm::raw_ostream &diagnostics);
// A replayed candidate, NOT a concrete security-oracle decision.
pass3::Solution searchNativeParameters(const NativeArithmeticProgram &, const pass3::Options &);

// Try distinct certified Pass 2 schedules against the unchanged Pass 3
// options. The frontier for a downstream activation depends on its parents.
// The budget limits attempts, not precision or security; never silently relax.
mlir::LogicalResult specializeNativeArithmetic(llvm::StringRef bytes, llvm::StringRef sourceName,
    llvm::StringRef inputSpec, const ArithmeticOptions &, const pass3::Options &, size_t maxAttempts,
    NativeArithmeticProgram &, pass3::Solution &, size_t &attempts, llvm::raw_ostream &diagnostics);
} // namespace snabf::native

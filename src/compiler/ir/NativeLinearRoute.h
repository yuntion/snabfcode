#pragma once

#include "NativeLinalgImport.h"
#include "pass1/SubgroupTower.h"
#include <cstddef>
#include <span>
#include <string>

namespace snabf::native {
// N1 only: one ciphertext, one vector repeated contiguously across all E
// encoded slots. The CKKS rotation group has order N/2, not necessarily E.
struct PeriodicSingleDomain {
  unsigned logN = 0, encodedSlots = 0;
  bool valid() const;
};

struct NativeDiagonal {
  int64_t signedOffset;
  std::vector<uint64_t> rowBits; // One logical period, before BSGS adjustment.
};

struct NativeMatvec {
  std::string resultID, weightID, inputID, outsID, outsKind, weightSHA256;
  unsigned elementBits = 0;
  std::map<unsigned, NativeDiagonal> diagonals; // Key: physical left rotation mod N/2.
  std::vector<SubgroupRoute> candidates;        // Pass 1 candidates, not a selection.
};

struct NativeLinearProgram {
  PeriodicSingleDomain domain{};
  unsigned width = 0;
  llvm::json::Object inspection; // Unmodified C-1 receipt; SSA IDs are scoped to it.
  std::vector<NativeMatvec> matvecs;

  // Expand a specific candidate's inverse-giant-adjusted public mask into
  // exactly E caller-owned words. f32 words are zero-extended, never promoted.
  mlir::LogicalResult materializeMask(size_t site, size_t candidate, size_t term,
                                      std::span<uint64_t> bits) const;
};

// One parse of the supplied bytes. On any source/domain failure, leave out
// untouched. No receipt input, CKKS parameters, pricing or HEIR dependency.
mlir::LogicalResult deriveNativeLinear(llvm::StringRef sourceBytes, llvm::StringRef sourceName,
                                      llvm::StringRef specBytes, PeriodicSingleDomain domain,
                                      NativeLinearProgram &out, llvm::raw_ostream &diagnostics);
// Consumes the inspection. Default-constructed or already consumed results
// cannot produce a detached receipt.
mlir::FailureOr<llvm::json::Object> takeNativeLinearReceipt(NativeLinearProgram &&program);
} // namespace snabf::native

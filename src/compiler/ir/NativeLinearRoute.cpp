#include "NativeLinearRoute.h"
#include "NativeLinalgImportInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <bit>

using namespace mlir;
namespace snabf::native {
bool PeriodicSingleDomain::valid() const {
  return logN >= 2 && logN <= 30 && std::has_single_bit(encodedSlots) &&
         encodedSlots <= (1u << (logN - 1));
}

namespace {
unsigned residue(int64_t d, unsigned modulus) {
  int64_t r = d % int64_t(modulus);
  return unsigned(r < 0 ? r + modulus : r);
}

LogicalResult collect(const Source &s, linalg::MatvecOp op, const ValueIDs &ids,
                      StringRef resultID, NativeLinearProgram &p) {
  auto ins = op.getInputs();
  auto weight = ins[0].getDefiningOp<arith::ConstantOp>();
  auto matrix = cast<DenseFPElementsAttr>(weight.getValue());
  const unsigned w = matrix.getType().getDimSize(0);
  if (p.domain.encodedSlots % w != 0)
    return s.reject(op, "[SNABF_NATIVE_LINEAR_LAYOUT] logical width must divide the encoded slot count");
  p.width = w; // The source verifier requires a single entry vector type throughout.
  Value outs = op.getOutputs().front();
  NativeMatvec m;
  m.resultID = resultID.str();
  m.weightID = ids.lookup(ins[0]);
  m.inputID = ids.lookup(ins[1]);
  m.outsID = ids.lookup(outs);
  m.outsKind = outs.getDefiningOp<linalg::FillOp>() ? "positive-zero-fill" : "positive-zero-dense";
  m.elementBits = matrix.getType().getElementTypeBitWidth();

  // Hash expanded row-major IEEE words in network byte order. In particular
  // a splat and its expansion have the same weight hash, not the same source hash.
  llvm::SHA256 hash;
  std::vector<uint64_t> words;
  words.reserve(w * w);
  for (auto v : matrix.getValues<llvm::APFloat>()) {
    auto word = v.bitcastToAPInt().getZExtValue();
    words.push_back(word);
    uint8_t bytes[8];
    unsigned n = m.elementBits / 8;
    for (unsigned k = 0; k < n; ++k) bytes[n - 1 - k] = uint8_t(word >> (8 * k));
    hash.update(llvm::ArrayRef<uint8_t>(bytes, n));
  }
  m.weightSHA256 = llvm::toHex(hash.final(), /*LowerCase=*/true);

  const unsigned group = 1u << (p.domain.logN - 1);
  std::vector<int64_t> offsets;
  for (unsigned r = 0; r < w; ++r) {
    // Choose the centered lift, including -w/2 at an even width's tie.
    int64_t d = 2 * r >= w ? int64_t(r) - w : int64_t(r);
    NativeDiagonal diag{d, {}};
    diag.rowBits.reserve(w);
    for (unsigned row = 0; row < w; ++row)
      diag.rowBits.push_back(words[row * w + (row + r) % w]);
    // Elide only all-positive-zero masks. Keep negative-zero payloads exact.
    if (llvm::all_of(diag.rowBits, [](uint64_t x) { return x == 0; })) continue;
    offsets.push_back(d);
    m.diagonals.emplace(residue(d, group), std::move(diag));
  }
  m.candidates = executableRoutes(p.domain.logN, offsets);
  p.matvecs.push_back(std::move(m));
  return success();
}
} // namespace

LogicalResult deriveNativeLinear(StringRef bytes, StringRef name, StringRef spec,
                                 PeriodicSingleDomain domain, NativeLinearProgram &out,
                                 llvm::raw_ostream &diagnostics) {
  if (!domain.valid()) {
    diagnostics << name << ":1:1: error: [SNABF_NATIVE_LINEAR_LAYOUT] require 2 <= logN <= 30 "
                           "and a power-of-two encoded slot count <= N/2\n";
    return failure();
  }
  NativeLinearProgram p;
  p.domain = domain;
  auto visit = [&](const Source &s, linalg::MatvecOp op, const ValueIDs &ids, StringRef id) {
    return collect(s, op, ids, id, p);
  };
  if (failed(inspectWithMatvecs(bytes, name, spec, p.inspection, diagnostics, visit))) return failure();
  if (p.matvecs.empty()) {
    diagnostics << name << ":1:1: error: [SNABF_NATIVE_LINEAR_EMPTY] no verified matvec\n";
    return failure();
  }
  out = std::move(p);
  return success();
}

LogicalResult NativeLinearProgram::materializeMask(size_t site, size_t candidate, size_t term,
                                                   std::span<uint64_t> bits) const {
  if (!domain.valid() || !width || domain.encodedSlots % width || bits.size() != domain.encodedSlots ||
      site >= matvecs.size() || candidate >= matvecs[site].candidates.size()) return failure();
  const auto &m = matvecs[site];
  const auto &terms = m.candidates[candidate].terms;
  if (term >= terms.size()) return failure();
  auto [d, baby, giant] = terms[term];
  const unsigned group = 1u << (domain.logN - 1), slots = domain.encodedSlots;
  auto it = m.diagonals.find(d);
  if (it == m.diagonals.end() || it->second.rowBits.size() != width ||
      residue(it->second.signedOffset, group) != d || baby >= group || giant >= group ||
      (baby + giant) % group != d) return failure();
  // Rot_g(mask * Rot_b(x)): the encoder needs Rot_-g(D_d), not D_d.
  for (unsigned j = 0; j < slots; ++j)
    bits[j] = it->second.rowBits[((j + slots - giant % slots) % slots) % width];
  return success();
}
} // namespace snabf::native

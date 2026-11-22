#include "NativeLinearRoute.h"
#include "RouteSelection.h"
#include "llvm/Support/MemoryBuffer.h"
#include <stdexcept>

using namespace snabf::native;
using mlir::failed;
using mlir::succeeded;
namespace {
void require(bool condition, const char *why) {
  if (!condition) throw std::runtime_error(why);
}

// Exercise the receiving Pass 1 C++ API unchanged. These synthetic costs
// exist only in the test binary, never in the source-bound candidate receipt.
void checkSelector(const NativeLinearProgram &p) {
  std::vector<snabf::LinearRegion> regions;
  snabf::KeySizes sizes;
  for (auto &site : p.matvecs) {
    auto &r = regions.emplace_back(snabf::LinearRegion{"test-only", 1, 2, site.candidates});
    for (auto &c : r.candidates)
      for (auto key : c.galoisKeys) sizes[{r.keyScope, key}] = 64;
  }
  auto picked = snabf::selectSubgroupRoutes(regions, sizes, 1.0);
  require(picked.selected.size() == regions.size(), "Pass 1 did not select one candidate per site");
  for (size_t i = 0; i < regions.size(); ++i)
    require(picked.selected[i] < p.matvecs[i].candidates.size(), "Pass 1 candidate index");
}

// Test-only binary witness for the independent source-column oracle. Stream
// actual materializeMask results in site/candidate/term/encoded-slot order.
int dump(const char *source, const char *spec, const char *n, const char *e) {
  auto src = llvm::MemoryBuffer::getFile(source, /*IsText=*/false);
  auto ports = llvm::MemoryBuffer::getFile(spec, /*IsText=*/false);
  PeriodicSingleDomain domain;
  if (!src || !ports || llvm::StringRef(n).getAsInteger(10, domain.logN) ||
      llvm::StringRef(e).getAsInteger(10, domain.encodedSlots)) return 2;
  NativeLinearProgram p;
  if (failed(deriveNativeLinear((*src)->getBuffer(), source, (*ports)->getBuffer(), domain, p, llvm::errs()))) return 1;
  checkSelector(p);
  std::vector<uint64_t> mask(domain.encodedSlots);
  for (size_t i = 0; i < p.matvecs.size(); ++i) {
    const auto &site = p.matvecs[i];
    unsigned bytes = site.elementBits / 8;
    std::string packed(mask.size() * bytes, '\0');
    for (size_t k = 0; k < site.candidates.size(); ++k) {
      for (size_t t = 0; t < site.candidates[k].terms.size(); ++t) {
        require(succeeded(p.materializeMask(i, k, t, mask)), "cannot materialize candidate mask");
        size_t pos = 0;
        for (uint64_t word : mask)
          for (int shift = int(bytes - 1) * 8; shift >= 0; shift -= 8)
            packed[pos++] = char(uint8_t(word >> shift));
        llvm::outs().write(packed.data(), packed.size());
      }
    }
  }
  return 0;
}

constexpr llvm::StringLiteral spec = R"({"schema":"snabf.native-input-spec.v1","inputs":[{"arg_index":0,"func_symbol":"f","classification":"secret","source_id":"x"}]})";
std::string source(llvm::StringRef matrix, llvm::StringRef init = "0.0") {
  return (llvm::Twine("module { func.func @f(%x: tensor<2xf64>) -> tensor<2xf64> {\n") +
          "%z = arith.constant dense<" + init + "> : tensor<2xf64>\n" +
          "%w = arith.constant dense<" + matrix + "> : tensor<2x2xf64>\n" +
          "%y = linalg.matvec ins(%w, %x : tensor<2x2xf64>, tensor<2xf64>) " +
          "outs(%z : tensor<2xf64>) -> tensor<2xf64>\nreturn %y : tensor<2xf64>\n}}\n").str();
}

void api() {
  NativeLinearProgram p;
  std::string diagnostics;
  llvm::raw_string_ostream os(diagnostics);
  auto run = [&](const std::string &src, PeriodicSingleDomain d = {5, 8}) {
    return deriveNativeLinear(src, "probe.mlir", spec, d, p, os);
  };
  require(succeeded(run(source("0.0"))), "zero-map import");
  const auto &z = p.matvecs.at(0);
  require(z.outsKind == "positive-zero-dense" && z.diagonals.empty() && z.candidates.size() == 1 &&
          z.candidates[0].terms.empty() && z.candidates[0].galoisKeys.empty(), "empty support or zero outs");
  require(succeeded(run(source("[[1.0000000000000002, 0.0], [0.0, 3.0]]"))), "identity support import");
  require(p.matvecs[0].diagonals.size() == 1 && p.matvecs[0].candidates[0].galoisKeys.empty(), "diagonal-only route");
  std::vector<uint64_t> mask(8);
  require(succeeded(p.materializeMask(0, 0, 0, mask)) && mask[6] == 0x3ff0000000000001ULL &&
          mask[7] == 0x4008000000000000ULL, "f64 must not narrow or zero-pad");
  require(failed(p.materializeMask(1, 0, 0, mask)) && failed(p.materializeMask(0, 1, 0, mask)) &&
          failed(p.materializeMask(0, 0, 1, mask)) && failed(p.materializeMask(0, 0, 0, std::span(mask).first(4))),
          "out-of-contract mask request");
  require(succeeded(run(source("[[0.0, -0.0], [0.0, 0.0]]"))), "signed-zero import");
  const auto &m = p.matvecs[0];
  const auto &d = m.diagonals.at(15);
  require(m.diagonals.size() == 1 && d.signedOffset == -1 && d.rowBits[0] == (1ULL << 63) &&
          d.rowBits[1] == 0, "signed-zero bits or signed physical lift lost");
  auto before = m.weightSHA256;
  auto late = source("1.0");
  late.insert(late.find("return"), "%bad = math.sqrt %y : tensor<2xf64>\n");
  require(failed(run(late)) && diagnostics.find("probe.mlir:5:") != std::string::npos &&
          failed(run(source("1.0", "-0.0"))) && failed(run(source("1.0"), {2, 1})) &&
          failed(run("module { func.func @f(%x: tensor<2xf64>) -> tensor<2xf64> { return %x : tensor<2xf64> } }")) &&
          p.matvecs.size() == 1 && p.matvecs[0].weightSHA256 == before, "partial output after refusal");
  require(succeeded(run(source("1.0"), {4, 8})) && p.matvecs[0].diagonals.contains(7) &&
          succeeded(run(source("1.0"), {5, 2})) && p.matvecs[0].diagonals.contains(15), "stale lift after domain change");
  require(failed(takeNativeLinearReceipt(NativeLinearProgram{})) && succeeded(takeNativeLinearReceipt(std::move(p))) &&
          failed(takeNativeLinearReceipt(std::move(p))), "cannot serialize absent or consumed inspection");
  llvm::outs() << "typed routes: zero-map, no-key diagonal, f64 ULP, signed zero, mask bounds, atomic refusal, domain change passed\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 6 && llvm::StringRef(argv[1]) == "--masks") return dump(argv[2], argv[3], argv[4], argv[5]);
    if (argc != 1) return 2;
    api();
    return 0;
  } catch (const std::exception &e) { llvm::errs() << e.what() << '\n'; return 1; }
}

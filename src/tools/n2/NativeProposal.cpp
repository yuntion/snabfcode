#include "NativeArithmetic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <charconv>
#include <stdexcept>
#include <string_view>

using namespace snabf;
using namespace snabf::native;
using llvm::json::Array;
using llvm::json::Object;

namespace {
int number(std::string_view text, int lo, int hi) {
  int value = 0;
  auto [pos, err] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (err != std::errc{} || pos != text.data() + text.size() || value < lo || value > hi)
    throw std::invalid_argument("invalid N2 integer argument");
  return value;
}
template<class T> Array nums(const T &xs) {
  Array a;
  for (const auto x : xs) a.emplace_back(int64_t(x));
  return a;
}
// Identify the actual pre-Pass-3 arithmetic DAG, including routed public masks.
std::string graphID(const NativeArithmeticProgram &p) {
  Array edges;
  for (size_t i = 0; i < p.graph.edges.size(); ++i) {
    const auto &e = p.graph.edges[i];
    Array gains, pt;
    for (double x : e.gains) gains.push_back(x);
    if (auto it = p.publicOperands.find(i); it != p.publicOperands.end())
      for (double x : it->second) pt.push_back(x);
    edges.push_back(Array{e.id, pass3::name(e.op), e.level, nums(e.inputs), std::move(gains),
                          e.magnitude, e.maskLevel, e.poly, e.node, e.shift, std::move(pt)});
  }
  Array shape{p.graph.n, p.graph.levels, p.graph.sink, p.graph.approximation,
              nums(p.routeIndices), std::move(edges)};
  auto text = llvm::formatv("{0}", llvm::json::Value(std::move(shape))).str();
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 11) {
    llvm::errs() << "usage: snabf-n2-propose SOURCE PORTS LOG_N SLOTS Q_COUNT D_MIN D_MAX ALPHA narrow|wide CAP_BITS\n";
    return 2;
  }
  try {
    auto source = llvm::MemoryBuffer::getFile(argv[1]), spec = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source || !spec) throw std::invalid_argument("missing original source or input spec");
    int logN = number(argv[3], 14, 16), slots = number(argv[4], 1, 1 << (logN-1));
    ArithmeticOptions cfg;
    cfg.domain = {unsigned(logN), unsigned(slots)};
    cfg.qCount = number(argv[5], 2, 32);
    cfg.polynomial.degrees = {3, 5, 7}; cfg.polynomial.composed = false;
    pass3::Options o;
    o.minScale = number(argv[6], 24, 52); o.maxScale = number(argv[7], o.minScale, 52);
    o.alpha = number(argv[8], 1, 2);
    const std::string mode(argv[9]);
    if (mode != "narrow" && mode != "wide") throw std::invalid_argument("invalid N2 scale mode");
    o.heterogeneous = mode == "wide"; o.heirScaleAnchor = false; o.maxK = 0;
    o.securityCap = number(argv[10], 32, (1 << logN)/16);
    cfg.activationTolerance = o.tolerance;
    NativeArithmeticProgram p;
    pass3::Solution s;
    size_t attempts = 0;
    if (mlir::failed(specializeNativeArithmetic((*source)->getBuffer(), argv[1], (*spec)->getBuffer(),
            cfg, o, 32, p, s, attempts, llvm::errs()))) return 1;
    Object result{{"schema", "snabf.n2.native-proposal.v1"}, {"source", *p.source.inspection.getString("source_sha256")},
      {"input_spec", *p.source.inspection.getString("input_spec_sha256")}, {"graph_sha256", graphID(p)},
      {"ring", p.graph.n}, {"encoded_slots", slots}, {"q_count", cfg.qCount}, {"alpha", o.alpha},
      {"scale_mode", mode}, {"heir_scale_anchor", o.heirScaleAnchor}, {"max_k", o.maxK}, {"scale", s.scale},
      {"min_scale", o.minScale}, {"max_scale", o.maxScale}, {"security_cap", int(o.securityCap)},
      {"wide_prime_bits", Array{o.primeWidthMin,o.primeWidthMax}}, {"margin_bits", o.auxMarginBits},
      {"q", nums(s.q)}, {"p", nums(s.p)}, {"k", nums(s.k)}, {"edges", int64_t(p.graph.edges.size())},
      {"attempts", int64_t(attempts)}, {"approximation", p.graph.approximation},
      {"arithmetic", s.errors.at(p.graph.sink)}, {"tolerance", o.tolerance}};
    llvm::outs() << llvm::formatv("{0}\n", llvm::json::Value(std::move(result)));
    return 0;
  } catch (const std::exception &e) {
    llvm::errs() << e.what() << "\n";
    return 1;
  }
}

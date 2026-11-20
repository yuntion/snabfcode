#include "GadgetNoiseContract.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace snabf::pass3 {
const char *name(Op op) {
  const char *names[] = {"source", "drop",        "add",       "mul",
                         "mask",   "coefficient", "add_plain", "rescale",
                         "baby",   "giant",       "relin", "ring_down"};
  return names[static_cast<int>(op)];
}
bool keyOp(Op o) { return o == Op::Baby || o == Op::Giant || o == Op::Relin; }
int activeDigitBlocks(int level, int k, int alpha) {
  if (k < 0 || k > level || alpha < 1)
    throw std::invalid_argument("invalid_fixed_gadget_partition");
  return 1 + (level / alpha - k / alpha);
}
long double specialModulus(const std::vector<int64_t> &p) {
  if (p.empty() || p.size() > 4) throw std::invalid_argument("unsupported_special_basis");
  long double product = 1;
  for (int64_t v : p) product *= v;
  return product;
}
long double switchKeyError(const std::vector<int64_t> &q, const std::vector<int64_t> &p,
                           int level, int k, int n) {
  const int alpha = int(p.size());
  const long double P = specialModulus(p);
  long double error = 0;
  // Even a partially zero block is a CRT digit in the ORIGINAL product basis.
  for (int b = k / alpha * alpha; b <= level; b += alpha) {
    long double digit = 1;
    for (int j = b; j < std::min(b + alpha, level + 1); ++j) digit *= q.at(j);
    error += 10.L * n * n * (digit + 1.L) / P;
  }
  return error;
}
// Hoeffding on the real and imaginary projections of an iid centered
// ternary key, union over N embeddings. No independence of gadget sites.
double secretBound(int n) {
  return std::min<double>(n,
                          2 * std::sqrt(n * std::log(std::ldexp(4. * n, 40))));
}
double rounding(int n) { return .5 * n * (1 + secretBound(n)); }
double residualFactor(Op op, int n) {
  double s = secretBound(n);
  return .5 * n * (op == Op::Relin ? s * s : s);
}
int scaleClass(const Graph &g, int i) {
  auto &e = g.edges.at(i);
  switch (e.op) {
  case Op::Source:
    return 1;
  case Op::Mask:
  case Op::Coefficient:
    return 1 + scaleClass(g, e.inputs[0]);
  case Op::Multiply:
    return scaleClass(g, e.inputs[0]) + scaleClass(g, e.inputs[1]);
  case Op::Rescale:
    return scaleClass(g, e.inputs[0]) - 1;
  default:
    return scaleClass(g, e.inputs[0]);
  }
}
void validate(const Graph &g, const Options &o) {
  if (g.n < 4 || (g.n & (g.n - 1)) || g.levels < 1 || g.sink < 0 ||
      g.sink >= int(g.edges.size()) ||
      !(o.minBase >= 20 && o.minBase <= 60 && o.minScale >= 24 &&
        o.minScale <= o.maxScale && o.maxScale <= 52 && o.maxK >= 0 &&
        o.alpha >= -1 && o.alpha <= 4 && (!o.heirScaleAnchor || o.heterogeneous) &&
        o.primeWidthMin >= 20. && o.primeWidthMin < o.primeWidthMax && o.primeWidthMax <= 59. &&
        o.auxMarginBits >= 2 && o.auxMarginBits <= 59 &&
        std::isfinite(o.securityCap) && o.securityCap > 0 &&
        std::isfinite(o.guardBits) && o.guardBits >= 1 &&
        std::isfinite(o.tolerance) && o.tolerance > g.approximation &&
        g.approximation >= 0))
    throw std::invalid_argument("invalid graph or Pass 3 budget");
  for (int i = 0; i < int(g.edges.size()); ++i) {
    auto &v = g.edges[i];
    if (v.level < 0 || v.level >= g.levels ||
        v.inputs.size() != v.gains.size() || !std::isfinite(v.magnitude) ||
        v.magnitude < 0)
      throw std::invalid_argument("invalid Pass 3 edge");
    for (size_t j = 0; j < v.inputs.size(); ++j)
      if (v.inputs[j] < 0 || v.inputs[j] >= i || !std::isfinite(v.gains[j]) ||
          v.gains[j] < 0)
        throw std::invalid_argument("invalid predecessor/gain");
  }
}
} // namespace snabf::pass3

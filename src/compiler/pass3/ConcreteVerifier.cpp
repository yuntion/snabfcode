#include "ConcreteVerifier.h"
#include "PrimeInstantiator.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
namespace snabf::pass3 {
namespace {
// Outward composition for the new nonnegative split. Preserve exact zero
// and identity gains; a positive sub-ulp contribution must not disappear.
double splitSum(double a, double b) {
  if (a == 0 || b == 0) return a + b;
  return std::nextafter(a + b, INFINITY);
}
double splitProduct(double a, double b) {
  if (a == 0 || b == 0) return 0;
  if (a == 1 || b == 1) return a * b;
  return std::nextafter(a * b, INFINITY);
}
std::vector<double> polyScales(const Poly &p, const Solution &s) {
  int n = p.nodes.size();
  std::vector<std::vector<double>> rows;
  auto row = [&](int i) -> auto & {
    auto &r = rows.emplace_back(n + 1, 0);
    r[i] = 1;
    return r;
  };
  row(0);
  row(p.output);
  for (int i = 1; i < n; ++i) {
    auto &v = p.nodes[i];
    if (v.op == "mul") {
      auto &r = row(i);
      for (int a : v.args)
        r[a]--;
      r[n] = s.scale - std::log2(double(s.q.at(p.level - v.depth + 1)));
    } else if (v.op == "add")
      for (int a : v.args)
        row(i)[a]--;
  }
  std::vector<int> piv;
  int r = 0;
  for (int c = 0; c < n && r < int(rows.size()); ++c) {
    int b = r;
    for (int j = r + 1; j < int(rows.size()); ++j)
      if (std::abs(rows[j][c]) > std::abs(rows[b][c]))
        b = j;
    if (std::abs(rows[b][c]) < 1e-12)
      continue;
    std::swap(rows[b], rows[r]);
    double x = rows[r][c];
    for (double &v : rows[r])
      v /= x;
    for (int j = 0; j < int(rows.size()); ++j)
      if (j != r) {
        x = rows[j][c];
        for (int k = c; k <= n; ++k)
          rows[j][k] -= x * rows[r][k];
      }
    piv.push_back(c);
    ++r;
  }
  for (; r < int(rows.size()); ++r)
    if (std::abs(rows[r][n]) > 1e-9)
      throw std::runtime_error(
          "incompatible physical scales in polynomial DAG");
  std::vector<double> result(n, s.scale);
  for (int i = 0; i < int(piv.size()); ++i)
    result[piv[i]] += rows[i][n];
  return result;
}
} // namespace
std::vector<double> replayScales(const Graph &g, const Solution &s) {
  std::vector<std::vector<double>> polys;
  for (auto &p : g.polys)
    polys.push_back(polyScales(p, s));
  std::vector<double> out;
  for (auto &e : g.edges) {
    double a = e.inputs.empty() ? s.scale : out.at(e.inputs[0]);
    switch (e.op) {
    case Op::Multiply:
      a += out.at(e.inputs[1]);
      break;
    case Op::Mask:
      a += std::log2(double(s.q.at(e.maskLevel)));
      break;
    case Op::Coefficient:
      a = e.poly < 0 ? a + s.scale
                     : polys.at(e.poly).at(e.node) +
                           std::log2(double(s.q.at(e.level)));
      break;
    case Op::Rescale:
      a -= std::log2(double(s.q.at(e.level + 1)));
      break;
    case Op::Add:
      for (int i : e.inputs)
        if (std::abs(a - out.at(i)) > 1e-8)
          throw std::runtime_error("unaligned physical add: " + e.id);
      break;
    default:
      break;
    }
    out.push_back(a);
  }
  return out;
}
Violation replay(const Graph &g, const Options &o, Solution &s) {
  validate(g, o);
  if (int(s.q.size()) != g.levels || !o.admits(int(s.p.size())) ||
      s.k.size() != g.edges.size() || s.scale < o.minScale ||
      s.scale > o.maxScale)
    return {"inconsistent parameter shape"};
  std::set<int64_t> seen;
  long double total = 0;
  std::vector<long double> prefix;
  for (auto v : s.q) {
    total += std::log2((long double)v);
    prefix.push_back(total);
  }
  long double auxWidth = 0;
  for (auto p : s.p) auxWidth += std::log2((long double)p);
  total += auxWidth;
  for (auto v : {s.q, s.p})
    for (int64_t q : v)
      if (q < 3 || q >= (1LL << 60) || q % (2LL * g.n) != 1 ||
          !seen.insert(q).second || !prime(q))
        return {"invalid concrete NTT basis"};
  if (total > o.securityCap)
    return {"QP exceeds security envelope", -2, double(total - o.securityCap)};
  if (o.heirScaleAnchor) {
    using boost::multiprecision::cpp_int;
    const cpp_int lower = cpp_int(1) << (2*s.scale-1), upper = lower << 2;
    for (size_t l = 1; l < s.q.size(); ++l) {
      cpp_int square = cpp_int(s.q[l]) * s.q[l];
      if (square < lower || square >= upper || std::llround(std::log2(static_cast<double>(s.q[l]))) != s.scale)
        return {"rescale prime outside HEIR integer scale anchor"};
    }
  }
  if (o.blockSearch()) {
    // Compare the requested margin on the concrete integer basis.
    using boost::multiprecision::cpp_int;
    cpp_int aux = 1;
    for (auto p : s.p) aux *= p;
    for (size_t b = 0; b < s.q.size(); b += s.p.size()) {
      cpp_int need = cpp_int(1) << o.auxMarginBits;
      long double bits = o.auxMarginBits;
      for (size_t j = b; j < std::min(b + s.p.size(), s.q.size()); ++j) {
        need *= s.q[j]; bits += std::log2((long double)s.q[j]);
      }
      if (aux < need) return {"concrete special-basis block margin", -3, double(bits - auxWidth), int(b)};
    }
  }
  s.physical = replayScales(g, s);
  s.errors.clear();
  std::vector<double> fresh(g.edges.size(), 0.);
  double P = double(specialModulus(s.p));
  for (size_t i = 0; i < g.edges.size(); ++i) {
    auto &e = g.edges[i];
    int N = e.ring ? e.ring : g.n;
    if (N<4 || N>g.n || (N&(N-1)) || g.n%N) return {"invalid edge ring"};
    double S=secretBound(N), round=rounding(N);
    int k = s.k[i];
    if (k < 0 || k > std::min(o.maxK, e.level) || (k && !keyOp(e.op)))
      return {"illegal truncation at " + e.id};
    double scale = std::exp2(s.physical[i]);
    double err = 0;
    const bool split = e.linearEntry >= 0;
    auto charge = [&](double x) { err = split ? splitSum(err,x) : err + x; };
    for (size_t j = 0; j < e.inputs.size(); ++j) {
      int a = e.inputs[j];
      if (split && a == e.linearEntry) continue;
      charge(split ? splitProduct(fresh.at(a),e.gains[j]) : s.errors.at(a)*e.gains[j]);
    }
    if (e.op == Op::Multiply)
      err += s.errors[e.inputs[0]] * s.errors[e.inputs[1]];
    if (e.op == Op::Source)
      err += (round + N + (20 * N * (N + 1 + S)) / P) / scale;
    if (e.op == Op::Mask || e.op == Op::Coefficient) {
      int a = e.inputs[0];
      double amp = split ? splitSum(g.edges[a].magnitude, s.errors[a])
                         : g.edges[a].magnitude + s.errors[a];
      charge(amp * N * std::exp2(s.physical[a]) / scale);
    }
    if (e.op == Op::AddPlain)
      err += N / scale;
    if (e.op == Op::Rescale)
      charge(round / scale);
    if (keyOp(e.op) || e.op==Op::Down) {
      long double key = switchKeyError(s.q, s.p, e.level, k, N), D = 1;
      if (!o.blockSearch()) {
        // v1 engineering receipts rounded the one-P denominator to binary64.
        key=0;
        for (int j=k;j<=e.level;++j) key+=10.L*N*N*(s.q[j]+1.L)/P;
      }
      for (int j = 0; j < k; ++j) D *= s.q[j];
      charge(double(key + round + (k ? residualFactor(e.op, N) * (D - 1) : 0)) / scale);
    }
    if (split) {
      // Encoding above used the TOTAL predecessor error, including entrance
      // error (the plaintext-error cross term). Never subtract two bounds.
      fresh[i] = err;
      err = splitSum(err, splitProduct(e.entryGain, s.errors.at(e.linearEntry)));
    }
    if (!std::isfinite(err) || err > 8 * o.tolerance)
      return {"arithmetic allowance exceeded at " + e.id};
    s.errors.push_back(err);
    double need = s.physical[i] + std::log2(std::max(1., e.magnitude + err)) +
                  o.guardBits;
    if (need > prefix[e.level])
      return {"concrete message/scale headroom at " + e.id, e.level,
              need - double(prefix[e.level])};
  }
  if (s.errors[g.sink] + g.approximation > o.tolerance)
    return {"sink approximation + arithmetic bound exceeds target"};
  return {};
}
} // namespace snabf::pass3

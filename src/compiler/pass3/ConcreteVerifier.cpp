#include "ConcreteVerifier.h"
#include "PrimeInstantiator.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
namespace snabf::pass3 {
namespace {
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
  if (int(s.q.size()) != g.levels || s.p.size() != 1 ||
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
  total += std::log2((long double)s.p[0]);
  for (auto v : {s.q, s.p})
    for (int64_t q : v)
      if (q < 3 || q >= (1LL << 60) || q % (2LL * g.n) != 1 ||
          !seen.insert(q).second || !prime(q))
        return {"invalid concrete NTT basis"};
  if (total > o.securityCap)
    return {"QP exceeds security envelope", -2, double(total - o.securityCap)};
  s.physical = replayScales(g, s);
  s.errors.clear();
  double P = s.p[0];
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
    for (size_t j = 0; j < e.inputs.size(); ++j)
      err += s.errors.at(e.inputs[j]) * e.gains[j];
    if (e.op == Op::Multiply)
      err += s.errors[e.inputs[0]] * s.errors[e.inputs[1]];
    if (e.op == Op::Source)
      err += (round + N + (20 * N * (N + 1 + S)) / P) / scale;
    if (e.op == Op::Mask || e.op == Op::Coefficient) {
      int a = e.inputs[0];
      err += (g.edges[a].magnitude + s.errors[a]) * N *
             std::exp2(s.physical[a]) / scale;
    }
    if (e.op == Op::AddPlain)
      err += N / scale;
    if (e.op == Op::Rescale)
      err += round / scale;
    if (keyOp(e.op) || e.op==Op::Down) {
      long double key = 0, D = 1;
      for (int j = 0; j <= e.level; ++j) {
        if (j < k)
          D *= s.q[j];
        else
          key += .5L * N * N * 20 * (s.q[j] + 1.L) / P;
      }
      err +=
          double(key + round + (k ? residualFactor(e.op, N) * (D - 1) : 0)) /
          scale;
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

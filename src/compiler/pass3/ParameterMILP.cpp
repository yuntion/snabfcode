#include "ParameterMILP.h"
#include "ConcreteVerifier.h"
#include "Highs.h"
#include "PrimeInstantiator.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
namespace snabf::pass3 {
namespace {
using Row = std::map<int, double>;
void ok(HighsStatus s) {
  if (s == HighsStatus::kError)
    throw std::runtime_error("invalid Pass 3 HiGHS model");
}
struct Model {
  Highs h;
  Model() {
    ok(h.setOptionValue("output_flag", false));
    ok(h.setOptionValue("threads", 1));
    ok(h.setOptionValue("mip_rel_gap", 1e-8));
  }
  int var(double lo, double hi, double cost = 0, bool binary = false) {
    int c = h.getNumCol();
    ok(h.addVar(lo, hi));
    ok(h.changeColCost(c, cost));
    if (binary)
      ok(h.changeColIntegrality(c, HighsVarType::kInteger));
    return c;
  }
  void row(double lo, Row r, double hi) {
    std::vector<HighsInt> c;
    std::vector<double> a;
    for (auto [j, v] : r)
      if (v) {
        c.push_back(j);
        a.push_back(v);
      }
    ok(h.addRow(lo, hi, c.size(), c.data(), a.data()));
  }
  // Epigraph of the convex piecewise secant interpolant, scaled into [0,1].
  // All uses of t are monotone upper-budget constraints. SOS2 binaries are
  // unnecessary.
  int upperExp(int x, double lo, double hi, double slope, double origin) {
    auto f = [&](double u) { return std::exp2(slope * (u - origin)); };
    int t = var(std::min(f(lo), f(hi)), std::max(f(lo), f(hi)));
    int num = std::max(1, int(std::ceil((hi - lo) * 2)));
    for (int j = 0; j < num; ++j) {
      double a = lo + (hi - lo) * j / num, b = lo + (hi - lo) * (j + 1) / num,
             m = (f(b) - f(a)) / (b - a);
      row(f(a) - m * a, {{t, 1}, {x, -m}}, kHighsInf);
    }
    return t;
  }
  int product(int b, int x, double a, double u) {
    int y = var(0, u);
    row(0, {{y, 1}, {b, -a}}, kHighsInf);
    row(-kHighsInf, {{y, 1}, {b, -u}}, 0);
    row(-u, {{y, 1}, {x, -1}, {b, -u}}, kHighsInf);
    row(-kHighsInf, {{y, 1}, {x, -1}, {b, -a}}, -a);
    return y;
  }
};
std::optional<Solution> fixed(const Graph &g, const Options &o, int d) {
  Model m;
  const double inf = kHighsInf;
  double b0 = o.minBase;
  for (int i = 0; i < int(g.edges.size()); ++i)
    if (g.edges[i].level == 0)
      b0 = std::max(b0,
                    std::floor(scaleClass(g, i) * d + o.guardBits +
                               std::log2(std::max(1., g.edges[i].magnitude))) -
                        .25);
  const double plo = d + 2, phi = std::min(59.9, double(d + 19));
  if (plo >= phi || b0 + 2 >= 60)
    return {};
  std::vector<int> q, sc, noise;
  std::vector<double> lb, ub;
  Row total;
  for (int l = 0; l < g.levels; l++) {
    lb.push_back(l ? d - .025 : b0);
    ub.push_back(l ? d + .025 : b0 + 2);
    q.push_back(m.var(lb[l], ub[l], .10));
    total[q.back()] = 1;
  }
  int p = m.var(plo, phi, .10);
  total[p] = 1;
  m.row(-inf, total, o.securityCap);
  int pinv = m.upperExp(p, plo, phi, -1, plo);
  std::vector<std::vector<int>> z(g.edges.size());
  int n = g.n;
  double S = secretBound(n);
  const double eta = o.tolerance;
  for (int i = 0; i < int(g.edges.size()); ++i) {
    auto &e = g.edges[i];
    int cls = scaleClass(g, i);
    double nominal = cls * d, slack = .20;
    int s = m.var(nominal - slack, nominal + slack);
    sc.push_back(s);
    int en = m.var(0, i == g.sink ? 1 - g.approximation / eta : 8);
    noise.push_back(en);
    Row eq{{s, 1}}, er{{en, 1}};
    double fixed = 0, inv = std::exp2(-nominal + slack) / eta;
    for (size_t j = 0; j < e.inputs.size(); ++j) {
      er[noise[e.inputs[j]]] -=
          e.gains[j] + (e.op == Op::Multiply ? 4 * eta : 0);
    }
    auto parent = [&]() { eq[sc[e.inputs[0]]]--; };
    switch (e.op) {
    case Op::Source:
      fixed = d;
      break;
    case Op::Multiply:
      for (int a : e.inputs)
        eq[sc[a]]--;
      break;
    case Op::Mask:
      parent();
      eq[q[e.maskLevel]]--;
      break;
    case Op::Coefficient: {
      parent();
      if (e.poly < 0)
        fixed = d;
      else {
        int ps = m.var(d - .45, d + .45);
        eq[ps]--;
      }
      break;
    }
    case Op::Rescale:
      parent();
      eq[q[e.level + 1]]++;
      break;
    case Op::Add:
      for (size_t j = 1; j < e.inputs.size(); ++j)
        m.row(0, {{s, 1}, {sc[e.inputs[j]], -1}}, 0);
      [[fallthrough]];
    default:
      parent();
      break;
    }
    m.row(fixed, eq, fixed);
    Row cap{{s, -1}};
    for (int j = 0; j <= e.level; ++j)
      cap[q[j]] = 1;
    m.row(o.guardBits + std::log2(std::max(1., e.magnitude + 8 * eta)), cap,
          inf);
    double base = 0;
    if (e.op == Op::Source) {
      base = (rounding(n) + n) * inv;
      er[pinv] -= 20. * n * (n + 1 + S) * std::exp2(-plo) * inv;
    }
    if (e.op == Op::Rescale)
      base = rounding(n) * inv;
    if (e.op == Op::AddPlain)
      base = n * inv;
    if (e.op == Op::Mask || e.op == Op::Coefficient)
      base = (g.edges[e.inputs[0]].magnitude + 8 * eta) * n *
             std::exp2(-d + .48) / eta;
    if (keyOp(e.op)) {
      base = rounding(n) * inv;
      double key = 0;
      for (int j = 0; j <= e.level; ++j)
        key += 10. * n * n * (std::exp2(ub[j]) + 1) * std::exp2(-plo) * inv;
      er[pinv] -= key;
      Row one;
      int kmax = std::min(o.maxK, e.level);
      double lo = 0, hi = 0;
      for (int k = 0; k <= kmax; ++k) {
        if (k) {
          lo += lb[k - 1];
          hi += ub[k - 1];
          if (residualFactor(e.op, n) * std::exp2(lo - nominal - slack) >
              8 * eta)
            break;
        }
        int choose = m.var(0, 1, (e.level + 1 - k) * (e.level + 2), true);
        z[i].push_back(choose);
        one[choose] = 1;
        if (k) {
          int x = m.var(lo, hi);
          Row prefix{{x, 1}};
          for (int j = 0; j < k; ++j)
            prefix[q[j]]--;
          m.row(0, prefix, 0);
          double a = std::exp2(lo - hi);
          int amp = m.upperExp(x, lo, hi, 1, hi),
              sq = m.upperExp(x, lo, hi, 2, hi);
          int y = m.product(choose, amp, a, 1),
              y2 = m.product(choose, sq, a * a, 1);
          double factor = residualFactor(e.op, n) * std::exp2(hi) * inv;
          er[y] -= factor;
          // Squared range is also a Hoeffding proxy for the centered residual;
          // constrain its canonical capacity independently of signed
          // cancellation.
          m.row(-inf, {{y2, factor * factor}}, 64);
        }
      }
      m.row(1, one, 1);
      for (int j = 0; j + 1 < int(z[i].size()); ++j) {
        int b = m.var(0, 1, 0, true);
        Row flag{{b, 1}};
        for (int k = j + 1; k < int(z[i].size()); ++k)
          flag[z[i][k]] = -1;
        m.row(0, flag, 0);
      }
    }
    m.row(base, er, inf);
  }
  std::vector<std::string> cuts;
  for (int round = 0; round < 12; ++round) {
    ok(m.h.run());
    auto status = m.h.getModelStatus();
    if (status == HighsModelStatus::kInfeasible)
      return {};
    if (status != HighsModelStatus::kOptimal)
      throw std::runtime_error(
          "HiGHS did not prove an optimal Pass 3 solution");
    auto v = m.h.getSolution().col_value;
    Solution s;
    s.scale = d;
    s.objective = m.h.getObjectiveValue();
    s.pBits = v[p];
    std::set<int64_t> used;
    for (auto col : q) {
      s.bits.push_back(v[col]);
      s.q.push_back(nearestPrime(n, v[col], used));
    }
    s.p = {nearestPrime(n, s.pBits, used)};
    for (int c : sc)
      s.scales.push_back(v[c]);
    for (auto &choices : z) {
      int k = 0;
      for (int j = 0; j < int(choices.size()); ++j)
        if (v[choices[j]] > .5)
          k = j;
      s.k.push_back(k);
    }
    auto bad = replay(g, o, s);
    if (bad.reason.empty()) {
      s.cuts = cuts;
      return s;
    }
    std::ostringstream why;
    why << bad.reason << "; bit deficit=" << std::setprecision(12) << bad.deficit;
    cuts.push_back(why.str());
    double step = std::max(.002, bad.deficit * 1.25);
    if (bad.level >= 0) {
      Row r;
      double sum = 0;
      for (int j = 0; j <= bad.level; ++j) {
        r[q[j]] = 1;
        sum += s.bits[j];
      }
      m.row(sum + step, r, inf);
    } else if (bad.level == -2) {
      double sum = s.pBits;
      for (double x : s.bits)
        sum += x;
      m.row(-inf, total, sum - step);
    } else
      return {};
  }
  return {};
}
} // namespace
Solution optimizeAtBoundary(const Graph &g,const Options &o,Solution s){
  validate(g,o);Model m;auto scales=replayScales(g,s);std::vector<int> err;std::vector<std::vector<int>> flags;
  double eta=o.tolerance;
  for(int i=0;i<int(g.edges.size());++i){const auto &e=g.edges[i];int n=e.ring?e.ring:g.n;
    double scale=std::exp2(scales[i]),p=s.p.at(0),round=rounding(n),S=secretBound(n);
    int y=m.var(0,i==g.sink?1.-g.approximation/eta:8.);Row row{{y,1}};double fixed=0;flags.emplace_back();
    for(size_t j=0;j<e.inputs.size();++j)row[err.at(e.inputs[j])]-=e.gains[j];
    if(e.op==Op::Multiply)row[err.at(e.inputs[0])]-=8*eta;
    if(e.op==Op::Source)fixed=(round+n+20.*n*(n+1+S)/p)/scale;
    if(e.op==Op::Mask||e.op==Op::Coefficient){auto a=e.inputs.at(0);double w=n*std::exp2(scales[a])/scale;fixed=g.edges[a].magnitude*w;row[err[a]]-=w;}
    if(e.op==Op::AddPlain)fixed=n/scale;
    if(e.op==Op::Rescale)fixed=round/scale;
    if(keyOp(e.op)||e.op==Op::Down){Row one;long double D=1;int maxK=e.op==Op::Down?0:std::min(o.maxK,e.level);
      for(int k=0;k<=maxK;++k){long double key=0;for(int j=k;j<=e.level;++j)key+=10.L*n*n*(s.q.at(j)+1.L)/p;
        long double inc=(key+round+(k?residualFactor(e.op,n)*(D-1):0))/(scale*eta);
        if(inc<=8){int z=m.var(0,1,double(n)/g.n*(e.level+1-k)*(e.level+2),true);flags.back().push_back(z);one[z]=1;row[z]-=double(inc);}else flags.back().push_back(-1);
        D*=s.q.at(k);
      }m.row(1,one,1);
    }
    m.row(fixed/eta,row,kHighsInf);err.push_back(y);
  }
  ok(m.h.run());if(m.h.getModelStatus()!=HighsModelStatus::kOptimal)throw std::runtime_error("boundary_milp_infeasible");
  auto vals=m.h.getSolution().col_value;s.k.assign(g.edges.size(),0);
  for(size_t i=0;i<flags.size();++i)for(size_t k=0;k<flags[i].size();++k)if(flags[i][k]>=0 && vals[flags[i][k]]>.5)s.k[i]=k;
  if(auto v=replay(g,o,s);!v.reason.empty())throw std::runtime_error("boundary_concrete_replay: "+v.reason);s.objective=m.h.getObjectiveValue();return s;
}
Solution optimize(const Graph &g, const Options &o) {
  validate(g, o);
  std::optional<Solution> best;
  for (int d = o.minScale; d <= o.maxScale; ++d) {
    auto s = fixed(g, o, d);
    if (s && (!best || s->objective < best->objective - 1e-8))
      best = std::move(s);
  }
  if (!best)
    throw std::runtime_error(
        "Pass 3 infeasible under requested scale, precision and QP bounds");
  return *best;
}
} // namespace snabf::pass3

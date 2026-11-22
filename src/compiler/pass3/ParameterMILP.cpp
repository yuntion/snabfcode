#include "ParameterMILP.h"
#include "ConcreteVerifier.h"
#include "CostProfile.h"
#include "Highs.h"
#include "PrimeInstantiator.h"
#include "RejectionIsolation.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
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
    if (hi == lo) return var(f(lo), f(lo));
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
// One-hot k names zero original limbs; u_b stays active until the entire
// original key block is zero. Zero weight retains the structural/noise rows
// without counting a second (proxy) objective in calibrated mode.
std::vector<int> priceBlocks(Model &m, int level, int alpha,
                             const std::vector<int> &z, double weight) {
  std::vector<int> active;
  for (int j = 0; j <= level; ++j) {
    int kept = m.var(0, 1, weight, true);
    Row relation{{kept, 1}};
    for (int k=0; k<int(z.size()) && k<=j; ++k)
      if (z[k] >= 0) relation[z[k]] = -1;
    m.row(0, relation, 0);
    if (j == level || (j+1)%alpha == 0) {
      int u = m.var(0, 1, weight * (level + 1 + alpha), true);
      m.row(0, {{u,1},{kept,-1}}, 0);
      active.push_back(u);
    }
  }
  return active;
}
void applyPrices(Model &m, const std::vector<PriceGroup> &groups,
                 const std::vector<std::vector<int>> &z) {
  for (const auto &group : groups) {
    if (group.edges.size()==1) {
      const auto &cols=z.at(group.edges.front());
      for(int k=0;k<int(cols.size());++k) if(cols[k]>=0) {
        auto it=group.prices->find({k});
        if(it==group.prices->end()) ok(m.h.changeColBounds(cols[k],0,0));
        else ok(m.h.changeColCost(cols[k],it->second));
      }
      continue;
    }
    // Whole hoisted call: one paid selector per measured vector, linked to
    // each member's original one-hot (which still drives the noise rows).
    Row one; std::map<int,Row> link;
    for(int edge:group.edges) for(int col:z.at(edge)) if(col>=0) link[col]={{col,1}};
    for (const auto &[ks,us]:*group.prices) {
      std::vector<int> cols;
      for (size_t j=0;j<ks.size();++j) {
        const auto &v=z[group.edges[j]];
        if(ks[j]>=int(v.size()) || v[ks[j]]<0) break;
        cols.push_back(v[ks[j]]);
      }
      if(cols.size()!=group.edges.size()) continue;
      int choose=m.var(0,1,us,true);one[choose]=1;
      for(int col:cols) link.at(col)[choose]=-1;
    }
    m.row(1,one,1);
    for(auto &[col,row]:link) m.row(0,row,0);
  }
}

void verifyPricedObjective(const Graph &g,const Options &o,const Solution &s) {
  if (!o.costProfile) return;
  double us=0; for(const auto &charge:replayPrices(g,*o.costProfile,s)) us+=charge.us;
  if (!std::isfinite(s.objective) || std::abs(us-s.objective)>1e-7+1e-9*us)
    throw std::runtime_error("gadget_objective_replay_mismatch");
}

// An attempted scale/basis can fail before HiGHS, inside HiGHS, or after
// concrete replay. None of these events proves why another attempt failed.
struct CandidateTrace {
  std::ostream *out;
  const Graph &g;
  const Options &o;
  int scale, alpha;
  const MeasuredBasis *basis;

  void record(const char *stage, const char *status, int round = 0,
              const Solution *physical = nullptr, int blockFirst = -1,
              const diagnostic::Tree *isolation = nullptr) const {
    if (!out) return;
    using boost::property_tree::ptree;
    ptree row;
    row.put("schema", "snabf.pass3.candidate-rejection.v1");
    row.put("stage", stage);
    row.put("status", status);
    row.put("reason", "unknown");
    row.put("n", g.n);
    row.put("q_count", g.levels);
    row.put("alpha", alpha);
    row.put("scale_bits", scale);
    row.put("max_k", o.maxK);
    row.put("tolerance", o.tolerance);
    row.put("security_cap_bits", o.securityCap);
    row.put("round", round); // 0: model construction; 1..12: solve/refine
    if (isolation) row.add_child("isolation", *isolation);
    if (physical || basis) {
      auto limbs = [](const std::vector<int64_t> &v) {
        ptree a;
        for (auto x : v) a.push_back({"", ptree(std::to_string(x))});
        return a;
      };
      row.put("basis_source", physical ? "concrete_replay" : "profile_input");
      row.add_child("q", limbs(physical ? physical->q : basis->q));
      row.add_child("p", limbs(physical ? physical->p : basis->p));
    }
    if (physical && blockFirst >= 0) {
      // Recheck the verifier's original (possibly short final) Q block using
      // integers. A continuous width, deficit or solver status is not proof.
      using boost::multiprecision::cpp_int;
      int end = std::min(blockFirst + alpha, int(physical->q.size()));
      cpp_int actual = 1, required = cpp_int(1) << o.auxMarginBits;
      for (auto p : physical->p) actual *= p;
      for (int j = blockFirst; j < end; ++j) required *= physical->q[j];
      if (actual < required) {
        row.put("reason", "aux_block_margin");
        row.put("witness.block_first", blockFirst);
        row.put("witness.block_end_exclusive", end);
        row.put("witness.margin_bits", o.auxMarginBits);
        row.put("witness.p_product", actual.str());
        row.put("witness.required_product", required.str());
      }
    }
    std::ostringstream line;
    boost::property_tree::write_json(line, row, false);
    *out << line.str();
  }

  std::nullopt_t reject(const char *stage, const char *status, int round = 0) const {
    record(stage, status, round);
    return std::nullopt;
  }
};

// Bits of nominal message headroom, before there are solver variables, an
// instantiated basis, or a decoded arithmetic error. Keep the actual operands
// of the guard; the diagnostic must not reconstruct a different expression.
constexpr double baseWidthOffset = -.25;
struct BaseWidthTerm {
  int edge, cls;
  double nominal, logMagnitude, beforeFloor, floored;
  double lower() const { return floored + baseWidthOffset; }
};
BaseWidthTerm baseWidthTerm(const Graph &g, int i, int d, double guard) {
  int cls = scaleClass(g, i);
  double nominal = cls * d;
  double logMagnitude = std::log2(std::max(1., g.edges[i].magnitude));
  double beforeFloor = nominal + guard + logMagnitude;
  return {i, cls, nominal, logMagnitude, beforeFloor, std::floor(beforeFloor)};
}

diagnostic::Tree baseExpression(const Graph &g, int d, double guard,
                                const std::vector<BaseWidthTerm> &terms,
                                const IRMagnitudeOrigins *origins) {
  diagnostic::Tree t, rows, values;
  t.put("schema", "snabf.pass3.base-expression.v1");
  if (origins) t.put("ir_interval_mode", origins->intervalMode);
  for (const auto &x : terms) {
    diagnostic::Tree row;
    row.put("edge", x.edge);
    row.put("scale_class", x.cls);
    row.put("nominal_scale_bits", x.nominal);
    row.put("guard_bits", guard);
    row.put("log2_clamped_decoded_abs_bits", x.logMagnitude);
    row.put("before_floor_bits", x.beforeFloor);
    row.put("floor_delta_bits", x.floored - x.beforeFloor);
    row.put("offset_bits", baseWidthOffset);
    row.put("lower_bits", x.lower());
    const auto &e = g.edges[x.edge];
    if (e.op == Op::Rescale) {
      row.put("nominal_numerator_bits", scaleClass(g, e.inputs.at(0)) * d);
      row.put("nominal_drop_bits", -d); // -1 class, NOT an observed log2(q)
      row.put("dropped_q_index", e.level + 1);
    }
    rows.push_back({"", std::move(row)});
  }
  if (!rows.empty()) t.add_child("level_zero_terms", std::move(rows));
  // Indexed by the original isolation.edges DAG; rescale preserves DECODED
  // ideal magnitude. The divisor is accounted for in its scale class above.
  for (size_t i = 0; i < g.edges.size(); ++i) {
    const auto &e = g.edges[i];
    diagnostic::Tree v;
    v.put("decoded_abs", e.magnitude);
    if (origins) {
      if (e.op == Op::Mask || e.op == Op::Coefficient)
        v.put("abs_weight", e.gains.at(0));
      if (auto it = origins->polynomialLiterals.find(i); it != origins->polynomialLiterals.end())
        v.put("signed_literal", it->second);
    }
    values.push_back({"", std::move(v)});
  }
  t.add_child("magnitude_transfers", std::move(values));
  return t;
}

std::optional<Solution> fixed(const Graph &g, const Options &o, int d, int alpha,
                              std::ostream *out, const IRMagnitudeOrigins *origins,
                              const MeasuredBasis *basis,
                              const std::vector<SourceCutBound> &bounds, SourceCutStats *stats) {
  if (stats) ++stats->candidates;
  const CandidateTrace trace{out, g, o, d, alpha, basis};
  Model m;
  const double inf = kHighsInf;
  double b0 = o.minBase;
  std::vector<BaseWidthTerm> terms;
  for (int i = 0; i < int(g.edges.size()); ++i) {
    if (g.edges[i].level != 0) continue;
    auto term = baseWidthTerm(g, i, d, o.guardBits);
    b0 = std::max(b0, term.lower());
    if (out) terms.push_back(term);
  }
  const bool block = o.blockSearch();
  const double bitWeight = basis ? 0. : .10;
  std::set<int64_t> occupied;
  // Constrain the continuous hint to the measured integer's nearest-prime
  // rounding cell. Two equal requested P widths may instantiate in the
  // opposite numeric order; pinning their exact logarithms would lose that
  // valid candidate. All concrete constraints are still replayed on Q/P.
  auto pin = [&](int col,int64_t z,double lo,double hi) {
    for(int direction:{-1,1}) {
      const int64_t step=direction*2LL*g.n;
      for(int64_t v=z+step;v>2 && v<(1LL<<60);v+=step) {
        if(occupied.contains(v) || !prime(v))continue;
        double midpoint=static_cast<double>(std::log2((static_cast<long double>(v)+z)/2));
        if(direction<0) lo=std::max(lo,midpoint); else hi=std::min(hi,midpoint);
        break;
      }
    }
    if(lo>hi) return false;
    occupied.insert(z);m.row(lo,{{col,1}},hi);return true;
  };
  if (b0 >= (block ? o.primeWidthMax : 60.)) {
    if (out) {
      auto input = diagnostic::baseWidth(g, o, b0);
      auto expr = baseExpression(g, d, o.guardBits, terms, origins);
      if (o.coSlotMagnitude) expr.put("ir_magnitude_mode", "co-slot-v1");
      input.add_child("base_expression", std::move(expr));
      trace.record("model_bounds", "base_width", 0, nullptr, -1, &input);
    }
    return {};
  }
  std::vector<int> q, sc, noise;
  std::vector<double> lb, ub;
  Row total;
  for (int l = 0; l < g.levels; l++) {
    lb.push_back(l ? (o.heterogeneous ? o.primeWidthMin : d - .025) : b0);
    ub.push_back(l ? (o.heterogeneous ? o.primeWidthMax : d + .025) : std::min(block ? o.primeWidthMax : 59.99, b0 + 2));
    if (l && block) {
      lb.back() = std::max(lb.back(), o.primeWidthMin);
      ub.back() = std::min(ub.back(), o.primeWidthMax);
    }
    if (l && o.heirScaleAnchor) {
      // The physical prime is instantiated later: leave room on both sides
      // of the half-integer rounding boundary and replay the actual prime.
      lb.back() = std::max(lb.back(), d - .49);
      ub.back() = std::min(ub.back(), d + .49);
    }
    if (lb[l] > ub[l]) return trace.reject("model_bounds", "empty_q_interval");
    q.push_back(m.var(lb[l], ub[l], bitWeight));
    if(basis && !pin(q.back(),basis->q.at(l),lb[l],ub[l])) return trace.reject("model_bounds", "q_rounding_cell");
    total[q.back()] = 1;
  }
  const double plo = block ? std::max(o.primeWidthMin * alpha, double(d+o.auxMarginBits)) : double(d+2);
  const double phi = block ? o.primeWidthMax * alpha : std::min(59.9, double(d+19));
  if (plo >= phi) return trace.reject("model_bounds", "nonpositive_p_width_span");
  std::vector<int> aux;
  int p = m.var(plo, phi, block ? 0 : bitWeight);
  if (block) {
    Row sum{{p,1}};
    for (int j=0; j<alpha; ++j) {
      int v=m.var(o.primeWidthMin,o.primeWidthMax,bitWeight);
      if(basis && !pin(v,basis->p.at(j),o.primeWidthMin,o.primeWidthMax)) return trace.reject("model_bounds", "p_rounding_cell");
      aux.push_back(v); total[v]=1; sum[v]=-1;
      if (j) m.row(0, {{aux[j-1],1},{v,-1}}, inf);
    }
    m.row(0,sum,0);
    for (int j=0;j<g.levels;j+=alpha) {
      Row margin{{p,1}};
      for (int t=j;t<std::min(j+alpha,g.levels);++t) margin[q[t]]=-1;
      m.row(o.auxMarginBits,margin,inf);
    }
  } else { aux.push_back(p); total[p]=1; }
  const int qpCapRow = m.h.getNumRow();
  m.row(-inf, total, o.securityCap);
  int pinv = m.upperExp(p, plo, phi, -1, plo);
  std::vector<std::vector<int>> z(g.edges.size());
  int n = g.n;
  double S = secretBound(n);
  const double eta = o.tolerance;
  for (int i = 0; i < int(g.edges.size()); ++i) {
    auto &e = g.edges[i];
    int cls = scaleClass(g, i);
    double nominal = cls * d, slack = o.heterogeneous ? 2. : .20;
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
        int ps = m.var(d - (o.heterogeneous ? 4. : .45), d + (o.heterogeneous ? 4. : .45));
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
    if (o.heterogeneous && e.op==Op::Mask) {
      int l=e.maskLevel;
      int invQ=m.upperExp(q[l],lb[l],ub[l],-1,lb[l]);
      er[invQ]-=(g.edges[e.inputs[0]].magnitude + 8*eta)*n*std::exp2(-lb[l])/eta;
    } else if (e.op == Op::Mask || e.op == Op::Coefficient)
      base = (g.edges[e.inputs[0]].magnitude + 8 * eta) * n *
             std::exp2(o.heterogeneous ? -d + 4. : -d + .48) / eta;
    if (keyOp(e.op)) {
      base = rounding(n) * inv;
      double key = 0;
      if (!block) {
        for (int j = 0; j <= e.level; ++j)
          key += 10. * n * n * (std::exp2(ub[j]) + 1) * std::exp2(-plo) * inv;
        er[pinv] -= key;
      }
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
        int choose = m.var(0, 1, block ? 0 : activeDigitBlocks(e.level,k,1)*(e.level+2), true);
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
      if (block) {
        auto u = priceBlocks(m,e.level,alpha,z[i],basis ? 0. : 1.);
        for (int first=0;first<=e.level;first+=alpha) {
          Row eq{{p,-1}};
          double lowQ=0, highQ=0;
          for (int j=first;j<std::min(first+alpha,e.level+1);++j) {
            eq[q[j]]=1; lowQ+=lb[j]; highQ+=ub[j];
          }
          double low=std::max(double(o.auxMarginBits),plo-highQ), high=phi-lowQ;
          if (low>high) return trace.reject("model_bounds", "empty_block_gap");
          int gap=m.var(low,high); eq[gap]=1; m.row(0,eq,0);
          int ratio=m.upperExp(gap,low,high,-1,low);
          int term=m.product(u.at(first/alpha),ratio,0.,1.);
          er[term]-=10.*n*n*inv*std::exp2(-low)*(1+std::exp2(-lowQ));
        }
      } else for (int j = 0; j + 1 < int(z[i].size()); ++j) {
        int b = m.var(0, 1, 0, true);
        Row flag{{b, 1}};
        for (int k = j + 1; k < int(z[i].size()); ++k)
          flag[z[i][k]] = -1;
        m.row(0, flag, 0);
      }
    }
    m.row(base, er, inf);
  }
  if(basis) applyPrices(m,priceGroups(g,*basis),z);
  for (const auto &bound : bounds) {
    Row row;
    for (auto [edge, weight] : bound.weights) row[noise.at(edge)] = weight;
    m.row(-inf, row, bound.budget / eta);
  }
  if (stats) ++stats->modelsWithCuts;
  std::vector<std::string> cuts;
  for (int round = 0; round < 12; ++round) {
    if (stats) ++stats->solverRuns;
    ok(m.h.run());
    auto status = m.h.getModelStatus();
    if (status != HighsModelStatus::kOptimal) {
      if (out) {
        auto input = diagnostic::solverRows(m.h, qpCapRow);
        const char *s = status == HighsModelStatus::kInfeasible ? "infeasible" :
                        status == HighsModelStatus::kUnknown ? "unknown" : "not_optimal";
        trace.record("solver", s, round + 1, nullptr, -1, &input);
      }
      if (status == HighsModelStatus::kInfeasible) {
        if (stats) ++stats->solverInfeasible;
        return {};
      }
      throw std::runtime_error(
          "HiGHS did not prove an optimal Pass 3 solution");
    }
    auto v = m.h.getSolution().col_value;
    Solution s;
    s.scale = d;
    s.objective = m.h.getObjectiveValue();
    s.pBits = v[p];
    std::set<int64_t> used;
    for (auto col : q) {
      s.bits.push_back(v[col]);
      s.q.push_back(basis ? basis->q.at(s.q.size()) : nearestPrime(n, v[col], used));
    }
    for (int col : aux) {
      s.auxBits.push_back(v[col]);
      s.p.push_back(basis ? basis->p.at(s.p.size()) : nearestPrime(n,v[col],used));
    }
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
      for (const auto &bound : bounds) {
        double demand = 0;
        for (auto [edge, weight] : bound.weights) demand += weight * s.errors.at(edge);
        if (!(demand <= bound.budget)) {
          if (stats) ++stats->concreteCutMisses;
          return {}; // includes concrete prime rounding
        }
      }
      verifyPricedObjective(g,o,s);
      s.cuts = cuts;
      return s;
    }
    trace.record("concrete_replay", "rejected", round + 1, &s,
                 bad.level == -3 ? bad.blockFirst : -1);
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
    } else if (bad.level == -3) {
      Row r{{p,1}};
      double current=s.pBits;
      for(int j=bad.blockFirst;j<std::min(bad.blockFirst+alpha,g.levels);++j) {
        r[q[j]]=-1; current-=s.bits[j];
      }
      m.row(current+step,r,inf);
    } else
      return {};
  }
  return trace.reject("refinement", "round_limit", 12);
}
} // namespace
double baseWidthLower(const Graph &g, int edge, int d, double guard) {
  if (g.edges.at(edge).level != 0) throw std::invalid_argument("base width needs a level-zero edge");
  return baseWidthTerm(g, edge, d, guard).lower();
}
Solution optimizeAtBoundary(const Graph &g,const Options &o,Solution s, BoundaryRows *capture){
  if (capture) *capture = {};
  validate(g,o);Model m;std::vector<int> err;std::vector<std::vector<int>> flags;
  std::vector<PriceGroup> prices;
  if(o.costProfile) prices=priceGroups(g,measuredBasis(*o.costProfile,g.n,s));
  auto scales=replayScales(g,s);
  double eta=o.tolerance;
  for(int i=0;i<int(g.edges.size());++i){const auto &e=g.edges[i];int n=e.ring?e.ring:g.n;
    double scale=std::exp2(scales[i]),p=double(specialModulus(s.p)),round=rounding(n),S=secretBound(n);
    int y=m.var(0,i==g.sink?1.-g.approximation/eta:8.);Row row{{y,1}};double fixed=0;flags.emplace_back();
    for(size_t j=0;j<e.inputs.size();++j)row[err.at(e.inputs[j])]-=e.gains[j];
    if(e.op==Op::Multiply)row[err.at(e.inputs[0])]-=8*eta;
    if(e.op==Op::Source)fixed=(round+n+20.*n*(n+1+S)/p)/scale;
    if(e.op==Op::Mask||e.op==Op::Coefficient){auto a=e.inputs.at(0);double w=n*std::exp2(scales[a])/scale;fixed=g.edges[a].magnitude*w;row[err[a]]-=w;}
    if(e.op==Op::AddPlain)fixed=n/scale;
    if(e.op==Op::Rescale)fixed=round/scale;
    if(keyOp(e.op)||e.op==Op::Down){Row one;long double D=1;int maxK=e.op==Op::Down?0:std::min(o.maxK,e.level);
      for(int k=0;k<=maxK;++k){long double key=switchKeyError(s.q,s.p,e.level,k,n);
        long double inc=(key+round+(k?residualFactor(e.op,n)*(D-1):0))/(scale*eta);
        if(inc<=8){int z=m.var(0,1,o.blockSearch()?0:double(n)/g.n*activeDigitBlocks(e.level,k,int(s.p.size()))*(e.level+2),true);flags.back().push_back(z);one[z]=1;row[z]-=double(inc);}else flags.back().push_back(-1);
        D*=s.q.at(k);
      }m.row(1,one,1);
      if (o.blockSearch()) priceBlocks(m,e.level,int(s.p.size()),flags.back(),o.costProfile ? 0. : double(n)/g.n);
    }
    if (capture) {
      // Read the submitted row, not ideal gains or values of slack solver columns.
      BoundaryRow r{fixed/eta, {}, {}};
      for (int a : e.inputs) r.parents[a] = -row.at(err.at(a));
      for (int k = 0; k < int(flags.back().size()); ++k)
        if (int z = flags.back()[k]; z >= 0) r.choices[k] = -row.at(z);
      capture->rows.push_back(std::move(r));
    }
    m.row(fixed/eta,row,kHighsInf);err.push_back(y);
  }
  if(o.costProfile) applyPrices(m,prices,flags);
  ok(m.h.run());
  if (m.h.getModelStatus() != HighsModelStatus::kOptimal) {
    if (capture) {
      capture->provenInfeasible = m.h.getModelStatus() == HighsModelStatus::kInfeasible;
      if (!capture->provenInfeasible) throw std::runtime_error("boundary_solver_not_optimal");
    }
    throw std::runtime_error("boundary_milp_infeasible");
  }
  auto vals=m.h.getSolution().col_value;s.k.assign(g.edges.size(),0);
  for(size_t i=0;i<flags.size();++i)for(size_t k=0;k<flags[i].size();++k)if(flags[i][k]>=0 && vals[flags[i][k]]>.5)s.k[i]=k;
  if(auto v=replay(g,o,s);!v.reason.empty())throw std::runtime_error("boundary_concrete_replay: "+v.reason);
  s.objective=m.h.getObjectiveValue();verifyPricedObjective(g,o,s);return s;
}
BoundaryProjection projectBoundaryCut(const Graph &g, const Options &o, const BoundaryRows &model,
                                      const std::vector<std::pair<int, int>> &cuts) {
  if (!model.provenInfeasible || model.rows.size() != g.edges.size() || cuts.empty())
    throw std::invalid_argument("missing_boundary_rows_or_cut");
  BoundaryProjection out;
  auto stop = [&](const char *why, int i = -1) {
    out.status = why; out.blockedEdge = i; return out;
  };
  // A measured hoist can couple K choices across sites. Do not claim that
  // independently minimizing its member rows realizes a permitted assignment.
  if (o.costProfile) return stop("coupled_measured_choices_unsupported");
  std::vector<double> local;
  std::map<int, int> downToCut;
  for (size_t j = 0; j < cuts.size(); ++j) {
    auto [producer, down] = cuts[j];
    const auto &e = g.edges.at(down);
    if (e.op != Op::Down || e.inputs.size() != 1 || !downToCut.emplace(down, j).second)
      throw std::invalid_argument("invalid_down_binding");
    int a = e.inputs[0];
    // The only supported implicit alignment is a cost-free, unit-gain Drop.
    if (a != producer && !(g.edges[a].op == Op::Drop && g.edges[a].inputs == std::vector<int>{producer} &&
                           model.rows[a].parents.at(producer) == 1 && model.rows[a].fixed == 0))
      throw std::invalid_argument("invalid_down_input_binding");
  }
  for (int i = 0; i < int(g.edges.size()); ++i) {
    const auto &e = g.edges[i];
    const auto &r = model.rows[i];
    double v = r.fixed;
    if (keyOp(e.op) || e.op == Op::Down) {
      if (r.choices.empty()) return stop("no_admissible_boundary_k", i);
      v += std::min_element(r.choices.begin(), r.choices.end(),
                            [](auto a, auto b) { return a.second < b.second; })->second;
    }
    local.push_back(v * o.tolerance);
    double floor = local.back();
    for (auto [a, gain] : r.parents) floor += gain * out.floors.at(a);
    if (!std::isfinite(floor)) return stop("nonfinite_boundary_projection", i);
    out.floors.push_back(floor);
    if (i != g.sink && floor > 8 * o.tolerance && out.internalCapEdge < 0)
      out.internalCapEdge = i;
  }
  // Reverse substitution stops at each Down *input*, not at the first visit
  // to its producer (one producer may also feed the other crossing upstream).
  std::vector<double> adj(g.edges.size()); adj[g.sink] = 1;
  out.gains.resize(cuts.size());
  for (int i = g.sink; i >= 0; --i) {
    if (!adj[i]) continue;
    if (g.edges[i].op == Op::Source) return stop("incomplete_boundary_cut", i);
    out.suffix += adj[i] * local[i];
    if (auto it = downToCut.find(i); it != downToCut.end()) {
      out.gains[it->second] = adj[i] * model.rows[i].parents.at(g.edges[i].inputs[0]);
    } else {
      if (g.edges[i].op == Op::Down) return stop("incomplete_boundary_cut", i);
      for (auto [a, gain] : model.rows[i].parents) adj[a] += adj[i] * gain;
    }
  }
  for (size_t j = 0; j < cuts.size(); ++j)
    out.demand += out.gains[j] * out.floors[cuts[j].first];
  out.jointBudget = o.tolerance - g.approximation - out.suffix;
  out.deficit = out.demand - out.jointBudget;
  if (!std::isfinite(out.demand) || !std::isfinite(out.deficit)) return stop("nonfinite_boundary_projection");
  out.status = "projected";
  return out;
}
namespace {
std::optional<Solution> search(const Graph &g, const Options &o, std::ostream *out,
                               const IRMagnitudeOrigins *origins,
                               const std::vector<SourceCutBound> &bounds = {}, SourceCutStats *stats = nullptr) {
  validate(g, o);
  std::optional<Solution> best;
  if(o.costProfile) {
    // Never extrapolate the timing of one concrete key to another basis.
    // The original MILP is solved on each covered basis, with all physical
    // constraints and replay intact. This is a finite measured search space.
    bool covered=false;std::exception_ptr missing;
    for(const auto &b:o.costProfile->bases()) {
      if(b.n!=g.n || int(b.q.size())!=g.levels || !o.admits(b.p.size())) continue;
      try { (void)priceGroups(g,b); } catch(const MissingCost &) {missing=std::current_exception();continue;}
      covered=true;
      for(int d=o.minScale;d<=o.maxScale;++d) {
        auto s=fixed(g,o,d,b.p.size(),out,origins,&b,bounds,stats);
        if(s && (!best || s->objective<best->objective-1e-8)) best=std::move(s);
      }
    }
    if(!covered) {
      if(missing)std::rethrow_exception(missing);
      throw MissingCost("no compatible N, Q count or alpha");
    }
  } else
  for (int alpha=1; alpha<=4; ++alpha) if(o.admits(alpha))
  for (int d = o.minScale; d <= o.maxScale; ++d) {
    auto s = fixed(g, o, d, alpha, out, origins,nullptr,bounds,stats);
    if (s && (!best || s->objective < best->objective - 1e-8))
      best = std::move(s);
  }
  return best;
}
Solution requireSource(std::optional<Solution> best) {
  if (!best)
    throw std::runtime_error(
        "Pass 3 infeasible under requested scale, precision and QP bounds");
  return std::move(*best);
}
} // namespace

Solution optimize(const Graph &g, const Options &o, std::ostream &rejections) {
  return requireSource(search(g, o, &rejections, nullptr));
}
Solution optimize(const Graph &g, const Options &o, std::ostream &rejections,
                  const IRMagnitudeOrigins &origins) {
  return requireSource(search(g, o, &rejections, &origins));
}
std::ostream *rejectionTraceStream() {
  const char *flag = std::getenv("SNABF_PASS3_TRACE_REJECTIONS");
  return flag && std::string_view(flag) == "1" ? &std::cerr : nullptr;
}
Solution optimize(const Graph &g, const Options &o) {
  return requireSource(search(g, o, rejectionTraceStream(), nullptr));
}
std::optional<Solution> optimizeForCuts(const Graph &g, const Options &o,
                                      const std::vector<SourceCutBound> &bounds, SourceCutStats &stats) {
  stats = {};
  if (bounds.empty()) throw std::invalid_argument("missing_source_cut_bound");
  for (const auto &b : bounds) {
    if (b.weights.empty() || !std::isfinite(b.budget) || b.budget < 0)
      throw std::invalid_argument("invalid_source_cut_bound");
    for (auto [i, w] : b.weights)
      if (i < 0 || i >= int(g.edges.size()) || !std::isfinite(w) || w <= 0)
        throw std::invalid_argument("invalid_source_cut_term");
  }
  return search(g, o, nullptr, nullptr, bounds, &stats);
}
} // namespace snabf::pass3

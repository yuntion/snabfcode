#include "PolyTemplates.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace snabf::pass2 {

const std::vector<PolyTemplateSpec>& templatesFor(std::string_view activation) {
  using F = StageFunction;
  // tanh(2z)/2 = tanh(z)/(1+tanh(z)^2); Phi(x)-1/2 = erf(x/sqrt(2))/2.
  static const std::vector<PolyTemplateSpec> silu{
      {"silu_direct", 0, {F::SiluDirect}},
      {"silu_tanh_rational", 0, {F::TanhQuarter, F::Rational}}};
  static const std::vector<PolyTemplateSpec> gelu{
      {"gelu_direct", 0, {F::GeluDirect}}};
  static const std::vector<PolyTemplateSpec> empty;
  if (activation == "silu") return silu;
  if (activation == "gelu") return gelu;
  return empty;
}

const StageSpec& stageSpec(StageFunction stage) {
  using F = StageFunction;
  constexpr double inf = std::numeric_limits<double>::infinity();
  static constexpr StageSpec silu{F::SiluDirect,"half_tanh_half","tanh(x/2)/2",{-inf,inf}};
  static constexpr StageSpec quarter{F::TanhQuarter,"tanh_quarter","tanh(x/4)",{-inf,inf}};
  static constexpr StageSpec rational{F::Rational,"u_over_one_plus_u2","u/(1+u*u)",{-1,1}};
  static constexpr StageSpec gelu{F::GeluDirect,"half_erf_sqrt2","erf(x/sqrt(2))/2",{-inf,inf}};
  switch (stage) {
    case F::SiluDirect: return silu;
    case F::TanhQuarter: return quarter;
    case F::Rational: return rational;
    case F::GeluDirect: return gelu;
  }
  throw std::invalid_argument("unknown Pass 2 reference stage");
}

std::string_view stageName(StageFunction stage) { return stageSpec(stage).id; }

namespace {
using Real = long double;
using Vec = std::vector<Real>;
using F = StageFunction;

struct ValueSlope { Real value, slope; };

ValueSlope reference(F stage, Real x) {
  switch (stage) {
    case F::SiluDirect:
    case F::TanhQuarter: {
      const Real a = stage == F::SiluDirect ? 0.5L : 1.L;
      const Real b = stage == F::SiluDirect ? 0.5L : 0.25L;
      const Real z = std::exp(-2 * b * std::abs(x));
      return {a * std::tanh(b * x), 4 * a * b * z / ((1 + z) * (1 + z))};
    }
    case F::Rational: {
      const Real q = 1 / (1 + x * x);
      return {x * q, (2 * q - 1) * q};
    }
    case F::GeluDirect: {
      static const Real norm = 1 / std::sqrt(2 * std::acos(-1.L));
      return {std::erf(x / std::sqrt(2.L)) / 2, norm * std::exp(-x*x/2)};
    }
  }
  throw std::invalid_argument("unknown Pass 2 reference stage");
}

// On a small domain solve the SAME minimax problem after removing an element
// of its approximation space. With t=x/B, the new target is
// (g(Bt)-Taylor_d(Bt))/(a_m B^(d+2)), evaluated without subtraction.
// This keeps the ripple observable even when g(Bt) is linear to binary64.
class Target {
 public:
  Target(F stage, unsigned degree, Real radius)
      : stage_(stage), d_(degree), B_(radius), m_((degree + 1) / 2) {
    if (B_ > 2) return;
    taylor_.push_back(reference(stage, 0).slope);
    Real power = 1;
    for (unsigned k = 1; k <= m_ + 96; ++k) {
      Real a = 0;
      switch (stage) {
        case F::Rational: a = -taylor_.back(); break;
        case F::GeluDirect:
          a = -taylor_.back() * (2*k-1) / (2*k*(2*k+1));
          break;
        case F::SiluDirect:
        case F::TanhQuarter:
          for (unsigned j = 0; j < k; ++j) a += taylor_[j] * taylor_[k-1-j];
          a *= -(stage == F::SiluDirect ? 1.L : 0.25L) / (2*k+1);
          break;
      }
      taylor_.push_back(a);
      if (k < m_) continue;
      if (stage == F::Rational) break;  // Closed-form geometric remainder.
      const Real term = (a / taylor_[m_]) * power;
      tail_.push_back(term);
      if (std::abs(term) < std::numeric_limits<Real>::epsilon() / 100) break;
      power *= B_ * B_;
    }
  }

  ValueSlope operator()(Real t) const {
    if (taylor_.empty()) {
      auto v = reference(stage_, B_ * t);
      v.slope *= B_;
      return v;
    }
    const Real s = t*t;
    Real f = 0, df = 0;
    if (stage_ == F::Rational) {
      f = 1 / (1 + B_*B_*s);
      df = -B_*B_*f*f;
    } else {
      for (auto i = tail_.rbegin(); i != tail_.rend(); ++i) {
        df = df*s + f;
        f = f*s + *i;
      }
    }
    const Real power = std::pow(t, static_cast<int>(d_ + 1));
    return {power*t*f, power*((d_+2)*f + 2*s*df)};
  }

  Real originalCoefficient(unsigned k, Real correction) const {
    if (taylor_.empty()) return correction / std::pow(B_, static_cast<int>(k));
    return taylor_[k/2] + correction * taylor_[m_] *
        std::pow(B_, static_cast<int>(d_+2-k));
  }

 private:
  F stage_;
  unsigned d_;
  Real B_;
  unsigned m_;
  Vec taylor_, tail_;
};

// Evaluate sum c_i T_(2i+1)(t) and its derivative without conversion to
// monomials. The linear solve uses the same Chebyshev recurrence.
ValueSlope cheb(const Vec& c, Real t) {
  Real u = 1, v = t, du = 0, dv = 1, y = 0, dy = 0;
  for (unsigned k = 1; k < 2*c.size(); ++k) {
    if (k % 2) { y += c[k/2]*v; dy += c[k/2]*dv; }
    const Real next = 2*t*v-u, dnext = 2*v+2*t*dv-du;
    u = v; v = next; du = dv; dv = dnext;
  }
  return {y, dy};
}

Vec solveAlternant(const Target& target, const Vec& nodes) {
  const unsigned n = static_cast<unsigned>(nodes.size());
  std::vector<Vec> rows(n, Vec(n+1));
  for (unsigned i = 0; i < n; ++i) {
    const Real t = nodes[i];
    Real u = 1, v = t;
    for (unsigned k = 1; k < 2*(n-1); ++k) {
      if (k % 2) rows[i][k/2] = v;
      const Real w = 2*t*v-u;
      u = v; v = w;
    }
    rows[i][n-1] = i % 2 ? -1 : 1;
    rows[i][n] = target(t).value;
  }
  for (unsigned j = 0; j < n; ++j) {
    unsigned pivot = j;
    for (unsigned i = j+1; i < n; ++i)
      if (std::abs(rows[i][j]) > std::abs(rows[pivot][j])) pivot = i;
    if (rows[pivot][j] == 0) throw std::runtime_error("singular Remez alternant");
    std::swap(rows[j], rows[pivot]);
    for (unsigned i = j+1; i < n; ++i) {
      const Real factor = rows[i][j]/rows[j][j];
      for (unsigned k = j+1; k <= n; ++k) rows[i][k] -= factor*rows[j][k];
    }
  }
  Vec x(n);
  for (unsigned i = n; i-- > 0;) {
    Real rhs = rows[i][n];
    for (unsigned j = i+1; j < n; ++j) rhs -= rows[i][j]*x[j];
    x[i] = rhs/rows[i][i];
  }
  return x;
}

template <class Function>
Real bisect(Function f, Real a, Real b, unsigned steps) {
  const Real fa = f(a);
  if (fa == 0) return a;
  for (unsigned i = 0; i < steps; ++i) {
    const Real c = std::midpoint(a, b), fc = f(c);
    if (fc == 0 || c == a || c == b) return c;
    if (std::signbit(fc) == std::signbit(fa)) a = c; else b = c;
  }
  return std::midpoint(a,b);
}

Vec remez(const Target& target, unsigned d, Real left) {
  const unsigned m = (d+1)/2;
  Vec nodes(m+1), zeros(m+2);
  for (unsigned i = 0; i <= m; ++i)
    nodes[i] = left + (1-left)*std::cos((m-i)*std::acos(-1.L)/(d+2));
  zeros.front() = left; zeros.back() = 1;
  for (unsigned iter = 0; iter < 32; ++iter) {
    Vec c = solveAlternant(target, nodes);
    const Real ripple = c.back(); c.pop_back();
    auto error = [&](Real t) {
      const auto g = target(t), p = cheb(c,t);
      return ValueSlope{g.value-p.value, g.slope-p.slope};
    };
    auto val = [&](Real t) { return error(t).value; };
    auto slope = [&](Real t) { return error(t).slope; };
    // One signed maximum per alternating lobe. The grid only brackets slope
    // roots for this exchange; none of these numbers is a certificate.
    for (unsigned i = 0; i < m; ++i)
      zeros[i+1] = bisect(val, nodes[i], nodes[i+1], 20);
    Vec next(nodes);
    Real peak = 0;
    for (unsigned i = 0; i <= m; ++i) {
      const Real sign = (std::signbit(ripple) != bool(i%2)) ? -1 : 1;
      Real best = sign*val(nodes[i]);
      auto consider = [&](Real t) {
        const Real e = sign*val(t);
        if (e > best) { best = e; next[i] = t; }
      };
      Real a = zeros[i], da = slope(a);
      consider(a); consider(zeros[i+1]);
      for (unsigned j = 1; j <= 12; ++j) {
        const Real b = j == 12 ? zeros[i+1] : a + (zeros[i+1]-a)/(13-j);
        const Real db = slope(b);
        if (da == 0 || db == 0 || std::signbit(da) != std::signbit(db))
          consider(db == 0 ? b : bisect(slope, a, b, 38));
        a = b; da = db;
      }
      peak = std::max(peak, best);
    }
    Real scale = std::abs(target(1).value);
    for (Real v : c) scale += std::abs(v);
    const Real roundoff = 128*std::numeric_limits<Real>::epsilon()*scale;
    if (peak <= std::abs(ripple)*(1+1e-9L) + roundoff) return c;
    nodes = std::move(next);
  }
  throw std::runtime_error("odd Chebyshev Remez exchange did not converge");
}

Vec chebToPowers(const Vec& c, unsigned degree) {
  Vec a(degree+1), prev(degree+1), curr(degree+1), next(degree+1);
  prev[0] = 1; curr[1] = 1;
  for (unsigned k = 1; k <= degree; ++k) {
    if (k % 2)
      for (unsigned j = 1; j <= k; j += 2) a[j] += c[k/2]*curr[j];
    if (k == degree) break;
    for (unsigned j = 0; j <= k+1; ++j)
      next[j] = (j ? 2*curr[j-1] : 0) - prev[j];
    prev.swap(curr); curr.swap(next);
  }
  return a;
}
}  // namespace

StagePolynomial generateStage(F stage, unsigned degree, Interval1D Iin) {
  const auto& degrees = stageSpec(stage).degrees;
  if (std::find(degrees.begin(), degrees.end(), degree) == degrees.end())
    throw std::invalid_argument("unsupported Remez degree");
  if (!std::isfinite(Iin.lo) || !std::isfinite(Iin.hi) || Iin.lo > Iin.hi)
    throw std::invalid_argument("invalid Remez interval");
  StagePolynomial out{degree, std::vector<double>(degree+1, 0)};
  const Real B = std::max(std::abs(Iin.lo), std::abs(Iin.hi));
  if (Iin.lo == Iin.hi) {
    const auto g = reference(stage, B);
    out.coefficients[1] = static_cast<double>(B == 0 ? g.slope : g.value/B);
    return out;  // On a singleton, interpolation attains the minimax error 0.
  }
  // Odd symmetry folds the interval onto its absolute image; translating x
  // to the midpoint would introduce even powers for asymmetric inputs.
  const Real left = std::max({0., Iin.lo, -Iin.hi}) / B;
  const Target target(stage, degree, B);
  const Vec powers = chebToPowers(remez(target, degree, left), degree);
  for (unsigned k = 1; k <= degree; k += 2) {
    out.coefficients[k] = static_cast<double>(target.originalCoefficient(k, powers[k]));
    if (!std::isfinite(out.coefficients[k]))
      throw std::overflow_error("Remez coefficient does not fit in binary64");
  }
  return out;
}

}  // namespace snabf::pass2

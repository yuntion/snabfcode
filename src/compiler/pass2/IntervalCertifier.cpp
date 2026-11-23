#include "IntervalCertifier.h"

#include <mpfi.h>
#include <mpfr.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace snabf::pass2 {
namespace {

constexpr mpfr_prec_t kPrecision = 160;
constexpr unsigned kOrder = 5;
static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits == 53);

struct Float {
  mpfr_t v;
  Float() { mpfr_init2(v, kPrecision); }
  ~Float() { mpfr_clear(v); }
  Float(const Float&) = delete;
  Float& operator=(const Float&) = delete;
};

struct Box {
  mpfi_t v;
  Box() { mpfi_init2(v, kPrecision); mpfi_set_si(v, 0); }
  explicit Box(double x) : Box() { mpfi_set_d(v, x); }
  Box(double a, double b) : Box() { mpfi_interv_d(v, a, b); }
  Box(const Box& other) : Box() { mpfi_set(v, other.v); }
  Box(Box&& other) noexcept : Box() { mpfi_swap(v, other.v); }
  Box& operator=(const Box& other) { mpfi_set(v, other.v); return *this; }
  Box& operator=(Box&& other) noexcept { mpfi_swap(v, other.v); return *this; }
  ~Box() { mpfi_clear(v); }

  Interval1D bounds() const {
    if (mpfi_nan_p(v) || mpfi_is_empty(v))
      throw std::overflow_error("invalid MPFI enclosure");
    Float tmp;
    mpfi_get_left(tmp.v, v);
    const double lo = mpfr_get_d(tmp.v, MPFR_RNDD);
    mpfi_get_right(tmp.v, v);
    const double hi = mpfr_get_d(tmp.v, MPFR_RNDU);
    if (!std::isfinite(lo) || !std::isfinite(hi))
      throw std::overflow_error("certificate bound does not fit in binary64");
    return {lo,hi};
  }
};

Box operator+(const Box& a, const Box& b) { Box r; mpfi_add(r.v,a.v,b.v); return r; }
Box operator-(const Box& a, const Box& b) { Box r; mpfi_sub(r.v,a.v,b.v); return r; }
Box operator-(const Box& a) { Box r; mpfi_neg(r.v,a.v); return r; }
Box operator*(const Box& a, const Box& b) { Box r; mpfi_mul(r.v,a.v,b.v); return r; }
Box operator/(const Box& a, const Box& b) { Box r; mpfi_div(r.v,a.v,b.v); return r; }
Box times(const Box& a, unsigned n) { Box r; mpfi_mul_ui(r.v,a.v,n); return r; }
Box over(const Box& a, unsigned n) { Box r; mpfi_div_ui(r.v,a.v,n); return r; }
Box square(const Box& a) { Box r; mpfi_sqr(r.v,a.v); return r; }

Interval1D hull(Interval1D a, Interval1D b) {
  return {std::min(a.lo,b.lo), std::max(a.hi,b.hi)};
}
double magnitude(Interval1D a) { return std::max(std::abs(a.lo), std::abs(a.hi)); }
double lowerMagnitude(Interval1D a) { return std::max({0., a.lo, -a.hi}); }
Interval1D folded(Interval1D a) { return {lowerMagnitude(a), magnitude(a)}; }
void checkDomain(Interval1D a) {
  if (!std::isfinite(a.lo) || !std::isfinite(a.hi) || a.lo > a.hi)
    throw std::invalid_argument("invalid certificate input interval");
}

using Jet = std::array<Box, kOrder+1>;
using Powers = std::vector<Box>;
using F = StageFunction;

// A separate interval implementation of the exact mathematical reference.
// No Remez coefficients, extremal points, or transcendental libm calls enter
// this proof. Each jet entry encloses the normalized derivative g^(k)/k!.
class Reference {
 public:
  explicit Reference(F stage) : stage_(stage) {
    switch (stage) {
      case F::SiluDirect: case F::TanhQuarter: case F::Rational: break;
      case F::GeluDirect: {
        Box two(2), pi;
        mpfi_sqrt(rootTwo_.v, two.v);
        mpfi_const_pi(pi.v);
        Box twoPi = times(pi,2);
        mpfi_sqrt(twoPi.v,twoPi.v);
        mpfi_inv(normalDensity_.v,twoPi.v);
        break;
      }
      default: throw std::invalid_argument("unknown certified reference stage");
    }
  }

  Box value(const Box& x) const {
    if (stage_ == F::Rational) return x / (Box(1)+square(x));
    if (stage_ == F::GeluDirect) {
      // MPFI 1.5 has no erf. erf is increasing, so directed MPFR calls
      // on the endpoints of x/sqrt(2) give its interval extension.
      const Box z = x/rootTwo_;
      Float lo, hi;
      mpfi_get_left(lo.v,z.v); mpfi_get_right(hi.v,z.v);
      mpfr_erf(lo.v,lo.v,MPFR_RNDD); mpfr_erf(hi.v,hi.v,MPFR_RNDU);
      Box r; mpfi_interv_fr(r.v,lo.v,hi.v);
      return over(r,2);
    }
    Box z = over(x, stage_ == F::TanhQuarter ? 4 : 2);
    mpfi_tanh(z.v,z.v);
    return stage_ == F::TanhQuarter ? z : over(z,2);
  }

  Interval1D range(Interval1D x) const {
    auto r = hull(value(Box(x.lo)).bounds(), value(Box(x.hi)).bounds());
    // The tanh and erf references are monotone. The only other extrema
    // of x/(1+x*x) are at -1 and +1, with exact dyadic values.
    if (stage_ == F::Rational) {
      if (x.lo <= -1 && x.hi >= -1) r = hull(r, {-0.5,-0.5});
      if (x.lo <= 1 && x.hi >= 1) r = hull(r, {0.5,0.5});
    }
    return r;
  }

  Jet jet(const Box& x) const {
    Jet y;
    y[0] = value(x);
    if (stage_ == F::Rational) {
      // (1+x^2)y=x, with strictly positive denominator even across zero.
      const Box den = Box(1)+square(x), twoX = times(x,2);
      y[1] = (Box(1)-twoX*y[0])/den;
      for (unsigned k=2; k<=kOrder; ++k) y[k] = -(twoX*y[k-1]+y[k-2])/den;
    } else if (stage_ == F::GeluDirect) {
      // y'=phi(x), y''=-x*y'.
      Box e = -over(square(x),2);
      mpfi_exp(e.v,e.v);
      y[1] = normalDensity_*e;
      for (unsigned k=1; k<kOrder; ++k)
        y[k+1] = -over(times(x*y[k],k)+times(y[k-1],k-1), k*(k+1));
    } else {
      // y'=1/4-beta*y^2; beta is 1 for half_tanh_half and 1/4 for tanh_quarter.
      const unsigned betaDen = stage_ == F::TanhQuarter ? 4 : 1;
      y[1] = Box(0.25)-over(square(y[0]),betaDen);
      for (unsigned k=1; k<kOrder; ++k) {
        Box sum;
        for (unsigned j=0; j<=k/2; ++j)
          sum = sum + (2*j == k ? square(y[j]) : times(y[j]*y[k-j],2));
        y[k+1] = -over(sum,(k+1)*betaDen);
      }
    }
    return y;
  }

 private:
  F stage_;
  Box rootTwo_, normalDensity_;
};

unsigned choose(unsigned n, unsigned k) {
  unsigned c = 1;
  for (unsigned i=1; i<=k; ++i) c = c*(n-i+1)/i;
  return c;  // n <= 15.
}

Box evaluate(const Powers& p, const Box& x) {
  Box r;
  for (auto i=p.rbegin(); i!=p.rend(); ++i) {
    mpfi_mul(r.v,r.v,x.v);
    mpfi_add(r.v,r.v,i->v);
  }
  return r;
}

// Convex-hull property of the Bernstein basis on [0,1]. Each split is an
// outward-rounded de Casteljau split. A split limit only loosens the hull.
Interval1D bernsteinRange(const Powers& b, unsigned& splits, unsigned depth) {
  const unsigned n = static_cast<unsigned>(b.size())-1;
  const auto first = b.front().bounds(), last = b.back().bounds();
  auto outer = hull(first,last);
  for (unsigned i=1; i<n; ++i) outer = hull(outer,b[i].bounds());
  const double tol = 2e-15 + 1e-11*magnitude(outer);
  if ((std::min(first.hi,last.hi)-outer.lo <= tol &&
       outer.hi-std::max(first.lo,last.lo) <= tol) || depth == 30 || !splits)
    return outer;
  --splits;
  Powers work = b, left(n+1), right(n+1);
  left[0] = b[0]; right[n] = b[n];
  for (unsigned j=1; j<=n; ++j) {
    for (unsigned i=0; i<=n-j; ++i) {
      mpfi_add(work[i].v,work[i].v,work[i+1].v);
      mpfi_div_2ui(work[i].v,work[i].v,1);
    }
    left[j] = work[0]; right[n-j] = work[n-j];
  }
  const auto lower = bernsteinRange(left,splits,depth+1);
  return hull(lower,bernsteinRange(right,splits,depth+1));
}

Interval1D polynomialRange(const Powers& p, Interval1D domain) {
  if (p.empty()) return {0,0};
  const Box a(domain.lo);
  if (domain.lo == domain.hi) return evaluate(p,a).bounds();
  const Box width = Box(domain.hi)-a;
  const unsigned n = static_cast<unsigned>(p.size())-1;
  Powers q(n+1), b(n+1);
  q[0] = p[n];
  // Horner shift to t in [0,1]: q(t)=p(a+width*t).
  for (unsigned i=n; i>0; --i) {
    for (unsigned j=n-i+1; j>0; --j) q[j] = q[j]*a + q[j-1]*width;
    q[0] = q[0]*a + p[i-1];
  }
  for (unsigned i=0; i<=n; ++i)
    for (unsigned k=0; k<=i; ++k)
      b[i] = b[i] + over(times(q[k],choose(i,k)),choose(n,k));
  unsigned splits = 256;
  return bernsteinRange(b,splits,0);
}

using Derivatives = std::array<Powers,kOrder+1>;

double residualBound(const Reference& g, const Derivatives& p, Interval1D domain) {
  auto point = [&](double x) {
    const Box c(x);
    return (evaluate(p[0],c)-g.value(c)).bounds();
  };
  struct Cell { double a, b; Interval1D ra, rb; unsigned depth; };
  const auto left = point(domain.lo);
  if (domain.lo == domain.hi) return magnitude(left);
  const auto right = point(domain.hi);
  std::deque<Cell> work{{domain.lo,domain.hi,left,right,0}};
  double upper = 0, witness = std::max(lowerMagnitude(left),lowerMagnitude(right));
  unsigned splits = 0;
  while (!work.empty()) {
    const Cell cell = work.front(); work.pop_front();
    const double mid = std::midpoint(cell.a, cell.b);
    const Box c(mid), X(cell.a,cell.b), h = X-c;
    Jet r = g.jet(c);
    for (unsigned k=0; k<kOrder; ++k) r[k] = evaluate(p[k],c)-r[k];
    r[kOrder] = evaluate(p[kOrder],X)-g.jet(X)[kOrder];
    const auto center = r[0].bounds();
    witness = std::max(witness,lowerMagnitude(center));

    // Taylor's theorem about c for r=p-g and r'. All lower coefficients
    // are subtracted at a point BEFORE multiplying by h. The last one is
    // an interval on the entire closed cell, not a point estimate.
    Box value = r[kOrder], slope = times(r[kOrder],kOrder);
    for (unsigned k=kOrder; k>0; --k) {
      value = value*h+r[k-1];
      if (k>1) slope = slope*h+times(r[k-1],k-1);
    }
    const auto derivative = slope.bounds();
    if (derivative.lo >= 0 || derivative.hi <= 0) {
      upper = std::max({upper,magnitude(cell.ra),magnitude(cell.rb)});
      continue;
    }
    const double bound = magnitude(value.bounds());
    if (bound <= witness*(1+5e-4)+2e-15 || cell.depth == 40 || splits >= 1024 ||
        mid == cell.a || mid == cell.b) {
      // Even at a work/precision limit, retain the ENTIRE interval enclosure.
      upper = std::max(upper,bound);
    } else {
      ++splits;
      work.push_back({cell.a,mid,cell.ra,center,cell.depth+1});
      work.push_back({mid,cell.b,center,cell.rb,cell.depth+1});
    }
  }
  return upper;
}

// Each z*Q(z^2) uses the output of the preceding stage; x is the original
// bypass operand. The signed residual on a closed half-axis is exactly
// x*(g(x)+1)/2 on the left, x*(g(x)-1)/2 on the right, also at zero.
// Stationary-point cover for literal polynomials. On any subinterval a
// polynomial's extrema occur at endpoints or inside these certified boxes.
// A subdivision cap keeps the whole unresolved box, never just a midpoint.
class CriticalRange {
 public:
  CriticalRange(const StagePolynomial& p, Interval1D axis) : domain_(axis) {
    Powers slope;
    for (unsigned i=0;i<=p.degree;++i) {
      coefficients_.emplace_back(p.coefficients[i]);
      if (i) slope.push_back(times(coefficients_.back(),i));
    }
    while (!slope.empty() && mpfi_is_zero(slope.back().v)) slope.pop_back();
    unsigned budget=4096;
    Powers curvature;
    for (unsigned i=1;i<slope.size();++i) curvature.push_back(times(slope[i],i));
    if (!slope.empty()) isolate(slope,curvature,axis,0,budget);
  }
  Interval1D operator()(Interval1D axis) const {
    // Intersect two proved enclosures of the same input: stationary points
    // were isolated on domain_, not outside its outward-rounded endpoints.
    axis={std::max(axis.lo,domain_.lo),std::min(axis.hi,domain_.hi)};
    checkDomain(axis);
    auto r=hull(evaluate(coefficients_,Box(axis.lo)).bounds(),evaluate(coefficients_,Box(axis.hi)).bounds());
    for (const auto& [cell,values]: critical_)
      if (cell.lo<=axis.hi && cell.hi>=axis.lo) r=hull(r,values);
    return r;
  }
 private:
  void isolate(const Powers& slope, const Powers& curvature, Interval1D cell, unsigned depth, unsigned& budget) {
    Box x(cell.lo,cell.hi);
    if (!mpfi_has_zero(evaluate(slope,x).v)) return;
    const double mid=std::midpoint(cell.lo,cell.hi);
    const Box c(mid);
    if (!mpfi_has_zero((evaluate(slope,c)+(x-c)*evaluate(curvature,x)).v)) return;
    if (depth==44 || !budget || mid==cell.lo || mid==cell.hi) {
      critical_.push_back({cell,evaluate(coefficients_,x).bounds()}); return;
    }
    --budget;
    isolate(slope,curvature,{cell.lo,mid},depth+1,budget);
    isolate(slope,curvature,{mid,cell.hi},depth+1,budget);
  }
  Powers coefficients_;
  Interval1D domain_;
  std::vector<std::pair<Interval1D,Interval1D>> critical_;
};

double reluBranchBound(const std::vector<Powers>& oddPowers, const std::vector<CriticalRange>& maps,
                       Interval1D domain, double offset) {
  const Box shift(offset);
  auto residual = [&](Interval1D cell) {
    const Box x(cell.lo,cell.hi);
    Box z(x);
    if (maps.empty()) {
      for (const auto& q : oddPowers) z = z*evaluate(q,square(z));
    } else {
      auto a=cell;
      for (const auto& map:maps) a=map(a);
      z=Box(a.lo,a.hi);
    }
    return over(x*(z+shift),2).bounds();
  };
  struct Cell {
    Interval1D axis;
    double upper;
    bool operator<(const Cell& b) const {
      return upper == b.upper ? axis.lo > b.axis.lo : upper < b.upper;
    }
  };
  std::priority_queue<Cell> leaves;
  auto cover = [&](Interval1D cell) { leaves.push({cell,magnitude(residual(cell))}); };
  cover(domain);
  if (domain.lo == domain.hi) return leaves.top().upper;
  auto pointLower = [&](double x) { return lowerMagnitude(residual({x,x})); };
  double witness = std::max(pointLower(domain.lo),pointLower(domain.hi));
  // Leaves always cover the whole branch. A point can stop refinement but
  // never supplies the upper bound, even at the work/precision limit.
  constexpr unsigned maxSplits = 512;
  for (unsigned i=0; i<maxSplits && leaves.top().upper>witness; ++i) {
    const auto [lo,hi] = leaves.top().axis;
    const double mid = std::midpoint(lo,hi);
    if (mid == lo || mid == hi) break;
    witness = std::max(witness,pointLower(mid));
    leaves.pop();
    cover({lo,mid}); cover({mid,hi});
  }
  return leaves.top().upper;
}
}  // namespace

StageCertificate certifyStage(F stage, const StagePolynomial& p, Interval1D Iin,
                              Interval1D Ain) {
  checkDomain(Iin); checkDomain(Ain);
  if (p.degree > kCandidateDegrees.back() || p.coefficients.size() != p.degree+1)
    throw std::invalid_argument("invalid certified polynomial dimension");
  Derivatives derivatives;
  bool odd = true;
  for (unsigned i=0; i<=p.degree; ++i) {
    const double a = p.coefficients[i];
    if (!std::isfinite(a)) throw std::invalid_argument("nonfinite certified coefficient");
    odd = odd && (i%2 || a == 0);
    for (unsigned k=0; k<=std::min(kOrder,i); ++k)
      derivatives[k].push_back(times(Box(a),choose(i,k)));
  }
  const Interval1D J = hull(Iin,Ain);
  const double L = magnitude(polynomialRange(derivatives[1],odd ? folded(J) : J));
  auto Aout = polynomialRange(derivatives[0],Ain);
  if (odd && Ain.lo == -Ain.hi) Aout = {-magnitude(Aout),magnitude(Aout)};
  // The stage reference here is the exact literal polynomial, NOT sign.
  // Only the composed reconstruction certifier can issue a ReLU residual.
  if (stage==F::ReluSignPolynomial) {
    if (!odd) throw std::invalid_argument("expected an odd literal ReLU stage");
    return {Iin,Ain,J,polynomialRange(derivatives[0],Iin),Aout,0,L};
  }
  const Reference g(stage);
  const double epsilon = residualBound(g,derivatives,odd ? folded(Iin) : Iin);
  return {Iin,Ain,J,g.range(Iin),Aout,epsilon,L};
}

ReluReconstructionCertificate certifyReluReconstruction(
    std::span<const StagePolynomial> chain, Interval1D input) {
  checkDomain(input);
  if (input.lo > 0 || input.hi < 0 || chain.empty())
    throw std::invalid_argument("ReLU certification needs a nonempty chain and a domain containing zero");
  std::vector<Powers> oddPowers;
  oddPowers.reserve(chain.size());
  for (const auto& p : chain) {
    if (p.degree > kCandidateDegrees.back() || p.coefficients.size() != p.degree+1)
      throw std::invalid_argument("invalid ReLU stage dimension");
    Powers& q = oddPowers.emplace_back();
    for (unsigned i=0; i<=p.degree; ++i) {
      const double a = p.coefficients[i];
      if (!std::isfinite(a) || (i%2 == 0 && a != 0))
        throw std::invalid_argument("ReLU stage must have finite odd-polynomial coefficients");
      if (i%2) q.emplace_back(a);
    }
  }
  std::vector<CriticalRange> maps;
  if (chain.size()>2) {
    auto axis=input;
    for (auto& p:chain) {
      maps.emplace_back(p,axis);
      axis=maps.back()(axis);
    }
  }
  const double left = reluBranchBound(oddPowers,maps,{input.lo,0},1);
  const double right = reluBranchBound(oddPowers,maps,{0,input.hi},-1);
  return {input,left,right,std::max(left,right)};
}

}  // namespace snabf::pass2

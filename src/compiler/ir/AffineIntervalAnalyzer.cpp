#include "AffineIntervalAnalyzer.h"

#include <mpfi.h>
#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace snabf::ir::affine {
namespace {
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53);

void check(bool condition, const char* reason) {
  if (!condition) throw std::invalid_argument(reason);
}
void valid(Interval x) {
  check(std::isfinite(x.lo) && std::isfinite(x.hi) && x.lo <= x.hi, "invalid affine interval");
}
void checkPrecision(unsigned p) {
  check(p >= 53 && p <= 4096, "affine precision must be in [53,4096]");
}
double finite(mpfr_srcptr v, mpfr_rnd_t rounding) {
  double d = mpfr_get_d(v, rounding);
  if (!std::isfinite(d)) throw std::overflow_error("affine bound does not fit in binary64");
  return d;
}
struct Float {
  mpfr_t v;
  explicit Float(unsigned p) { mpfr_init2(v, p); }
  ~Float() { mpfr_clear(v); }
  Float(const Float&) = delete;
  Float& operator=(const Float&) = delete;
};
struct Box {
  mpfi_t v;
  explicit Box(unsigned p, double d = 0) {
    check(std::isfinite(d), "nonfinite affine constant");
    mpfi_init2(v, p); mpfi_set_d(v, d);
  }
  Box(unsigned p, Interval x) : Box(p) { valid(x); mpfi_interv_d(v, x.lo, x.hi); }
  Box(const Box& b) : Box(b.precision()) { mpfi_set(v, b.v); }
  Box(Box&& b) noexcept : Box(b.precision()) { mpfi_swap(v, b.v); }
  Box& operator=(const Box&) = delete;
  ~Box() { mpfi_clear(v); }
  unsigned precision() const { return static_cast<unsigned>(mpfi_get_prec(v)); }
  Interval bounds() const {
    if (mpfi_nan_p(v) || mpfi_is_empty(v)) throw std::logic_error("invalid MPFI affine enclosure");
    Float lo(precision()), hi(precision());
    mpfi_get_left(lo.v, v); mpfi_get_right(hi.v, v);
    return {finite(lo.v, MPFR_RNDD), finite(hi.v, MPFR_RNDU)};
  }
};
Box operator+(const Box& a, const Box& b) { Box r(a.precision()); mpfi_add(r.v,a.v,b.v); return r; }
Box operator-(const Box& a, const Box& b) { Box r(a.precision()); mpfi_sub(r.v,a.v,b.v); return r; }
Box operator*(const Box& a, const Box& b) { Box r(a.precision()); mpfi_mul(r.v,a.v,b.v); return r; }
Box half(const Box& a) { Box r(a.precision()); mpfi_div_2ui(r.v,a.v,1); return r; }
void add(Box& a, const Box& b) { mpfi_add(a.v, a.v, b.v); }
void addDouble(Box& a, double d) { mpfi_add_d(a.v, a.v, d); }
void madd(Box& a, double w, const Box& x) {
  Box t(a.precision()); mpfi_mul_d(t.v, x.v, w); add(a, t);
}
Interval intersection(Interval a, Interval b) {
  Interval both{std::max(a.lo,b.lo), std::min(a.hi,b.hi)};
  if (both.lo > both.hi) throw std::logic_error("inconsistent affine and interval proofs");
  return both;
}

// The unknown exact coefficient is somewhere in b, not necessarily at its
// computed midpoint. Charge both the MPFI operation errors and the conversion
// to a stored binary64 by measuring the distance to BOTH endpoints.
double quantize(const Box& b, Box& localError) {
  Float lo(b.precision()), hi(b.precision()), mid(b.precision());
  mpfi_mid(mid.v, b.v);
  double d = finite(mid.v, MPFR_RNDN);
  mpfi_get_left(lo.v, b.v); mpfi_get_right(hi.v, b.v);
  mpfr_d_sub(lo.v, d, lo.v, MPFR_RNDU);
  mpfr_sub_d(hi.v, hi.v, d, MPFR_RNDU);
  mpfr_max(hi.v, hi.v, lo.v, MPFR_RNDU);
  mpfi_add_fr(localError.v, localError.v, hi.v);
  return d;
}

struct Accumulator {
  const unsigned p;
  Box center, remainder, naive, tight;
  std::map<std::uint64_t, Box> coeff;

  Accumulator(unsigned p, double bias = 0)
      : p(p), center(p,bias), remainder(p), naive(p,bias), tight(p,bias) {}
  void term(const SoundValue& x, double w) {
    check(std::isfinite(w), "nonfinite affine weight");
    if (w == 0) return;
    madd(center, w, Box(p,x.form.center));
    madd(remainder, std::abs(w), Box(p,x.form.remainder));
    for (auto [id, c] : x.form.coefficients)
      madd(coeff.try_emplace(id,p).first->second, w, Box(p,c));
    madd(naive, w, Box(p,x.naive));
    madd(tight, w, Box(p,x.enclosure));
  }
  // Add the set [lo,hi] using a fresh signed symbol. Its midpoint and
  // half-width remain boxes until all arithmetic is rounded and charged.
  void independent(Interval x, std::uint64_t id) {
    valid(x);
    const Box lo(p,x.lo), hi(p,x.hi);
    add(center, half(lo+hi));
    coeff.emplace(id, half(hi-lo));
  }
  SoundValue finish(std::size_t limit, Interval baseline, Interval independentBox) {
    SoundValue out{};
    Box rounding(p);
    out.form.center = quantize(center, rounding);
    for (const auto& [id, c] : coeff) {
      double d = quantize(c, rounding);
      if (d != 0) out.form.coefficients.emplace(id,d);
    }
    out.roundingRadius = rounding.bounds().hi;
    add(remainder, rounding);
    if (out.form.coefficients.size() > limit) {
      std::vector<std::pair<double,std::uint64_t>> order;
      for (auto [id,c] : out.form.coefficients) order.emplace_back(std::abs(c), id);
      std::sort(order.begin(), order.end());
      out.condensed = order.size() - limit;
      for (std::size_t i=0; i<out.condensed; ++i) {
        addDouble(remainder, order[i].first);
        out.form.coefficients.erase(order[i].second);
      }
    }
    out.form.remainder = remainder.bounds().hi;
    out.naive = baseline;
    // Intersect only projections. The form's coefficients are still needed
    // with their original meanings when this SSA value meets a sibling.
    out.enclosure = intersection(intersection(independentBox, baseline), out.form.toSoundIntervalMPFR(p));
    return out;
  }
  SoundValue finish(std::size_t limit) { return finish(limit, naive.bounds(), tight.bounds()); }
};

// Monotone odd *core*, not a claim that the full activation is monotone.
Box core(const Box& x, Activation f) {
  unsigned p = x.precision();
  if (f == Activation::SiLU) {
    Box r = half(x); mpfi_tanh(r.v,r.v); return half(r);
  }
  check(f == Activation::ErfGeLU, "unknown affine activation");
  Box root(p,2), z(p), r(p);
  mpfi_sqrt(root.v, root.v); mpfi_div(z.v,x.v,root.v);
  Float lo(p), hi(p);
  mpfi_get_left(lo.v,z.v); mpfi_get_right(hi.v,z.v);
  mpfr_erf(lo.v,lo.v,MPFR_RNDD); mpfr_erf(hi.v,hi.v,MPFR_RNDU);
  mpfi_interv_fr(r.v,lo.v,hi.v);
  return half(r);
}
Interval image(unsigned p, Interval x, Activation f, double error) {
  Box band(p,Interval{-error,error});
  if (f == Activation::ReLU)
    return (Box(p,Interval{std::max(0.,x.lo), std::max(0.,x.hi)}) + band).bounds();
  Box X(p,x);
  return (X * (Box(p,0.5)+core(X,f)) + band).bounds();
}
// f(x) = x/2 + h(|x|). On t>=0, h(t)=t*core(t) is nonnegative
// nondecreasing (both factors are); for ReLU, h(t)=t/2. This certifies the
// deviation of the chosen line on the full closed interval, including zero.
Interval residual(unsigned p, Interval x, Activation f, double error) {
  const double a = std::max({0.,x.lo,-x.hi}), b = std::max(-x.lo,x.hi);
  Box t(p,Interval{a,b}), band(p,Interval{-error,error});
  return ((f==Activation::ReLU ? half(t) : t*core(t,f)) + band).bounds();
}
}  // namespace

Interval AffineRange::toSoundIntervalMPFR(unsigned p) const {
  checkPrecision(p);
  check(std::isfinite(center) && std::isfinite(remainder) && remainder>=0, "invalid affine form");
  Float radius(p), lo(p), hi(p);
  mpfr_set_d(radius.v, remainder, MPFR_RNDU);
  for (auto [id,c] : coefficients) {
    (void)id; check(std::isfinite(c), "nonfinite affine coefficient");
    mpfr_add_d(radius.v,radius.v,std::abs(c),MPFR_RNDU);
  }
  mpfr_d_sub(lo.v,center,radius.v,MPFR_RNDD);
  mpfr_add_d(hi.v,radius.v,center,MPFR_RNDU);
  return {finite(lo.v,MPFR_RNDD), finite(hi.v,MPFR_RNDU)};
}

AffineIntervalAnalyzer::AffineIntervalAnalyzer(unsigned p, std::size_t limit)
    : precision_(p), maxSymbols_(limit) {
  checkPrecision(p); check(limit<=256, "affine symbol limit exceeds 256");
}
AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::store(Slots slots) {
  check(!slots.empty(), "empty affine value");
  nodes_.push_back(std::move(slots)); return Value(this,nodes_.size()-1);
}
std::uint64_t AffineIntervalAnalyzer::fresh() {
  if (next_==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("affine symbol ID exhausted");
  return next_++;
}
const AffineIntervalAnalyzer::Slots& AffineIntervalAnalyzer::slots(Value v) const {
  check(v.owner_==this, "foreign affine SSA value"); return nodes_.at(v.index_);
}
Interval AffineIntervalAnalyzer::domain(Value v) const {
  const auto& s=slots(v); auto r=s.front().enclosure;
  for (const auto& x:s) { r.lo=std::min(r.lo,x.enclosure.lo); r.hi=std::max(r.hi,x.enclosure.hi); }
  return r;
}
AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::source(std::span<const Interval> domains) {
  Slots out; out.reserve(domains.size());
  for (auto d:domains) {
    Accumulator a(precision_); a.independent(d,fresh());
    out.push_back(a.finish(maxSymbols_,d,d));
  }
  return store(std::move(out));
}

AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::binary(Value x, Value y, bool minus) {
  const auto &xs=slots(x), &ys=slots(y);
  check(xs.size()==ys.size(), "affine add/sub shape mismatch");
  Slots out; out.reserve(xs.size());
  for (std::size_t i=0;i<xs.size();++i) {
    Accumulator a(precision_);
    if (minus && x==y) {
      // Only literal SSA identity witnesses cancellation of even the anonymous remainder.
      auto n=Box(precision_,xs[i].naive)-Box(precision_,ys[i].naive);
      out.push_back(a.finish(maxSymbols_,n.bounds(),{0,0}));
    } else {
      a.term(xs[i],1); a.term(ys[i],minus?-1:1); out.push_back(a.finish(maxSymbols_));
    }
  }
  return store(std::move(out));
}
AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::add(Value a, Value b) { return binary(a,b,false); }
AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::subtract(Value a, Value b) { return binary(a,b,true); }
AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::negate(Value x) {
  Slots out;
  for (const auto& s:slots(x)) {
    Accumulator a(precision_); a.term(s,-1); out.push_back(a.finish(maxSymbols_));
  }
  return store(std::move(out));
}
AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::linear(Value x, std::size_t rows,
    std::span<const double> w, std::span<const double> bias) {
  const auto& xs=slots(x); auto n=xs.size();
  check(rows>0 && w.size()/n==rows && w.size()%n==0 && (bias.empty() || bias.size()==rows),
        "affine matrix/bias shape mismatch");
  Slots out; out.reserve(rows);
  for (std::size_t i=0;i<rows;++i) {
    Accumulator a(precision_,bias.empty()?0:bias[i]);
    for (std::size_t j=0;j<n;++j) a.term(xs[j],w[i*n+j]);
    out.push_back(a.finish(maxSymbols_));
  }
  return store(std::move(out));
}
AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::periodicDiagonals(Value x, std::size_t period,
    std::span<const std::int64_t> offsets, std::span<const double> table) {
  const auto& xs=slots(x); auto n=xs.size();
  check(period>0 && period%n==0 && !offsets.empty() && table.size()/period==offsets.size() &&
        table.size()%period==0, "invalid affine periodic diagonal dimensions");
  std::vector<std::int64_t> sorted(offsets.begin(),offsets.end());
  std::sort(sorted.begin(),sorted.end());
  check(sorted.front()>=0 && std::uint64_t(sorted.back())<period &&
        std::adjacent_find(sorted.begin(),sorted.end())==sorted.end(), "invalid affine diagonal offsets");
  for (std::size_t d=0;d<offsets.size();++d) {
    auto row=table.subspan(d*period,period);
    for (std::size_t i=0;i<period;++i)
      check(std::isfinite(row[i]) && row[i]==row[i%n], "unproven periodic diagonal replica");
  }
  Slots out; out.reserve(n);
  for (std::size_t i=0;i<n;++i) {
    Accumulator a(precision_);
    for (std::size_t d=0;d<offsets.size();++d) {
      auto off=static_cast<std::size_t>(offsets[d])%n;
      auto j=i>=n-off ? i-(n-off) : i+off;
      a.term(xs[j],table[d*period+i]);
    }
    out.push_back(a.finish(maxSymbols_));
  }
  return store(std::move(out));
}
AffineIntervalAnalyzer::Value AffineIntervalAnalyzer::activate(Value x, Activation f, double error,
    NonlinearMode mode) {
  check(std::isfinite(error) && error>=0, "invalid affine activation error");
  check(mode==NonlinearMode::AffineResidual || mode==NonlinearMode::IndependentInterval,
        "unknown affine nonlinear mode");
  Slots out;
  for (const auto& s:slots(x)) {
    auto baseline=image(precision_,s.naive,f,error);
    auto tight=image(precision_,s.enclosure,f,error);
    Accumulator a(precision_);
    if (mode==NonlinearMode::IndependentInterval) {
      // No correlation claim about the nonlinear image. A new symbol per
      // result/slot still lets later consumers share this particular value.
      a.independent(tight,fresh());
    } else if (f==Activation::ReLU && (s.enclosure.lo>=0 || s.enclosure.hi<=0)) {
      if (s.enclosure.lo>=0) a.term(s,1);
      a.independent({-error,error},fresh());
    } else {
      a.term(s,0.5); a.independent(residual(precision_,s.enclosure,f,error),fresh());
    }
    out.push_back(a.finish(maxSymbols_,baseline,tight));
  }
  return store(std::move(out));
}

}  // namespace snabf::ir::affine

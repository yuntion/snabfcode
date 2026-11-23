#include "pass2/IntervalCertifier.h"
#include "pass2/PolyTemplates.h"

#include <mpfr.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace snabf::pass2;
using F = StageFunction;

void check(bool result, const char* what) {
  if (!result) throw std::runtime_error(what);
}

template <class Fn> void invalid(Fn fn) {
  try { fn(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid input was accepted");
}

long double reference(F f, long double x) {
  switch (f) {
    case F::SiluDirect: return std::tanh(x/2)/2;
    case F::TanhQuarter: return std::tanh(x/4);
    case F::Rational: return x/(1+x*x);
    case F::GeluDirect: return std::erf(x/std::sqrt(2.L))/2;
  }
  throw std::logic_error("bad test reference");
}

void registry() {
  check(templatesFor("silu").size() == 2 && templatesFor("gelu").size() == 1 &&
        templatesFor("unsupported").empty(), "registry lookup");
  for (const std::string name : {"silu", "gelu"}) {
    for (const auto& spec : templatesFor(name)) {
      check(spec.templateRefError == 0 && !spec.id.empty() && !spec.stages.empty() &&
            spec.reconstruction == Reconstruction::MultiplyOriginalInputByOffsetPlusCore &&
            spec.reconstructionOffset == 0.5, "template metadata");
      for (F f : spec.stages) {
        const auto& s = stageSpec(f);
        check(s.function == f && s.id == stageName(f) && !s.referenceExpression.empty() &&
              s.degrees == kCandidateDegrees && s.basis == FitBasis::OddChebyshevRadiusNormalized &&
              s.fitDomain == FitDomain::PropagatedIdealInput &&
              s.coefficientFormat == CoefficientFormat::AscendingBinary64OriginalPowers,
              "missing stage metadata");
        check(f == F::Rational ? (s.templateInputEnvelope.lo == -1 && s.templateInputEnvelope.hi == 1) :
              (std::isinf(s.templateInputEnvelope.lo) && std::isinf(s.templateInputEnvelope.hi)),
              "wrong ideal domain envelope");
      }
      for (long double x : {-8.L,-1.L,0.L,0.5L,4.L}) {
        long double u=x;
        for (F f : spec.stages) u=reference(f,u);
        const long double exact = name == "silu" ? x/(1+std::exp(-x)) : x*std::erfc(-x/std::sqrt(2.L))/2;
        check(std::abs(x*(spec.reconstructionOffset+u)-exact) < 2e-15L,
              "reconstruction does not equal the activation");
      }
    }
  }
}

struct Real {
  mpfr_t v;
  Real() { mpfr_init2(v,256); }
  ~Real() { mpfr_clear(v); }
  Real(const Real&) = delete;
  Real& operator=(const Real&) = delete;
};
constexpr auto nearest = MPFR_RNDN;

// Independent high-precision point checks, not a proof of a supremum. SiLU
// and GeLU use logistic/erfc identities rather than the certifier's tanh/erf.
void referencePoint(Real& y, F f, double x) {
  Real t;
  mpfr_set_d(t.v,x,nearest);
  switch (f) {
    case F::SiluDirect:
      mpfr_neg(y.v,t.v,nearest); mpfr_exp(y.v,y.v,nearest);
      mpfr_add_ui(y.v,y.v,1,nearest); mpfr_ui_div(y.v,1,y.v,nearest);
      mpfr_sub_d(y.v,y.v,0.5,nearest); return;
    case F::TanhQuarter:
      mpfr_div_ui(y.v,t.v,4,nearest); mpfr_tanh(y.v,y.v,nearest); return;
    case F::Rational:
      mpfr_sqr(y.v,t.v,nearest); mpfr_add_ui(y.v,y.v,1,nearest);
      mpfr_div(y.v,t.v,y.v,nearest); return;
    case F::GeluDirect:
      mpfr_sqrt_ui(y.v,2,nearest); mpfr_div(y.v,t.v,y.v,nearest);
      mpfr_erfc(y.v,y.v,nearest); mpfr_ui_sub(y.v,1,y.v,nearest);
      mpfr_div_ui(y.v,y.v,2,nearest); return;
  }
}

void polynomialPoint(Real& y, const StagePolynomial& p, double x, unsigned derivative=0) {
  Real a, t;
  mpfr_set_d(t.v,x,nearest); mpfr_set_zero(y.v,1);
  for (unsigned j=p.degree+1; j-- > derivative;) {
    mpfr_set_d(a.v,p.coefficients[j],nearest);
    if (derivative) mpfr_mul_ui(a.v,a.v,j,nearest);
    mpfr_fma(y.v,y.v,t.v,a.v,nearest);
  }
}

void witnesses(F f, const StagePolynomial& p, const StageCertificate& c) {
  auto contains = [](Interval1D i, const Real& t) {
    return mpfr_cmp_d(t.v,i.lo) >= 0 && mpfr_cmp_d(t.v,i.hi) <= 0;
  };
  check(c.J.lo == std::min(c.Iin.lo,c.Ain.lo) && c.J.hi == std::max(c.Iin.hi,c.Ain.hi),
        "incorrect ideal/actual hull");
  Real g, y;
  for (unsigned k=0; k<=64; ++k) {
    auto x = [&](Interval1D i) { return std::lerp(i.lo,i.hi,k/64.); };
    referencePoint(g,f,x(c.Iin));
    check(contains(c.Iout,g),"ideal range excludes an MPFR value");
    polynomialPoint(y,p,x(c.Iin));
    mpfr_sub(y.v,y.v,g.v,nearest); mpfr_abs(y.v,y.v,nearest);
    check(mpfr_cmp_d(y.v,c.epsilon) <= 0,"residual upper bound excludes an MPFR value");
    polynomialPoint(y,p,x(c.J),1); mpfr_abs(y.v,y.v,nearest);
    check(mpfr_cmp_d(y.v,c.L) <= 0,"derivative bound excludes an MPFR value");
    polynomialPoint(y,p,x(c.Ain));
    check(contains(c.Aout,y),"actual range excludes an MPFR value");
  }
}

// A fit on m odd Chebyshev terms has m+1 alternating extrema on the folded
// interval. Detecting the ripple rejects interpolation or a one-shot fit at
// the initial nodes even when both would pass a loose residual threshold.
void alternation(F f, const StagePolynomial& p, double B, double epsilon) {
  std::vector<long double> peaks;
  bool lastNegative = false;
  for (unsigned i=1; i<=4096; ++i) {
    const long double x = static_cast<long double>(B)*i/4096;
    long double y=0;
    for (auto it=p.coefficients.rbegin(); it!=p.coefficients.rend(); ++it) y=y*x+*it;
    const long double err = y-reference(f,x);
    if (err == 0) continue;
    if (peaks.empty() || std::signbit(err) != lastNegative) peaks.push_back(0);
    peaks.back()=std::max(peaks.back(),std::abs(err)); lastNegative=std::signbit(err);
  }
  check(peaks.size() == (p.degree+1)/2+1,"Remez alternant has wrong number of lobes");
  const auto [low,high]=std::minmax_element(peaks.begin(),peaks.end());
  check(*low >= 0.995L * *high,"Remez ripple does not equioscillate");
  check(epsilon <= 1.003L * *high + 2e-15L,"certificate too loose on broad domain");
}

void broadDomains() {
  for (double B : {4.,8.}) {
    const Interval1D axis{-B,B};
    for (F f : {F::SiluDirect,F::TanhQuarter,F::Rational,F::GeluDirect}) {
      for (unsigned d : kCandidateDegrees) {
        try {
          const auto p=generateStage(f,d,axis);
          check(p.degree == d && p.coefficients.size() == d+1,"wrong coefficient count");
          for (unsigned i=0; i<=d; i+=2) check(p.coefficients[i] == 0,"non-odd output");
          const auto c=certifyStage(f,p,axis,axis);
          witnesses(f,p,c); alternation(f,p,B,c.epsilon);
        } catch (const std::exception& ex) {
          throw std::runtime_error(std::string(stageName(f))+" d="+std::to_string(d)+
                                   " B="+std::to_string(B)+": "+ex.what());
        }
      }
    }
    const auto p1=generateStage(F::TanhQuarter,5,axis);
    const auto c1=certifyStage(F::TanhQuarter,p1,axis,axis);
    const auto p2=generateStage(F::Rational,7,c1.Iout);
    const auto c2=certifyStage(F::Rational,p2,c1.Iout,c1.Aout);
    witnesses(F::Rational,p2,c2);
    check(c2.Iin.hi == c1.Iout.hi && c2.Ain.lo == c1.Aout.lo,"lost composite stage input");
  }
}

void smallDomains() {
  const Interval1D small{-0.75,0.75};
  const auto p=generateStage(F::TanhQuarter,5,small);
  const auto c=certifyStage(F::TanhQuarter,p,small,{-0.76,0.8});
  witnesses(F::TanhQuarter,p,c);
  check(c.epsilon > 4e-9 && c.epsilon < 1e-8,"tiny d5 residual not resolved");
  std::cout << "tanh(x/4) d5 [-0.75,0.75]: epsilon=" << c.epsilon << '\n';
  for (Interval1D i : {Interval1D{-1e-10,1e-10},{-0.125,0.1},{0.125,1},{-1,-0.125},{0,0}}) {
    const auto q=generateStage(F::SiluDirect,7,i);
    const auto cert=certifyStage(F::SiluDirect,q,i,i);
    witnesses(F::SiluDirect,q,cert);
    check(cert.epsilon < (i.hi <= 0.1 && i.lo >= -0.125 ? 3e-15 : 1e-6),"small interval fit");
  }
}

void independentCertificate() {
  // The maximum |p'| = 3 is in the gap between the ideal and actual inputs.
  const StagePolynomial cubic{3,{0,0,-3,1}};
  const auto c=certifyStage(F::Rational,cubic,{0,0.25},{1.75,2});
  check(c.L >= 3 && c.L < 3.000000001 && c.Aout.lo == -4 && c.Aout.hi < -3,
        "derivative on J or range on Ain is incorrect");
  check(c.epsilon < 0.42,"residual evaluated on hull instead of ideal input");
  witnesses(F::Rational,cubic,c);

  // Non-odd supplied polynomials and interior extrema must not be ignored.
  const StagePolynomial quadratic{2,{4,-8,4}};
  const auto q=certifyStage(F::TanhQuarter,quadratic,{0,0},{0,2});
  check(q.epsilon == 4 && q.L == 8 && q.Aout.lo == 0 && q.Aout.hi == 4,
        "interior polynomial range extremum or even coefficient lost");
  const auto r=certifyStage(F::Rational,{0,{0}},{-3,3},{-3,3});
  check(r.Iout.lo == -0.5 && r.Iout.hi == 0.5 && r.epsilon >= 0.5 && r.epsilon < 0.501,
        "rational reference has interior extrema at +/-1");

  StagePolynomial changed{1,{0,0.5}};
  auto exact=certifyStage(F::Rational,changed,{1,1},{1,1});
  check(exact.epsilon == 0 && exact.Aout.lo == 0.5,"exact dyadic point certificate");
  changed.coefficients[1]=std::nextafter(0.5,1.);
  auto mutated=certifyStage(F::Rational,changed,{1,1},{1,1});
  check(mutated.epsilon == changed.coefficients[1]-0.5 && mutated.L == changed.coefficients[1] &&
        mutated.Aout.lo == changed.coefficients[1],"certificate ignored a one-ULP mutation");
}

void reluReconstruction() {
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<StagePolynomial> zero{{0,{0}}};
  for (const double B : {0.,0.75,4.}) {
    const auto c = certifyReluReconstruction(zero,{-B,B});
    const double exact = B/2, upper = std::nextafter(exact,inf);
    check(c.input.lo == -B && c.input.hi == B && c.leftEpsilon >= exact && c.leftEpsilon <= upper &&
          c.rightEpsilon >= exact && c.rightEpsilon <= upper && c.epsilon == std::max(c.leftEpsilon,c.rightEpsilon),
          "zero chain: expected B/2 within one outward ULP on both closed halves");
    if (B == 4.) std::cout << "ReLU reconstruction: g=0 [-4,4] epsilon=" << c.epsilon << '\n';
  }
  const auto endpoint = certifyReluReconstruction(zero,{0,4});
  check(endpoint.leftEpsilon == 0 && endpoint.rightEpsilon == 2 && endpoint.epsilon == 2,
        "zero endpoint or right half missing from ReLU certificate");
  const double tiny = std::numeric_limits<double>::denorm_min();
  const auto sub = certifyReluReconstruction(zero,{-tiny,tiny});
  check(sub.leftEpsilon == tiny && sub.rightEpsilon == tiny && sub.epsilon == tiny,
        "positive subnormal half-error rounded inward");

  // p1=2x, p2=z^3/32 gives g=x^3/4; reversing these stages is different.
  // At t=|x| on [0,1.5] the magnitude is t/2-t^4/8. Its maximum
  // on [0,1.5] is 3/8 at t=1, and on [0,0.5] it is 31/128 at t=0.5.
  const std::vector<StagePolynomial> chain{{1,{0,2}},{3,{0,0,0,1./32}}};
  const auto c = certifyReluReconstruction(chain,{-1.5,0.5});
  auto enclosesTightly = [](double bound, double exact) { return bound >= exact && bound <= exact+1e-4; };
  check(enclosesTightly(c.leftEpsilon,3./8) && enclosesTightly(c.rightEpsilon,31./128) &&
        c.epsilon == c.leftEpsilon,"composed ReLU: interior extremum, stage order or side bound lost");
  const auto mirror = certifyReluReconstruction(chain,{-0.5,1.5});
  check(enclosesTightly(mirror.leftEpsilon,31./128) && enclosesTightly(mirror.rightEpsilon,3./8) &&
        mirror.epsilon == mirror.rightEpsilon,"larger positive half lost");

  // g=3x-4x^3 stays in [-1,1] (extrema at +/-1/2). Thus |r|<=|x|<=1,
  // with equality at both endpoints. Both odd coefficients matter here.
  const std::vector<StagePolynomial> mixedSigns{{1,{0,2}},{3,{0,1.5,0,-0.5}}};
  const auto s = certifyReluReconstruction(mixedSigns,{-1,1});
  check(s.leftEpsilon == 1 && s.rightEpsilon == 1 && s.epsilon == 1,
        "mixed-sign cubic reconstruction lost an endpoint or coefficient");
  // Exercise the longer-chain stationary-point cover on a non-symmetric
  // interval, including two different stationary points of the stage.
  auto longChain=chain; longChain.insert(longChain.begin(),{1,{0,1}});
  const auto longCert=certifyReluReconstruction(longChain,{-1.5,0.5});
  check(enclosesTightly(longCert.leftEpsilon,3./8) && enclosesTightly(longCert.rightEpsilon,31./128),
        "long-chain certificate omitted an interior maximum");
  auto stationary=mixedSigns; stationary.push_back({1,{0,1}});
  check(certifyReluReconstruction(stationary,{-1,1}).epsilon>=1,"stationary cover lost endpoint");

  // Old maximum 1; moving the actual coefficient one ULP makes the exact
  // maximum 1+2^-53, which must round UP, not to the nearest even double.
  std::vector<StagePolynomial> perturbed{{1,{0,-1}}};
  check(certifyReluReconstruction(perturbed,{-1,1}).epsilon == 1,"wrong linear-chain maximum");
  perturbed[0].coefficients[1] = std::nextafter(-1.,-inf);
  check(certifyReluReconstruction(perturbed,{-1,1}).epsilon == std::nextafter(1.,inf),
        "ReLU certificate ignored binary64 coefficient mutation or rounded inward");
  const auto precision = std::cout.precision(std::numeric_limits<double>::max_digits10);
  std::cout << "ReLU composed [-1.5,0.5] left=" << c.leftEpsilon << " right=" << c.rightEpsilon << '\n';
  std::cout.precision(precision);
}

void invalidInputs() {
  const double inf=std::numeric_limits<double>::infinity(), nan=std::numeric_limits<double>::quiet_NaN();
  invalid([&]{generateStage(F::SiluDirect,4,{-1,1});});
  invalid([&]{generateStage(F::Rational,3,{1,-1});});
  invalid([&]{generateStage(F::GeluDirect,5,{-inf,1});});
  invalid([&]{certifyStage(F::Rational,{3,{0,1}},{0,1},{0,1});});
  invalid([&]{certifyStage(F::TanhQuarter,{1,{0,nan}},{0,1},{0,1});});
  invalid([&]{certifyStage(F::SiluDirect,{1,{0,1}},{0,1},{nan,1});});
  invalid([&]{certifyStage(static_cast<F>(77),{1,{0,1}},{0,1},{0,1});});

  const std::vector<StagePolynomial> identity{{1,{0,1}}};
  for (auto axis : {Interval1D{1,-1},{1,2},{-2,-1},{-inf,1},{-1,inf},{nan,0}})
    invalid([&]{certifyReluReconstruction(identity,axis);});
  invalid([&]{certifyReluReconstruction({}, {-1,1});});
  // Validate every stage, even when the input is exactly zero and the first
  // stage already gives the zero polynomial.
  for (auto bad : {StagePolynomial{3,{0,1}}, {0,{1}}, {3,{0,1,1e-300,0}},
                   {1,{0,nan}}, {1,{0,inf}}, {17,std::vector<double>(18,0)}}) {
    std::vector<StagePolynomial> chain{{0,{0}},bad};
    invalid([&]{certifyReluReconstruction(chain,{0,0});});
  }
  const std::vector<StagePolynomial> huge{{1,{0,std::numeric_limits<double>::max()}}};
  bool overflow = false;
  try { (void)certifyReluReconstruction(huge,{-2,2}); }
  catch (const std::overflow_error&) { overflow = true; }
  check(overflow,"unrepresentable ReLU bound was accepted");
}
}  // namespace

int main() {
  try {
    const auto start=std::chrono::steady_clock::now();
    registry(); smallDomains(); independentCertificate(); broadDomains(); reluReconstruction(); invalidInputs();
    std::cout << "Pass2CertTest passed (" << std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count() << " s)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Pass2CertTest failed: " << e.what() << '\n';
    return 1;
  }
}

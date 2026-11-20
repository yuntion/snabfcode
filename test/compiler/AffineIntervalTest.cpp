#include "ir/AffineIntervalAnalyzer.h"

#include <mpfr.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace {
using namespace snabf::ir::affine;
using Engine = AffineIntervalAnalyzer;
using Value = Engine::Value;
void expect(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f, const char* why) {
  try { f(); } catch (const std::exception&) { return; } throw std::runtime_error(why);
}
bool contains(Interval a, long double x) { return a.lo<=x && x<=a.hi; }
Value source(Engine& a, std::initializer_list<Interval> d) { return a.source(d); }
Value linear(Engine& a, Value x, size_t rows, std::initializer_list<double> w,
             std::initializer_list<double> b={}) { return a.linear(x,rows,w,b); }
const SoundValue& one(const Engine& a, Value v) { return a.slots(v).at(0); }
void handwritten() {
  Engine a; auto x=source(a,{{-1,1}}), y=source(a,{{-1,1}});
  expect(a.domain(a.subtract(x,x))==Interval{0,0},"same SSA did not cancel");
  expect(a.domain(a.subtract(x,y))==Interval{-2,2},"independent ports falsely cancelled");
  auto xy=source(a,{{-1,1},{-2,2}});
  auto mixed=linear(a,xy,2,{-1,1, 1,-1},{0.25,-0.5});
  auto join=a.add(xy,mixed);
  expect(a.slots(join)[0].enclosure==Interval{-1.75,2.25} &&
         a.slots(join)[1].enclosure==Interval{-1.5,0.5},"signed residual/bias");
  expect(a.slots(join)[0].naive==Interval{-3.75,4.25},"naive path was overwritten by affine");
  expect(a.domain(a.add(join,a.negate(join)))==Interval{0,0},"residual sign lost");
  auto relu=a.activate(x,Activation::ReLU); const auto& s=one(a,relu);
  expect(s.form.toSoundIntervalMPFR()==Interval{-0.5,1} && s.enclosure==Interval{0,1},"intersection modified the form");
  expect(a.domain(a.subtract(relu,linear(a,x,1,{0.5})))==Interval{0,0.5},"intersection discarded correlation");
  auto f1=a.activate(x,Activation::ReLU,0,NonlinearMode::IndependentInterval);
  auto f2=a.activate(x,Activation::ReLU,0,NonlinearMode::IndependentInterval);
  expect(a.domain(a.subtract(f1,f2))==Interval{-1,1} && a.domain(a.subtract(f1,f1))==Interval{0,0},
         "nonlinear fallback reused a fresh symbol, or forgot its own SSA identity");
  // Different logical coordinates must never share a signed symbol.
  auto v=source(a,{{-1,1},{-1,1},{-1,1}});
  expect(a.domain(linear(a,v,1,{1,0,-1}))==Interval{-2,2},"independent slots falsely cancelled");
  auto fixed=source(a,{{1,1},{2,2},{4,4}});
  std::vector<int64_t> off{1,2}; std::vector<double> weights{1,2,3,1,2,3, -1,0,1,-1,0,1};
  auto diag=a.periodicDiagonals(fixed,6,off,weights);
  expect(a.slots(diag)[0].enclosure==Interval{-2,-2} && a.slots(diag)[1].enclosure==Interval{8,8} &&
         a.slots(diag)[2].enclosure==Interval{5,5},"signed diagonal orientation/row order");
  weights.back()=2;
  rejects([&]{a.periodicDiagonals(fixed,6,off,weights);},"nonperiodic diagonal accepted");
  rejects([&]{a.periodicDiagonals(fixed,6,std::array<int64_t,2>{1,1},weights);},"duplicate diagonals accepted");
  rejects([&]{a.add(x,v);},"shape mismatch accepted");
  Engine other; rejects([&]{other.subtract(x,x);},"foreign SSA accepted");
  rejects([&]{source(a,{{2,1}});},"inverted interval accepted");
  rejects([&]{source(a,{{0,INFINITY}});},"unbounded source accepted");
  rejects([&]{linear(a,x,1,{NAN});},"NaN weight accepted");
  rejects([&]{a.activate(x,static_cast<Activation>(123));},"unknown activation accepted");
  rejects([&]{a.activate(x,Activation::SiLU,-1);},"negative polynomial cap accepted");
  rejects([&]{Engine e(32);},"precision below 53 accepted");
  rejects([&]{linear(a,source(a,{{2,2}}),1,{std::numeric_limits<double>::max()});},"overflow accepted");
}

void condensation() {
  for (size_t n:{256u,257u,768u}) {
    Engine a; auto x=a.source(std::vector<Interval>(n,{-1,1}));
    std::vector<double> w(n,1); w[0]=-2; w[1]=0.5;
    const auto& s=one(a,a.linear(x,1,w));
    expect(s.form.coefficients.size()==256 && s.condensed==n-256,"wrong condensation threshold");
    expect(s.form.coefficients.contains(0),"largest negative term evicted");
    if (n>256) {
      expect(!s.form.coefficients.contains(1) && s.form.coefficients.contains(n-256+1),"unstable condensation tie");
      expect(s.form.remainder==double(n-256)-0.5,"evicted signed terms not absorbed by magnitude");
    }
    expect(s.enclosure==Interval{-double(n)-0.5,double(n)+0.5},"condensation lost an independent corner");
  }
  Engine none(53,0); auto x=source(none,{{-1,1}}),y=source(none,{{-1,1}});
  expect(none.domain(none.subtract(x,y))==Interval{-2,2},"anonymous remainders falsely cancelled");
  expect(none.domain(none.subtract(x,x))==Interval{0,0},"literal SSA identity lost after condensation");
}

struct Ref { mpfr_t v; Ref(){mpfr_init2(v,512);} ~Ref(){mpfr_clear(v);}
             Ref(const Ref&)=delete; Ref& operator=(const Ref&)=delete; };
void rounding() {
  // Independently evaluate the exact dyadic expression (under 220 significant
  // bits) at both corners. Check the form with the SAME symbol assignment,
  // not just the scalar hull which might hide an uncharged coefficient error.
  for (unsigned p:{53u,160u,384u}) {
    Engine a(p); auto x=source(a,{{-1,1}});
    auto v=linear(a,linear(a,x,1,{0.1},{0.3}),1,{-0.7},{0.125});
    const auto& s=one(a,v);
    expect(s.roundingRadius>0 && s.form.remainder>=s.roundingRadius,"no binary64 storage error charged");
    for (int sign:{-1,1}) {
      Ref exact, approx, diff;
      mpfr_set_d(exact.v,sign*0.1,MPFR_RNDN); mpfr_add_d(exact.v,exact.v,0.3,MPFR_RNDN);
      mpfr_mul_d(exact.v,exact.v,-0.7,MPFR_RNDN); mpfr_add_d(exact.v,exact.v,0.125,MPFR_RNDN);
      mpfr_set_d(approx.v,sign*s.form.coefficients.at(0),MPFR_RNDN);
      mpfr_add_d(approx.v,approx.v,s.form.center,MPFR_RNDN);
      mpfr_sub(diff.v,exact.v,approx.v,MPFR_RNDN); mpfr_abs(diff.v,diff.v,MPFR_RNDN);
      expect(mpfr_cmp_d(diff.v,s.form.remainder)<=0 && mpfr_cmp_d(exact.v,s.enclosure.lo)>=0 &&
             mpfr_cmp_d(exact.v,s.enclosure.hi)<=0,"directed coefficient or center bound missed exact dyadic corner");
    }
    double tiny=std::numeric_limits<double>::denorm_min();
    auto sub=source(a,{{0,tiny},{-tiny,0},{0,0}});
    expect(a.slots(sub)[0].enclosure==Interval{0,tiny} && a.slots(sub)[1].enclosure==Interval{-tiny,0} &&
           a.slots(sub)[0].roundingRadius>=tiny,"subnormal midpoint/radius rounded inward");
    for (auto f:{Activation::SiLU,Activation::ErfGeLU,Activation::ReLU}) {
      auto result=a.activate(sub,f);
      expect(contains(a.slots(result)[0].enclosure,tiny*0.5L) && a.slots(result)[2].enclosure==Interval{0,0},
             "closed zero/subnormal activation interval");
    }
  }
}
long double reference(long double x, Activation f) {
  if (f==Activation::SiLU) return x/(1+std::exp(-x));
  if (f==Activation::ErfGeLU) return x*std::erfc(-x/std::sqrt(2.L))/2;
  return std::max(0.L,x);
}
void nonlinear() {
  for (auto f:{Activation::SiLU,Activation::ErfGeLU,Activation::ReLU}) {
    Engine a; auto x=source(a,{{-4,1}}); constexpr double err=0.03125;
    for (auto mode:{NonlinearMode::AffineResidual,NonlinearMode::IndependentInterval}) {
      auto r=a.domain(a.activate(x,f,err,mode));
      // Includes the interior stationary regions of the FULL negative SiLU
      // and erf-GeLU, not only the endpoints of their monotone cores.
      for (auto t:{-4.L,-2.L,-1.27846454L,-0.751791525L,-0.1L,0.L,1.L}) {
        auto g=reference(t,f); expect(contains(r,g-err)&&contains(r,g+err),"lost nonlinear/actual trajectory");
      }
      auto z=source(a,{{0,0}}); expect(a.domain(a.activate(z,f,err,mode))==Interval{-err,err},
                                      "ideal singleton incorrectly narrowed a real polynomial trajectory");
    }
  }
}

constexpr size_t N=4, layers=6;
using Point=std::array<long double,N>;
using Matrix=std::array<double,N*N>;
struct Layer {
  Matrix w;
  std::array<double,N> bias;
  Activation f;
  Value pre, act, join;
};
void residualNetwork() {
  constexpr std::uint64_t seed=0xc20261120ULL;
  const std::array<double,N> rad{0.375,0.5,0.25,0.625};
  std::array<Interval,N> box;
  for (size_t i=0;i<N;++i) box[i]={-rad[i],rad[i]};
  std::mt19937_64 gen(seed);
  Engine a; Value input=a.source(box), h=input;
  std::vector<Layer> net;
  size_t peakSymbols=0;
  for (size_t k=0;k<layers;++k) {
    Matrix w; std::array<double,N> b;
    for (size_t i=0;i<N;++i) {
      b[i]=(int(gen()%5)-2)/64.;
      for (size_t j=0;j<N;++j)
        w[i*N+j]=i==j ? -1.25 : (gen()&1?1.:-1.)*(1+(gen()%2))/16.;
    }
    auto f=k%2?Activation::ErfGeLU:Activation::SiLU;
    auto pre=a.linear(h,N,w,b), act=a.activate(pre,f), join=a.add(h,act);
    net.push_back({w,b,f,pre,act,join}); h=join;
    for (auto v:{pre,act,join}) for (const auto& s:a.slots(v))
      peakSymbols=std::max(peakSymbols,s.form.coefficients.size());
    if (k<3) continue;
    double widestNaive=0,widestAffine=0,minReduction=1;
    for (const auto& s:a.slots(pre)) {
      double naive=s.naive.hi-s.naive.lo, affine=s.enclosure.hi-s.enclosure.lo;
      expect(naive>0 && affine>=0 && affine<=naive,"invalid fixture widths");
      minReduction=std::min(minReduction,1-affine/naive);
      widestNaive=std::max(widestNaive,naive); widestAffine=std::max(widestAffine,affine);
    }
    std::cout<<std::setprecision(10)<<"layer="<<k+1<<" width_naive_max="<<widestNaive
             <<" width_affine_max="<<widestAffine<<" reduction_of_max="<<1-widestAffine/widestNaive
             <<" min_slot_reduction="<<minReduction<<'\n';
    expect(minReduction>=0.40,"six-layer fixture missed its per-slot 40% target");
  }
  std::vector<Point> corners;
  for (size_t m=0;m<(1u<<N);++m) {
    Point p; for (size_t j=0;j<N;++j) p[j]=(m&(1u<<j))?rad[j]:-rad[j];
    corners.push_back(p);
  }
  // For each layer and each pre-activation slot, maximize and minimize the
  // affine contribution of the four original inputs (symbols 0..3).
  for (const auto& l:net) for (const auto& s:a.slots(l.pre)) for (int direction:{-1,1}) {
    Point p;
    for (size_t j=0;j<N;++j) {
      auto it=s.form.coefficients.find(j);
      p[j]=(it==s.form.coefficients.end() || direction*it->second>=0)?rad[j]:-rad[j];
    }
    corners.push_back(p);
  }
  std::uint64_t checks=0;
  auto check=[&](Value v,const Point& p) {
    const auto& slots=a.slots(v);
    for (size_t j=0;j<N;++j) {
      expect(contains(slots[j].enclosure,p[j]) && contains(slots[j].naive,p[j]),"sample outside node/slot bounds");
      ++checks;
    }
  };
  for (size_t trial=0;trial<50000;++trial) {
    Point p;
    if (trial<corners.size()) p=corners[trial];
    else if (trial<2*corners.size()) {
      p=corners[trial-corners.size()]; for (auto& t:p) t=std::nextafter(t,0.L);
    } else {
      for (size_t j=0;j<N;++j) p[j]=rad[j]*(2.L*std::ldexp(static_cast<long double>(gen()>>11),-53)-1);
    }
    check(input,p);
    for (const auto& l:net) {
      Point pre{},act{},join{};
      for (size_t i=0;i<N;++i) {
        pre[i]=l.bias[i]; for (size_t j=0;j<N;++j) pre[i]+=l.w[i*N+j]*p[j];
        act[i]=reference(pre[i],l.f); join[i]=p[i]+act[i];
      }
      check(l.pre,pre); check(l.act,act); check(l.join,join); p=join;
    }
  }
  std::cout<<"seed="<<seed<<" samples=50000 corners=16 directed_corners=48 near_corners=64"
           <<" node_slot_checks="<<checks<<" violations=0 peak_symbols_per_slot="<<peakSymbols;
#if defined(__linux__)
  rusage ru{}; if (getrusage(RUSAGE_SELF,&ru)==0) std::cout<<" process_peak_rss_kib="<<ru.ru_maxrss;
#endif
  std::cout<<" (sampling is a counterexample search, not a proof)\n";
}
}  // namespace

int main() {
  try { handwritten(); condensation(); rounding(); nonlinear(); residualNetwork(); }
  catch (const std::exception& e) { std::cerr<<"AffineIntervalTest: "<<e.what()<<'\n'; return 1; }
  std::cout<<"AffineIntervalTest PASS\n";
}

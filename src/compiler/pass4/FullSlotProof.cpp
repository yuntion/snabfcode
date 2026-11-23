#include "FullSlotProof.h"

namespace snabf::pass4 {
namespace {
using R=ExactRational;
constexpr int N=16384;
const R eps(1,4096), cutMag(17,256), approx(1,196608);
void need(bool ok,const char *msg){if(!ok) throw std::invalid_argument(msg);}
R qi(int64_t n){return R(std::to_string(n));}
// Any ternary secret, including an embedded one, has |s(zeta)| <= n.
// Do not spend the source and target's separate probabilistic tail events.
R rnd(int n){return R(n)*(1+n)/2;}
R ks(const pass3::Solution &s,int n,int l){
  R sum=0;for(int i=0;i<=l;++i) sum+=qi(s.q.at(i))+1;
  return 10*R(n)*n*sum/qi(s.p.at(0));
}
R rs(const R &m,const R &q){return pass2::roundCKKSScale(pass2::roundCKKSScale(m*q)/q);}
}

FullSlotShape analyzeFullSlot(const FullSlotWeights &w,int s){
  need(s>0 && s<=N/4 && std::has_single_bit(unsigned(s)) && w.inputWidth>s && w.outputWidth>s &&
       w.inputWidth<=2*s && w.outputWidth<=2*s && !w.diagonals.empty(),"invalid full-slot extents");
  std::vector<R> norms(2*s); FullSlotShape a{}; std::array<std::array<std::set<int>,2>,2> keys;
  for(const auto &[k,v]:w.diagonals){
    need(k>=0 && k<2*s && int(v.size())==2*s,"invalid original diagonal"); bool live=false;
    for(int r=0;r<2*s;++r){
      int c=(r+k)%(2*s);need(std::isfinite(v[r]),"nonfinite original weight");
      need(std::abs(v[r])<=1,"unsupported public weight magnitude");
      if(!v[r]) continue;
      need(r<w.outputWidth && c<w.inputWidth,"nonzero weight in padding");
      live=true; norms[r]+=abs(pass2::exactBinary(v[r])); keys[r/s][c/s].insert(k%s);
    }
    need(live,"noncanonical empty source diagonal"); ++a.sourceTerms;
  }
  a.norm=*std::max_element(norms.begin(),norms.end());
  for(int o=0;o<2;++o){int count=0;for(int i=0;i<2;++i){
    a.childOffsets[o][i].assign(keys[o][i].begin(),keys[o][i].end()); count+=int(keys[o][i].size());
  } need(count>0,"empty output half unsupported"); a.maxChildTerms=std::max(a.maxChildTerms,count);}
  return a;
}

pass3::Graph neighborSiluPrefix(){
  using O=pass3::Op;
  // |u|<=1/8. SiLU(u)=u*(1+tanh(u/2))/2; integrating tanh(t)^2<=t^2
  // gives |SiLU(u)-u*(1/2+u/4)|<=|u|^4/48, also at padded boundaries.
  pass3::Graph g;g.n=N;g.levels=6;g.sink=11;g.approximation=pass2::upperDouble(approx);
  g.edges={{"x",O::Source,5,{},{},.5},{"r",O::Baby,5,{0},{1},.5},
    {"x+r",O::Add,5,{0,1},{1,1},1},{"/8",O::Mask,5,{2},{.125},.125,5},
    {"u",O::Rescale,4,{3},{1},.125},{"/4",O::Mask,4,{4},{.25},.03125,4},
    {"v0",O::Rescale,3,{5},{1},.03125},{"v",O::AddPlain,3,{6},{1},.53125},
    {"u.drop",O::Drop,3,{4},{1},.125},{"uv",O::Multiply,3,{8,7},{.53125,.125},.06640625},
    {"rl",O::Relin,3,{9},{1},.06640625},{"cut",O::Rescale,2,{10},{1},.06640625}};
  g.edges[1].shift=1;return g;
}

FullSlotMargin fullSlotMargin(const FullSlotShape &a,const pass3::Solution &s,bool split){
  need(s.q.size()==6 && s.p.size()==1 && s.scale>=36 && s.scale<=49,"unsupported full-slot basis");
  ScaleDomain domain{"source",N,s.scale,{s.q.begin(),s.q.end()},{s.p.begin(),s.p.end()}};domain.check(5);
  for(size_t i=1;i<s.q.size();++i) need(s.q[i]<(1LL<<53),"mask scale not exact in binary64");
  const R M(mpz_class(1)<<s.scale), P=qi(s.p[0]),rn=rnd(N),secret=N;
  std::vector<R> q,mod;R prod=1;for(auto v:s.q){q.push_back(qi(v));prod*=qi(v);mod.push_back(prod);}
  bool room=true;auto fit=[&](int level,R scale,R bound){room=room && 2*scale*bound<mod.at(level);};
  R ex=(rn+N+20*R(N)*(N+1+secret)/P)/M;
  fit(5,M,R(1,2)+ex);
  R sum=2*ex+(ks(s,N,5)+rn)/M;fit(5,M,1+sum);
  R u=sum/8+(1+sum)*N/q[5];fit(5,M*q[5],R(1,8)+u);u+=rn/M;fit(4,M,R(1,8)+u);
  R v=u/4+(R(1,8)+u)*N/q[4];fit(4,M*q[4],R(1,32)+v);v+=(rn+N)/M;fit(3,M,R(17,32)+v);
  fit(3,M,R(1,8)+u); // Drop is not a rescale.
  R ec=R(17,32)*u+R(1,8)*v+u*v+(ks(s,N,3)+rn)/(M*M);fit(3,M*M,cutMag+ec);
  const R G=M*M/q[3], cm=pass2::roundCKKSScale(G);ec+=rn/G;fit(2,G,cutMag+ec);
  // Store error in ideal effective-scale units until the final physical decode.
  // All q used as mask scales are exactly representable in binary64. At a
  // fold output the two rotated mask entries are complementary, so the ideal
  // entrance gain is one. Both mask errors, rescale rounds and KS survive.
  R fresh=0,gain=1,meta=cm,down=0;
  if(split){
    R de=R(N)/q[2], key=(ks(s,N,1)+rn)/G;
    fresh=2*cutMag*de+2*rn/G+2*key;gain=1+2*de;down=key;
    fit(2,G*q[2],(cutMag+ec)*(1+de));fit(1,G,cutMag+fresh+gain*ec);meta=rs(meta,q[2]);
  }
  const int n=split?N/2:N,level=split?1:2,L=split?a.maxChildTerms:a.sourceTerms;
  const R delta=R(n)/q[level], key=(ks(s,n,level)+rnd(n))/G;
  // All route terms are charged one possible baby and one possible giant.
  // No cancellation between quadrants, groups, encodings or shared keys.
  fit(level,G,cutMag+fresh+gain*ec+key);
  R f=a.norm*(fresh+key)+L*delta*(cutMag+fresh+key)+L*key/q[level];
  R h=(a.norm+L*delta)*gain;fit(level,G*q[level],a.norm*cutMag+f+h*ec);
  f+=rnd(n)/G;fit(level-1,G,a.norm*cutMag+f+h*ec);
  const R linearInput=meta;meta=rs(meta,q[level]);R ratio=G/meta;
  f=ratio*f+abs(ratio-1)*a.norm*cutMag+a.norm*approx;h*=ratio;
  // gap is the maximum arithmetic error permitted at the prefix result,
  // not an observed margin and not a security or probability certificate.
  R gap=(eps-f)/h;
  return {cm,linearInput,meta,ec,f,h,gap,f+h*ec,down,room};
}

FullSlotFeedback solveFullSlot(const FullSlotShape &shape,int cap){
  need(cap>=256 && cap<=512,"invalid source cap");
  auto g=neighborSiluPrefix();
  pass3::Options o;
  o.maxScale=49;
  o.maxK=0;
  o.alpha=1;
  o.minBase=48;
  o.securityCap=cap;
  // A nonpositive suffix-only remainder cannot be repaired by shrinking
  // just the prefix error. Try a different basis within the same epsilon/cap.
  std::vector<std::string> rejected;
  for(int floor=40;floor<=46;++floor) for(int margin:{2,4,6}){
    o.minScale=floor;
    o.auxMarginBits=margin;
    auto reject=[&](const char *why){
      rejected.push_back(std::to_string(floor)+":"+std::to_string(margin)+":"+why);
    };
    FullSlotFeedback r{};
    r.initial=pass3::optimize(g,o);
    r.marginBits=margin;
    r.oldKeep=fullSlotMargin(shape,r.initial,false);
    r.oldDown=fullSlotMargin(shape,r.initial,true);
    R b=std::min(r.oldKeep.budget,r.oldDown.budget);
    if(b<=0){reject("nonpositive-suffix-budget");continue;}

    // Round the submitted double DOWN. A row transferred to a new basis only
    // proposes parameters. Both complete physical ledgers are rederived below.
    r.submittedBudget=std::nextafter(b.get_d(),0.);
    const pass3::SourceCutBound row{{{g.sink,1.}},r.submittedBudget};
    auto sol=pass3::optimizeForCuts(g,o,{row},r.stats);
    if(!sol || !r.stats.modelsWithCuts || !r.stats.solverRuns){reject("source-row-rejected");continue;}
    r.basis=*sol;
    r.keep=fullSlotMargin(shape,*sol,false);
    r.down=fullSlotMargin(shape,*sol,true);
    if(!r.keep.headroom || !r.down.headroom || r.keep.total>=eps || r.down.total>=eps){
      reject("physical-ledger-rejected");continue;
    }
    r.rejected=std::move(rejected);
    return r;
  }
  throw std::invalid_argument("no physically replayed source/suffix pair in finite full-slot search");
}
} // namespace snabf::pass4

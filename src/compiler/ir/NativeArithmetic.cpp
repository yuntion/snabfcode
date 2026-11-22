#include "NativeArithmetic.h"
#include "AffineIntervalAnalyzer.h"
#include "RouteSelection.h"
#include "pass3/ConcreteVerifier.h"
#include "pass3/ParameterMILP.h"
#include "llvm/ADT/APFloat.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace snabf::native {
namespace {
namespace af = ir::affine;
using O = pass3::Op;
using V = af::AffineIntervalAnalyzer::Value;
using Rows = std::vector<double>;
void need(bool ok, const char *why) { if (!ok) throw std::invalid_argument(why); }
double up(double a, double b, bool times) {
  llvm::APFloat x(a);
  if (times) x.multiply(llvm::APFloat(b), llvm::APFloat::rmTowardPositive);
  else x.add(llvm::APFloat(b), llvm::APFloat::rmTowardPositive);
  need(x.isFinite(), "overflow in decoded message bound");
  return x.convertToDouble();
}
double add(double a,double b) { return up(a,b,false); }
double mul(double a,double b) { return up(a,b,true); }
double absmax(Rows const &r) { double m=0; for(double x:r) m=std::max(m,std::abs(x)); return m; }
template<class I> double radius(I i) { return std::max(std::abs(i.lo),std::abs(i.hi)); }
double word(uint64_t bits, unsigned width) {
  return width==32 ? static_cast<double>(std::bit_cast<float>(uint32_t(bits))) : std::bit_cast<double>(bits);
}
struct Data { int e; V value; double approximation; };
struct Builder {
  NativeArithmeticProgram &p;
  const ArithmeticOptions &config;
  af::AffineIntervalAnalyzer af;
  std::map<std::string,Data> data;
  size_t site=0;
  const std::vector<size_t> *choices=nullptr;
  std::vector<size_t> counts;
  int edge(std::string id, O op, int level, std::vector<int> args, Rows gains, double magnitude) {
    need(level>=0 && level<p.graph.levels, "insufficient active Q levels");
    need(std::isfinite(magnitude) && magnitude>=0,"invalid message enclosure");
    int i=p.graph.edges.size();
    p.graph.edges.push_back({std::move(id),op,level,std::move(args),std::move(gains),magnitude});
    return i;
  }
  int level(int e) const { return p.graph.edges.at(e).level; }
  double mag(int e) const { return p.graph.edges.at(e).magnitude; }
  int drop(int e,int l,const std::string &id) {
    need(level(e)>=l,"cannot raise level at join");
    return level(e)==l?e:edge(id,O::Drop,l,{e},{1},mag(e));
  }
  int sum(const std::string &id, const std::vector<int> &xs, double bound) {
    need(!xs.empty(),"empty arithmetic sum");
    return xs.size()==1?xs.front():edge(id,O::Add,level(xs.front()),xs,Rows(xs.size(),1),bound);
  }
  void bind(const std::string &id,Data v) {
    need(data.emplace(id,v).second,"duplicate encrypted SSA value");
    p.sourceValues.emplace(id,v.e);
  }
  Data linear(const std::string &id,Data x) {
    const auto &m=p.source.matvecs.at(site);
    need(m.resultID==id,"route not bound to source matvec");
    int l=level(x.e); need(l>=1,"no level for linear rescale");
    size_t choice=p.routeIndices.at(site), w=p.source.width;
    const auto &route=m.candidates.at(choice);
    auto const &slots=af.slots(x.value);
    std::vector<int64_t> offsets;
    Rows table, rowNorm(w), finalBound(w);
    for (auto &[d,diag]:m.diagonals) {
      offsets.push_back(d%w);
      for(size_t j=0;j<w;++j) {
        double weight=word(diag.rowBits.at(j),m.elementBits);
        table.push_back(weight); rowNorm[j]=add(rowNorm[j],std::abs(weight));
      }
    }
    if(offsets.empty()) {offsets.push_back(0); table.resize(w);}
    V result=af.periodicDiagonals(x.value,w,offsets,table);
    std::map<unsigned,int> babies{{0,x.e}};
    struct Group { std::vector<int> terms; Rows bounds; };
    std::map<unsigned,Group> groups;
    std::vector<uint64_t> bits(p.source.domain.encodedSlots);
    auto rotation=[&](int parent,unsigned shift,O op) {
      int e=edge(id+(op==O::Baby?".baby":".giant")+std::to_string(shift),op,l,{parent},{1},mag(parent));
      p.graph.edges[e].shift=shift; p.graph.edges[e].routeSite=site;
      return e;
    };
    // Empty support is an actual zero plaintext product (including its rescale).
    for(size_t i=0;i<std::max<size_t>(1,route.terms.size());++i) {
      RotationTerm t{0,0,0};
      if(!route.terms.empty()) {
        t=route.terms[i];
        need(mlir::succeeded(p.source.materializeMask(site,choice,i,bits)),"invalid selected rotation/mask");
      }
      if(!babies.contains(t.baby)) babies[t.baby]=rotation(x.e,t.baby,O::Baby);
      auto &g=groups[t.giant]; if(g.bounds.empty()) g.bounds.resize(w);
      Rows mask(w),term(w);
      for(size_t j=0;j<w;++j) {
        mask[j]=word(bits[j],m.elementBits);
        term[j]=mul(std::abs(mask[j]),radius(slots[(j+t.baby)%w].enclosure));
        g.bounds[j]=add(g.bounds[j],term[j]);
      }
      int e=edge(id+".d"+std::to_string(t.diagonal),O::Mask,l,{babies.at(t.baby)},{absmax(mask)},absmax(term));
      p.graph.edges[e].maskLevel=l;
      p.publicOperands.emplace(e,std::move(mask)); g.terms.push_back(e);
    }
    std::vector<int> giants;
    for(auto &[shift,g]:groups) {
      int e=sum(id+".coset"+std::to_string(shift),g.terms,absmax(g.bounds));
      giants.push_back(shift?rotation(e,shift,O::Giant):e);
      for(size_t j=0;j<w;++j) finalBound[j]=add(finalBound[j],g.bounds[(j+shift)%w]);
    }
    double bound=std::min(absmax(finalBound),radius(af.domain(result)));
    int merged=sum(id+".merge",giants,bound);
    ++site;
    return {edge(id,O::Rescale,l-1,{merged},{1},bound),result,mul(absmax(rowNorm),x.approximation)};
  }
  Data activation(const std::string &id, Data x, const llvm::json::Object &lock) {
    auto in=af.domain(x.value);
    const auto &decl=*lock.getArray("declared_domain");
    need(in.lo>=*decl[0].getAsNumber() && in.hi<=*decl[1].getAsNumber(),"computed activation domain exceeds source declaration");
    bool gelu=lock.getString("kind")=="gelu_exact_erf";
    const std::string key=gelu?"gelu":"silu";
    auto opt=config.polynomial;
    opt.inputLevel=level(x.e); opt.maxDepth=std::min(opt.maxDepth,opt.inputLevel);
    opt.maxError=config.activationTolerance; opt.inputError=0;
    auto dp=pass2::synthesizePolynomials(key,{in.lo,in.hi},opt);
    need(!dp.frontier.empty(),"no certified polynomial at available depth/tolerance");
    auto rank=[](const auto &c){return std::tie(c.cost.depth,c.cost.ctMults,c.cost.ptMults,c.approximationError,c.id);};
    std::sort(dp.frontier.begin(),dp.frontier.end(),[&](const auto &a,const auto &b){return rank(a)<rank(b);});
    int pi=p.polynomials.size(); counts.push_back(dp.frontier.size());
    size_t pick=choices && size_t(pi)<choices->size()?choices->at(pi):0;
    p.polynomials.push_back(std::move(dp.frontier.at(pick)));
    const auto &c=p.polynomials.back();
    double local=c.approximationError;
    if(gelu) {
      // The imported 1/sqrt(2) is the correctly rounded SOURCE literal. The
      // ideal erf fit uses the real constant. |erf'| <= 2 and half an ULP
      // at 1/sqrt(2) is 2^-25 (f32) or 2^-54 (f64).
      bool found=false;
      for(const auto &v:*lock.getArray("literals")) {
        const auto &lit=*v.getAsObject(); if(lit.getString("role")!="inv_sqrt2") continue;
        local=add(local,mul(mul(radius(in),radius(in)),lit.getString("type")=="f32"?0x1p-25:0x1p-54)); found=true;
      }
      need(found,"unbound erf literal");
    }
    auto value=af.activate(x.value,gelu?af::Activation::ErfGeLU:af::Activation::SiLU,local);
    std::vector<int> values;
    pass3::Poly poly{level(x.e),int(c.dag.output),{}};
    for(size_t i=0;i<c.dag.nodes.size();++i) {
      const auto &n=c.dag.nodes[i]; const auto prefix=id+".p"+std::to_string(i)+".";
      bool lin=n.op=="linear", product=n.op=="mul";
      int end=poly.level-int(n.depth),start=end+(lin||product);
      std::vector<int> args, refs(n.inputs.begin(),n.inputs.end());
      for(int ref:refs) args.push_back(drop(values.at(ref),start,prefix+"drop"+std::to_string(args.size())));
      poly.nodes.push_back({n.op,refs,int(n.depth)});
      int out=x.e;
      if(product) {
        int a=args.at(0),b=args.at(1);
        out=edge(prefix+"mul",O::Multiply,start,{a,b},{mag(b),mag(a)},mul(mag(a),mag(b)));
        out=edge(prefix+"relin",O::Relin,start,{out},{1},mag(out));
      } else if(lin || n.op=="add") {
        std::vector<int> terms; double bound=0;
        for(size_t j=0;j<args.size();++j) {
          int t=args[j];
          if(lin) {
            double weight=n.weights.at(j), a=std::abs(weight);
            t=edge(prefix+"coef"+std::to_string(j),O::Coefficient,start,{t},{a},mul(mag(t),a));
            p.graph.edges[t].poly=pi; p.graph.edges[t].node=i;
            p.publicOperands.emplace(t,Rows{weight});
          }
          terms.push_back(t); bound=add(bound,mag(t));
        }
        out=sum(prefix+"sum",terms,bound);
      } else need(n.op=="input" && i==0,"unknown polynomial primitive");
      if(lin||product) out=edge(prefix+"rs",O::Rescale,end,{out},{1},mag(out));
      if(n.constant!=0) {
        out=edge(prefix+"constant",O::AddPlain,end,{out},{1},add(mag(out),std::abs(n.constant)));
        p.publicOperands.emplace(out,Rows{n.constant});
      }
      // Only narrow a node this instruction owns, never a reused SSA input.
      if(out!=x.e && (product||lin||n.constant!=0||args.size()>1))
        p.graph.edges[out].magnitude=std::min(mag(out),radius(c.noise.nodes.at(i).range));
      values.push_back(out);
    }
    p.graph.polys.push_back(std::move(poly));
    // Both exact reference activations have |f'| <= 1+|x| (no claim of
    // independence between this approximation and a residual bypass).
    return {values.at(c.dag.output),value,add(local,mul(add(1,radius(in)),x.approximation))};
  }
  void build() {
    auto &s=p.source; const auto &ins=s.inspection;
    const auto &port=*ins.getArray("inputs")->front().getAsObject();
    auto *dom=port.getArray("domain"); need(dom,"secret input has no bound domain");
    af::Interval d{*(*dom)[0].getAsNumber(),*(*dom)[1].getAsNumber()};
    p.graph.n=1<<s.domain.logN; p.graph.levels=config.qCount;
    auto v=af.source(std::vector<af::Interval>(s.width,d));
    bind("arg0",{edge("arg0",O::Source,config.qCount-1,{},{},radius(d)),v,0});
    std::map<std::string,const llvm::json::Object*> locks;
    for(auto &j:*ins.getArray("activation_locks")) {auto &l=*j.getAsObject();locks.emplace(l.getString("site")->str(),&l);}
    std::vector<LinearRegion> regions; KeySizes units;
    for(const auto &m:s.matvecs) {
      regions.push_back({"source",1,1,m.candidates});
      for(auto &r:m.candidates) for(auto key:r.galoisKeys) units[{"source",key}]=1;
    }
    p.routeIndices=selectSubgroupRoutes(regions,units,1).selected;
    for(auto &j:*ins.getArray("nodes")) {
      const auto &n=*j.getAsObject(); auto id=n.getString("id")->str(),kind=n.getString("kind")->str();
      if(kind=="constant"||kind=="empty"||kind=="zero_fill") continue;
      const auto &args=*n.getArray("inputs"); auto arg=[&](size_t i){return args[i].getAsString()->str();};
      if(kind=="matvec") bind(id,linear(id,data.at(arg(1))));
      else if(kind=="silu"||kind=="gelu_exact_erf") {
        const auto &lock=*locks.at(id); need(lock.getString("x")==arg(0),"unbound activation SSA input");
        bind(id,activation(id,data.at(arg(0)),lock));
      } else if(kind=="add" && data.contains(arg(0)) && data.contains(arg(1))) {
        Data a=data.at(arg(0)), b=data.at(arg(1)); int l=std::min(level(a.e),level(b.e));
        int lhs=drop(a.e,l,id+".lhs"),rhs=drop(b.e,l,id+".rhs");
        V y=af.add(a.value,b.value);
        bind(id,{edge(id,O::Add,l,{lhs,rhs},{1,1},radius(af.domain(y))),y,add(a.approximation,b.approximation)});
      } else throw std::invalid_argument("unsupported encrypted source operation: "+kind);
    }
    need(site==s.matvecs.size(),"unused source route");
    auto out=data.at(ins.getArray("outputs")->front().getAsObject()->getString("value")->str());
    p.graph.sink=out.e; p.graph.approximation=out.approximation;
  }
};
} // namespace

mlir::LogicalResult deriveNativeArithmetic(llvm::StringRef bytes,llvm::StringRef name,llvm::StringRef spec,
    const ArithmeticOptions &config,NativeArithmeticProgram &out,llvm::raw_ostream &diagnostics) {
  try {
    need(config.qCount>0 && config.qCount<=64 && config.activationTolerance>0 && std::isfinite(config.activationTolerance),"invalid Q count/activation tolerance");
    NativeArithmeticProgram p;
    if(mlir::failed(deriveNativeLinear(bytes,name,spec,config.domain,p.source,diagnostics))) return mlir::failure();
    Builder builder{p,config}; builder.build();
    out=std::move(p); return mlir::success();
  } catch(const std::exception &e) {
    diagnostics<<name<<": error: [SNABF_NATIVE_ARITHMETIC] "<<e.what()<<"\n"; return mlir::failure();
  }
}
pass3::Solution searchNativeParameters(const NativeArithmeticProgram &p,const pass3::Options &options) {
  auto candidate=pass3::optimize(p.graph,options);
  auto result=pass3::replay(p.graph,options,candidate);
  if(!result.reason.empty()) throw std::invalid_argument("native concrete replay: "+result.reason);
  return candidate;
}

mlir::LogicalResult specializeNativeArithmetic(llvm::StringRef bytes,llvm::StringRef name,llvm::StringRef spec,
    const ArithmeticOptions &config,const pass3::Options &options,size_t limit,NativeArithmeticProgram &out,
    pass3::Solution &solution,size_t &attempts,llvm::raw_ostream &diagnostics) {
  try {
    need(limit>0 && config.qCount>0 && config.qCount<=64 && config.activationTolerance>0 && std::isfinite(config.activationTolerance),"invalid native search budget");
    NativeLinearProgram source;
    if(mlir::failed(deriveNativeLinear(bytes,name,spec,config.domain,source,diagnostics))) return mlir::failure();
    std::vector<size_t> choices;
    for(size_t n=0;n<limit;++n) {
      NativeArithmeticProgram p; p.source=std::move(source);
      Builder b{p,config}; b.choices=&choices;
      try {
        b.build();
        if(p.graph.approximation<options.tolerance) {
          auto s=searchNativeParameters(p,options);
          out=std::move(p); solution=std::move(s); attempts=n+1; return mlir::success();
        }
      } catch(const std::exception &e) {
        llvm::StringRef why=e.what();
        if(why!="Pass 3 infeasible under requested scale, precision and QP bounds" &&
           why!="no certified polynomial at available depth/tolerance" &&
           why!="no level for linear rescale" && why!="computed activation domain exceeds source declaration") throw;
        if(b.counts.empty()) throw;
      }
      source=std::move(p.source);
      choices.resize(b.counts.size());
      while(!choices.empty()) {
        const size_t i=choices.size()-1;
        if(++choices[i]<b.counts[i]) break;
        choices.pop_back();
      }
      if(choices.empty()) break;
    }
    throw std::runtime_error("no concrete native schedule within bounded certified search");
  } catch(const std::exception &e) {
    diagnostics<<name<<": error: [SNABF_NATIVE_ARITHMETIC] "<<e.what()<<"\n"; return mlir::failure();
  }
}
} // namespace snabf::native

#include "NativeArithmetic.h"
#include "NativeReluIsolation.h"
#include "pass3/ConcreteVerifier.h"
#include "Target/Lattigo/NativeSingleGo.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cmath>
#include <stdexcept>

using namespace snabf;
using namespace snabf::native;
namespace {
using Rows=std::vector<double>;
void check(bool ok,const char *reason){if(!ok)throw std::runtime_error(reason);}
// Evaluate graph *edges*, not PolyCandidate::evaluateDAG or matvec source
// values. The independent Python oracle uses the original dense row-major W.
Rows run(const NativeArithmeticProgram &p,int seed) {
  using O=pass3::Op;
  size_t w=p.source.width;
  std::vector<Rows> values;
  for(size_t i=0;i<p.graph.edges.size();++i) {
    auto &e=p.graph.edges[i]; Rows v(w);
    for(size_t j=0;j<w;++j) {
      auto a=[&](int t){return values.at(e.inputs.at(t)).at(j);};
      auto pt=[&](){auto &r=p.publicOperands.at(i);return r.at(j%r.size());};
      switch(e.op) {
        case O::Source: v[j]=((int(j)*17+seed*23)%101-50)/50.; break;
        case O::Drop: case O::Rescale: case O::Relin: v[j]=a(0); break;
        case O::Mask: case O::Coefficient: v[j]=a(0)*pt(); break;
        case O::AddPlain: v[j]=a(0)+pt(); break;
        case O::Add: for(size_t k=0;k<e.inputs.size();++k)v[j]+=a(k); break;
        case O::Multiply: v[j]=a(0)*a(1); break;
        case O::Baby: case O::Giant: v[j]=values.at(e.inputs.at(0)).at((j+e.shift)%w); break;
        default: throw std::runtime_error("unsupported semantic evaluator edge");
      }
      check(std::abs(v[j])<=e.magnitude+1e-13,"an edge exceeded its decoded bound");
    }
    values.push_back(std::move(v));
  }
  return values.at(p.graph.sink);
}
}
int main(int argc,char **argv) {
  try {
    if(argc!=3 && argc!=4 && argc!=5 && argc!=6) return 2;
    auto bytes=llvm::MemoryBuffer::getFile(argv[1]),spec=llvm::MemoryBuffer::getFile(argv[2]);
    if(!bytes||!spec) return 2;
    ArithmeticOptions opt; opt.domain={14,4096};
    // Bounded, reproducible production DP search; no frozen plan or Q/P.
    opt.polynomial.degrees={3,5,7}; opt.polynomial.composed=false;
    NativeArithmeticProgram p;
    pass3::Options o; o.maxK=0; // Native exact-arithmetic arm; do not claim a truncation kernel.
    opt.activationTolerance=o.tolerance;
    if((argc==4 && (llvm::StringRef(argv[3])=="--relu-contract" || llvm::StringRef(argv[3])=="--relu-isolate")) ||
       (argc==5 && llvm::StringRef(argv[3])=="--relu-recheck")) {
      configureNativeRelu(opt,o);
      if(mlir::failed(deriveNativeArithmetic((*bytes)->getBuffer(),argv[1],(*spec)->getBuffer(),opt,p,llvm::errs())))return 1;
      if(llvm::StringRef(argv[3])=="--relu-isolate") {
        llvm::outs()<<llvm::formatv("{0}\n",llvm::json::Value(test::isolate(p,o)));return 0;
      }
      if(argc==5) {
        auto file=llvm::MemoryBuffer::getFile(argv[4]); if(!file)return 2;
        auto claim=llvm::json::parse((*file)->getBuffer());
        if(!claim)throw std::invalid_argument(llvm::toString(claim.takeError()));
        checkNativeReluWitness(p,claim->getAsObject());return 0;
      }
      auto w=nativeReluWitness(p); auto candidate=nativeReluFixedBasis(p); llvm::json::Array q,aux;
      for(auto x:candidate.q)q.push_back(x); for(auto x:candidate.p)aux.push_back(x);
      std::string why;
      try { why=pass3::replay(p.graph,o,candidate).reason; } catch(const std::exception &e){why=e.what();}
      w["first_arm"]=llvm::json::Object{{"ring",p.graph.n},{"q",std::move(q)},{"p",std::move(aux)},
        {"scale",candidate.scale},{"security_cap",o.securityCap},{"replay_refusal",why}};
      if(candidate.errors.size()<p.graph.edges.size() && candidate.physical.size()==p.graph.edges.size()) {
        const size_t i=candidate.errors.size(); const auto &e=p.graph.edges[i];
        if(e.op==pass3::Op::Coefficient && why=="arithmetic allowance exceeded at "+e.id) {
          auto parent=e.inputs.at(0); double pe=candidate.errors.at(parent);
          double inherited=pe*e.gains.at(0);
          double encoding=(p.graph.edges.at(parent).magnitude+pe)*p.graph.n*std::exp2(candidate.physical[parent])/std::exp2(candidate.physical[i]);
          w["first_blocked_edge"]=llvm::json::Object{{"id",e.id},{"parent",p.graph.edges.at(parent).id},
            {"parent_error",pe},{"gain",e.gains.at(0)},{"inherited",inherited},{"encoding",encoding},{"internal_limit",8*o.tolerance}};
        }
      }
      w["approximation"]=p.graph.approximation; w["sink_level"]=p.graph.edges.at(p.graph.sink).level;
      llvm::outs()<<llvm::formatv("{0}\n",llvm::json::Value(std::move(w)));return 0;
    }
    pass3::Solution s; size_t attempts=0;
    if(mlir::failed(specializeNativeArithmetic((*bytes)->getBuffer(),argv[1],(*spec)->getBuffer(),opt,o,32,p,s,attempts,llvm::errs())))return 1;
    check(s.errors.at(p.graph.sink)+p.graph.approximation<=o.tolerance,"concrete error budget");
    const ExactRational joinBudget=ExactRational(1)/(mpz_class(1)<<40); // decoded absolute allocation, NOT a tail probability
    if(argc==4 && llvm::StringRef(argv[3])=="--physical") {
      llvm::outs()<<llvm::formatv("{0}\n",llvm::json::Value(lowerNativePhysical(p,s,o,joinBudget)));return 0;
    }
    if(argc==6) {
      auto plan=llvm::MemoryBuffer::getFile(argv[4]),sec=llvm::MemoryBuffer::getFile(argv[5]);
      if(!plan||!sec)return 2;
      auto text=(*plan)->getBuffer(),proof=(*sec)->getBuffer();
      if(llvm::StringRef(argv[3])=="--check")checkNativeSingle(p,s,o,joinBudget,text,proof);
      else if(llvm::StringRef(argv[3])=="--emit")llvm::outs()<<emitNativeSingleGo(p,s,o,joinBudget,text,proof);
      else if(llvm::StringRef(argv[3])=="--runtime") {checkNativeSingle(p,s,o,joinBudget,text,proof);llvm::outs()<<nativeSingleRuntimeSource();}
      else return 2;
      return 0;
    }
    if(argc!=3)return 2;
    llvm::json::Array runs, polys, q, aux;
    for(auto v:s.q)q.push_back(v); for(auto v:s.p)aux.push_back(v);
    for(int seed=0;seed<8;++seed){llvm::json::Array y;for(double x:run(p,seed))y.push_back(x);runs.push_back(std::move(y));}
    for(auto &c:p.polynomials)polys.push_back(c.templateName);
    llvm::json::Object result{{"source",p.source.inspection.getString("source_sha256")->str()},
      {"outputs",std::move(runs)},{"polys",std::move(polys)},{"approximation",p.graph.approximation},
      {"edges",int64_t(p.graph.edges.size())},{"sink_level",p.graph.edges.at(p.graph.sink).level},
      {"arithmetic_bound",s.errors.at(p.graph.sink)}, {"Q_count",int64_t(s.q.size())},
      {"q",std::move(q)},{"p",std::move(aux)},{"ring",p.graph.n},{"security_cap",o.securityCap},
      {"attempts",int64_t(attempts)}};
    llvm::outs()<<llvm::formatv("{0}\n",llvm::json::Value(std::move(result)));
    return 0;
  } catch(const std::exception &e){ llvm::errs()<<e.what()<<"\n";return 1; }
}

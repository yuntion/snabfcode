#include "FullSlotProduct.h"
#include "FullSlotProof.h"
#include "RouteSelection.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <filesystem>

namespace snabf::pass4 {
namespace {
using namespace llvm;
using O=json::Object; using A=json::Array; using V=json::Value;
using mlir::ModuleOp;
constexpr StringRef src="snabf.a2b_source",arm="snabf.a2b_arm",stage="snabf.a2b_stage",
                    plan="snabf.a2b_plan",seal="snabf.a2b_seal";
constexpr int n=16384,s=4096,cap=352;
constexpr StringRef schema="snabf.a2b.neighbor-silu-linear.v1";
void need(bool x,StringRef why){if(!x)throw std::invalid_argument(why.str());}
std::string sha(StringRef x){SHA256 h;h.update(x);return toHex(h.final(),true);}
std::string dump(const O &o){return formatv("{0}",V(O(o))).str();}
V parse(StringRef text){auto v=json::parse(text);if(!v)throw std::invalid_argument(toString(v.takeError()));return std::move(*v);}
const O &obj(const V &v){auto *o=v.getAsObject();need(o,"expected full-slot object");return *o;}
int integer(const O &o,StringRef f,int lo,int hi){auto x=o.getInteger(f);need(x && *x>=lo && *x<=hi,"invalid full-slot integer");return int(*x);}
StringRef field(ModuleOp m,StringRef k){auto v=m->getAttrOfType<mlir::StringAttr>(k);need(bool(v),"missing full-slot field");return v.getValue();}
void set(ModuleOp m,StringRef k,StringRef v){m->setAttr(k,mlir::StringAttr::get(m.getContext(),v));}
void carrier(ModuleOp m,StringRef state){
  std::set<StringRef> fields{src,arm,stage};if(state!="source")fields.insert(plan);if(state=="checked")fields.insert(seal);
  need(m.getBody()->empty() && field(m,stage)==state,"invalid full-slot body/stage");
  for(auto a:m->getAttrs())need(fields.erase(a.getName().getValue())==1,"foreign full-slot attribute");
  need(fields.empty(),"incomplete full-slot carrier");
}
template<class C>A array(const C &c){A a;for(const auto &x:c)a.push_back(x);return a;}
// Restrict each standard embedding to Y=X^2. Match the two representatives
// and their order to the independently constructed runtime slot map.
std::string slotMap(){
  static constexpr char tag[]="snabf.standard-ckks-half-slots.v1";
  SHA256 h;h.update(StringRef(tag,sizeof(tag)));
  auto word=[&](uint64_t x){uint8_t b[8];for(int i=0;i<8;++i)b[i]=uint8_t(x>>(8*i));h.update(ArrayRef(b));};
  const uint64_t root=2*n,small=n,fold=rotationElement(s,14);word(root);word(small);word(fold);
  std::set<uint64_t> seen;uint64_t a=1,d=1;
  for(int j=0;j<s;++j){uint64_t b=a*fold%root;need(a!=b && a%small==d && b%small==d && seen.insert(d).second,"nonbijective standard trace orbit");
    word(j);word(j+s);word(d);a=a*5%root;d=d*5%small;}
  return toHex(h.final(),true);
}
FullSlotWeights weights(StringRef raw){
  need(raw.size()<4000000,"full-slot source too large");auto v=parse(raw);const auto &o=obj(v);
  auto *ds=o.getArray("diagonals");auto *ops=o.getArray("ops");
  need(o.size()==7 && o.getString("schema")==schema && o.getString("input_linf")=="1/2" && o.getString("input_type")=="real-zero-padded" &&
       ops && *ops==A{"rotate(x,1)","u=(x+rot)/8","h=SiLU(u)","y=W*h"} && ds && !ds->empty() && ds->size()<=16,
       "unsupported full-slot original graph or input contract");
  FullSlotWeights w{integer(o,"input_width",s+1,2*s),integer(o,"output_width",s+1,2*s),{}};
  for(const auto &d:*ds){const auto &diag=obj(d);int k=integer(diag,"offset",0,2*s-1);auto *a=diag.getArray("w64");
    need(diag.size()==2 && !w.diagonals.contains(k) && a && a->size()==2*s,"invalid original diagonal");
    auto &vv=w.diagonals[k];for(auto &e:*a){auto x=e.getAsInteger();need(x && *x>=-64 && *x<=64,"weight not in signed /64 domain");vv.push_back(double(*x)/64.);}}
  return w;
}
O basis(const pass3::Solution &p){return O{{"q",array(p.q)},{"p",array(p.p)},{"d",p.scale}};}
O ledger(const FullSlotMargin &m){return O{{"cut_scale",m.cutScale.get_str()},{"linear_input_scale",m.linearInputScale.get_str()},{"sink_scale",m.sinkScale.get_str()},
  {"cut_error",m.cutError.get_str()},{"fresh",m.fresh.get_str()},{"gain",m.gain.get_str()},
  {"budget",m.budget.get_str()},{"total",m.total.get_str()},{"down_key_charge",m.downKeyCharge.get_str()},{"headroom",m.headroom}};}
struct Routes {A steps;std::set<uint64_t> elements;};
Routes routes(const std::vector<std::vector<int>> &diags,unsigned logN,unsigned level,StringRef scope){
  std::vector<LinearRegion> rs;KeySizes sizes;
  // Cost proxy in coefficient bytes (not serialized bytes or latency).
  const uint64_t coeffBytes=8ull*(1u<<logN)*2*(level+1)*(level+2);
  for(const auto &d:diags)if(!d.empty()){
    auto cs=executableRoutes(logN,{d.begin(),d.end()});for(auto &c:cs)for(auto k:c.galoisKeys)sizes[{scope.str(),k}]=coeffBytes;
    rs.push_back({scope.str(),(level+1)*.625,double(level+1),std::move(cs)});
  }
  auto sel=selectSubgroupRoutes(rs,sizes,1./(1<<20));Routes r;int j=0;
  for(const auto &d:diags){if(d.empty()){r.steps.push_back(nullptr);continue;}
    auto &c=rs.at(j).candidates.at(sel.selected.at(j));++j;A terms;
    for(auto t:c.terms)terms.push_back(A{t.diagonal,t.baby,t.giant});
    r.steps.push_back(O{{"Terms",std::move(terms)},{"Level",level},{"LogN",logN},{"LogSlots",logN-1}});
    r.elements.insert(c.galoisKeys.begin(),c.galoisKeys.end());}
  return r;
}
struct Physical {
  FullSlotWeights w;FullSlotShape shape;FullSlotFeedback f;O document;std::string sourceId,mode,selection;
  explicit Physical(ModuleOp m):w(weights(field(m,src))),shape(analyzeFullSlot(w,s)),f(solveFullSlot(shape,cap)),sourceId(sha(field(m,src))),mode(field(m,arm).str()){
    need(mode=="keep-source" || mode=="force-down" || mode=="adaptive","invalid full-slot arm");
    std::vector<int> parent;for(auto &[k,v]:w.diagonals)parent.push_back(k);
    auto keep=routes({parent},14,2,"source");std::vector<std::vector<int>> ds;for(auto &row:shape.childOffsets)for(auto &d:row)ds.push_back(d);
    auto down=routes(ds,13,1,"target");auto extra=keep.elements;extra.erase(rotationElement(1,14));
    auto words=[](int ring,int level){return int64_t(ring)*2*(level+1)*(level+2);};
    auto kw=int64_t(extra.size())*words(n,2),dw=2*words(n,1)+int64_t(down.elements.size())*words(n/2,1);
    selection=mode=="adaptive"?(dw<kw?"force-down":"keep-source"):mode;
    A children;for(int o=0;o<2;++o)for(int i=0;i<2;++i)children.push_back(O{{"out",o},{"in",i},{"route",std::move(down.steps[2*o+i])}});
    auto ports=[&](int width,int level,const std::string &scale){A z;for(int i=0;i<2;++i)z.push_back(O{
      {"key_domain","target"},{"n",n/2},{"start",s*i},{"used",std::min(s,width-i*s)},
      {"level",level},{"q",A{f.basis.q[0],f.basis.q[1]}},{"active_q",array(std::span(f.basis.q).first(level+1))},{"p",array(f.basis.p)},{"scale",scale}});return z;};
    document=O{{"schema","snabf.a2b.fullslot-plan.v1"},{"source_sha256",sourceId},{"arm",mode},{"selected",selection},
      {"basis",basis(f.basis)},{"source_n",n},{"target_n",n/2},{"cut_level",2},{"input_width",w.inputWidth},{"output_width",w.outputWidth},
      {"fold",O{{"from","source"},{"to","target"},{"offsets",A{s,s}},{"mask_halves",A{0,1}},
         {"galois",rotationElement(s,14)},{"slot_map_sha256",slotMap()},{"q_level",1},{"p_level",0},{"projection","normalized-even-trace"}}},
      {"split_outputs",ports(w.inputWidth,1,f.down.linearInputScale.get_str())},{"outputs_down",ports(w.outputWidth,0,f.down.sinkScale.get_str())},
      {"keep_route",std::move(keep.steps[0])},{"children",std::move(children)},{"keep_galois",array(keep.elements)},{"target_galois",array(down.elements)},
      {"preference",O{{"metric","marginal-expanded-key-coefficients"},{"keep",kw},{"down",dw}}},
      {"keep",ledger(f.keep)},{"down",ledger(f.down)},{"feedback",O{{"initial",basis(f.initial)},{"initial_keep",ledger(f.oldKeep)},
        {"initial_down",ledger(f.oldDown)},{"source_edge",11},{"submitted",pass2::exactBinary(f.submittedBudget).get_str()},
        {"models",f.stats.modelsWithCuts},{"runs",f.stats.solverRuns},{"margin_bits",f.marginBits},{"rejected",array(f.rejected)}}},
      {"precision_bits",12},{"cap_bits",cap},{"proof_scope","conditional-engineering-primitive-allowances"}};
  }
  void security(StringRef path) {
    need(!path.empty() && path!="-","independent full-slot security file required");auto b=MemoryBuffer::getFile(path);need(bool(b),"cannot read full-slot security");
    auto v=parse((*b)->getBuffer());const auto &r=obj(v);auto *rows=r.getObject("rows");
    need(r.size()==3 && r.getString("original_sha256")==sourceId && r.getString("arm")==mode && rows && rows->size()==6,"wrong full-slot trust binding");
    bool downOK=true;
    for(StringRef role:{"source.Q","source.QP","target.Q","target.QP","switch.small","switch.large"}){
      auto *entry=rows->getObject(role);need(entry && entry->size()==2,"missing estimator row");
      auto *req=entry->getObject("request"),*a=entry->getObject("attacks");bool whole=role.starts_with("source"),sw=role.starts_with("switch");
      auto q=f.basis.q;if(!whole)q.resize(2);mpz_class mod=1;for(auto x:q)mod*=mpz_class(std::to_string(x));if(!role.ends_with(".Q"))mod*=mpz_class(std::to_string(f.basis.p[0]));
      O expected{{"schema","snabf.measure.v1"},{"estimator","ae45d740ceb138af87a4f92edc592a0ab0d4f43f"},
        {"model","LWE.estimate.rough/classical ADPS16; GSA uSVP, dual_hybrid"},{"secret","iid uniform ternary"},
        {"error","discrete Gaussian sigma=3.2 (estimator model for bounded sampler)"},{"samples","unbounded"},{"role",("a2b."+role).str()},
        {"ring",whole || role=="switch.large"?n:n/2},{"q",array(q)},{"p",array(f.basis.p)},{"modulus",mod.get_str()}};
      if(sw)expected["switch_basis"]=O{{"q",array(q)},{"p",array(f.basis.p)},{"physical_ring",n},{"embedded_secret_ring",n/2}};
      need(req && *req==expected && a && a->getNumber("usvp") && a->getNumber("dual_hybrid"),"wrong measured instance or attacks");
      for(auto &kv:*a){auto x=kv.second.getAsNumber();need(x && std::isfinite(*x) && *x>0,"invalid estimate");
        if(whole)need(*x>=128,"actual source instance below 128 bits");else downOK &= *x>=128;}
    }
    if(!downOK){need(mode!="force-down","actual target/switch instance below 128 bits");selection="keep-source";document["selected"]=selection;}
  }
};
void newFile(const std::filesystem::path &p,StringRef text){std::error_code e;raw_fd_ostream out(p.string(),e,sys::fs::CD_CreateNew);need(!e,"new output file required");out<<text;out.close();need(!out.has_error(),"output write failed");}
template<class F>mlir::LogicalResult diagnosed(ModuleOp m,F f){try{f();return mlir::success();}catch(const std::exception &e){return m.emitError(e.what());}}
}
bool isFullSlotProduct(ModuleOp m){for(auto a:m->getAttrs())if(a.getName().getValue().starts_with("snabf.a2b_"))return true;return false;}
mlir::LogicalResult planFullSlotProduct(ModuleOp m,StringRef proposal){return diagnosed(m,[&]{carrier(m,"source");Physical p(m);auto raw=dump(p.document);
  if(!proposal.empty())newFile(proposal.str(),raw);set(m,plan,raw);set(m,stage,"planned");});}
mlir::LogicalResult checkFullSlotProduct(ModuleOp m,StringRef security,bool requireDown){return diagnosed(m,[&]{bool sealed=field(m,stage)=="checked";
  carrier(m,sealed?"checked":"planned");Physical p(m);auto raw=field(m,plan);if(!sealed)need(raw==dump(p.document),"full-slot plan differs from source replay");p.security(security);
  auto checked=dump(p.document);if(sealed)need(raw==checked,"full-slot plan differs from source replay");
  need(!requireDown || p.selection=="force-down","missing required physical down");auto id=sha(checked);need(!sealed || id==field(m,seal),"stale full-slot seal");set(m,plan,checked);set(m,seal,id);set(m,stage,"checked");});}
mlir::LogicalResult lowerFullSlotProduct(ModuleOp m,StringRef security,StringRef output){return diagnosed(m,[&]{carrier(m,"checked");Physical p(m);auto raw=field(m,plan);
  p.security(security);need(raw==dump(p.document) && sha(raw)==field(m,seal),"full-slot lowering source replay mismatch");
  // Embedded immutable original and checked schedule feed a real two-result runtime.
  std::string go="package fullslotprogram\nimport (\n _ \"embed\"\n r \"snabf.local/compiler/src/backend/ringswitch\"\n)\n//go:embed source.json\nvar source []byte\n//go:embed program.json\nvar program []byte\n";
  go+="func Prepare(keys r.GraphKeys, pin [32]byte)(*r.PreparedGraph,error){return r.PrepareCheckedGraph(source,program,\""+p.sourceId+"\",\""+sha(raw)+"\",keys,pin)}\n";
  if(!output.empty()){std::filesystem::path d(output.str());need(std::filesystem::create_directory(d),"new full-slot output directory required");newFile(d/"program.json",raw);newFile(d/"source.json",field(m,src));newFile(d/"model.go",go);}
  set(m,"snabf.a2b_go",go);set(m,stage,"lowered");});}
} // namespace snabf::pass4

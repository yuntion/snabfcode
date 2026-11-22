#include "DirectEmitter.h"
#include "EmitGraph.h"
#include <charconv>
#include <cmath>

namespace snabf::ir::lattigo_target::detail {
namespace {
std::string quoted(StringRef s) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string out(1,'"');
  for (unsigned char c:s.bytes()) {
    if (c=='\\' || c=='"') {out+='\\'; out+=c;}
    else if (c>=32 && c<127) out+=c;
    else {out+="\\x"; out+=hex[c>>4]; out+=hex[c&15];}
  }
  return out+'"';
}
// Round-trip the actual binary64 bound/coefficient, independent of locale.
std::string number(double x) {
  require(std::isfinite(x),"nonfinite_go_literal");
  char buf[64]; auto r=std::to_chars(buf,buf+sizeof(buf),x);
  require(r.ec==std::errc(),"unrepresentable_go_literal");
  return std::string(buf,r.ptr);
}
template<class Range> void list(llvm::raw_ostream &o,StringRef type,const Range &xs) {
  o<<"[]"<<type<<"{";
  for(auto &x:xs) {
    if constexpr (std::is_floating_point_v<std::decay_t<decltype(x)>>) {
      // Untyped Go constants have no signed zero; keep public payload bits.
      if (x == 0 && std::signbit(x)) o<<"math.Float64frombits(1<<63)";
      else o<<number(x);
    }
    else o<<x;
    o<<",";
  }
  o<<"}";
}
void layout(llvm::raw_ostream &o,DictionaryAttr d) {
  o<<"epoch.Layout{N:"<<num(d,"ring_dim")<<", Capacity:"<<num(d,"num_slots")
   <<", Width:"<<num(d,"logical_width")<<", Period:"<<num(d,"period")<<", Stride:"<<num(d,"stride")
   <<", Level:"<<num(d,"active_level")<<", Lanes:"<<num(d,"lane_count")<<", Packing:"<<quoted(str(d,"packing_mode"))
   <<", LogScale:"<<number(real(d,"log2_scale"))<<", Offsets:";list(o,"int",ints(d,"offsets"));
  o<<", Active:";list(o,"int",ints(d,"active_lanes"));o<<", Mask:";list(o,"int",ints(d,"mask_signature"));o<<"}";
}
void bind(llvm::raw_ostream &o,const EmitGraph &g,StringRef digest) {
  o<<"const PlanSHA256 = "<<quoted(digest)<<"\n\nfunc New() (*epoch.Runtime, error) {\n"
    "\treturn epoch.Bind(epoch.Program{Fingerprint: PlanSHA256, Source:"<<g.source<<", Target:"<<g.target
   <<", Width:"<<g.width<<", Slots:"<<g.slots<<", GaloisBytes:"<<num(g.program,"galois_bytes")<<",\n\t\tEpochs: []epoch.Epoch{\n";
  for (auto [n,d]:g.families) {
    o<<"\t\t\t{N:"<<n<<", Scale:"<<num(d,"scale")<<", KeyID:"<<quoted(str(d,"key_id"))<<", KeyBytes:"
     <<num(d,"key_bytes")<<", Relin:"<<(g.relin.count(n)?"true":"false")<<", Q:";list(o,"uint64",ints(d,"q"));
    o<<", P:";list(o,"uint64",ints(d,"p"));o<<", Galois:";list(o,"uint64",ints(d,"galois"));o<<"},\n";
  }
  o<<"\t\t},\n";
  if (auto d=g.transition) {
    o<<"\t\tSwitch: &epoch.Transition{FromKey:"<<quoted(str(d,"source_key"))<<", ToKey:"<<quoted(str(d,"target_key"))
     <<", LevelQ:"<<num(d,"switch_level_q")<<", LevelP:"<<num(d,"switch_level_p")<<", KeyBytes:"<<num(d,"switch_bytes")
     <<", Parallel:"<<num(d,"parallel_factor")<<", Operands:[]epoch.CutOperand{";
    for (const auto &op:g.operands) {
      auto cut=op.value, use=cast<DictionaryAttr>(field<ArrayAttr>(cut,"uses")[0]);
      o<<"{SSA:"<<quoted(str(cut,"id"))<<", Edge:"<<quoted(op.input)<<", KeyID:"<<quoted(str(cut,"key_domain"))
       <<", User:"<<quoted(str(use,"user"))<<", UserIndex:"<<num(use,"user_index")<<", Operand:"<<num(use,"operand")<<", Provided:";
      layout(o,op.provided);o<<", Required:";layout(o,op.required);
      if (op.consumers) {
        o<<", Consumers:[]epoch.CutUse{";
        for(auto a:op.consumers) { auto u=cast<DictionaryAttr>(a); o<<"{User:"<<quoted(str(u,"user"))<<", UserIndex:"<<num(u,"user_index")<<", Operand:"<<num(u,"operand")<<"},"; }
        o<<"}";
      }
      o<<"},";
    }
    o<<"}"; auto ext=g.switchingDomain;
    o<<", Basis:epoch.SwitchBasis{Ring:"<<num(ext,"ring_dim")<<", EmbeddedRing:"<<num(ext,"embedded_secret_ring")<<", Q:";
    list(o,"uint64",ints(ext,"q"));o<<", P:";list(o,"uint64",ints(ext,"p"));o<<"}},\n";
  }
  o<<"\t\tPorts: []epoch.ExactPort{\n";
  for (const auto &e : g.edges) {
    o<<"\t\t\t{ID:"<<quoted(e.id)<<", N:"<<e.n<<", Level:"<<e.level<<", Output:"<<quoted(e.outputScale);
    if (auto d = e.encoding)
      o<<", Call:"<<quoted(str(d,"call"))<<", Metadata:"<<quoted(str(d,"metadata_scale"))
       <<", Effective:"<<quoted(str(d,"effective_encoder_scale"))<<", Precision:"<<num(d,"precision")
       <<", Count:"<<num(d,"payload_count")<<", PayloadSHA256:"<<quoted(str(d,"payload_sha256"));
    o<<"},\n";
  }
  o<<"\t\t},\n\t\tPlain: []epoch.PlainBinding{\n";
  for (auto [j,e]:llvm::enumerate(g.edges)) if (!e.values.empty()) {
    o<<"\t\t\t{Index:"<<j<<", N:"<<e.n<<", Level:"<<e.level<<", Values:";
    list(o,"float64",e.values);o<<"},\n";
  }
  const bool masks = llvm::any_of(g.edges, [](const Edge &e) { return e.op == Op::Mask && e.values.size() > 1; });
  o<<"\t\t},\n\t}, evaluate, epoch."<<(masks ? "RequireDiagonalEncodingV1" : "RequireExactEncoding")<<")\n}\n\n";
}
std::string site(const Edge &e) {
  return "gadget.Site{ID:"+quoted(e.id)+", Level:"+std::to_string(e.level)+", K:"+std::to_string(e.k)+"}";
}
std::string value(int64_t i) {return "v"+std::to_string(i);}
void evaluation(llvm::raw_ostream &o,const EmitGraph &g) {
  o<<"func evaluate(r *epoch.Runtime, input *rlwe.Ciphertext) (result *rlwe.Ciphertext, err error) {\n"
     "\tsite := \"input\"\n\tdefer epoch.HEIRError(&site, &err)\n";
  for(auto [j,e]:llvm::enumerate(g.edges)) {
    std::string v=value(j), ev="r.Evaluator("+std::to_string(e.n)+")";
    std::string a=e.inputs.empty()?"input":value(e.inputs[0]);
    std::string pt="r.Plain("+std::to_string(j)+")";
    o<<"\tsite = "<<quoted(e.id)<<"\n";
    auto checked=[&] {o<<"\tif err != nil { return nil, fmt.Errorf(\"%s: %w\", site, err) }\n";};
    switch(e.op) {
      case Op::Source: o<<"\t"<<v<<" := input\n"; break;
      case Op::Mask: case Op::Coefficient: o<<"\t"<<v<<" := heir.MulPlain("<<ev<<", "<<a<<", "<<pt<<")\n"; break;
      case Op::AddPlain: o<<"\t"<<v<<" := heir.AddPlain("<<ev<<", "<<a<<", "<<pt<<")\n"; break;
      case Op::Mul: o<<"\t"<<v<<" := heir.Mul("<<ev<<", "<<a<<", "<<value(e.inputs[1])<<")\n"; break;
      case Op::Add:
        if (int64_t(j)==g.sink && e.inputs.size()==2)
          o<<"\tif err = r.CheckJoin(site, "<<a<<", "<<value(e.inputs[1])<<"); err != nil { return nil, err }\n";
        o<<"\t"<<v<<" := heir.Add("<<ev<<", "<<a<<", "<<value(e.inputs[1])<<")\n";
        for(auto i:e.inputs.drop_front(2)) o<<"\t"<<v<<" = heir.Add("<<ev<<", "<<v<<", "<<value(i)<<")\n";
        break;
      case Op::Rescale: o<<"\t"<<v<<" := heir.Rescale("<<ev<<", "<<a<<")\n"; break;
      case Op::Drop: // The HEIR op takes a static attribute; the depth gap is fixed by this plan.
        o<<"\t"<<v<<" := "<<ev<<".DropLevelNew("<<a<<", "<<g.edges[e.inputs[0]].level-e.level<<")\n";break;
      case Op::Relin: o<<"\t"<<v<<", err := gadget.RelinearizeNew("<<ev<<", "<<a<<", "<<site(e)<<")\n";checked();break;
      case Op::Giant: o<<"\t"<<v<<", err := gadget.Giant("<<ev<<", "<<a<<".CopyNew(), "<<e.shift<<", "<<site(e)<<")\n";checked();break;
      case Op::Baby: {
        const auto &group=g.hoists.at({e.site,e.inputs[0]});
        auto h="hoist"+std::to_string(group.front());
        if (group.front()==j) {
          std::vector<int64_t> steps; for (auto i:group) steps.push_back(g.edges[i].shift);
          o<<"\t"<<h<<", err := gadget.Babies("<<ev<<", "<<a<<", ";list(o,"int",steps);
          o<<", map[int]gadget.Site{";for(auto i:group) o<<g.edges[i].shift<<":"<<site(g.edges[i])<<",";
          o<<"})\n";checked();
        }
        o<<"\t"<<v<<" := "<<h<<"["<<e.shift<<"]\n";break;
      }
      case Op::Down: o<<"\t"<<v<<", err := r.Down("<<quoted(g.edges[e.inputs[0]].id)<<", "<<a<<")\n";checked();break;
    }
    o<<"\tif err = r.Check(site, "<<v<<", "<<e.n<<", "<<e.level<<", "<<number(e.scale)<<"); err != nil { return nil, err }\n";
  }
  o<<"\treturn "<<value(g.sink)<<", nil\n}\n";
}
} // namespace
std::string renderGo(DictionaryAttr p, StringRef digest) {
  EmitGraph g(p);
  bool gadget=false, standard=false;
  for (auto &e:g.edges) switch(e.op) {
    case Op::Baby: case Op::Giant: case Op::Relin: gadget=true;break;
    case Op::Source: case Op::Down: case Op::Drop: break;
    default: standard=true; break;
  }
  std::string go;llvm::raw_string_ostream o(go);
  o<<"// Code generated by snabf-bind-lattigo (direct-go.v1). DO NOT EDIT.\npackage planned\n\nimport (\n";
  if (gadget||g.transition) o<<"\t\"fmt\"\n";
  if (llvm::any_of(g.edges, [](const Edge &e) {
        return llvm::any_of(e.values, [](double x) { return x == 0 && std::signbit(x); });
      })) o<<"\t\"math\"\n";
  o<<"\t\"github.com/tuneinsight/lattigo/v6/core/rlwe\"\n\tepoch \"snabf.local/compiler/src/backend/ringswitch/epochs\"\n";
  if (gadget) o<<"\t\"snabf.local/compiler/src/backend/gadget\"\n";
  if (standard) o<<"\their \"snabf.local/compiler/src/backend/ringswitch/epochs/heirckks\"\n";
  o<<")\n\n";bind(o,g,digest);evaluation(o,g);return go;
}
} // namespace snabf::ir::lattigo_target::detail

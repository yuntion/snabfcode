#include "Pipeline.h"
#include "Pass2IR.h"
#include "Pass3IR.h"
#include "Pass4IR.h"
#include "Target/Lattigo/DirectEmitter.h"
#include "Pass2Lowering.h"
#include "pass1/NativeRouting.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringSet.h"
#include <bit>
#include <map>
#include <set>

namespace snabf::ir {
void lowerCKKS(ModuleOp m) {
  if (isMixed(m) || m->hasAttr("snabf.ring_search") || m->hasAttr("snabf.ckks_direct") || m->hasAttr("snabf.ckks_epochs")) {
    lattigo_target::lowerDirect(m); return;
  }
  verifyPlan(m);
  verifyPass3(m);
  if(auto rec=executableBootstrap(m)) {
    m->setAttr("snabf.export_bootstrap",rec); return; // already a typed Lattigo bootstrap; bind its evaluator after HEIR translation
  }
  m.walk([&](Operation *o) {
    if (auto d = o->getAttrOfType<DictionaryAttr>("snabf.ring_switch"))
      require(!field<BoolAttr>(d, "enabled").getValue(),
              "selected ring-switched bootstrap has no executable lowering");
  });
  IRRewriter rw(m.getContext());
  SmallVector<Operation *> ops;
  SmallVector<Attribute> records;
  ProfileOp profile;
  m.walk([&](Operation *o) {
    if (o->getDialect() && o->getDialect()->getNamespace() == "snabf") {
      if (auto p = dyn_cast<ProfileOp>(o)) {
        require(!profile, "backend expects one physical family");
        profile = p;
      } else
        ops.push_back(o);
    }
  });
  require(bool(profile), "missing profile");
  Pass2Lowering p2(m, profile, rw);
  DictionaryAttr p2Audit;
  if (p2.active()) p2Audit = pass2Manifest(m);
  for (auto *o : ops) {
    rw.setInsertionPoint(o);
    if (auto src = dyn_cast<SourceOp>(o)) {
      if (p2.active()) p2.bindSource(src);
      rw.replaceOp(o, src.getCipher());
      continue;
    }
    if (auto sink = dyn_cast<SinkOp>(o)) {
      if (p2.active()) p2.bindSink(sink);
      require(o->getOperand(0).getType() == o->getResult(0).getType(),
              "sink ABI mismatch");
      rw.replaceOp(o, o->getOperand(0));
      continue;
    }
    if (auto poly = dyn_cast<PolyEvalOp>(o); poly && o->hasAttr("activation")) {
      rw.replaceOp(o, p2.emit(poly));
      continue;
    }
    StringRef kind;
    auto fields = NamedAttrList(field<DictionaryAttr>(o, "native_props"));
    if (auto lin = dyn_cast<LintransOp>(o)) {
      kind = "kernel.apply_linear_transform";
      fields.set("diagonal_indices", lin.getDiagonalsAttr());
      auto table = preparedWeights(lin.getTable(), lin.getDiagonals());
      auto in = decodeCT(o->getOperand(0).getType());
      NamedAttrList r;
      r.set("diagonals", lin.getDiagonalsAttr());
      r.set("weights", rw.getStringAttr(weightId(table)));
      r.set("selected", field<DictionaryAttr>(o, "selected"));
      r.set("candidates",field<ArrayAttr>(o,"candidates"));
      r.set("charge",field<DictionaryAttr>(o,"route_charge"));
      r.set("context",field<DictionaryAttr>(o,"route_context"));
      r.set("level", rw.getI64IntegerAttr(in.level));
      r.set("site", field<IntegerAttr>(o, "site"));
      records.push_back(r.getDictionary(m.getContext()));
    } else if (auto poly = dyn_cast<PolyEvalOp>(o)) {
      kind = "kernel.eval_chebyshev";
      fields.set("coefficients", poly.getCoefficients());
    } else if (auto step = dyn_cast<StepOp>(o))
      kind = step.getKind();
    else
      throw ContractError("no executable selected lowering for " +
                          name(o).str());
    OperationState state(o->getLoc(), kind);
    state.addTypes(p2.active() ? p2.nativeResult(o, fields)
                               : field<TypeAttr>(o, "native").getValue());
    state.addOperands(o->getOperands());
    if (!fields.empty())
      state.propertiesAttr = fields.getDictionary(m.getContext());
    state.addAttributes(field<DictionaryAttr>(o, "native_attrs").getValue());
    rw.replaceOp(o, rw.create(state)->getResults());
  }
  NamedAttrList manifest;
  if(auto p3=m->getAttrOfType<DictionaryAttr>("snabf.pass3"))manifest.set("pass3",p3);
  if (p2.active()) {
    p2.finish();
    manifest.set("pass2", p2Audit);
    int64_t encodes = 0;
    m.walk([&](Operation *op) { encodes += op->hasAttr("snabf.physical_scale"); });
    manifest.set("p2_encodes", rw.getI64IntegerAttr(encodes));
  }
  manifest.set("regions", rw.getArrayAttr(records));
  manifest.set("inventory", field<DictionaryAttr>(m, "snabf.inventory"));
  manifest.set("q", profile.getQAttr());
  manifest.set("p", profile.getPAttr());
  manifest.set("ring", profile.getRingAttr());
  manifest.set("slots", profile.getSlotsAttr());
  manifest.set("scale", profile.getDefaultLogScaleAttr());
  manifest.set("key_id", profile.getKeyIdAttr());
  manifest.set("key_bytes", profile.getKeyBytesAttr());
  manifest.set("arm", field<StringAttr>(m, "snabf.arm"));
  manifest.set("rounds", field<IntegerAttr>(m, "snabf.solve_rounds"));
  manifest.set("route_rounds", field<ArrayAttr>(m,"snabf.route_rounds"));
  m->setAttr("snabf.export", manifest.getDictionary(m.getContext()));
  rw.eraseOp(profile);
}

llvm::json::Object bindBackend(ModuleOp m) {
  using namespace llvm::json;
  IRRewriter rw(m.getContext());
  if(auto rec=m->getAttrOfType<DictionaryAttr>("snabf.export_bootstrap")) {
    require(rec==executableBootstrap(m),"stale bootstrap binding");
    return Object{{"bootstrap",jsonPass2(rec)}};
  }
  auto manifest = field<DictionaryAttr>(m, "snabf.export");
  auto regions = field<ArrayAttr>(manifest, "regions");
  auto inv = field<DictionaryAttr>(manifest, "inventory");
  auto n = field<IntegerAttr>(manifest, "ring").getInt(),
       slots = field<IntegerAttr>(manifest, "slots").getInt();
  std::map<StringRef, DictionaryAttr> byWeight;
  Array audits;
  for (auto a : regions) {
    auto d = cast<DictionaryAttr>(a);
    NamedAttrList audit;
    for(auto key:{"site","level","selected","candidates","charge"}) audit.set(key,d.get(key));
    auto ctx=field<DictionaryAttr>(d,"context"); NamedAttrList context(ctx);
    for(auto key:{"profile","layout","layout_feedback"}) if(auto v=ctx.get(key)) context.set(key,rw.getStringAttr(spelling(v)));
    audit.set("context",context.getDictionary(m.getContext()));
    audits.push_back(jsonPass2(audit.getDictionary(m.getContext())));
    require(
        byWeight.emplace(field<StringAttr>(d, "weights").getValue(), d).second,
        "ambiguous backend weight identity");
  }
  SmallVector<Operation *> old;
  Operation *set = nullptr, *pair = nullptr, *literal = nullptr;
  Array backend;
  std::map<int64_t, int64_t> refs;
  const llvm::StringSet<> allowed = {"ckks.apply_linear_transform",
                                     "ckks.prepare_linear_transform",
                                     "ckks.rescale",
                                     "ckks.mul_new",
                                     "ckks.mul",
                                     "ckks.new_polynomial_evaluator",
                                     "ckks.chebyshev",
                                     "ckks.relinearize",
                                     "ckks.encode",
                                     "ckks.decode",
                                     "ckks.new_plaintext",
                                     "ckks.new_encoder",
                                     "ckks.new_evaluator",
                                     "ckks.new_parameters_from_literal",
                                     "rlwe.drop_level",
                                     "rlwe.encrypt",
                                     "rlwe.decrypt",
                                     "rlwe.new_key_generator",
                                     "rlwe.gen_key_pair",
                                     "rlwe.new_encryptor",
                                     "rlwe.new_decryptor",
                                     "rlwe.gen_relinearization_key",
                                     "rlwe.gen_galois_key",
                                     "rlwe.new_evaluation_key_set"};
  auto unique = [&](Operation *&slot, Operation *o) {
    require(!slot, "multiple backend key families not bound");
    slot = o;
  };
  m.walk([&](Operation *o) {
    auto k = name(o);
    if (!k.consume_front("lattigo.")) return;
    bool dagPrimitive = manifest.get("pass2") &&
        (k == "ckks.add" || k == "ckks.add_new" ||
         k == "ckks.rescale_new" || k == "ckks.relinearize_new" ||
         k == "rlwe.drop_level_new");
    require(allowed.contains(k) || dagPrimitive, "unmodeled Lattigo consumer: " + k);
    if (k == "rlwe.gen_key_pair")
      unique(pair, o);
    else if (k == "rlwe.new_evaluation_key_set")
      unique(set, o);
    else if (k == "rlwe.gen_galois_key")
      old.push_back(o);
    else if (k == "ckks.new_parameters_from_literal")
      unique(literal, o);
    else if (k == "ckks.prepare_linear_transform") {
      auto def = o->getOperand(2).getDefiningOp();
      require(def && name(def) == "memref.get_global", "lost diagonal weights");
      auto g = m.lookupSymbol(field<FlatSymbolRefAttr>(def, "name").getValue());
      require(g, "missing weight global");
      auto id = weightId(field<DenseFPElementsAttr>(g, "initial_value"));
      auto it = byWeight.find(id);
      require(it != byWeight.end(), "unmatched backend weight payload");
      auto d = it->second;
      byWeight.erase(it);
      auto route = field<DictionaryAttr>(d, "selected");
      auto level = field<IntegerAttr>(o, "levelQ").getInt();
      auto logs = field<IntegerAttr>(o, "logSlots").getInt();
      auto diags = field<DenseI32ArrayAttr>(o, "diagonal_indices").asArrayRef();
      require(
          logs == std::countr_zero(uint64_t(slots)) &&
              level == field<IntegerAttr>(d, "level").getInt() &&
              llvm::equal(
                  diags, field<DenseI64ArrayAttr>(d, "diagonals").asArrayRef()),
          "backend level, layout or diagonal mismatch");
      auto ts = field<DenseI64ArrayAttr>(route, "terms");
      auto rung = field<IntegerAttr>(route, "rung");
      auto ctx=field<DictionaryAttr>(d,"context");
      auto rebuilt=executableRoutes(std::countr_zero(uint64_t(n)),{diags.begin(),diags.end()},field<StringAttr>(ctx,"pricing")!="dense_tower");
      require(llvm::any_of(rebuilt,[&](const auto& c){return route==routeAttr(rw,c);}),"emitted tower disagrees with bound diagonal support");
      o->setAttr("snabf_pass1_terms", ts);
      o->setAttr("snabf_pass1_tower", rung);
      Array terms;
      std::set<uint64_t> keys;
      require(ts.size() == 3 * int64_t(diags.size()), "invalid tower length");
      for (int64_t i = 0; i < ts.size(); i += 3) {
        auto b = ts[i + 1], h = ts[i + 2];
        require(ts[i] == diags[i / 3] && b >= 0 && b < n / 2 && h >= 0 &&
                    h < n / 2 && (h + b) % (n / 2) == ts[i],
                "invalid tower term");
        terms.push_back(Array{ts[i], b, h});
        for (auto v : {h, b})
          if (v) {
            keys.insert(rotationElement(v,std::countr_zero(uint64_t(n))));
          }
      }
      for (auto k : keys) ++refs[k];
      backend.push_back(
          Object{{"LogN", int64_t(std::countr_zero(uint64_t(n)))},
                 {"Level", level},
                 {"LogSlots", logs},
                 {"Terms", std::move(terms)},
                 {"S", rung.getInt()},
                 {"site", field<IntegerAttr>(d, "site").getInt()}});
    }
  });
  require(byWeight.empty() && pair && set && literal,
          "incomplete HEIR backend module");
  require(field<StringAttr>(manifest, "key_id")
              .getValue()
              .starts_with(
                  pair->getParentOfType<func::FuncOp>().getName().str() + ":"),
          "backend key family mismatch");
  auto p = decodeParameters(field<Attribute>(literal, "paramsLiteral"));
  require((1LL << p.logN) == n &&
              p.logScale == field<IntegerAttr>(manifest, "scale").getInt() &&
              llvm::equal(
                  p.q, field<DenseI64ArrayAttr>(manifest, "q").asArrayRef()) &&
              llvm::equal(p.p,
                          field<DenseI64ArrayAttr>(manifest, "p").asArrayRef()),
          "changed backend Q/P or scale");
  SmallVector<mlir::Value> retained;
  for (auto v : set->getOperands())
    if (!llvm::is_contained(old, v.getDefiningOp())) retained.push_back(v);
  for (auto o : old)
    require(o->getOperand(0) == pair->getOperand(0) &&
                o->getOperand(1) == pair->getResult(0) &&
                o->getResult(0).hasOneUse() &&
                *o->getResult(0).getUsers().begin() == set,
            "unexpected key use or key family");
  auto keys = field<ArrayAttr>(inv, "keys");
  require(refs.size() == keys.size(), "wrong emitted key count");
  rw.setInsertionPoint(set);
  Array publicKeys;
  for (auto a : keys) {
    auto d = cast<DictionaryAttr>(a);
    auto key = field<IntegerAttr>(d, "galois");
    auto count = field<IntegerAttr>(d, "refs").getInt();
    auto ref = refs.find(key.getInt());
    require(ref != refs.end() && ref->second == count,
            "unexpected key or reference count");
    refs.erase(ref);
    auto type = parseType("!lattigo.rlwe.galois_key<galoisElement = " +
                              std::to_string(key.getInt()) + " : i64>",
                          m.getContext());
    require(bool(type), "invalid Galois key type");
    OperationState st(set->getLoc(), "lattigo.rlwe.gen_galois_key");
    st.addTypes(type);
    st.addOperands({pair->getOperand(0), pair->getResult(0)});
    st.propertiesAttr =
        rw.getDictionaryAttr({rw.getNamedAttr("galoisElement", key)});
    retained.push_back(rw.create(st)->getResult(0));
    publicKeys.push_back(Object{{"galois", key.getInt()}, {"refs", count}});
  }
  set->setOperands(retained);
  for (auto o : old) rw.eraseOp(o);
  auto bytes = field<IntegerAttr>(inv, "bytes").getInt();
  require(bytes == int64_t(keys.size()) *
                       field<IntegerAttr>(manifest, "key_bytes").getInt(),
          "physical key size mismatch");
  if (manifest.get("pass2")) {
    require(!backend.empty(), "missing transform to bind Pass 2 preprocessing");
    (*backend.front().getAsObject())["PolyScales"] =
        pass2ScaleBindings(m, field<IntegerAttr>(manifest, "p2_encodes").getInt());
  }
  if(auto p3=manifest.getAs<DictionaryAttr>("pass3"))bindPass3(p3,backend);
  Object report{
      {"route_sites",std::move(audits)},
      {"route_feedback",jsonPass2(rw.getDictionaryAttr({rw.getNamedAttr("rounds",field<ArrayAttr>(manifest,"route_rounds"))}))},
      {"backend", std::move(backend)},
      {"keys", std::move(publicKeys)},
      {"key_bytes", bytes},
      {"objective", field<FloatAttr>(inv, "objective").getValueAsDouble()},
      {"rounds", field<IntegerAttr>(manifest, "rounds").getInt()},
      {"arm", field<StringAttr>(manifest, "arm").str()}};
  if (auto extra = manifest.getAs<DictionaryAttr>("pass2"))
    for (auto &entry : jsonPass2(extra))
      report[entry.first.str()] = std::move(entry.second);
  return report;
}
}  // namespace snabf::ir

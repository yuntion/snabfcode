#include "Pass2IR.h"
#include "Pass4IR.h"
#include <map>
#include <set>

namespace snabf::ir {
// Bind the signed coefficients and actual HEIR diagonal data that the error
// graph deliberately represents only by magnitudes. No decryption in Eval.
DictionaryAttr ringProgram(ModuleOp m) {
  Builder b(m.getContext());
  bool mixed = isMixed(m);
  auto plan = field<DictionaryAttr>(m, mixed ? "snabf.mixed_parameters"
                                             : "snabf.pass3");
  std::map<std::string, Attribute> payload;
  std::map<std::string, std::pair<int,int64_t>> rotations;
  std::map<int64_t, std::set<int64_t>> keys;
  NamedAttrList out;
  int poly = 0, step = 0;
  int64_t cut = -1, large = 0, small = 0;
  m.walk([&](SourceOp s) {
    require(large==0,"multiple_direct_sources");
    auto t = s.getOut().getType();
    large = t.getN();
    auto width=t.getLogical().getNumElements();
    out.set("width", b.getI64IntegerAttr(width));
    auto prof=SymbolTable::lookupNearestSymbolFrom<ProfileOp>(s,t.getProfile());
    require(bool(prof) && width>0,"unbound_source_layout");
    // This encoder tiles one logical vector. Other HEIR packing maps must
    // not be admitted merely because their slot and vector counts divide.
    auto expected = "#tensor_ext.original_type<originalType = " + spelling(t.getLogical()) +
        ", layout = #tensor_ext.layout<\"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod " +
        std::to_string(width) + " = 0 and 0 <= i0 <= " + std::to_string(width-1) +
        " and 0 <= slot <= " + std::to_string(prof.getSlots()-1) + " }\">>";
    require(spelling(prof.getLayout())==expected,"unsupported_direct_source_packing_map");
  });
  m.walk([&](ConvertOp c) {
    cut = c.getInput().getType().getLevel();
    small = c.getOut().getType().getN();
  });
  out.set("source_ring", b.getI64IntegerAttr(large));
  out.set("target_ring", b.getI64IntegerAttr(small ? small : large));
  out.set("cut_level", b.getI64IntegerAttr(cut));
  SmallVector<Attribute> epochs, routes;
  m.walk([&](LintransOp op) {
    auto r = field<DictionaryAttr>(op, "selected");
    auto site = field<IntegerAttr>(op, "site").getInt();
    auto pre = "lin" + std::to_string(site) + ".";
    int64_t n = op.getInput().getType().getN();
    NamedAttrList route;
    route.set("site", op->getAttr("site"));
    route.set("n", b.getI64IntegerAttr(n));
    route.set("level", b.getI64IntegerAttr(op.getInput().getType().getLevel()));
    route.set("selected", r);
    routes.push_back(route.getDictionary(m.getContext()));
    for (auto k : field<DenseI64ArrayAttr>(r, "keys").asArrayRef())
      keys[n].insert(k);
    auto t = field<DenseI64ArrayAttr>(r, "terms").asArrayRef();
    auto w = preparedWeights(op.getTable(), op.getDiagonals());
    auto shape = w.getType().getShape();
    require(shape.size() == 2 && t.size() == 3 * size_t(shape[0]),
            "unbound_suffix_diagonals");
    int slots = shape[1];
    auto raw = w.getValues<APFloat>();
    auto it = raw.begin();
    for (int i = 0; i < shape[0]; ++i) {
      int diag = t[3 * i], baby = t[3 * i + 1], giant = t[3 * i + 2];
      require(diag == op.getDiagonals()[i], "diagonal_weight_order_mismatch");
      SmallVector<double> values(slots);
      for (int j = 0; j < slots; ++j, ++it)
        values[(j + giant) % slots] = (*it).convertToDouble();
      payload[pre + "mask" + std::to_string(diag)] =
          b.getDenseF64ArrayAttr(values);
      if (baby)
        rotations[pre + "baby" + std::to_string(baby)] = {baby,site};
      if (giant)
        rotations[pre + "giant" + std::to_string(giant)] = {giant,site};
    }
  });
  m.walk([&](PolyEvalOp op) {
    auto sel = field<DictionaryAttr>(op, "poly_selected");
    auto pre = "poly" + std::to_string(poly++) + ".n";
    for (auto [i, node] : llvm::enumerate(field<ArrayAttr>(sel, "nodes"))) {
      auto d = cast<DictionaryAttr>(node);
      auto id = pre + std::to_string(i) + ".";
      auto w = field<DenseF64ArrayAttr>(d, "weights");
      if (field<StringAttr>(d, "op") == "linear")
        for (auto [j, x] : llvm::enumerate(w.asArrayRef()))
          payload[id + "pt" + std::to_string(j)] = b.getDenseF64ArrayAttr({x});
      double x = field<FloatAttr>(d, "constant").getValueAsDouble();
      if (x != 0)
        payload[id + "constant"] = b.getDenseF64ArrayAttr({x});
    }
  });
  m.walk([&](StepOp op) {
    auto id = "step" + std::to_string(step++);
    if (op.getKind() == "ckks.mul_plain") {
      SmallVector<double> v;
      for (auto x : plainWeights(op.getInputs()[1]).getValues<APFloat>())
        v.push_back(x.convertToDouble());
      payload[id] = b.getDenseF64ArrayAttr(v);
    }
  });
  m.walk([&](ProfileOp p) {
    require(p.getRing()==uint64_t(large)||(mixed&&p.getRing()==uint64_t(small)),"unaccounted_key_domain");
    NamedAttrList e;
    e.set("n", p.getRingAttr());
    e.set("q", p.getQAttr());
    e.set("p", p.getPAttr());
    e.set("scale", p.getDefaultLogScaleAttr());
    e.set("key_id", p.getKeyIdAttr());
    auto &k = keys[p.getRing()];
    e.set("galois",
          b.getDenseI64ArrayAttr(SmallVector<int64_t>(k.begin(), k.end())));
    e.set("key_bytes", p.getKeyBytesAttr());
    out.set("slots", p.getSlotsAttr());
    epochs.push_back(e.getDictionary(m.getContext()));
  });
  require(epochs.size()==(mixed?2u:1u),"unaccounted_key_domain");
  out.set("epochs", b.getArrayAttr(epochs));
  out.set("sink", plan.get("sink"));
  out.set("galois_bytes",
          field<DictionaryAttr>(m, "snabf.inventory").get("bytes"));
  out.set("routes", b.getArrayAttr(routes));
  if (auto s = m->getAttr("snabf.ring_search")) out.set("ring_search", s);
  if (mixed) {
    out.set("ring_decision", field<DictionaryAttr>(m, "snabf.ring_decision"));
    out.set(
        "switch_bytes",
        field<DictionaryAttr>(m, "snabf.ring_decision").get("switch_bytes"));
  }
  SmallVector<Attribute> nodes;
  for (auto a : field<ArrayAttr>(plan, "edges")) {
    auto d = cast<DictionaryAttr>(a);
    NamedAttrList e;
    for (auto k : {"ID", "op", "Level", "K", "inputs", "physical_log_scale"})
      e.set(k, d.get(k));
    auto id = field<StringAttr>(d, "ID").str();
    auto op = field<StringAttr>(d, "op").getValue();
    // Pass 3 records the kernel domain. A conversion runs in the source
    // domain but returns a value in the target domain. Never infer domain
    // from the textual position of an edge relative to the conversion.
    int64_t kernel = mixed ? field<IntegerAttr>(d, "kernel_ring").getInt() : large;
    int64_t n = op == "ring_down" ? small : kernel;
    e.set("kernel_ring", b.getI64IntegerAttr(kernel));
    if (mixed) for (auto k : {"input_key_domain","output_key_domain","kernel_domain"})
      e.set(k,field<StringAttr>(d,k));
    e.set("n", b.getI64IntegerAttr(n));
    if (op == "coefficient" || op == "mask" || op == "add_plain") {
      require(payload.contains(id), "unbound_plaintext_payload");
      e.set("values", payload.at(id));
      payload.erase(id);
    }
    if (op == "baby" || op == "giant") {
      require(rotations.contains(id), "unbound_rotation");
      auto [offset, site] = rotations.at(id);
      e.set("rotation", b.getI64IntegerAttr(offset));
      e.set("route_site", b.getI64IntegerAttr(site));
      rotations.erase(id);
    }
    nodes.push_back(e.getDictionary(m.getContext()));
  }
  require(payload.empty(), "orphan_plaintext_payload");
  require(rotations.empty(), "orphan_rotation_binding");
  out.set("instructions", b.getArrayAttr(nodes));
  return out.getDictionary(m.getContext());
}
} // namespace snabf::ir

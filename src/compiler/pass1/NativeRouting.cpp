#include "NativeRouting.h"
#include <bit>
#include <cmath>
namespace snabf::ir {
Catalog::Catalog(ModuleOp m) {
  auto price = [&](StringRef key) {
    double x = field<FloatAttr>(m, key).getValueAsDouble();
    require(std::isfinite(x) && x >= 0, "invalid cost " + key);
    return x;
  };
  eta = price("snabf.key_units_per_mib") / (1 << 20);
  double baby = price("snabf.hoist"), giant = price("snabf.giant");
  auto mode = m->getAttrOfType<StringAttr>("snabf.route_pricing");
  require(!mode || mode == "occupied" || mode == "dense_tower",
          "invalid Pass 1 price model");
  bool dense = mode && mode == "dense_tower";
  m.walk([&](LintransOp op) {
    auto in = op.getInput().getType();
    auto p =
        SymbolTable::lookupNearestSymbolFrom<ProfileOp>(op, in.getProfile());
    require(bool(p), "unresolved profile");
    auto scope = p.getKeyId().str() + ":" + spelling(p.getQAttr()) + ":" +
                 spelling(p.getPAttr()) + ":" + std::to_string(p.getRing());
    ops.push_back(op);
    auto &r = regions.emplace_back(LinearRegion{scope,
                                                (1 + in.getLevel()) * baby,
                                                (1 + in.getLevel()) * giant,
                                                {},
                                                dense});
    Builder b(m.getContext());
    NamedAttrList ctx;
    ctx.set("physical_scope", b.getStringAttr(scope));
    ctx.set("input_level", b.getI64IntegerAttr(in.getLevel()));
    ctx.set("profile", in.getProfile());
    ctx.set("layout", p.getLayout());
    ctx.set("slots", p.getSlotsAttr());
    ctx.set("hoist_price", b.getF64FloatAttr(r.hoistedStepCost));
    ctx.set("giant_price", b.getF64FloatAttr(r.giantStepCost));
    ctx.set("key_price_per_byte", b.getF64FloatAttr(eta));
    ctx.set("bytes_per_key", p.getKeyBytesAttr());
    ctx.set("pricing", b.getStringAttr(dense ? "dense_tower" : "occupied"));
    if (auto f = op->getAttrOfType<DictionaryAttr>("snabf.route_feedback")) {
      require(field<StringAttr>(f, "source") == "pass4_layout" &&
                  f.get("profile") == in.getProfile() &&
                  f.get("layout") == p.getLayout() &&
                  field<IntegerAttr>(f, "ring").getInt() == in.getN(),
              "route feedback does not match the current physical layout");
      auto cap = field<IntegerAttr>(f, "max_serial_rotations").getInt();
      require(cap >= 0 && cap <= INT32_MAX, "invalid layout route budget");
      layoutCuts.push_back({{ops.size() - 1}, unsigned(cap)});
      ctx.set("layout_feedback", f);
    }
    auto d = op.getDiagonals();
    auto logN = std::countr_zero(uint64_t(in.getN()));
    SmallVector<int64_t> lifts;
    for (auto x : d)
      lifts.push_back(rotationElement(x, logN));
    ctx.set("diagonal_galois", b.getDenseI64ArrayAttr(lifts));
    ctx.set("group_order", b.getI64IntegerAttr(in.getN() / 2));
    contexts.push_back(ctx.getDictionary(m.getContext()));
    auto routes = executableRoutes(logN, {d.begin(), d.end()}, !dense);
    for (auto &c : routes) {
      for (auto g : c.galoisKeys) {
        auto [it, added] =
            sizes.emplace(ScopedGaloisKey{scope, g}, p.getKeyBytes());
        require(added || it->second == p.getKeyBytes(),
                "inconsistent physical key sizes");
      }
      r.candidates.push_back(std::move(c));
    }
  });
  require(!ops.empty(), "no snabf linear regions");
}
DictionaryAttr routeAttr(Builder &b, const SubgroupRoute &c) {
  SmallVector<int64_t> terms, splits;
  auto array = [&](const auto &values) {
    SmallVector<int64_t> out(values.begin(), values.end());
    return b.getDenseI64ArrayAttr(out);
  };
  SmallVector<Attribute> cosets;
  for (auto h : c.splits)
    splits.append({h.rung, h.stride, h.order, h.index});
  for (auto &[g, ds] : c.cosets)
    cosets.push_back(
        b.getDictionaryAttr({b.getNamedAttr("gamma", b.getI64IntegerAttr(g)),
                             b.getNamedAttr("diagonals", array(ds))}));
  for (auto t : c.terms)
    terms.append({t.diagonal, t.baby, t.giant});
  return b.getDictionaryAttr(
      {b.getNamedAttr("rung", b.getI64IntegerAttr(c.towerIndex)),
       b.getNamedAttr("terms", b.getDenseI64ArrayAttr(terms)),
       b.getNamedAttr("keys", array(c.galoisKeys)),
       b.getNamedAttr("equivalent_splits", array(splits)),
       b.getNamedAttr("b_v", b.getI64IntegerAttr(c.splits.front().order)),
       b.getNamedAttr("g_v", b.getI64IntegerAttr(c.splits.front().index)),
       b.getNamedAttr("subgroup_stride", b.getI64IntegerAttr(c.subgroupStride)),
       b.getNamedAttr("cosets", b.getArrayAttr(cosets)),
       b.getNamedAttr("baby_steps", array(c.babyRotations)),
       b.getNamedAttr("giant_steps", array(c.giantRotations)),
       b.getNamedAttr("rotation_depth",
                      b.getI64IntegerAttr(rotationDepth(c)))});
}
ArrayAttr alternatives(Builder &b, const LinearRegion &r) {
  SmallVector<Attribute> list;
  for (auto &c : r.candidates) {
    NamedAttrList a(routeAttr(b, c));
    auto cost = operationCharge(r, c);
    a.set("hoist_work", b.getF64FloatAttr(cost.hoistedWork));
    a.set("giant_work", b.getF64FloatAttr(cost.giantWork));
    list.push_back(a.getDictionary(b.getContext()));
  }
  return b.getArrayAttr(list);
}
void publishRoutes(const Catalog &catalog, Builder &b) {
  for (auto [i, op] : llvm::enumerate(catalog.ops)) {
    op->setAttr("site", b.getI64IntegerAttr(i));
    op->setAttr("route_context", catalog.contexts[i]);
    op->setAttr("candidates", alternatives(b, catalog.regions[i]));
  }
}
DictionaryAttr chargeRecord(Builder &b, const RoutingPlan &p, size_t i,
                            double eta) {
  const auto &c = p.charges.at(i);
  NamedAttrList a;
  a.set("hoist_work", b.getF64FloatAttr(c.hoistedWork));
  a.set("giant_work", b.getF64FloatAttr(c.giantWork));
  a.set("amortized_key_bytes", b.getF64FloatAttr(c.amortizedBytes));
  a.set("key_cost", b.getF64FloatAttr(eta * c.amortizedBytes));
  return a.getDictionary(b.getContext());
}
} // namespace snabf::ir

#include "IR.h"
#include "pass4/LayoutContract.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cmath>
#include <set>
#include "Dialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Attrs.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
namespace snabf::ir {
void SNABF::initialize() {
  addTypes<CtType>();
  addAttributes<BudgetAttr,SlotLayoutAttr>();
  addOperations<ProfileOp, SourceOp, LintransOp, PolyEvalOp, StepOp, ConvertOp,
                SinkOp>();
}
static bool pow2(int64_t n) { return n > 0 && !(n & (n - 1)); }
LogicalResult CtType::verify(function_ref<InFlightDiagnostic()> err,
                             RankedTensorType logical, int64_t n, int64_t level,
                             FlatSymbolRefAttr profile) {
  if (!logical.hasStaticShape() || logical.getNumElements() < 1 || !pow2(n) ||
      n < 4 || level < 0 || !profile)
    return err() << "expected a static nonempty tensor, power-of-two ring and "
                    "nonnegative level";
  return success();
}
LogicalResult BudgetAttr::verify(function_ref<InFlightDiagnostic()> err,
                                 DictionaryAttr d) {
  auto interval = d.getAs<DenseF64ArrayAttr>("interval");
  if (!interval || interval.size() != 2 || !std::isfinite(interval[0]) ||
      !std::isfinite(interval[1]) || interval[0] > interval[1])
    return err() << "invalid decoded interval";
  for (auto name : {"canonical_bias", "variance_proxy", "validity_failure",
                    "decoded_error"}) {
    auto a = d.get(name);
    if (isa_and_nonnull<UnitAttr>(a)) continue;  // unit = unknown
    auto f = dyn_cast_or_null<FloatAttr>(a);
    if (!f || !std::isfinite(f.getValueAsDouble()) ||
        f.getValueAsDouble() < 0 ||
        (StringRef(name) == "validity_failure" && f.getValueAsDouble() > 1))
      return err() << "invalid or missing " << name;
  }
  return success();
}
LogicalResult ProfileOp::verify() {
  int64_t n = getRing();
  if (!pow2(n) || n < 4 || n > (1LL << 30) || !pow2(getSlots()) ||
      int64_t(getSlots()) > n / 2 || !getQ().size() || !getP().size() ||
      getKeyBytesAttr().getInt() <= 0 || getKeyId().empty() ||
      getDefaultLogScaleAttr().getInt() <= 0)
    return emitOpError("invalid ring, slots, key family, scale or chain");
  std::set<int64_t> all;
  for (auto list : {getQ(), getP()})
    for (auto prime : list)
      if (prime < 3 || prime % (2 * n) != 1 || !all.insert(prime).second)
        return emitOpError("invalid or repeated Q/P modulus");
  return success();
}
LogicalResult SlotLayoutAttr::verify(function_ref<InFlightDiagnostic()> err,DictionaryAttr d) {
  using namespace ::snabf::pass4;
  for(auto k:{"ring_dim","num_slots","logical_width","stride","period","active_level","lane_count"})if(!d.getAs<IntegerAttr>(k))return err()<<"missing layout field: "<<k;
  auto num=[&](StringRef k){return d.getAs<IntegerAttr>(k).getInt();};auto mode=d.getAs<StringAttr>("packing_mode");auto s=d.getAs<FloatAttr>("log2_scale");
  if(!mode||!s)return err()<<"invalid packing mode or scale";
  Packing packing;if(mode=="periodic_single")packing=Packing::PeriodicSingle;else if(mode=="single_strided")packing=Packing::SingleStrided;else if(mode=="packed_multi")packing=Packing::PackedMulti;else return err()<<"unknown packing mode";
  auto offsets=d.getAs<DenseI64ArrayAttr>("offsets"),active=d.getAs<DenseI64ArrayAttr>("active_lanes"),mask=d.getAs<DenseI64ArrayAttr>("mask_signature");
  if(!offsets||!active||!mask||num("lane_count")!=offsets.size())return err()<<"invalid_lane_map";
  auto vec=[](DenseI64ArrayAttr a){auto v=a.asArrayRef();return std::vector<int64_t>(v.begin(),v.end());};
  auto reason=validate({num("ring_dim"),num("num_slots"),num("logical_width"),num("stride"),num("period"),num("active_level"),s.getValueAsDouble(),packing,vec(offsets),vec(active),vec(mask)});
  if(!reason.empty())return err()<<reason;return success();
}
static LogicalResult checkType(Operation *op, CtType t) {
  auto p = SymbolTable::lookupNearestSymbolFrom<ProfileOp>(op, t.getProfile());
  if (!p || int64_t(p.getRing()) != t.getN() ||
      t.getLevel() >= int64_t(p.getQ().size()))
    return op->emitOpError("ciphertext ring or level disagrees with profile ")
           << t.getProfile();
  return success();
}
static LogicalResult ordinary(Operation *op) {
  auto out = cast<CtType>(op->getResult(0).getType());
  if (failed(checkType(op, out))) return failure();
  for (auto operand : op->getOperands())
    if (auto in = dyn_cast<CtType>(operand.getType())) {
      if (failed(checkType(op, in))) return failure();
      if (in.getProfile() != out.getProfile() ||
          in.getLogical() != out.getLogical() || out.getLevel() > in.getLevel())
        return op->emitOpError(
            "incompatible key/layout profile, shape or level; use an explicit "
            "conversion");
    }
  return success();
}
LogicalResult SourceOp::verify() { return ordinary(*this); }
LogicalResult SinkOp::verify() {
  return checkType(*this, getInput().getType());
}
LogicalResult StepOp::verify() {
  auto k = getKind();
  auto vs = getInputs();
  bool unary = k == "ckks.rescale" || k == "ckks.level_reduce" ||
               k == "ckks.relinearize";
  if ((!unary && k != "ckks.mul_plain" && k != "ckks.mul" && k != "ckks.add") ||
      vs.size() != (unary ? 1u : 2u) || !isa<CtType>(vs[0].getType()))
    return emitOpError("invalid CKKS operation or ciphertext arity");
  if (!unary && ((k == "ckks.mul" || k == "ckks.add") != isa<CtType>(vs[1].getType())))
    return emitOpError("invalid multiply operand type");
  if (k == "ckks.mul" && cast<CtType>(vs[0].getType()).getLevel() !=
                             cast<CtType>(vs[1].getType()).getLevel())
    return emitOpError("unaligned ciphertext levels");
  return ordinary(*this);
}
LogicalResult LintransOp::verify() {
  if (failed(ordinary(*this))) return failure();
  auto p = SymbolTable::lookupNearestSymbolFrom<ProfileOp>(
      *this, getInput().getType().getProfile());
  std::set<int64_t> seen;
  for (auto d : getDiagonals())
    if (d < 0 || d >= int64_t(p.getSlots()) || !seen.insert(d).second)
      return emitOpError("non-canonical or duplicate packed diagonal");
  return success(!seen.empty());
}
LogicalResult PolyEvalOp::verify() {
  if (auto activation = (*this)->getAttrOfType<StringAttr>("activation")) {
    if (activation.getValue() != "silu" || !getCoefficients().empty())
      return emitOpError("a logical SiLU owns its full DAG, not an imported Chebyshev series");
    return ordinary(*this);
  }
  if (getCoefficients().empty() ||
      !llvm::all_of(getCoefficients(),
                    [](Attribute a) { return isa<FloatAttr>(a); }))
    return emitOpError("expected floating-point Chebyshev coefficients");
  return ordinary(*this);
}
LogicalResult ConvertOp::verify() {
  auto in = getInput().getType(), out = getOut().getType();
  if (failed(checkType(*this, in)) || failed(checkType(*this, out)))
    return failure();
  if(bool(getProvidedAttr())!=bool(getRequiredAttr()))return emitOpError("incomplete_layout_contract");
  if(getProvidedAttr())for(auto [a,t]:{std::pair{getProvidedAttr(),in},std::pair{getRequiredAttr(),out}}){auto f=a.getFields();if(f.getAs<IntegerAttr>("ring_dim").getInt()!=t.getN()||f.getAs<IntegerAttr>("active_level").getInt()!=t.getLevel()||f.getAs<IntegerAttr>("logical_width").getInt()!=t.getLogical().getNumElements())return emitOpError("stale_layout_type");}
  if (in.getN() <= out.getN() || in.getN() % out.getN() ||
      in.getLogical() != out.getLogical() || out.getLevel() > in.getLevel() ||
      (getLayoutContract() != "trace-invariant" &&
       getLayoutContract() != "split-repack"))
    return emitOpError(
        "invalid DownConv ring, level, shape or layout contract");
  return success();
}
}  // namespace snabf::ir

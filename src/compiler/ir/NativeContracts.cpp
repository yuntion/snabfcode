#include "NativeContracts.h"
#include <cmath>
#include <gmpxx.h>
#include <mpfr.h>
#include <set>
#include <vector>

using namespace mlir;
namespace snabf::ir {
namespace {
using Diag = function_ref<InFlightDiagnostic()>;
StringRef text(DictionaryAttr d, StringRef key) {
  auto a = d.getAs<StringAttr>(key);
  return a ? a.getValue() : StringRef{};
}
int64_t integer(DictionaryAttr d, StringRef key) {
  auto a = d.getAs<IntegerAttr>(key);
  return a && a.getType().isSignlessInteger(64) ? a.getInt() : -1;
}
bool digest(StringRef s) {
  return s.size() == 64 && llvm::all_of(s, [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
bool positiveDecimal(StringRef s) {
  return !s.empty() && s[0] != '0' &&
         llvm::all_of(s, [](char c) { return c >= '0' && c <= '9'; });
}
mpq_class value(ExactScaleAttr s) {
  return {mpz_class(s.getNumerator().str()),
          mpz_class(s.getDenominator().str())};
}
bool dyadic(ExactScaleAttr a, unsigned precision) {
  if (!a)
    return false;
  auto x = value(a);
  auto n = x.get_num_mpz_t();
  return mpz_popcount(x.get_den_mpz_t()) == 1 &&
         mpz_sizeinbase(n, 2) - mpz_scan1(n, 0) <= precision;
}
bool distinctIn(DenseI64ArrayAttr a, int64_t limit) {
  if (!a)
    return false;
  std::set<int64_t> seen;
  return llvm::all_of(a.asArrayRef(), [&](int64_t i) {
    return i >= 0 && i < limit && seen.insert(i).second;
  });
}
bool matches(LayoutSpecAttr layout, RankedTensorType tensor) {
  return layout && tensor && tensor.hasStaticShape() &&
         layout.getFields()
                 .getAs<DenseI64ArrayAttr>("logical_dims")
                 .asArrayRef() == tensor.getShape();
}
bool matches(LayoutSpecAttr layout, CtType ct) {
  return matches(layout, ct.getLogical()) &&
         integer(layout.getFields(), "num_slots") == ct.getN() / 2;
}
bool ports(ArrayAttr list, bool input) {
  if (!list || list.empty())
    return false;
  std::set<std::pair<std::string, int64_t>> seen;
  for (auto item : list) {
    auto d = dyn_cast<DictionaryAttr>(item);
    if (!d || d.get("arg_name") || d.get(input ? "result_index" : "arg_index"))
      return false;
    auto sym = d.getAs<FlatSymbolRefAttr>("func_symbol");
    auto i = integer(d, input ? "arg_index" : "result_index");
    auto t = d.getAs<TypeAttr>("logical_type");
    if (!sym || i < 0 || !t ||
        !matches(d.getAs<LayoutSpecAttr>("layout"),
                 dyn_cast<RankedTensorType>(t.getValue())) ||
        !seen.emplace(sym.getValue().str(), i).second)
      return false;
    if (input) {
      if (text(d, "source_id").empty())
        return false;
      // Missing or unit means unknown. Never manufacture an input domain.
      if (auto domain = d.get("input_domain");
          domain && !isa<UnitAttr>(domain)) {
        auto range = dyn_cast<DenseF64ArrayAttr>(domain);
        if (!range || range.size() != 2 || !std::isfinite(range[0]) ||
            !std::isfinite(range[1]) || range[0] > range[1])
          return false;
      }
    }
  }
  return true;
}
bool moduli(DenseI64ArrayAttr a, std::set<int64_t> &seen) {
  return a && a.size() && llvm::all_of(a.asArrayRef(), [&](int64_t q) {
           return q > 2 && (q & 1) && seen.insert(q).second;
         });
}
bool useSite(DictionaryAttr d) {
  return !text(d, "consumer").empty() && integer(d, "operand_index") >= 0;
}
bool boundTo(EncodingCallReceiptAttr receipt, StringRef site) {
  auto f = receipt.getFields();
  return text(f, "site") == site && integer(f, "operand_index") == 1;
}
bool rational(StringRef spelling, mpq_class &x) {
  if (spelling.empty() || x.set_str(spelling.str(), 10) != 0 ||
      !mpz_sgn(x.get_den_mpz_t()))
    return false;
  x.canonicalize();
  return spelling == x.get_str();
}
bool nativeAttr(Attribute a) {
  if (isa<FrontendProvenanceAttr, LayoutSpecAttr, ExactScaleAttr,
          EncodingCallReceiptAttr, ScalePlanAttr>(a))
    return true;
  if (auto list = dyn_cast<ArrayAttr>(a))
    return llvm::any_of(list, nativeAttr);
  if (auto dict = dyn_cast<DictionaryAttr>(a))
    return llvm::any_of(dict, [](NamedAttribute f) {
      auto k = f.getName().getValue();
      return k == "snabf.frontend_provenance" || k == "snabf.scale_plan" ||
             k == "snabf.exact_scale" || k == "snabf.layout_spec" ||
             k == "snabf.encoding_call_receipt" || k == "snabf.physical_scale_exact" || nativeAttr(f.getValue());
    });
  return false;
}
LogicalResult compatible(Operation *op, bool sameShape = true) {
  auto out = cast<CtType>(op->getResult(0).getType());
  auto p =
      SymbolTable::lookupNearestSymbolFrom<ProfileOp>(op, out.getProfile());
  if (!p || out.getN() != int64_t(p.getRing()) ||
      out.getLevel() >= int64_t(p.getQ().size()))
    return op->emitOpError("invalid native result profile");
  for (auto v : op->getOperands())
    if (auto t = dyn_cast<CtType>(v.getType())) {
      if (t.getN() != out.getN() || t.getProfile() != out.getProfile() ||
          t.getLevel() >= int64_t(p.getQ().size()) ||
          out.getLevel() > t.getLevel() ||
          (sameShape && t.getLogical() != out.getLogical()))
        return op->emitOpError("incompatible native ciphertext types");
    }
  return success();
}
LogicalResult sameLevel(Operation *op) {
  if (failed(compatible(op)))
    return failure();
  auto level = cast<CtType>(op->getResult(0).getType()).getLevel();
  for (Value v : op->getOperands())
    if (auto t = dyn_cast<CtType>(v.getType()); t && t.getLevel() != level)
      return op->emitOpError(
          "requires equal levels; represent rescale/drop explicitly");
  return success();
}
} // namespace

LogicalResult rejectNativeExecution(ModuleOp module) {
  auto result = module.walk([](Operation *op) -> WalkResult {
    bool native = isa<AlignScaleOp, SubOp, NegateOp, AddPlainOp, MulPlainOp,
                      CtMulOp, RotateOp, ReduceSumOp>(op);
    if (auto lin = dyn_cast<LintransOp>(op))
      native |= bool(lin.getOutsInit() || lin.getInLayoutAttr() ||
                     lin.getOutLayoutAttr());
    if (native || nativeAttr(op->getAttrDictionary())) {
      op->emitError("[SNABF_NATIVE_FRONTEND_UNAVAILABLE] RFC-004 N0 contracts "
                    "are not executable");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult verifyNativeLintrans(LintransOp op) {
  if (failed(compatible(op, false)))
    return failure();
  auto in = op.getInput().getType(), out = op.getOut().getType();
  if (!matches(op.getInLayoutAttr(), in) ||
      !matches(op.getOutLayoutAttr(), out) || !op.getOutsInit() ||
      op.getOutsInit().getType() != out.getLogical())
    return op.emitOpError("native lintrans needs both layouts and the original "
                          "typed outs operand");
  auto table = dyn_cast<RankedTensorType>(op.getTable().getType());
  if (!table ||
      table.getShape() != ArrayRef<int64_t>({out.getLogical().getNumElements(),
                                             in.getLogical().getNumElements()}))
    return op.emitOpError("native weight matrix must have shape [M_out, M_in]");
  return success();
}
LogicalResult AlignScaleOp::verify() {
  if (failed(compatible(*this)))
    return failure();
  auto plan = getScalePlanAttr().getFields();
  auto in = getInput().getType(), out = getOut().getType();
  auto profile =
      SymbolTable::lookupNearestSymbolFrom<ProfileOp>(*this, in.getProfile());
  if (!boundTo(getEncodingAttr(), getSite()) ||
      text(plan, "mode") != "explicit_align_mul" ||
      integer(plan, "level_in") != in.getLevel() ||
      integer(plan, "level_out") != out.getLevel() ||
      plan.getAs<DenseI64ArrayAttr>("ordered_q") != profile.getQAttr() ||
      plan.getAs<DenseI64ArrayAttr>("ordered_p") != profile.getPAttr())
    return emitOpError(
        "explicit align must match the live levels and ordered profile Q/P");
  return success();
}
LogicalResult NegateOp::verify() { return sameLevel(*this); }
LogicalResult SubOp::verify() {
  if (failed(sameLevel(*this)))
    return failure();
  auto f = getScalePlanAttr().getFields();
  auto in = getLhs().getType();
  auto p =
      SymbolTable::lookupNearestSymbolFrom<ProfileOp>(*this, in.getProfile());
  if (integer(f, "level_in") != in.getLevel() ||
      integer(f, "level_out") != in.getLevel() ||
      f.getAs<DenseI64ArrayAttr>("ordered_q") != p.getQAttr() ||
      f.getAs<DenseI64ArrayAttr>("ordered_p") != p.getPAttr())
    return emitOpError("join scale plan does not match levels and ordered Q/P");
  return success();
}
template <class Op> static LogicalResult plain(Op op) {
  if (!boundTo(op.getEncodingAttr(), op.getSite()))
    return op.emitOpError("encoding receipt site/operand mismatch");
  if (op.getPlain().getType() != op.getInput().getType().getLogical())
    return op.emitOpError("explicit broadcast/layout required");
  return sameLevel(op);
}
LogicalResult AddPlainOp::verify() { return plain(*this); }
LogicalResult MulPlainOp::verify() { return plain(*this); }
LogicalResult CtMulOp::verify() {
  if (failed(sameLevel(*this)))
    return failure();
  if (value(getRequestedProductAttr()) !=
      value(getLhsScaleAttr()) * value(getRhsScaleAttr()))
    return emitOpError(
        "requested product must equal lhs_scale times rhs_scale");
  return success();
}
LogicalResult RotateOp::verify() {
  if (failed(sameLevel(*this)))
    return failure();
  if (!matches(getLayoutAttr(), getInput().getType()) ||
      getOffsetAttr().getInt() < 0 ||
      getOffsetAttr().getInt() >= getInput().getType().getN() / 2)
    return emitOpError("invalid layout or rotation");
  return success();
}
LogicalResult ReduceSumOp::verify() {
  if (failed(compatible(*this, false)))
    return failure();
  auto in = getInput().getType(), out = getOut().getType();
  if (!matches(getInLayoutAttr(), in) || !matches(getOutLayoutAttr(), out) ||
      in.getLevel() != out.getLevel())
    return emitOpError("invalid reduction layouts or levels");
  auto shape = in.getLogical().getShape();
  if (getAxes().empty() || !distinctIn(getAxesAttr(), shape.size()))
    return emitOpError("invalid reduction axes");
  std::set<int64_t> axes(getAxes().begin(), getAxes().end());
  std::vector<int64_t> outShape;
  for (unsigned a = 0; a < shape.size(); ++a) {
    if (!axes.count(a))
      outShape.push_back(shape[a]);
    else if (getKeepDims())
      outShape.push_back(1);
  }
  auto ins = getInLayoutAttr()
                 .getFields()
                 .getAs<DenseI64ArrayAttr>("slot_map")
                 .asArrayRef();
  auto outs = getOutLayoutAttr()
                  .getFields()
                  .getAs<DenseI64ArrayAttr>("slot_map")
                  .asArrayRef();
  if (out.getLogical().getShape() != ArrayRef<int64_t>(outShape) ||
      getValidSlots() != ins || getOutputSlots() != outs ||
      getGroups().size() != outs.size() ||
      getEdgeDivisors().size() != outs.size())
    return emitOpError("invalid reduction shape, valid slots, output placement "
                       "or group count");
  // slot_map is indexed by logical row-major coordinate, independent of
  // physical packing.
  std::vector<std::vector<int64_t>> expected(outs.size());
  for (size_t j = 0; j < ins.size(); ++j) {
    size_t idx = j, group = 0, stride = 1;
    for (size_t a = shape.size(); a--;) {
      if (!axes.count(a)) {
        group += (idx % shape[a]) * stride;
        stride *= shape[a];
      }
      idx /= shape[a];
    }
    expected[group].push_back(ins[j]);
  }
  for (size_t j = 0; j < outs.size(); ++j) {
    auto group = dyn_cast<DenseI64ArrayAttr>(getGroups()[j]);
    if (!group || group.asArrayRef() != ArrayRef<int64_t>(expected[j]) ||
        getEdgeDivisors()[j] != int64_t(expected[j].size()))
      return emitOpError("cross-row reduction group or incorrect edge divisor");
  }
  return success();
}

LogicalResult ExactScaleAttr::verify(Diag err, StringRef n, StringRef d) {
  if (!positiveDecimal(n) || !positiveDecimal(d) ||
      gcd(mpz_class(n.str()), mpz_class(d.str())) != 1)
    return err() << "expected a reduced positive rational in canonical decimal";
  return success();
}
LogicalResult FrontendProvenanceAttr::verify(Diag err, DictionaryAttr d) {
  auto mode = text(d, "mode"), policy = text(d, "scale_policy");
  if ((mode != "native-linalg" && mode != "legacy-heir") ||
      !digest(text(d, "source_sha256")) ||
      (policy != "narrow" && policy != "heir-anchor" &&
       policy != "wide-heterogeneous") ||
      !ports(d.getAs<ArrayAttr>("inputs"), true) ||
      !ports(d.getAs<ArrayAttr>("outputs"), false))
    return err() << "invalid frontend provenance, input or result port";
  return success();
}
LogicalResult LayoutSpecAttr::verify(Diag err, DictionaryAttr d) {
  auto dims = d.getAs<DenseI64ArrayAttr>("logical_dims"),
       axes = d.getAs<DenseI64ArrayAttr>("axis_order");
  auto slots = d.getAs<DenseI64ArrayAttr>("slot_map");
  auto n = integer(d, "num_slots"), period = integer(d, "period"),
       used = integer(d, "used_slots");
  auto kind = text(d, "kind");
  if (!dims || !axes || axes.size() != dims.size() ||
      !distinctIn(axes, dims.size()) || n <= 0 || (n & (n - 1)) ||
      period <= 0 || n % period || !distinctIn(slots, period) ||
      used != slots.size() || text(d, "rotate_direction") != "left" ||
      (kind != "periodic_single" && kind != "packed_multi" &&
       kind != "block_bundle"))
    return err() << "invalid layout, axis order or slot map";
  int64_t count = 1;
  for (auto size : dims.asArrayRef()) {
    if (size <= 0 || count > period / size)
      return err() << "logical shape exceeds slot period";
    count *= size;
  }
  if (count != used)
    return err() << "slot map must cover each logical element once";
  return success();
}
LogicalResult EncodingCallReceiptAttr::verify(Diag err, DictionaryAttr d) {
  auto batch = d.getAs<BoolAttr>("batched");
  auto p = integer(d, "encoder_precision");
  if (text(d, "call_id").empty() || text(d, "site").empty() ||
      integer(d, "operand_index") < 0 || !batch || !batch.getValue() ||
      text(d, "arg_type") != "[]float64" || integer(d, "arg_precision") != 53 ||
      p <= 0 || !digest(text(d, "payload_sha256")) ||
      !d.getAs<ExactScaleAttr>("requested") ||
      !dyadic(d.getAs<ExactScaleAttr>("metadata"), 128) ||
      !dyadic(d.getAs<ExactScaleAttr>("effective"), p <= 53 ? 53 : 128))
    return err() << "unbound encoding call or unsupported encoding branch";
  bool f64 = p <= 53;
  if (text(d, "branch") != (f64 ? "embed_double_f64" : "bigfloat_value") ||
      text(d, "quantization") != (f64 ? "batched-f64" : "batched-bigfloat"))
    return err()
           << "encoding branch/quantization disagrees with encoder precision";
  auto m = value(d.getAs<ExactScaleAttr>("metadata"));
  auto e = value(d.getAs<ExactScaleAttr>("effective"));
  if (f64) {
    mpfr_t v;
    mpfr_init2(v, 128);
    int inexact = mpfr_set_q(v, m.get_mpq_t(), MPFR_RNDN);
    double encoded = mpfr_get_d(v, MPFR_RNDN);
    mpfr_clear(v);
    if (inexact || !std::isfinite(encoded) || encoded <= 0 ||
        e != mpq_class(encoded))
      return err() << "effective encoding scale is not RN-binary64(metadata)";
  } else if (m != e)
    return err() << "BigFloatValue requires E=M";
  return success();
}
LogicalResult ScalePlanAttr::verify(Diag err, DictionaryAttr d) {
  auto mode = text(d, "mode");
  auto q = d.getAs<DenseI64ArrayAttr>("ordered_q"),
       p = d.getAs<DenseI64ArrayAttr>("ordered_p");
  auto dropped = d.getAs<DenseI64ArrayAttr>("dropped_q");
  auto depth = integer(d, "mul_depth_delta"),
       levels = integer(d, "levels_consumed"), level = integer(d, "level_in");
  auto lhs = d.getAs<ExactScaleAttr>("in_scale"),
       rhs = d.getAs<ExactScaleAttr>("out_scale");
  std::set<int64_t> primes;
  if (text(d, "producer").empty() || !useSite(d) || !dyadic(lhs, 128) ||
      !dyadic(rhs, 128) || !moduli(q, primes) || !moduli(p, primes) ||
      level < 0 || level >= q.size() || !dropped || levels != dropped.size() ||
      levels > level || integer(d, "level_out") != level - levels)
    return err() << "incomplete scale plan or invalid levels/ordered Q/P";
  for (int64_t i = 0; i < levels; ++i)
    if (dropped[i] != q[level - i])
      return err() << "dropped_q is not in execution order";
  if (mode == "explicit_align_mul") {
    if (depth != 1 || levels != 1 || text(d, "plaintext") != "encoded-one")
      return err() << "Tier 4 requires encoded plaintext one and one dropped "
                      "Q; multi-prime rescale unavailable";
  } else {
    if (depth != 0 || levels != 0)
      return err() << "non-explicit coordination cannot charge extra levels";
    if (mode == "none" || mode == "exact_prime_match") {
      if (lhs != rhs)
        return err() << "exact scale match required";
    } else if (mode == "fold_into_plain") {
      auto receipt = d.getAs<EncodingCallReceiptAttr>("encoding");
      auto uses = d.getAs<ArrayAttr>("uses");
      mpq_class residual;
      if (!receipt || !uses || uses.empty() ||
          !llvm::all_of(uses,
                        [](Attribute u) {
                          auto s = dyn_cast<DictionaryAttr>(u);
                          return s && useSite(s);
                        }) ||
          !d.getAs<ExactScaleAttr>("requested") ||
          !d.getAs<ExactScaleAttr>("realized") ||
          !rational(text(d, "residual"), residual) ||
          !digest(text(d, "logical_payload_sha256")) ||
          text(d, "logical_payload_sha256") !=
              text(receipt.getFields(), "payload_sha256"))
        return err() << "fold requires an existing encoding, unchanged payload "
                        "and per-use bindings";
      if (d.get("requested") != receipt.getFields().get("requested") ||
          d.get("realized") != receipt.getFields().get("effective") ||
          residual != value(d.getAs<ExactScaleAttr>("realized")) /
                              value(d.getAs<ExactScaleAttr>("requested")) -
                          1)
        return err()
               << "inconsistent requested, realized or relative fold residual";
    } else if (mode == "bounded_mismatch_bias") {
      // No implicit tolerance or free precision budget. Numeric proof belongs
      // to N1 replay.
      mpq_class x;
      for (auto k : {"lhs_abs_bound", "rhs_abs_bound", "lhs_multiplier",
                     "rhs_multiplier"}) {
        if (!rational(text(d, k), x) || x < 0 ||
            (StringRef(k).ends_with("multiplier") &&
             (x < 1 || x.get_den() != 1)))
          return err() << "invalid two-sided mismatch accounting: " << k;
      }
      if (d.get("allocated_bias") &&
          (!rational(text(d, "allocated_bias"), x) || x < 0))
        return err() << "invalid allocated bias (absent means zero)";
    } else
      return err() << "unknown scale coordination mode";
  }
  return success();
}
} // namespace snabf::ir

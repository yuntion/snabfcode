#include "NativeLinalgImportInternal.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

using namespace mlir;
namespace snabf::native {
namespace {
std::string sha256(StringRef bytes) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes)), true);
}

enum class Role { Unknown, Data, Literal, Empty, ZeroInit };
struct Graph {
  const Source &s;
  Entry &entry;
  CheckedMatvec visit;
  ValueIDs ids;
  llvm::DenseMap<Value, Role> roles;
  json::Array nodes, locks, outputs;

  bool positiveZero(Value v) const {
    auto c = v.getDefiningOp<arith::ConstantOp>();
    if (!c) return false;
    auto zero = [](const llvm::APFloat &n) { return n.isZero() && !n.isNegative(); };
    if (auto f = dyn_cast<FloatAttr>(c.getValue())) return zero(f.getValue());
    if (auto d = dyn_cast<DenseFPElementsAttr>(c.getValue())) return llvm::all_of(d.getValues<llvm::APFloat>(), zero);
    return false;
  }

  bool zeroOut(Value v) const {
    return v.getType() == entry.vector && (roles.lookup(v) == Role::ZeroInit || positiveZero(v));
  }

  LogicalResult constant(arith::ConstantOp c, json::Object &node) {
    if (failed(s.attrs(c, {"value"}))) return failure();
    Type t = c.getType(), elt = entry.vector.getElementType();
    json::Array bits, shape;
    auto add = [&](const llvm::APFloat &v) -> LogicalResult {
      if (!v.isFinite()) return s.reject(c, "[SNABF_NATIVE_LITERAL] expected finite public constant");
      bits.push_back(floatBits(v));
      return success();
    };
    if (auto f = dyn_cast<FloatAttr>(c.getValue()); f && t == elt) {
      if (failed(add(f.getValue()))) return failure();
    } else {
      auto d = dyn_cast<DenseFPElementsAttr>(c.getValue());
      auto tensor = dyn_cast<RankedTensorType>(t);
      int64_t width = entry.vector.getDimSize(0);
      if (!d || !tensor || tensor.getEncoding() || tensor.getElementType() != elt ||
          (t != entry.vector && !(tensor.getRank() == 2 && tensor.getDimSize(0) == width && tensor.getDimSize(1) == width)))
        return s.reject(c, "[SNABF_NATIVE_LITERAL] expected a scalar, vector or square public dense matrix of the entry element type");
      for (int64_t dim : tensor.getShape()) shape.push_back(dim);
      for (const auto &v : d.getValues<llvm::APFloat>()) if (failed(add(v))) return failure();
    }
    node["kind"] = "constant";
    node["element_type"] = typeText(elt);
    node["shape"] = std::move(shape);
    node["bits"] = std::move(bits); // expanded, row-major IEEE words, including signed zeros
    return success();
  }

  LogicalResult fill(linalg::FillOp f) {
    if (failed(s.attrs(f, {"operandSegmentSizes"}))) return failure();
    if (f.getInputs().size()!=1 || f.getOutputs().size()!=1 || f->getNumResults()!=1 ||
        f.getInputs()[0].getType()!=entry.vector.getElementType() || !positiveZero(f.getInputs()[0]) ||
        roles.lookup(f.getOutputs()[0])!=Role::Empty || f->getResult(0).getType()!=entry.vector || !f.getRegion().hasOneBlock())
      return s.reject(f, "[SNABF_NATIVE_OUTS] expected positive-zero scalar fill into a fresh tensor.empty");
    auto &b = f.getRegion().front();
    if (b.getNumArguments()!=2 || !llvm::hasSingleElement(b) || !isa<linalg::YieldOp>(b.front()) ||
        b.front().getNumOperands()!=1 || b.front().getOperand(0)!=b.getArgument(0))
      return s.reject(f, "[SNABF_NATIVE_OUTS] fill must yield the scalar input unchanged");
    return s.attrs(&b.front(), {});
  }

  LogicalResult matvec(linalg::MatvecOp m) {
    if (failed(s.attrs(m, {"operandSegmentSizes", "linalg.memoized_indexing_maps"}))) return failure();
    auto ins = m.getInputs(), outs = m.getOutputs();
    if (ins.size()!=2 || outs.size()!=1 || m->getNumResults()!=1 || m->getResult(0).getType()!=entry.vector ||
        roles.lookup(ins[0])!=Role::Literal || ins[1].getType()!=entry.vector || roles.lookup(ins[1])!=Role::Data)
      return s.reject(m, "[SNABF_NATIVE_MATVEC] expected a public constant matrix and a data vector");
    auto w = dyn_cast<RankedTensorType>(ins[0].getType());
    if (!w || w.getRank()!=2 || w.getDimSize(0)!=entry.vector.getDimSize(0) || w.getDimSize(1)!=w.getDimSize(0))
      return s.reject(m, "[SNABF_NATIVE_MATVEC] weight type is not the matching square matrix");
    if (!zeroOut(outs[0])) return s.reject(m, "[SNABF_NATIVE_OUTS] matvec must accumulate into proven positive zero");
    auto linalgOp = cast<linalg::LinalgOp>(m.getOperation());
    auto *ctx = m.getContext();
    auto i = getAffineDimExpr(0, ctx), j = getAffineDimExpr(1, ctx);
    SmallVector<AffineMap> expected{AffineMap::get(2, 0, {i,j}, ctx), AffineMap::get(2, 0, j), AffineMap::get(2, 0, i)};
    if (linalgOp.getIndexingMapsArray()!=expected || linalgOp.getIteratorTypesArray()!=
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::reduction} || !m.getRegion().hasOneBlock())
      return s.reject(m, "[SNABF_NATIVE_MATVEC] expected row-by-column matvec indexing and reduction");
    auto &b = m.getRegion().front();
    if (b.getNumArguments()!=3 || std::distance(b.begin(), b.end())!=3)
      return s.reject(m, "[SNABF_NATIVE_MATVEC] expected the scalar multiply-accumulate region");
    auto *mul = &b.front(), *add = mul->getNextNode(), *y = &b.back();
    if (!isa<arith::MulFOp>(mul) || mul->getOperand(0)!=b.getArgument(0) || mul->getOperand(1)!=b.getArgument(1) ||
        !isa<arith::AddFOp>(add) || add->getOperand(0)!=b.getArgument(2) || add->getOperand(1)!=mul->getResult(0) ||
        !isa<linalg::YieldOp>(y) || y->getNumOperands()!=1 || y->getOperand(0)!=add->getResult(0))
      return s.reject(m, "[SNABF_NATIVE_MATVEC] scalar region is not acc + weight * x");
    if (failed(s.strictFloatOp(mul)) || failed(s.strictFloatOp(add))) return failure();
    return s.attrs(y, {});
  }

  LogicalResult operation(Operation &op) {
    auto name = op.getName().getStringRef();
    json::Array operands;
    for (Value v : op.getOperands()) {
      auto it = ids.find(v);
      if (it == ids.end()) return s.reject(&op, "[SNABF_NATIVE_SSA] expected an earlier source definition");
      operands.push_back(it->second);
    }
    if (auto r = dyn_cast<func::ReturnOp>(op)) {
      if (failed(s.attrs(r, {}))) return failure();
      if (r.getNumOperands()!=1 || roles.lookup(r.getOperand(0))!=Role::Data)
        return s.reject(r, "[SNABF_NATIVE_ABI] expected one defined data result");
      outputs.push_back(json::Object{{"func_symbol", entry.func.getSymName().str()}, {"result_index", 0},
                                    {"value", ids.lookup(r.getOperand(0))}, {"logical_type", typeText(entry.vector)},
                                    {"layout", "periodic"}, {"period", entry.vector.getDimSize(0)}, {"location", s.location(r)}});
      return success();
    }
    if (op.getNumResults()!=1) return s.reject(&op, "[SNABF_NATIVE_OP] unsupported operation " + name);
    Value result = op.getResult(0);
    std::string id = "n" + std::to_string(nodes.size());
    json::Object node{{"id", id}, {"op", name.str()}, {"location", s.location(&op)}, {"inputs", std::move(operands)},
                      {"logical_type", typeText(result.getType())}};
    Role role = Role::Data;
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      role = Role::Literal;
      if (failed(constant(c, node))) return failure();
    } else if (isa<tensor::EmptyOp>(op)) {
      if (failed(s.attrs(&op, {}))) return failure();
      if (op.getNumOperands()!=0 || result.getType()!=entry.vector)
        return s.reject(&op, "[SNABF_NATIVE_SHAPE] expected the static entry vector");
      role = Role::Empty;
      node["kind"] = "empty";
    } else if (auto f = dyn_cast<linalg::FillOp>(op)) {
      if (failed(fill(f))) return failure();
      role = Role::ZeroInit;
      node["kind"] = "zero_fill";
    } else if (auto m = dyn_cast<linalg::MatvecOp>(op)) {
      if (failed(matvec(m))) return failure();
      if (visit && failed(visit(s, m, ids, id))) return failure();
      node["kind"] = "matvec";
      node["outs"] = ids.lookup(m.getOutputs().front());
    } else if (auto g = dyn_cast<linalg::GenericOp>(op)) {
      if (g.getInputs().size()!=1 || g.getOutputs().size()!=1 || g.getInputs()[0].getType()!=entry.vector ||
          result.getType()!=entry.vector || roles.lookup(g.getInputs()[0])!=Role::Data ||
          !(roles.lookup(g.getOutputs()[0])==Role::Empty || zeroOut(g.getOutputs()[0])))
        return s.reject(g, "[SNABF_NATIVE_GENERIC] expected a data vector and an unused empty/zero destination");
      json::Object lock;
      if (failed(lockActivation(s, g, ids, id, lock))) return failure();
      node["kind"] = lock.getString("kind")->str();
      locks.push_back(std::move(lock));
    } else if (isa<arith::AddFOp, arith::SubFOp, arith::NegFOp, arith::MulFOp>(op)) {
      if (failed(s.strictFloatOp(&op))) return failure();
      unsigned data = 0, plain = 0;
      for (Value v : op.getOperands()) {
        data += roles.lookup(v)==Role::Data;
        plain += roles.lookup(v)==Role::Literal;
      }
      if (result.getType()!=entry.vector || !data || data+plain!=op.getNumOperands() || (isa<arith::MulFOp>(op) && plain!=1))
        return s.reject(&op, "[SNABF_NATIVE_ELEMENTWISE] expected defined vectors; multiply requires one public literal");
      node["kind"] = isa<arith::AddFOp>(op) ? "add" : isa<arith::SubFOp>(op) ? "sub" : isa<arith::NegFOp>(op) ? "negate" : "mul_plain";
    } else {
      return s.reject(&op, "[SNABF_NATIVE_OP] unsupported operation " + name);
    }
    ids[result] = id;
    roles[result] = role;
    nodes.push_back(std::move(node));
    return success();
  }
};
} // namespace

LogicalResult inspectWithMatvecs(StringRef bytes, StringRef sourceName, StringRef specBytes,
                                 json::Object &receipt, llvm::raw_ostream &diagnostics,
                                 CheckedMatvec visit) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect, linalg::LinalgDialect,
                  tensor::TensorDialect, cf::ControlFlowDialect, scf::SCFDialect>();
  MLIRContext ctx(registry, MLIRContext::Threading::DISABLED);
  ctx.printOpOnDiagnostic(false);
  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(bytes, sourceName), llvm::SMLoc());
  SourceMgrDiagnosticHandler handler(mgr, &ctx, diagnostics);
  if (auto pos = bytes.find('\0'); pos != StringRef::npos) {
    auto *start = mgr.getMemoryBuffer(mgr.getMainFileID())->getBufferStart();
    auto [line, column] = mgr.getLineAndColumn(llvm::SMLoc::getFromPointer(start + pos));
    emitError(FileLineColLoc::get(&ctx, sourceName, line, column)) << "[SNABF_NATIVE_PARSE] embedded NUL in text source";
    return failure();
  }
  AsmParserState parser;
  Block top;
  if (failed(parseAsmSourceFile(mgr, &top, ParserConfig(&ctx, /*verifyAfterParse=*/false), &parser)))
    return failure();
  Source s{mgr, parser, sha256(bytes)};
  if (!llvm::hasSingleElement(top) || !isa<ModuleOp>(top.front())) {
    if (!top.empty()) return s.reject(&top.front(), "[SNABF_NATIVE_MODULE] expected exactly one explicit module");
    diagnostics << sourceName << ":1:1: error: [SNABF_NATIVE_MODULE] empty source\n";
    return failure();
  }
  auto module = cast<ModuleOp>(top.front());
  auto preflight = module.walk([&](Operation *op) -> WalkResult {
    // Point even upstream verifier diagnostics into these bytes, not an
    // optional source-supplied loc(...). This does not transform the SSA.
    auto [line, column] = s.point(op);
    op->setLoc(FileLineColLoc::get(&ctx, sourceName, line, column));
    auto ns = op->getName().getDialectNamespace();
    if (ns == "scf" || ns == "cf" || ns == "secret" || ns == "mgmt" || isa<func::CallOp>(op)) {
      (void)s.reject(op, "[SNABF_NATIVE_OP] unsupported control flow, call or HEIR operation: " + op->getName().getStringRef());
      return WalkResult::interrupt();
    }
    // Linalg caches derived maps during verification. A source-supplied
    // cache must not alter (or crash) the verifier's interpretation.
    if (op->hasAttr("linalg.memoized_indexing_maps")) {
      (void)s.reject(op, "[SNABF_NATIVE_ATTR] source-supplied linalg.memoized_indexing_maps");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (preflight.wasInterrupted() || failed(verify(module))) return failure();
  Entry entry;
  if (failed(inspectEntry(s, module, specBytes, entry))) return failure();
  Graph g{s, entry, visit, {}, {}, {}, {}, {}};
  Value x = entry.func.getArgument(0);
  g.ids[x] = "arg0";
  g.roles[x] = Role::Data;
  for (auto &op : entry.func.getBody().front()) if (failed(g.operation(op))) return failure();
  json::Array inputs;
  inputs.push_back(std::move(entry.input));
  receipt = json::Object{{"schema", "snabf.native-import.v1"}, {"frontend", "native-linalg"}, {"stage", "structural-inspection"},
                         {"source_sha256", s.digest}, {"input_spec_sha256", sha256(specBytes)},
                         {"func_symbol", entry.func.getSymName().str()}, {"inputs", std::move(inputs)}, {"outputs", std::move(g.outputs)},
                         {"nodes", std::move(g.nodes)}, {"activation_locks", std::move(g.locks)}};
  return success();
}

LogicalResult inspectNativeLinalg(StringRef bytes, StringRef sourceName, StringRef specBytes,
                                 json::Object &receipt, llvm::raw_ostream &diagnostics) {
  return inspectWithMatvecs(bytes, sourceName, specBytes, receipt, diagnostics, nullptr);
}
} // namespace snabf::native

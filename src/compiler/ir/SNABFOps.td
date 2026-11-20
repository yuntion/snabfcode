class Op_<string name, list<Trait> traits = []> : Op<SNABF, name, traits>;
def ProfileOp : Op_<"profile", [Symbol]> {
  let extraClassDeclaration = [{
    ::mlir::StringAttr getNameAttr() { return getSymNameAttr(); }
    ::mlir::SymbolTable::Visibility getVisibility() { return ::mlir::SymbolTable::getSymbolVisibility(*this); }
    void setVisibility(::mlir::SymbolTable::Visibility v) { ::mlir::SymbolTable::setSymbolVisibility(*this,v); }
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$ring, DenseI64ArrayAttr:$q,
    DenseI64ArrayAttr:$p, I64Attr:$default_log_scale, StrAttr:$key_id, AnyAttr:$layout,
    I64Attr:$slots, I64Attr:$key_bytes);
  let hasVerifier = 1;
}
class CtOp<string name, dag args> : Op_<name> {
  let arguments = args;
  let results = (outs CT:$out);
  let hasVerifier = 1;
}
def SourceOp : CtOp<"source", (ins AnyType:$cipher)>;
def LintransOp : CtOp<"lintrans", (ins CT:$input, AnyType:$table, DenseI64ArrayAttr:$diagonals)>;
def PolyEvalOp : CtOp<"poly_eval", (ins CT:$input, ArrayAttr:$coefficients,
    OptionalAttr<StrAttr>:$activation, OptionalAttr<I64Attr>:$depth_budget,
    OptionalAttr<F64Attr>:$approx_budget, OptionalAttr<ArrayAttr>:$poly_frontier,
    OptionalAttr<DictionaryAttr>:$poly_selected)> {
  let builders = [OpBuilder<(ins "::mlir::Type":$result, "::mlir::Value":$input,
      "::mlir::ArrayAttr":$coefficients), [{
    build($_builder, $_state, result, input, coefficients, nullptr, nullptr, nullptr, nullptr, nullptr);
  }]>];
  let description = [{For an imported raw kernel, coefficients specify the HEIR
    Chebyshev series. A source-proven activation instead takes the original x;
    its coefficients list is empty and its poly_frontier and poly_selected
    implementation records own all coefficients, SSA paths and level costs.}];
}
def StepOp : CtOp<"ckks_step", (ins Variadic<AnyType>:$inputs, StrAttr:$kind)>;
def ConvertOp : CtOp<"convert", (ins CT:$input, StrAttr:$layout_contract,
    OptionalAttr<SlotLayout>:$provided, OptionalAttr<SlotLayout>:$required)>;
def SinkOp : Op_<"sink"> {
  let arguments = (ins CT:$input);
  let results = (outs AnyType:$cipher);
  let hasVerifier = 1;
}

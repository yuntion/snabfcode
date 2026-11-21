def CT : TypeDef<SNABF, "Ct"> {
  let mnemonic = "ct";
  let parameters = (ins "::mlir::RankedTensorType":$logical, "int64_t":$n,
                        "int64_t":$level, "::mlir::FlatSymbolRefAttr":$profile);
  let assemblyFormat = "`<` $logical `,` $n `,` $level `,` $profile `>`";
  let genVerifyDecl = 1;
}
def Budget : AttrDef<SNABF, "Budget"> {
  let mnemonic = "budget";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}
def SlotLayout : AttrDef<SNABF, "SlotLayout"> {
  let mnemonic = "slot_layout";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}

// RFC-004 N0: native records are opt-in; no defaults on sealed legacy IR.
class NativeRecord<string cpp, string spelling> : AttrDef<SNABF, cpp> {
  let mnemonic = spelling;
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}
def FrontendProvenance : NativeRecord<"FrontendProvenance", "frontend_provenance">;
def LayoutSpec : NativeRecord<"LayoutSpec", "layout_spec">;
def EncodingCallReceipt : NativeRecord<"EncodingCallReceipt", "encoding_call_receipt">;
def ScalePlan : NativeRecord<"ScalePlan", "scale_plan">;
// Canonical positive rational. The field containing it determines its role:
// requested vs physical_scale_exact (metadata) vs effective encoding scale.
def ExactScale : AttrDef<SNABF, "ExactScale"> {
  let mnemonic = "exact_scale";
  let parameters = (ins StringRefParameter<>:$numerator, StringRefParameter<>:$denominator);
  let assemblyFormat = "`<` $numerator `,` $denominator `>`";
  let genVerifyDecl = 1;
}

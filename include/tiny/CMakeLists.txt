#add_mlir_doc(TinyDialect TinyDialect tiny/ -gen-dialect-doc)
#add_mlir_doc(TinyOps TinyOps tiny/ -gen-dialect-doc)
set(LLVM_TARGET_DEFINITIONS TinyOps.td)
mlir_tablegen(TinyOps.inc -gen-rewriters)
#add_mlir_dialect(ShapeInferenceInterface tiny)

#set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
#mlir_tablegen(ShapeInferenceInterface.h.inc -gen-op-interface-decls)
#mlir_tablegen(ShapeInferenceInterface.cpp.inc -gen-op-interface-defs)

add_mlir_dialect(TinyOps tiny)
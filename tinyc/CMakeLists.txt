get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRTiny
        Graph
        MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRef
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
)

add_llvm_executable(tinyc main.cpp ../parser/AST.cpp  ../Graph/LowerGraphPass.cpp)

llvm_update_compile_flags(tinyc)

target_link_libraries(tinyc 
    PRIVATE ${LIBS}
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    
    #LowerGraphPass
    )

mlir_check_all_link_libraries(tinyc)

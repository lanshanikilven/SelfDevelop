add_mlir_conversion_library(RelayMLIRConversion
    RelayToAffine.cpp
    AffineToSCF.cpp
    SCFToLLVM.cpp

    DEPENDS
    RelayMLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        #MLIRMemRefDialect
        #MLIRLLVMDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIROptLib
    )
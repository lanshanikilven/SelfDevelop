#include "tiny/Parser.h"
#include "tiny/TinyDialect.h"
#include "tiny/MLIRGen.h"
#include "tiny/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
//#include "mlir/InitAllPasses.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include "Graph/GraphDialect.h"
#include "Graph/GraphOps.h"

using namespace tiny;

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input tiny file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
enum Action { None, DumpAST, DumpMLIR, DumpMLIRAffine, DumpMLIRLLVM, RunJIT, DumpLLVMFROMMLIR};

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
               cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
               cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the LLVM dump")),
               cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", "output the affine pass dump")),
               cl::values(clEnumValN(DumpLLVMFROMMLIR, "mutate-mlir-to-dst", "mutate MLIR to dst IR")),
               cl::values(clEnumValN(RunJIT, "jit", "JIT the code and run it by invoke the main function")));

std::unique_ptr<tiny::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    bool isLoweringMLIRAffine = emitAction >= Action::DumpMLIRAffine;
    bool isLoweringMLIRLLVM = emitAction >= Action::DumpMLIRLLVM;

    if (isLoweringMLIRAffine) {

        // Inline all functions into main and then delete them.
        pm.addPass(mlir::createInlinerPass());

        mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();

        // Partially lower the tiny dialect with a few cleanups afterwards.
        optPM.addPass(mlir::tiny::createLowerToAffinePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
    }

    if (isLoweringMLIRLLVM) {
         // add LowerToLLVMPass
        pm.addPass(mlir::tiny::createLowerToLLVMPass());
    }

    if (mlir::failed(pm.run(*module)))
        return -1;
    return 0;

}

int runJIT(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  mlir::registerLLVMDialectTranslation(*module->getContext());
  // Convert the module to LLVM IR in a new LLVM IR context.

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  // Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::outs() << *llvmModule << "\n";
  return 0;
}

int loadMLIRFromFile(mlir::MLIRContext &context, mlir::OwningModuleRef& module) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef& module) {
  if (int error = loadMLIRFromFile(context, module)) {
    return error;
  }
  // Register passes to be applied in this compile process
  mlir::PassManager passManager(&context);
  mlir::applyPassManagerCLOptions(passManager);
  std::cout << "111111111111111111111" << std::endl;
  module->dump();
  //注意这些pass的变化，针对的都是ModuleOp

  //下面这个标准化pass既可以针对ModuleOp，也可以针对FuncOp
  //passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());
  //将 inline pass添加到优化过程中，这个pass是针对ModuleOp的pass
  passManager.addPass(mlir::createInlinerPass());
  passManager.addPass(mlir::graph::registerLowerGraphPass());
  //passManager.addPass(mlir::tiny::createShapeInferencePass());
  //下面这两个是MLIR自带的pass，分别完成了相同循环边界融合优化和对于MemRef的数据流优化功能。
  mlir::OpPassManager &optPM = passManager.nest<mlir::FuncOp>();
  //createLoopFusionPass这个pass需要在FuncOp进行变换，所有需要先嵌套一层
  optPM.addPass(mlir::createLoopFusionPass());
  //passManager.addPass(mlir::tiny::createLowerToAffinePass());
  //passManager.addPass(mlir::tiny::createLowerToLLVMPass());
  //passManager.addPass(mlir::createMemRefDataFlowOptPass());

  if (mlir::failed(passManager.run(*module))) {
    return 4;
  }

  return 0;
}

int main(int argc, char** argv) {

    //这里也可以直接暴力将所有MLIR原生的pass全部引入近来，但是会影响整个代码工程的编译和运行效率
    //mlir::registerAllPasses();
    //但是我们常规会自己去识别我们需要依赖或者用到的MLIR的原生Pass，然后手动注册添加到这里
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "tiny compiler!\n");
    mlir::MLIRContext context;
    if (emitAction >= Action::DumpLLVMFROMMLIR){
        std::cout << "BBBBBBBBBBBBBBBB" <<std::endl;
        context.getOrLoadDialect<mlir::tiny::TinyDialect>();
        context.getOrLoadDialect<mlir::StandardOpsDialect>();
        context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        
        mlir::OwningModuleRef module;
        if (int error = loadAndProcessMLIR(context, module)) {
          return error;
        }
        std::cout << "44444444444444" << std::endl;
        module->dump();
        //std::cout << "55555555555555" << std::endl;
        //dumpLLVMIR(*module);
        //std::cout << "66666666666666" << std::endl;
        //runJIT(*module);
        return 0;
    } 
    // the following code is designed for DSL for future.
    /*
    else {
    std::cout << "AAAAAAAAAAAAA" << std::endl;
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return 1;
    if (emitAction == Action::DumpAST) {
        tiny::dump(*moduleAST);
        return 0;
    }
    context.getOrLoadDialect<mlir::tiny::TinyDialect>();
    mlir::OwningModuleRef module = mlirGen(context, *moduleAST);
    if (!module)
        return 1;
    if (loadMLIR(context, module) != 0) {
        llvm::errs() << "load MLIR failed\n";
        return 1;
    }
    if (emitAction == Action::RunJIT) {
        return runJIT(*module);
    } else {
        module->dump();
        return 0;
    }
    }
    */
    return 0;
}
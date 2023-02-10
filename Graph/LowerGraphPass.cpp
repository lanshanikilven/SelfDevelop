//====- LowerGraphPass.cpp - graph Dialect Lowering Pass
//---------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines Graph dialect lowering pass.
//
//===----------------------------------------------------------------------===//
#include "tiny/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
//#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
//#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "Graph/GraphDialect.h"
#include "Graph/GraphOps.h"


//using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//
class GraphBFSLowering : public mlir::OpRewritePattern<graph::BFSOp> {
public:
  using mlir::OpRewritePattern<graph::BFSOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(graph::BFSOp op,
                                mlir::PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    // Create constant indices.
    mlir::Value c0 = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = rewriter.create<mlir::ConstantIndexOp>(loc, 1);

    // Register operand values.
    mlir::Value m1 = op.getOperand(0);
    mlir::Value m2 = op.getOperand(1);
    mlir::Value m3 = op.getOperand(2);

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  int64_t stride;
};


void populateLowerGraphConversionPatterns(mlir::RewritePatternSet &patterns,
                                          int64_t stride) {
  patterns.add<GraphBFSLowering>(patterns.getContext(), stride);
}

//===----------------------------------------------------------------------===//
// LowerGraphPass
//===----------------------------------------------------------------------===//

namespace {
class LowerGraphPass
    : public mlir::PassWrapper<LowerGraphPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  LowerGraphPass() = default;
  LowerGraphPass(const LowerGraphPass &) {}
  explicit LowerGraphPass(int64_t strideParam) { stride = strideParam; }

  mlir::StringRef getArgument() const final { return "lower-graph"; }
  mlir::StringRef getDescription() const final { return "Lower Graph Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<graph::GraphDialect, mlir::StandardOpsDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    mlir::AffineDialect>();
  }

  Option<int64_t> stride{*this, "Graph-strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};
};
} // end anonymous namespace.

void LowerGraphPass::runOnOperation() {
  mlir::MLIRContext *context = &getContext();
  mlir::ModuleOp module = getOperation();

  mlir::ConversionTarget target(*context);
  target.addLegalDialect<mlir::AffineDialect, mlir::scf::SCFDialect, mlir::StandardOpsDialect,
                         mlir::memref::MemRefDialect, mlir::vector::VectorDialect>();
  target.addLegalOp<mlir::ModuleOp,mlir::FuncOp, mlir::LLVM::ReturnOp>();

  mlir::RewritePatternSet patterns(context);
  populateLowerGraphConversionPatterns(patterns, stride);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}


std::unique_ptr<mlir::Pass> mlir::graph::registerLowerGraphPass() {
  return std::make_unique<LowerGraphPass>();
}


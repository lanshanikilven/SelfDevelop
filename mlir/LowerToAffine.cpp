//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "tiny/TinyOps.h"
#include "tiny/TinyDialect.h"
#include "tiny/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/BuiltinOps.h.inc"
#include "mlir/IR/BuiltinDialect.h.inc"

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static mlir::MemRefType convertTensorToMemRef(mlir::TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc,
                                   mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

struct ConstantOpLowering : public mlir::OpRewritePattern<mlir::tiny::ConstantOp> {
  using OpRewritePattern<mlir::tiny::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tiny::ConstantOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::DenseIntElementsAttr constantValue = op.value();
    mlir::Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<mlir::TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    mlir::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(rewriter.create<mlir::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<mlir::ConstantIndexOp>(loc, 0));
    }
    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    mlir::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.getValues<mlir::IntegerAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::AffineStoreOp>(
            loc, rewriter.create<mlir::ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};


struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::tiny::ReturnOp> {
  using OpRewritePattern<mlir::tiny::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::tiny::ReturnOp op, mlir::PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been inlined.
    if (op.hasOperand())
      return mlir::failure();

    // We lower "toy.return" directly to "std.return".
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
    return mlir::success();
  }
};


//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass : public mlir::PassWrapper<ToyToAffineLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::memref::MemRefDialect, mlir::StandardOpsDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace.

void ToyToAffineLoweringPass::runOnOperation() {
  // auto function = getFunction();

  // // We only lower the main function as we expect that all other functions have
  // // been inlined.
  // if (function.getName() != "main")
  //   return;

  // // Verify that the given main has no inputs and results.
  // if (function.getNumArguments() || function.getType().getNumResults()) {
  //   function.emitError("expected 'main' to have 0 inputs and 0 results");
  //   return signalPassFailure();
  // }

  // The first thing to is defining the conversion target for this lowering.
  // getContext()函数是Pass基类中的一个纯虚函数，所有自定义的Pass都会继承自该Pass基类
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for this lowering. 
  // 这里声明了lower过程中能够合法使用的dialect，需要将非法dialect中的op转换到这些dialect中的合法op
  target.addLegalDialect<mlir::AffineDialect, mlir::memref::MemRefDialect, mlir::BuiltinDialect, mlir::StandardOpsDialect>();

  // We also define the TinyDialect as illegal so that the conversion will fail if any of these operations are not converted. 
  // we explicitly mark the PrintOp that don't want to be lowered, `tiny.print`, as `legal`.
  target.addIllegalDialect<mlir::tiny::TinyDialect>();

  target.addLegalOp<mlir::tiny::PrintOp>(); 
  // 至此从187-191行，仅仅定义了我们转换的Target目标，或者说相当于转换了namespace
  // 但是具体的转换过程并不在这里，需要在下面的rewritePatternSet里面来定义具体的转换过程

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ConstantOpLowering, ReturnOpLowering>(&getContext());  //这里添加了两种rewrite patterns，里面定义了怎么将输入的IR转换成我们期望的IR

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  // 这里执行部分转换，会保留没有被标记为illegal的operations，转换完成后，隶属于不同dialect的op可以共存在一个大的FuncOp或者ModuleOp中
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure(); 
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> mlir::tiny::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

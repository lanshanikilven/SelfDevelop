#ifndef TINY_PASSES_H
#define TINY_PASSES_H

#include <memory>

namespace mlir {
class Pass;
namespace tiny {
/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
std::unique_ptr<mlir::Pass> createShapeInferencePass();
}   // namespace tiny

namespace graph {
std::unique_ptr<mlir::Pass> registerLowerGraphPass();
}   // namespace graph

}   // namespace mlir

#endif
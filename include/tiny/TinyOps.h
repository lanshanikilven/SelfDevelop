//===- StandaloneDialect.h - Standalone dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TINYOPS_H
#define TINY_TINYOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "tiny/ShapeInferenceInterface.h.inc"

// Include the auto-generated header file containing the declarations of the toy operations.
// 这两行代码属于固定写法
#define GET_OP_CLASSES
#include "tiny/TinyOps.h.inc"

#endif // TINY_TINYDIALECT_H

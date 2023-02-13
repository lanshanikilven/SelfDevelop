//===- StandaloneDialect.h - Standalone dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TINY_TINYDIALECT_H
#define TINY_TINYDIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/IR/OpDefinition.h"


// Include the auto-generated header file containing the declaration of the toy dialect.
// 下面这行代码也是属于固定写法
#include "tiny/TinyOpsDialect.h.inc"

#endif // TINY_TINYDIALECT_H

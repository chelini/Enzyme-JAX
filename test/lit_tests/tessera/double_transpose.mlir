// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl | FileCheck %s

// Linnea's transpose(Transpose(A)) returns A. No numerical reassociation or
// matrix-property assumption is needed for this data permutation.
!mat = !llvm.array<3 x array<3 x f64>>
module {
  tessera.define private @la.transpose(!mat) -> !mat attributes {argModes = [unit], pure = true}
  llvm.func @consume(!mat)
  tessera.optimizations {
    tessera.optimization "la.transpose(la.transpose(a)) -> a"
  }

  // CHECK-LABEL: llvm.func @double_transpose
  // CHECK-NEXT: llvm.return %arg0
  llvm.func @double_transpose(%a: !mat) -> !mat {
    %t = tessera.call @la.transpose(%a) : (!mat) -> !mat
    %r = tessera.call @la.transpose(%t) : (!mat) -> !mat
    llvm.return %r : !mat
  }

  // A shared intermediate is still needed by the other consumer.
  // CHECK-LABEL: llvm.func @shared
  // CHECK-NEXT: %[[T:.*]] = tessera.call @la.transpose(%arg0)
  // CHECK-NEXT: llvm.call @consume(%[[T]])
  // CHECK-NEXT: llvm.return %arg0
  llvm.func @shared(%a: !mat) -> !mat {
    %t = tessera.call @la.transpose(%a) : (!mat) -> !mat
    llvm.call @consume(%t) : (!mat) -> ()
    %r = tessera.call @la.transpose(%t) : (!mat) -> !mat
    llvm.return %r : !mat
  }
}

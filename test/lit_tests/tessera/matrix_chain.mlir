// RUN: enzymexlamlir-opt %s -tessera-optimize-matrix-chain -split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s -pass-pipeline="builtin.module(tessera-optimize-matrix-chain,tessera-optimize-matrix-chain)" -split-input-file | FileCheck %s

!m2x3 = !llvm.array<2 x array<3 x f64>>
!m2x30 = !llvm.array<2 x array<30 x f64>>
!m20x2 = !llvm.array<20 x array<2 x f64>>
!m20x3 = !llvm.array<20 x array<3 x f64>>
!m20x30 = !llvm.array<20 x array<30 x f64>>
!m30x3 = !llvm.array<30 x array<3 x f64>>
!other = !llvm.struct<"Competing", (array<2 x array<3 x f64>>)>
module {
  tessera.define private @competing_bc(!m2x30, !m30x3) -> !other attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_20_2_30(!m20x2, !m2x30) -> !m20x30 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_20_30_3(!m20x30, !m30x3) -> !m20x3 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_2_30_3(!m2x30, !m30x3) -> !m2x3 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_20_2_3(!m20x2, !m2x3) -> !m20x3 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @unmarked_ab(!m20x2, !m2x30) -> !m20x30 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @unmarked_root(!m20x30, !m30x3) -> !m20x3 attributes {argModes = [unit, unit], pure = true}
  llvm.func @consume(!m20x30)

  // CHECK-LABEL: llvm.func @cheaper
  // CHECK-NEXT: %[[BC:.*]] = tessera.call @mm_2_30_3(%arg1, %arg2)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_20_2_3(%arg0, %[[BC]])
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @cheaper(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 {
    %ab = tessera.call @mm_20_2_30(%a, %b) : (!m20x2, !m2x30) -> !m20x30
    %r = tessera.call @mm_20_30_3(%ab, %c) : (!m20x30, !m30x3) -> !m20x3
    llvm.return %r : !m20x3
  }

  // CHECK-LABEL: llvm.func @already_optimal
  // CHECK-NEXT: %[[BC:.*]] = tessera.call @mm_2_30_3(%arg1, %arg2)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_20_2_3(%arg0, %[[BC]])
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @already_optimal(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 {
    %bc = tessera.call @mm_2_30_3(%b, %c) : (!m2x30, !m30x3) -> !m2x3
    %r = tessera.call @mm_20_2_3(%a, %bc) : (!m20x2, !m2x3) -> !m20x3
    llvm.return %r : !m20x3
  }

  // CHECK-LABEL: llvm.func @shared
  // CHECK: %[[AB:.*]] = tessera.call @mm_20_2_30(%arg0, %arg1)
  // CHECK-NEXT: llvm.call @consume(%[[AB]])
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_20_30_3(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @shared(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 {
    %ab = tessera.call @mm_20_2_30(%a, %b) : (!m20x2, !m2x30) -> !m20x30
    llvm.call @consume(%ab) : (!m20x30) -> ()
    %r = tessera.call @mm_20_30_3(%ab, %c) : (!m20x30, !m30x3) -> !m20x3
    llvm.return %r : !m20x3
  }

  // CHECK-LABEL: llvm.func @unmarked
  // CHECK: %[[AB:.*]] = tessera.call @unmarked_ab(%arg0, %arg1)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_20_30_3(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @unmarked(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 {
    %ab = tessera.call @unmarked_ab(%a, %b) : (!m20x2, !m2x30) -> !m20x30
    %r = tessera.call @mm_20_30_3(%ab, %c) : (!m20x30, !m30x3) -> !m20x3
    llvm.return %r : !m20x3
  }

  // CHECK-LABEL: llvm.func @root_unmarked
  // CHECK: %[[AB:.*]] = tessera.call @mm_20_2_30(%arg0, %arg1)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @unmarked_root(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @root_unmarked(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 {
    %ab = tessera.call @mm_20_2_30(%a, %b) : (!m20x2, !m2x30) -> !m20x30
    %r = tessera.call @unmarked_root(%ab, %c) : (!m20x30, !m30x3) -> !m20x3
    llvm.return %r : !m20x3
  }

  // CHECK-LABEL: llvm.func @cross_block
  // CHECK: %[[AB:.*]] = tessera.call @mm_20_2_30(%arg0, %arg1)
  // CHECK-NEXT: llvm.br
  // CHECK: ^bb1:
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_20_30_3(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @cross_block(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 {
    %ab = tessera.call @mm_20_2_30(%a, %b) : (!m20x2, !m2x30) -> !m20x30
    llvm.br ^next
  ^next:
    %r = tessera.call @mm_20_30_3(%ab, %c) : (!m20x30, !m30x3) -> !m20x3
    llvm.return %r : !m20x3
  }

  // CHECK-LABEL: llvm.func @protected
  // CHECK: %[[AB:.*]] = tessera.call @mm_20_2_30(%arg0, %arg1)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_20_30_3(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @protected(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 attributes {tessera.no_rewrite} {
    %ab = tessera.call @mm_20_2_30(%a, %b) : (!m20x2, !m2x30) -> !m20x30
    %r = tessera.call @mm_20_30_3(%ab, %c) : (!m20x30, !m30x3) -> !m20x3
    llvm.return %r : !m20x3
  }
}
// -----

// A missing kernel cannot be synthesized by reassociation.
!m2x3 = !llvm.array<2 x array<3 x f64>>
!m2x30 = !llvm.array<2 x array<30 x f64>>
!m20x2 = !llvm.array<20 x array<2 x f64>>
!m20x3 = !llvm.array<20 x array<3 x f64>>
!m20x30 = !llvm.array<20 x array<30 x f64>>
!m30x3 = !llvm.array<30 x array<3 x f64>>
module {
  tessera.define private @mm_20_2_30(!m20x2, !m2x30) -> !m20x30 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_20_30_3(!m20x30, !m30x3) -> !m20x3 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // CHECK-LABEL: llvm.func @missing_kernel
  // CHECK-NEXT: %[[AB:.*]] = tessera.call @mm_20_2_30
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_20_30_3(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @missing_kernel(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 {
    %ab = tessera.call @mm_20_2_30(%a, %b) : (!m20x2, !m2x30) -> !m20x30
    %r = tessera.call @mm_20_30_3(%ab, %c) : (!m20x30, !m30x3) -> !m20x3
    llvm.return %r : !m20x3
  }
}
// -----

// Equal shapes with incompatible representations cannot be substituted.
!m2x3 = !llvm.array<2 x array<3 x f64>>
!m2x30 = !llvm.array<2 x array<30 x f64>>
!m20x2 = !llvm.array<20 x array<2 x f64>>
!m20x3 = !llvm.array<20 x array<3 x f64>>
!m20x30 = !llvm.array<20 x array<30 x f64>>
!m30x3 = !llvm.array<30 x array<3 x f64>>
!other = !llvm.struct<"Other", (array<2 x array<3 x f64>>)>
module {
  tessera.define private @mm_20_2_30(!m20x2, !m2x30) -> !m20x30 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_20_30_3(!m20x30, !m30x3) -> !m20x3 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_20_2_3(!m20x2, !m2x3) -> !m20x3 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @other_bc(!m2x30, !m30x3) -> !other attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // CHECK-LABEL: llvm.func @incompatible_abi
  // CHECK-NEXT: %[[AB:.*]] = tessera.call @mm_20_2_30
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_20_30_3(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @incompatible_abi(%a: !m20x2, %b: !m2x30, %c: !m30x3) -> !m20x3 {
    %ab = tessera.call @mm_20_2_30(%a, %b) : (!m20x2, !m2x30) -> !m20x30
    %r = tessera.call @mm_20_30_3(%ab, %c) : (!m20x30, !m30x3) -> !m20x3
    llvm.return %r : !m20x3
  }
}
// -----

// Equal or saturated costs keep the original tree.
!m2x2 = !llvm.array<2 x array<2 x f64>>
module {
  tessera.define private @mm_2_2_2(!m2x2, !m2x2) -> !m2x2 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // CHECK-LABEL: llvm.func @tie
  // CHECK-NEXT: %[[AB:.*]] = tessera.call @mm_2_2_2(%arg0, %arg1)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_2_2_2(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @tie(%a: !m2x2, %b: !m2x2, %c: !m2x2) -> !m2x2 {
    %ab = tessera.call @mm_2_2_2(%a, %b) : (!m2x2, !m2x2) -> !m2x2
    %r = tessera.call @mm_2_2_2(%ab, %c) : (!m2x2, !m2x2) -> !m2x2
    llvm.return %r : !m2x2
  }
}
// -----

// Equal or saturated costs keep the original tree.
!m4194304x4194304 = !llvm.array<4194304 x array<4194304 x f64>>
module {
  tessera.define private @mm_4194304_4194304_4194304(!m4194304x4194304, !m4194304x4194304) -> !m4194304x4194304 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // CHECK-LABEL: llvm.func @overflow
  // CHECK-NEXT: %[[AB:.*]] = tessera.call @mm_4194304_4194304_4194304(%arg0, %arg1)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_4194304_4194304_4194304(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @overflow(%a: !m4194304x4194304, %b: !m4194304x4194304, %c: !m4194304x4194304) -> !m4194304x4194304 {
    %ab = tessera.call @mm_4194304_4194304_4194304(%a, %b) : (!m4194304x4194304, !m4194304x4194304) -> !m4194304x4194304
    %r = tessera.call @mm_4194304_4194304_4194304(%ab, %c) : (!m4194304x4194304, !m4194304x4194304) -> !m4194304x4194304
    llvm.return %r : !m4194304x4194304
  }
}

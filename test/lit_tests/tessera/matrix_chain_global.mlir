// RUN: enzymexlamlir-opt %s -tessera-optimize-matrix-chain | FileCheck %s
// Optimal cost is 15125 scalar multiplies: (A (B C)) ((D E) F).
!m5x10 = !llvm.array<5 x array<10 x f64>>
!m5x20 = !llvm.array<5 x array<20 x f64>>
!m5x25 = !llvm.array<5 x array<25 x f64>>
!m10x20 = !llvm.array<10 x array<20 x f64>>
!m10x25 = !llvm.array<10 x array<25 x f64>>
!m15x5 = !llvm.array<15 x array<5 x f64>>
!m15x10 = !llvm.array<15 x array<10 x f64>>
!m15x20 = !llvm.array<15 x array<20 x f64>>
!m15x25 = !llvm.array<15 x array<25 x f64>>
!m20x25 = !llvm.array<20 x array<25 x f64>>
!m30x5 = !llvm.array<30 x array<5 x f64>>
!m30x10 = !llvm.array<30 x array<10 x f64>>
!m30x15 = !llvm.array<30 x array<15 x f64>>
!m30x20 = !llvm.array<30 x array<20 x f64>>
!m30x25 = !llvm.array<30 x array<25 x f64>>
!m30x35 = !llvm.array<30 x array<35 x f64>>
!m35x5 = !llvm.array<35 x array<5 x f64>>
!m35x10 = !llvm.array<35 x array<10 x f64>>
!m35x15 = !llvm.array<35 x array<15 x f64>>
!m35x20 = !llvm.array<35 x array<20 x f64>>
!m35x25 = !llvm.array<35 x array<25 x f64>>
module {
  tessera.define private @mm_5_10_20(!m5x10, !m10x20) -> !m5x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_5_10_25(!m5x10, !m10x25) -> !m5x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_5_20_25(!m5x20, !m20x25) -> !m5x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_10_20_25(!m10x20, !m20x25) -> !m10x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_15_5_10(!m15x5, !m5x10) -> !m15x10 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_15_5_20(!m15x5, !m5x20) -> !m15x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_15_5_25(!m15x5, !m5x25) -> !m15x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_15_10_20(!m15x10, !m10x20) -> !m15x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_15_10_25(!m15x10, !m10x25) -> !m15x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_15_20_25(!m15x20, !m20x25) -> !m15x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_5_10(!m30x5, !m5x10) -> !m30x10 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_5_20(!m30x5, !m5x20) -> !m30x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_5_25(!m30x5, !m5x25) -> !m30x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_10_20(!m30x10, !m10x20) -> !m30x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_10_25(!m30x10, !m10x25) -> !m30x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_15_5(!m30x15, !m15x5) -> !m30x5 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_15_10(!m30x15, !m15x10) -> !m30x10 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_15_20(!m30x15, !m15x20) -> !m30x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_15_25(!m30x15, !m15x25) -> !m30x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_20_25(!m30x20, !m20x25) -> !m30x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_35_5(!m30x35, !m35x5) -> !m30x5 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_35_10(!m30x35, !m35x10) -> !m30x10 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_35_15(!m30x35, !m35x15) -> !m30x15 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_35_20(!m30x35, !m35x20) -> !m30x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_30_35_25(!m30x35, !m35x25) -> !m30x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_5_10(!m35x5, !m5x10) -> !m35x10 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_5_20(!m35x5, !m5x20) -> !m35x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_5_25(!m35x5, !m5x25) -> !m35x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_10_20(!m35x10, !m10x20) -> !m35x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_10_25(!m35x10, !m10x25) -> !m35x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_15_5(!m35x15, !m15x5) -> !m35x5 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_15_10(!m35x15, !m15x10) -> !m35x10 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_15_20(!m35x15, !m15x20) -> !m35x20 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_15_25(!m35x15, !m15x25) -> !m35x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @mm_35_20_25(!m35x20, !m20x25) -> !m35x25 attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // CHECK-LABEL: llvm.func @six_matrices
  // CHECK-NEXT: %[[BC:.*]] = tessera.call @mm_35_15_5(%arg1, %arg2)
  // CHECK-NEXT: %[[ABC:.*]] = tessera.call @mm_30_35_5(%arg0, %[[BC]])
  // CHECK-NEXT: %[[DE:.*]] = tessera.call @mm_5_10_20(%arg3, %arg4)
  // CHECK-NEXT: %[[DEF:.*]] = tessera.call @mm_5_20_25(%[[DE]], %arg5)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @mm_30_5_25(%[[ABC]], %[[DEF]])
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @six_matrices(%a0: !m30x35, %a1: !m35x15, %a2: !m15x5, %a3: !m5x10, %a4: !m10x20, %a5: !m20x25) -> !m30x25 {
    %p1 = tessera.call @mm_30_35_15(%a0, %a1) : (!m30x35, !m35x15) -> !m30x15
    %p2 = tessera.call @mm_30_15_5(%p1, %a2) : (!m30x15, !m15x5) -> !m30x5
    %p3 = tessera.call @mm_30_5_10(%p2, %a3) : (!m30x5, !m5x10) -> !m30x10
    %p4 = tessera.call @mm_30_10_20(%p3, %a4) : (!m30x10, !m10x20) -> !m30x20
    %p5 = tessera.call @mm_30_20_25(%p4, %a5) : (!m30x20, !m20x25) -> !m30x25
    llvm.return %p5 : !m30x25
  }
}

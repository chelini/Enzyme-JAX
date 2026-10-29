// RUN: enzymexlamlir-opt %s -tessera-optimize-matrix-chain | FileCheck %s
// Check both sides of the planning limit without allocating the matrices.
!s = !llvm.array<2 x array<2 x f64>>
!v = !llvm.array<2 x array<1 x f64>>
module {
  tessera.define private @square(!s, !s) -> !s attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @matvec(!s, !v) -> !v attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // CHECK-LABEL: llvm.func @chain_64
  // CHECK-NOT: tessera.call @square
  // CHECK-COUNT-63: tessera.call @matvec
  // CHECK-NEXT: llvm.return
  llvm.func @chain_64(%a0: !s, %a1: !s, %a2: !s, %a3: !s, %a4: !s, %a5: !s, %a6: !s, %a7: !s, %a8: !s, %a9: !s, %a10: !s, %a11: !s, %a12: !s, %a13: !s, %a14: !s, %a15: !s, %a16: !s, %a17: !s, %a18: !s, %a19: !s, %a20: !s, %a21: !s, %a22: !s, %a23: !s, %a24: !s, %a25: !s, %a26: !s, %a27: !s, %a28: !s, %a29: !s, %a30: !s, %a31: !s, %a32: !s, %a33: !s, %a34: !s, %a35: !s, %a36: !s, %a37: !s, %a38: !s, %a39: !s, %a40: !s, %a41: !s, %a42: !s, %a43: !s, %a44: !s, %a45: !s, %a46: !s, %a47: !s, %a48: !s, %a49: !s, %a50: !s, %a51: !s, %a52: !s, %a53: !s, %a54: !s, %a55: !s, %a56: !s, %a57: !s, %a58: !s, %a59: !s, %a60: !s, %a61: !s, %a62: !s, %v: !v) -> !v {
    %p1 = tessera.call @square(%a0, %a1) : (!s, !s) -> !s
    %p2 = tessera.call @square(%p1, %a2) : (!s, !s) -> !s
    %p3 = tessera.call @square(%p2, %a3) : (!s, !s) -> !s
    %p4 = tessera.call @square(%p3, %a4) : (!s, !s) -> !s
    %p5 = tessera.call @square(%p4, %a5) : (!s, !s) -> !s
    %p6 = tessera.call @square(%p5, %a6) : (!s, !s) -> !s
    %p7 = tessera.call @square(%p6, %a7) : (!s, !s) -> !s
    %p8 = tessera.call @square(%p7, %a8) : (!s, !s) -> !s
    %p9 = tessera.call @square(%p8, %a9) : (!s, !s) -> !s
    %p10 = tessera.call @square(%p9, %a10) : (!s, !s) -> !s
    %p11 = tessera.call @square(%p10, %a11) : (!s, !s) -> !s
    %p12 = tessera.call @square(%p11, %a12) : (!s, !s) -> !s
    %p13 = tessera.call @square(%p12, %a13) : (!s, !s) -> !s
    %p14 = tessera.call @square(%p13, %a14) : (!s, !s) -> !s
    %p15 = tessera.call @square(%p14, %a15) : (!s, !s) -> !s
    %p16 = tessera.call @square(%p15, %a16) : (!s, !s) -> !s
    %p17 = tessera.call @square(%p16, %a17) : (!s, !s) -> !s
    %p18 = tessera.call @square(%p17, %a18) : (!s, !s) -> !s
    %p19 = tessera.call @square(%p18, %a19) : (!s, !s) -> !s
    %p20 = tessera.call @square(%p19, %a20) : (!s, !s) -> !s
    %p21 = tessera.call @square(%p20, %a21) : (!s, !s) -> !s
    %p22 = tessera.call @square(%p21, %a22) : (!s, !s) -> !s
    %p23 = tessera.call @square(%p22, %a23) : (!s, !s) -> !s
    %p24 = tessera.call @square(%p23, %a24) : (!s, !s) -> !s
    %p25 = tessera.call @square(%p24, %a25) : (!s, !s) -> !s
    %p26 = tessera.call @square(%p25, %a26) : (!s, !s) -> !s
    %p27 = tessera.call @square(%p26, %a27) : (!s, !s) -> !s
    %p28 = tessera.call @square(%p27, %a28) : (!s, !s) -> !s
    %p29 = tessera.call @square(%p28, %a29) : (!s, !s) -> !s
    %p30 = tessera.call @square(%p29, %a30) : (!s, !s) -> !s
    %p31 = tessera.call @square(%p30, %a31) : (!s, !s) -> !s
    %p32 = tessera.call @square(%p31, %a32) : (!s, !s) -> !s
    %p33 = tessera.call @square(%p32, %a33) : (!s, !s) -> !s
    %p34 = tessera.call @square(%p33, %a34) : (!s, !s) -> !s
    %p35 = tessera.call @square(%p34, %a35) : (!s, !s) -> !s
    %p36 = tessera.call @square(%p35, %a36) : (!s, !s) -> !s
    %p37 = tessera.call @square(%p36, %a37) : (!s, !s) -> !s
    %p38 = tessera.call @square(%p37, %a38) : (!s, !s) -> !s
    %p39 = tessera.call @square(%p38, %a39) : (!s, !s) -> !s
    %p40 = tessera.call @square(%p39, %a40) : (!s, !s) -> !s
    %p41 = tessera.call @square(%p40, %a41) : (!s, !s) -> !s
    %p42 = tessera.call @square(%p41, %a42) : (!s, !s) -> !s
    %p43 = tessera.call @square(%p42, %a43) : (!s, !s) -> !s
    %p44 = tessera.call @square(%p43, %a44) : (!s, !s) -> !s
    %p45 = tessera.call @square(%p44, %a45) : (!s, !s) -> !s
    %p46 = tessera.call @square(%p45, %a46) : (!s, !s) -> !s
    %p47 = tessera.call @square(%p46, %a47) : (!s, !s) -> !s
    %p48 = tessera.call @square(%p47, %a48) : (!s, !s) -> !s
    %p49 = tessera.call @square(%p48, %a49) : (!s, !s) -> !s
    %p50 = tessera.call @square(%p49, %a50) : (!s, !s) -> !s
    %p51 = tessera.call @square(%p50, %a51) : (!s, !s) -> !s
    %p52 = tessera.call @square(%p51, %a52) : (!s, !s) -> !s
    %p53 = tessera.call @square(%p52, %a53) : (!s, !s) -> !s
    %p54 = tessera.call @square(%p53, %a54) : (!s, !s) -> !s
    %p55 = tessera.call @square(%p54, %a55) : (!s, !s) -> !s
    %p56 = tessera.call @square(%p55, %a56) : (!s, !s) -> !s
    %p57 = tessera.call @square(%p56, %a57) : (!s, !s) -> !s
    %p58 = tessera.call @square(%p57, %a58) : (!s, !s) -> !s
    %p59 = tessera.call @square(%p58, %a59) : (!s, !s) -> !s
    %p60 = tessera.call @square(%p59, %a60) : (!s, !s) -> !s
    %p61 = tessera.call @square(%p60, %a61) : (!s, !s) -> !s
    %p62 = tessera.call @square(%p61, %a62) : (!s, !s) -> !s
    %r = tessera.call @matvec(%p62, %v) : (!s, !v) -> !v
    llvm.return %r : !v
  }
  // CHECK-LABEL: llvm.func @chain_65
  // CHECK-COUNT-63: tessera.call @square
  // CHECK-NEXT: %{{.*}} = tessera.call @matvec
  // CHECK-NEXT: llvm.return
  llvm.func @chain_65(%a0: !s, %a1: !s, %a2: !s, %a3: !s, %a4: !s, %a5: !s, %a6: !s, %a7: !s, %a8: !s, %a9: !s, %a10: !s, %a11: !s, %a12: !s, %a13: !s, %a14: !s, %a15: !s, %a16: !s, %a17: !s, %a18: !s, %a19: !s, %a20: !s, %a21: !s, %a22: !s, %a23: !s, %a24: !s, %a25: !s, %a26: !s, %a27: !s, %a28: !s, %a29: !s, %a30: !s, %a31: !s, %a32: !s, %a33: !s, %a34: !s, %a35: !s, %a36: !s, %a37: !s, %a38: !s, %a39: !s, %a40: !s, %a41: !s, %a42: !s, %a43: !s, %a44: !s, %a45: !s, %a46: !s, %a47: !s, %a48: !s, %a49: !s, %a50: !s, %a51: !s, %a52: !s, %a53: !s, %a54: !s, %a55: !s, %a56: !s, %a57: !s, %a58: !s, %a59: !s, %a60: !s, %a61: !s, %a62: !s, %a63: !s, %v: !v) -> !v {
    %p1 = tessera.call @square(%a0, %a1) : (!s, !s) -> !s
    %p2 = tessera.call @square(%p1, %a2) : (!s, !s) -> !s
    %p3 = tessera.call @square(%p2, %a3) : (!s, !s) -> !s
    %p4 = tessera.call @square(%p3, %a4) : (!s, !s) -> !s
    %p5 = tessera.call @square(%p4, %a5) : (!s, !s) -> !s
    %p6 = tessera.call @square(%p5, %a6) : (!s, !s) -> !s
    %p7 = tessera.call @square(%p6, %a7) : (!s, !s) -> !s
    %p8 = tessera.call @square(%p7, %a8) : (!s, !s) -> !s
    %p9 = tessera.call @square(%p8, %a9) : (!s, !s) -> !s
    %p10 = tessera.call @square(%p9, %a10) : (!s, !s) -> !s
    %p11 = tessera.call @square(%p10, %a11) : (!s, !s) -> !s
    %p12 = tessera.call @square(%p11, %a12) : (!s, !s) -> !s
    %p13 = tessera.call @square(%p12, %a13) : (!s, !s) -> !s
    %p14 = tessera.call @square(%p13, %a14) : (!s, !s) -> !s
    %p15 = tessera.call @square(%p14, %a15) : (!s, !s) -> !s
    %p16 = tessera.call @square(%p15, %a16) : (!s, !s) -> !s
    %p17 = tessera.call @square(%p16, %a17) : (!s, !s) -> !s
    %p18 = tessera.call @square(%p17, %a18) : (!s, !s) -> !s
    %p19 = tessera.call @square(%p18, %a19) : (!s, !s) -> !s
    %p20 = tessera.call @square(%p19, %a20) : (!s, !s) -> !s
    %p21 = tessera.call @square(%p20, %a21) : (!s, !s) -> !s
    %p22 = tessera.call @square(%p21, %a22) : (!s, !s) -> !s
    %p23 = tessera.call @square(%p22, %a23) : (!s, !s) -> !s
    %p24 = tessera.call @square(%p23, %a24) : (!s, !s) -> !s
    %p25 = tessera.call @square(%p24, %a25) : (!s, !s) -> !s
    %p26 = tessera.call @square(%p25, %a26) : (!s, !s) -> !s
    %p27 = tessera.call @square(%p26, %a27) : (!s, !s) -> !s
    %p28 = tessera.call @square(%p27, %a28) : (!s, !s) -> !s
    %p29 = tessera.call @square(%p28, %a29) : (!s, !s) -> !s
    %p30 = tessera.call @square(%p29, %a30) : (!s, !s) -> !s
    %p31 = tessera.call @square(%p30, %a31) : (!s, !s) -> !s
    %p32 = tessera.call @square(%p31, %a32) : (!s, !s) -> !s
    %p33 = tessera.call @square(%p32, %a33) : (!s, !s) -> !s
    %p34 = tessera.call @square(%p33, %a34) : (!s, !s) -> !s
    %p35 = tessera.call @square(%p34, %a35) : (!s, !s) -> !s
    %p36 = tessera.call @square(%p35, %a36) : (!s, !s) -> !s
    %p37 = tessera.call @square(%p36, %a37) : (!s, !s) -> !s
    %p38 = tessera.call @square(%p37, %a38) : (!s, !s) -> !s
    %p39 = tessera.call @square(%p38, %a39) : (!s, !s) -> !s
    %p40 = tessera.call @square(%p39, %a40) : (!s, !s) -> !s
    %p41 = tessera.call @square(%p40, %a41) : (!s, !s) -> !s
    %p42 = tessera.call @square(%p41, %a42) : (!s, !s) -> !s
    %p43 = tessera.call @square(%p42, %a43) : (!s, !s) -> !s
    %p44 = tessera.call @square(%p43, %a44) : (!s, !s) -> !s
    %p45 = tessera.call @square(%p44, %a45) : (!s, !s) -> !s
    %p46 = tessera.call @square(%p45, %a46) : (!s, !s) -> !s
    %p47 = tessera.call @square(%p46, %a47) : (!s, !s) -> !s
    %p48 = tessera.call @square(%p47, %a48) : (!s, !s) -> !s
    %p49 = tessera.call @square(%p48, %a49) : (!s, !s) -> !s
    %p50 = tessera.call @square(%p49, %a50) : (!s, !s) -> !s
    %p51 = tessera.call @square(%p50, %a51) : (!s, !s) -> !s
    %p52 = tessera.call @square(%p51, %a52) : (!s, !s) -> !s
    %p53 = tessera.call @square(%p52, %a53) : (!s, !s) -> !s
    %p54 = tessera.call @square(%p53, %a54) : (!s, !s) -> !s
    %p55 = tessera.call @square(%p54, %a55) : (!s, !s) -> !s
    %p56 = tessera.call @square(%p55, %a56) : (!s, !s) -> !s
    %p57 = tessera.call @square(%p56, %a57) : (!s, !s) -> !s
    %p58 = tessera.call @square(%p57, %a58) : (!s, !s) -> !s
    %p59 = tessera.call @square(%p58, %a59) : (!s, !s) -> !s
    %p60 = tessera.call @square(%p59, %a60) : (!s, !s) -> !s
    %p61 = tessera.call @square(%p60, %a61) : (!s, !s) -> !s
    %p62 = tessera.call @square(%p61, %a62) : (!s, !s) -> !s
    %p63 = tessera.call @square(%p62, %a63) : (!s, !s) -> !s
    %r = tessera.call @matvec(%p63, %v) : (!s, !v) -> !v
    llvm.return %r : !v
  }
}

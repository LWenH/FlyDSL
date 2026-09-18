// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering --canonicalize --fly-convert-atom-call-to-ssa-form --fly-promote-regmem-to-vectorssa --convert-fly-to-rocdl --canonicalize | FileCheck %s

// Data operands promote independently of pred. A shared-memory predicate stays
// a memref in copy_atom_call_ssa and must be loaded before entering the emitter.
// Loads preserve the old destination on the false branch; stores/atomics issue
// only in the true branch.
gpu.module @copy_mixed_pred {

  // CHECK-LABEL: gpu.func @universal_store(
  // CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
  // CHECK: scf.if %[[PRED]] {
  // CHECK: llvm.store
  // CHECK-NEXT: }
  gpu.func @universal_store(%atom: !fly.copy_atom<!fly.universal_copy<32>, 32>, %dst: !fly.memref<f32, global, 1:1>, %pred: !fly.memref<i1, shared, 1:1>, %value: f32) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<f32, register>
    fly.ptr.store(%value, %ptr) : (f32, !fly.ptr<f32, register>) -> ()
    %src = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, register>, !fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly.universal_copy<32>, 32>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, global, 1:1>, !fly.memref<i1, shared, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @buffer_store(
  // CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
  // CHECK: scf.if %[[PRED]] {
  // CHECK: rocdl.raw.ptr.buffer.store
  // CHECK-NEXT: }
  gpu.func @buffer_store(%atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, %dst: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %pred: !fly.memref<i1, shared, 1:1>, %value: f32) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<f32, register>
    fly.ptr.store(%value, %ptr) : (f32, !fly.ptr<f32, register>) -> ()
    %src = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, register>, !fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<i1, shared, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @universal_atomic(
  // CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
  // CHECK: scf.if %[[PRED]] {
  // CHECK: llvm.atomicrmw fadd
  // CHECK-NEXT: }
  gpu.func @universal_atomic(%atom: !fly.copy_atom<!fly.universal_atomic<#fly<atomic_op add>, f32>, 32>, %dst: !fly.memref<f32, global, 1:1>, %pred: !fly.memref<i1, shared, 1:1>, %value: f32) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<f32, register>
    fly.ptr.store(%value, %ptr) : (f32, !fly.ptr<f32, register>) -> ()
    %src = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, register>, !fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly.universal_atomic<#fly<atomic_op add>, f32>, 32>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, global, 1:1>, !fly.memref<i1, shared, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @buffer_atomic(
  // CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
  // CHECK: scf.if %[[PRED]] {
  // CHECK: rocdl.raw.ptr.buffer.atomic.fadd
  // CHECK-NEXT: }
  gpu.func @buffer_atomic(%atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_atomic<#fly<atomic_op add>, f32>, 32>, %dst: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %pred: !fly.memref<i1, shared, 1:1>, %value: f32) kernel {
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%one, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<f32, register>
    fly.ptr.store(%value, %ptr) : (f32, !fly.ptr<f32, register>) -> ()
    %src = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, register>, !fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_atomic<#fly<atomic_op add>, f32>, 32>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<i1, shared, 1:1>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @universal_load(
  // CHECK-SAME: %[[OLD:[a-zA-Z0-9_]+]]: f32
  // CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
  // CHECK: %[[RESULT:.*]] = scf.if %[[PRED]] -> (f32) {
  // CHECK: llvm.load
  // CHECK: scf.yield
  // CHECK: } else {
  // CHECK-NEXT: scf.yield %[[OLD]] : f32
  // CHECK: llvm.store %[[RESULT]],
  gpu.func @universal_load(%atom: !fly.copy_atom<!fly.universal_copy<32>, 32>, %src: !fly.memref<f32, global, 1:1>, %pred: !fly.memref<i1, shared, 1:1>, %out: !fly.ptr<f32, global>, %old: f32) kernel {
    %shape = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%shape, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<f32, register>
    fly.ptr.store(%old, %ptr) : (f32, !fly.ptr<f32, register>) -> ()
    %dst = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, register>, !fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly.universal_copy<32>, 32>, !fly.memref<f32, global, 1:1>, !fly.memref<f32, register, 1:1>, !fly.memref<i1, shared, 1:1>) -> ()
    %value = fly.ptr.load(%ptr) : (!fly.ptr<f32, register>) -> f32
    fly.ptr.store(%value, %out) : (f32, !fly.ptr<f32, global>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @buffer_load(
  // CHECK-SAME: %[[OLD:[a-zA-Z0-9_]+]]: f32
  // CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
  // CHECK: %[[RESULT:.*]] = scf.if %[[PRED]] -> (f32) {
  // CHECK: rocdl.raw.ptr.buffer.load
  // CHECK: scf.yield
  // CHECK: } else {
  // CHECK-NEXT: scf.yield %[[OLD]] : f32
  // CHECK: llvm.store %[[RESULT]],
  gpu.func @buffer_load(%atom: !fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, %src: !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, %pred: !fly.memref<i1, shared, 1:1>, %out: !fly.ptr<f32, global>, %old: f32) kernel {
    %shape = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%shape, %one) : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<f32, register>
    fly.ptr.store(%old, %ptr) : (f32, !fly.ptr<f32, register>) -> ()
    %dst = fly.make_view(%ptr, %layout) : (!fly.ptr<f32, register>, !fly.layout<1:1>) -> !fly.memref<f32, register, 1:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>, !fly.memref<f32, #fly_rocdl.buffer_desc, 1:1>, !fly.memref<f32, register, 1:1>, !fly.memref<i1, shared, 1:1>) -> ()
    %value = fly.ptr.load(%ptr) : (!fly.ptr<f32, register>) -> f32
    fly.ptr.store(%value, %out) : (f32, !fly.ptr<f32, global>) -> ()
    gpu.return
  }

  // CHECK-LABEL: gpu.func @lds_transpose(
  // CHECK-SAME: %[[OLD:[a-zA-Z0-9_]+]]: vector<2xi32>
  // CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
  // CHECK: %[[RESULT:.*]] = scf.if %[[PRED]] -> (vector<2xi32>) {
  // CHECK: rocdl.ds.read.tr4.b64
  // CHECK: scf.yield
  // CHECK: } else {
  // CHECK-NEXT: scf.yield %[[OLD]] : vector<2xi32>
  // CHECK: llvm.store %[[RESULT]],
  gpu.func @lds_transpose(%atom: !fly.copy_atom<!fly_rocdl.cdna4.lds_read_trans<trans = 4b, 64>, 32>, %src: !fly.memref<i32, shared, 2:1>, %pred: !fly.memref<i1, shared, 1:1>, %out: !fly.ptr<i32, global>, %old: vector<2xi32>) kernel {
    %shape = fly.make_int_tuple() : () -> !fly.int_tuple<2>
    %one = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%shape, %one) : (!fly.int_tuple<2>, !fly.int_tuple<1>) -> !fly.layout<2:1>
    %ptr = fly.make_ptr() {dictAttrs = {allocSize = 2 : i64}} : () -> !fly.ptr<i32, register>
    fly.ptr.store(%old, %ptr) : (vector<2xi32>, !fly.ptr<i32, register>) -> ()
    %dst = fly.make_view(%ptr, %layout) : (!fly.ptr<i32, register>, !fly.layout<2:1>) -> !fly.memref<i32, register, 2:1>
    fly.copy_atom_call(%atom, %src, %dst, %pred) : (!fly.copy_atom<!fly_rocdl.cdna4.lds_read_trans<trans = 4b, 64>, 32>, !fly.memref<i32, shared, 2:1>, !fly.memref<i32, register, 2:1>, !fly.memref<i1, shared, 1:1>) -> ()
    %value = fly.ptr.load(%ptr) : (!fly.ptr<i32, register>) -> vector<2xi32>
    fly.ptr.store(%value, %out) : (vector<2xi32>, !fly.ptr<i32, global>) -> ()
    gpu.return
  }
}

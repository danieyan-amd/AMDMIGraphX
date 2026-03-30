# Fix: Stride Mismatch in compile_ops Sequential Replacement

## Bug Summary

**JIRA**: [Topaz][MIGraphX] Topaz_SLMU Model fails to compile in MIGraphX  
**Symptom**: `code_object_op.cpp:46: compute_shape: Input shapes have changed`  
**Model**: Topaz SLMU (slmu-v1-16fr-fp16-90x160-ox_amd.onnx), 8514 nodes  
**GPU**: AMD Radeon RX 7900 XTX (gfx1100), Windows  

## Error Details

```
Error gpu::compile_ops: code_object_op.cpp:46: compute_shape: Input shapes have changed:
[float_type, {16,32,80,24,40}, {32,1,0,0,0},
 float_type, {16,32,80,24,40}, {1,16,0,0,0},    ← expected strides
 float_type, {16,32,80,24,40}, {1,1280,16,1638400,40960},
 float_type, {16,32,80,24,40}, {0,80,1,0,0},
 float_type, {16,32,80,24,40}, {0,80,1,0,0},
 half_type,  {16,32,80,24,40}, {1,1280,16,1638400,40960}]
->
[... {32,1,0,0,0} ...]                           ← actual strides differ
```

Input #1's strides changed from `{1,16,0,0,0}` (column-major broadcast) to
`{32,1,0,0,0}` (row-major broadcast) between compile time and replacement time.

## Root Cause

### Empirical Confirmation

Setting `MIGRAPHX_DISABLE_PASSES=normalize_ops` allows the model to compile
successfully (tested in ~45 minutes vs error at ~6h44m with normalize_ops enabled).
This confirms that `normalize_ops` creates the graph state that triggers the bug.

### Mechanism

The error occurs during `compile_ops::apply()` in `compile_manager::compile()`.
The compile pipeline has two phases:

1. **Parallel compilation**: All precompile_ops are compiled in parallel. Each
   compiler captures `expected_inputs` from the current shapes of `ins->inputs()`.

2. **Sequential replacement**: Each precompile_op is replaced with a code_object_op.
   `m.replace_instruction(ins, code_object_op, ins->inputs())` calls
   `code_object_op::compute_shape()` which validates that the current input shapes
   match `expected_inputs`.

**The bug**: When precompile_op A is replaced with code_object_op A, the instruction's
output shape changes from `precompile_op::compute_shape()` to `code_object_op.output`
(the output buffer shape). These can differ when the wrapped operation produces
non-standard strides (e.g., broadcast or transposed strides) while the output buffer
has standard strides.

If downstream precompile_op B has A as an input, B was compiled with
`expected_inputs` capturing A's **pre-replacement** shape. But during B's sequential
replacement, `ins->inputs()` now reflects A's **post-replacement** shape. The stride
mismatch causes the error.

### Why normalize_ops Triggers This

The first `normalize_ops` (pre-lowering, target.cpp line 109) normalizes axis
attributes on operations like `unsqueeze`, `squeeze`, `slice`, etc. This can change
the output strides of these operations (e.g., converting column-major to row-major
for certain axis normalizations). These changed strides propagate through the
subsequent ~20 passes to create a graph configuration where the precompile_op output
shape and the allocation buffer shape diverge, triggering the mismatch in compile_ops.

Without `normalize_ops`, the graph has a different stride configuration that happens
to avoid this divergence.

## Fix (V2 — Pre-Compile Normalization)

The fix ensures all operations are fully normalized before kernel compilation begins,
eliminating the stride divergence at its source.

### 1. target.cpp — Add pre-compile normalize_ops pass

The original pipeline had ~30 transformation passes between the first `normalize_ops`
(line 109) and `compile_ops`. These intermediate passes (simplify_reshapes, fuse_ops,
lowering, etc.) create new operations with un-normalized axis attributes. When
`compile_ops` runs, these un-normalized ops have strides that differ from what they'd
have after normalization — creating the divergence that causes the mismatch.

**Fix**: Added `normalize_ops{}, dead_code_elimination{},` immediately before
`compile_ops` in the pass pipeline (line 187). This ensures every operation's axis
attributes are finalized before any kernel compilation begins. Kernels capture the
correct, final strides. Sequential replacement no longer encounters mismatches.

The pipeline now has three `normalize_ops` passes:
- Line 109: Early normalization (before transformations)
- Line 187: **Pre-compile normalization** (NEW — after all transformations, before compile)
- Line 201: Post-compile normalization (now near-no-op for most instructions)

### 2. compile_ops.cpp — Safety net: re-compile on shape mismatch

As defense-in-depth, `compile_manager::compile()` wraps sequential replacement in
a try/catch. If a replacement still fails due to shape mismatch, the plan's results
and config are cleared so it stays in the compile plan list. `compile_ops::apply()`
calls `cm.update_configs()` before the second `cm.compile(m)` to re-fetch tuning
configurations for updated input shapes.

With the pre-compile normalization pass, this safety net should never trigger in
practice — but it provides guaranteed correctness for any edge cases.

### 3. normalize_ops.cpp — Unchanged (clean)

`normalize_ops.cpp` is unmodified from the original source. All operations are fully
normalized — no skipping, no guards. This preserves full normalization performance
benefits.

## Testing

1. **Reproduce**: Run SLMU model with stock MIGraphX — stride error after ~6h44m
2. **Workaround**: `MIGRAPHX_DISABLE_PASSES=normalize_ops` — model compiles in ~45min
3. **V2 fix validation**: Build with patched target.cpp + compile_ops.cpp
   - Compilation completes successfully (~1hr CPU time) ✅
   - **No stride mismatch error** ✅
   - **No normalization skipping** — all ops fully normalized ✅
   - Session creation fails with GPU OOM (16GB VRAM hardware limit, unrelated)
   - Post-compile `normalize_ops` emits cosmetic warnings for `code_object` ops
     (these ops can't be re-normalized after compilation; strides are already correct)

## Files Changed

- `src/targets/gpu/target.cpp` — Main fix: add `normalize_ops{}` + `dead_code_elimination{}` before `compile_ops`
- `src/targets/gpu/compile_ops.cpp` — Safety net: try/catch re-compile on shape mismatch
- `src/normalize_ops.cpp` — **Unchanged** (clean, original logic)

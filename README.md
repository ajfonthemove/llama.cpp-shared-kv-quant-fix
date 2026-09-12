# Shared-KV quantization fix for llama.cpp

Gemma 4 E2B and E4B reuse the KV cache from earlier layers for everything past
`n_layer_kv_from_start`. The `attn_k` and `attn_v` tensors for those layers still
exist in the GGUF, but inference never touches them.

So the imatrix collector never records importance data for them. Quantize to
anything that needs an imatrix — IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M,
IQ3_XXS, Q2_K_S — and the quantizer dies:

```
Missing importance matrix for tensor blk.N.attn_k.weight in a very low-bit quantization
```

I hit this quantizing Gemma 4 at 2 bits.

## Fix

Detect shared-KV attention tensors with the existing `hparams.has_kv()` and assign
them Q8_0, which needs no imatrix. They are unused at inference, so the type costs
nothing in quality.

Three files, 165 lines, with a test.

- `src/llama-quant.cpp` — the fix
- `tests/test-shared-kv-quant-imatrix.cpp` — covers shared-KV and non-shared-KV models
- `tests/CMakeLists.txt` — registers the test

## Apply

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply ../shared-kv-quant.patch
```

The test is also here on its own if you just want to read it.

MIT, same as llama.cpp.

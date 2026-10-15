import pytest
import torch

from kv_attention_lab.attention import (
    ModelConfig,
    TinyCausalLM,
    attention,
    cached_decode,
    full_recomputation,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_cached_logits_equal_full_prefixes(backend):
    model = TinyCausalLM(ModelConfig(width=32, heads=4, layers=2, max_sequence=64)).eval()
    tokens = torch.randint(0, 256, (2, 20))
    with torch.inference_mode():
        expected = full_recomputation(model, tokens, 11, 8, backend)
        actual = cached_decode(model, tokens, 11, 8, backend)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_manual_equals_sdpa_prefill_and_cached_chunk():
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 9, 8)
    v = torch.randn(2, 4, 9, 8)
    torch.testing.assert_close(attention(q, k, v, "manual", 6), attention(q, k, v, "sdpa", 6), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_causal_attention_cannot_see_future(backend):
    q = torch.randn(1, 2, 6, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    original = attention(q, k, v, backend)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, :, 3:] += 100
    changed_v[:, :, 3:] -= 500
    changed = attention(q, changed_k, changed_v, backend)
    torch.testing.assert_close(changed[:, :, :3], original[:, :, :3], atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_single_decode_query_attends_all_cached_keys(backend):
    q = torch.zeros(1, 1, 1, 2)
    k = torch.zeros(1, 1, 5, 2)
    v = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
    actual = attention(q, k, v, backend, query_offset=4)
    torch.testing.assert_close(actual, v.mean(dim=-2, keepdim=True))


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_chunked_cache_matches_full_model(backend):
    model = TinyCausalLM(ModelConfig(width=32, heads=4, layers=2, max_sequence=32)).eval()
    tokens = torch.randint(0, 256, (1, 13))
    with torch.inference_mode():
        full = model(tokens, backend=backend)
        cache = model.new_cache(1, 16)
        chunks = [model(tokens[:, :5], backend=backend, cache=cache), model(tokens[:, 5:9], backend=backend, cache=cache), model(tokens[:, 9:], backend=backend, cache=cache)]
    assert cache.length == 13
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=2e-6, rtol=2e-5)
    assert cache.allocated_tensor_bytes == 2 * 2 * 1 * 4 * 16 * 8 * 4


def test_capacity_rejection_preserves_cache_length_and_payload():
    model = TinyCausalLM(ModelConfig(width=32, heads=4, layers=1, max_sequence=16)).eval()
    cache = model.new_cache(1, 4)
    with torch.inference_mode():
        model(torch.tensor([[1, 2, 3]]), cache=cache)
        before = cache.keys[0][:, :, :3].clone()
        with pytest.raises(ValueError, match="capacity"):
            model(torch.tensor([[4, 5]]), cache=cache)
    assert cache.length == 3
    torch.testing.assert_close(cache.keys[0][:, :, :3], before)

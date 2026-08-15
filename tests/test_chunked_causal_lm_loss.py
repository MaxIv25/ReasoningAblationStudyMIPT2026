import torch
import torch.nn.functional as F

from src.sft_chunked_loss import chunked_causal_lm_loss


def _full_causal_lm_loss(hidden, weight, labels, bias=None):
    logits = F.linear(hidden[:, :-1], weight, bias)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def test_chunked_causal_lm_loss_matches_full_forward_and_backward():
    torch.manual_seed(7)
    hidden_full = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
    weight_full = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    bias_full = torch.randn(11, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 7))

    hidden_chunked = hidden_full.detach().clone().requires_grad_(True)
    weight_chunked = weight_full.detach().clone().requires_grad_(True)
    bias_chunked = bias_full.detach().clone().requires_grad_(True)

    expected = _full_causal_lm_loss(hidden_full, weight_full, labels, bias_full)
    actual = chunked_causal_lm_loss(
        hidden_chunked,
        weight_chunked,
        labels,
        bias=bias_chunked,
        chunk_tokens=3,
        checkpoint_chunks=True,
    )

    expected.backward()
    actual.backward()

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(hidden_chunked.grad, hidden_full.grad, atol=1e-10, rtol=1e-10)
    assert torch.allclose(weight_chunked.grad, weight_full.grad, atol=1e-10, rtol=1e-10)
    assert torch.allclose(bias_chunked.grad, bias_full.grad, atol=1e-10, rtol=1e-10)


def test_chunked_causal_lm_loss_respects_ignore_index_across_chunks():
    torch.manual_seed(11)
    hidden_full = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
    weight_full = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 9, (2, 6))
    labels[0, :3] = -100
    labels[1, 4:] = -100

    hidden_chunked = hidden_full.detach().clone().requires_grad_(True)
    weight_chunked = weight_full.detach().clone().requires_grad_(True)

    expected = _full_causal_lm_loss(hidden_full, weight_full, labels)
    actual = chunked_causal_lm_loss(
        hidden_chunked,
        weight_chunked,
        labels,
        chunk_tokens=2,
        checkpoint_chunks=True,
    )

    expected.backward()
    actual.backward()

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(hidden_chunked.grad, hidden_full.grad, atol=1e-10, rtol=1e-10)
    assert torch.allclose(weight_chunked.grad, weight_full.grad, atol=1e-10, rtol=1e-10)

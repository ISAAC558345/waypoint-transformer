"""Expected regret of a temperature-1 soft selection. This is not ListNet."""
from __future__ import annotations

import torch
from torch import Tensor


def expected_regret_loss(scores: Tensor, labels: Tensor) -> Tensor:
    """Minimize the soft selection's normalized payload regret.

    y is detached. Equal labels produce r = 0 and a zero gradient. The softmax
    temperature is 1. This does not optimize the hard argmax directly.
    """
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise RuntimeError("expected one score and one label per candidate")
    y = labels.detach().to(dtype=scores.dtype)
    span = (y.max() - y.min()).clamp_min(1.0e-6)
    regret = ((y.max() - y) / span).detach()
    probability = torch.softmax(scores, dim=0)
    return (probability * regret).sum()


def check_loss() -> dict:
    """Move the higher-payload score upward. Equal labels must have zero gradient."""
    from wt.algorithms.ranking_loss.losses import LOSSES, top_regret_loss

    if LOSSES["c"] is not top_regret_loss:
        raise RuntimeError("historical ListNet entry was replaced")
    higher = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    payload = torch.tensor([1.0, 4.0], dtype=torch.float64)
    loss = expected_regret_loss(higher, payload)
    loss.backward()
    before = higher.detach().clone()
    with torch.no_grad():
        higher -= higher.grad
    after = higher.detach()
    if not (float(after[1]) > float(before[1]) and float(after[0]) < float(before[0])):
        raise RuntimeError("higher-payload score did not move upward")
    equal_scores = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    equal_loss = expected_regret_loss(equal_scores, torch.tensor([7.0, 7.0, 7.0], dtype=torch.float64))
    equal_loss.backward()
    if float(equal_loss.detach()) != 0.0 or torch.count_nonzero(equal_scores.grad).item() != 0:
        raise RuntimeError("equal labels did not produce zero loss and zero gradient")
    left = torch.tensor([0.2, -0.4], dtype=torch.float64)
    right = left.clone()
    if float(expected_regret_loss(left, payload)) == float(top_regret_loss(right, payload)):
        raise RuntimeError("expected regret matched ListNet on the check tensors")
    return {
        "status": "passed",
        "higher_payload_score_increased": True,
        "equal_labels_zero_gradient": True,
        "distinct_from_listnet": True,
        "environment_steps": 0,
        "scoring_forwards": 0,
    }
import numpy as np
import torch
import torch.nn.functional as F

from sanvia.training.losses import MaskedWeightedCrossEntropy, MultiTaskLoss, compute_class_weights


def test_class_weights_balanced_train_only_and_absent_class():
    w = compute_class_weights(np.array([0, 0, 0, 0, 1, 2, 2]), 4)
    assert w[0] < w[1] and w[3] == 0.0 and abs(w[:3].mean() - 1.0) < 1e-9


def test_masked_loss_ignores_masked_samples():
    logits = torch.randn(4, 5)
    targets = torch.tensor([0, 1, 2, 3])
    mask = torch.tensor([True, True, False, False])
    loss = MaskedWeightedCrossEntropy()(logits, targets, mask)
    assert torch.allclose(loss, F.cross_entropy(logits[:2], targets[:2]))
    changed = targets.clone()
    changed[2:] = 4
    assert torch.allclose(loss, MaskedWeightedCrossEntropy()(logits, changed, mask))  # masked targets have no effect


def test_all_masked_gives_zero_with_gradient():
    logits = torch.randn(3, 5, requires_grad=True)
    loss = MaskedWeightedCrossEntropy()(logits, torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0 and logits.grad is not None


def test_weighted_loss_and_total_is_sum():
    w = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0])
    crit = MultiTaskLoss(w, None)
    out = {"birads": torch.randn(4, 5), "density": torch.randn(4, 4)}
    batch = {
        "birads": torch.tensor([0, 1, 1, 2]),
        "density": torch.tensor([0, 1, 2, 3]),
        "birads_mask": torch.ones(4, dtype=torch.bool),
        "density_mask": torch.tensor([True, True, True, False]),
    }
    total, parts = crit(out, batch)
    expected_b = F.cross_entropy(out["birads"], batch["birads"], weight=w)
    expected_d = F.cross_entropy(out["density"][:3], batch["density"][:3])
    assert torch.allclose(total, expected_b + expected_d) and torch.allclose(parts["birads"], expected_b)


def test_one_task_missing_still_trains_other():
    crit = MultiTaskLoss(None, None)
    out = {"birads": torch.randn(2, 5, requires_grad=True), "density": torch.randn(2, 4, requires_grad=True)}
    batch = {
        "birads": torch.tensor([1, 2]),
        "density": torch.tensor([0, 0]),
        "birads_mask": torch.ones(2, dtype=torch.bool),
        "density_mask": torch.zeros(2, dtype=torch.bool),
    }
    total, _ = crit(out, batch)
    total.backward()
    assert out["birads"].grad.abs().sum() > 0 and out["density"].grad.abs().sum() == 0

"""Calculs pédagogiques reproductibles ; aucune performance réelle d'IAnimaux."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def main():
    # In CNNs, the filter is applied without reversal (cross-correlation convention).
    image = torch.tensor([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]])
    kernel = torch.tensor([[[[1., 0.], [0., -1.]]]])
    convolution = F.conv2d(image, kernel)
    assert torch.equal(convolution, torch.full((1, 1, 2, 2), -4.))

    logits = torch.tensor([[2., 1.]], requires_grad=True)
    target = torch.tensor([0])
    probabilities = logits.softmax(dim=1)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    # This toy SGD step updates logits as parameters; the real model uses Adam on weights.
    updated_logits = logits.detach() - 0.1 * logits.grad
    updated_loss = nn.CrossEntropyLoss()(updated_logits, target)
    assert updated_loss.item() < loss.item()
    assert torch.allclose(logits.grad, probabilities.detach() - torch.tensor([[1., 0.]]))

    # Artificial confusion matrix: rows=true Chat/Chien, columns=predicted Chat/Chien.
    matrix = [[8, 1], [2, 9]]
    true_positive, false_negative = matrix[0]
    false_positive, true_negative = matrix[1]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    result = {
        "purpose": "Exemples pédagogiques artificiels ; ce ne sont pas les performances du modèle.",
        "convolution": {"image": image.squeeze().tolist(), "filter": kernel.squeeze().tolist(), "output": convolution.squeeze().tolist(), "first_cell_calculation": "1*1 + 2*0 + 4*0 + 5*(-1) = -4"},
        "normalization": {"red_pixel": 150, "scaled": 150/255, "normalized": (150/255 - 0.485)/0.229},
        "softmax_and_loss": {"logits": logits.detach().tolist(), "target": "Chat", "probabilities": probabilities.detach().tolist(), "cross_entropy": loss.item()},
        "toy_gradient_step": {"gradient": logits.grad.tolist(), "learning_rate": 0.1, "updated_logits": updated_logits.tolist(), "updated_loss": updated_loss.item(), "note": "Mise à jour pédagogique des logits ; le réseau réel ajuste les poids du classificateur avec Adam."},
        "artificial_metrics": {"confusion_matrix": matrix, "accuracy": (true_positive + true_negative)/20, "precision_chat": precision, "recall_chat": recall, "f1_chat": 2*precision*recall/(precision+recall)},
    }
    output_dir = Path(__file__).resolve().parents[1] / "results" / "pedagogie"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calculs_illustratifs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

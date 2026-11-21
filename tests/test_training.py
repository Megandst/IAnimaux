"""Contrôle les mises à jour et le checkpoint sans télécharger de poids."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ianimaux.model import build_model, CLASS_NAMES, load_checkpoint, PREPROCESSING, preprocessing_metadata
from ianimaux.training import run_epoch


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = build_model(pretrained=False)
        images = torch.rand(2, 3, 224, 224)
        self.loader = DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=2)

    def test_optimizer_updates_classifier_but_not_features_or_batchnorm(self):
        before_features = {key: value.clone() for key, value in self.model.features.state_dict().items()}
        before_classifier = self.model.classifier[-1].weight.detach().clone()
        optimizer = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad), lr=0.01)
        metrics, _ = run_epoch(self.model, self.loader, nn.CrossEntropyLoss(), optimizer)
        self.assertEqual(metrics["images"], 2)
        self.assertFalse(torch.equal(before_classifier, self.model.classifier[-1].weight))
        for key, value in self.model.features.state_dict().items():
            self.assertTrue(torch.equal(value, before_features[key]), key)
        self.assertTrue(all(p.grad is None for p in self.model.features.parameters()))

    def test_validation_changes_no_model_state_and_probabilities_sum_to_one(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        metrics, rows = run_epoch(self.model, self.loader, nn.CrossEntropyLoss())
        self.assertEqual(metrics["images"], 2)
        for row in rows:
            self.assertAlmostEqual(row["p_chat"] + row["p_chien"], 1, places=6)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_checkpoint_restores_same_logits_without_pretrained_weights(self):
        self.model.eval()
        images, _ = next(iter(self.loader))
        with torch.inference_mode():
            expected = self.model(images)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({
                "architecture": "mobilenet_v3_small", "class_names": CLASS_NAMES,
                "preprocessing": PREPROCESSING, "model_state_dict": self.model.state_dict(),
            }, path)
            restored, _ = load_checkpoint(path)
            with torch.inference_mode():
                actual = restored(images)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(tuple(actual.shape), (2, 2))

    def test_checkpoint_preserves_letterbox_contract_and_still_loads_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("center_crop", "letterbox"):
                path = Path(directory) / f"{name}.pt"
                torch.save({"architecture": "mobilenet_v3_small", "class_names": CLASS_NAMES,
                            "preprocessing": preprocessing_metadata(name),
                            "model_state_dict": self.model.state_dict()}, path)
                restored, _ = load_checkpoint(path)
                self.assertEqual(restored.preprocessing_name, name)


if __name__ == "__main__":
    unittest.main()

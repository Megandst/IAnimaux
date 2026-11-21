"""La décision de changement de modèle dépend d'exemples de validation réservés."""

import unittest

import pandas as pd

from scripts.compare_runs import promotion_checks, validation_results


class ComparisonTests(unittest.TestCase):
    def test_sphynx_improvement_cannot_hide_global_regression(self):
        baseline = {"accuracy": 0.98, "f1_macro": 0.97, "sphynx": {"recall": 0.85}}
        candidate = {"accuracy": 0.97, "f1_macro": 0.96, "sphynx": {"recall": 0.95}}
        self.assertFalse(all(promotion_checks(baseline, candidate).values()))
        candidate.update(accuracy=0.99, f1_macro=0.98)
        self.assertTrue(all(promotion_checks(baseline, candidate).values()))

    def test_missing_predictions_cannot_improve_scores_by_omission(self):
        annotations = pd.DataFrame({"image_id": ["a", "b"], "breed": ["Sphynx", "Beagle"], "target": [0, 1]})
        predictions = pd.DataFrame({"image_id": ["a"], "target": [0], "prediction": [0], "p_chat": [0.9], "p_chien": [0.1]})
        with self.assertRaisesRegex(ValueError, "toute la validation"):
            validation_results(predictions, annotations)

    def test_probability_roundoff_is_tolerated_but_invalid_probabilities_are_rejected(self):
        annotations = pd.DataFrame({"image_id": ["a", "b"], "breed": ["Sphynx", "Beagle"], "target": [0, 1]})
        predictions = pd.DataFrame({"image_id": ["a", "b"], "target": [0, 1], "prediction": [0, 1], "p_chat": [0.90000001, 0.1], "p_chien": [0.1, 0.9]})
        self.assertEqual(validation_results(predictions, annotations)[0]["accuracy"], 1)
        predictions.loc[0, "p_chat"] = 0.5
        with self.assertRaisesRegex(ValueError, "Probabilités"):
            validation_results(predictions, annotations)


if __name__ == "__main__":
    unittest.main()

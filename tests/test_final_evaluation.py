"""Calculs vérifiés sur exemples connus, sans lire les images du test officiel."""
import unittest
import pandas as pd
from scripts.evaluate_final import summarize_predictions


class FinalEvaluationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'image_id':['a','b','c','d'], 'target':[0,0,1,1],
            'prediction':[0,1,1,1], 'p_chat':[.9,.4,.1,.2], 'p_chien':[.1,.6,.9,.8]})

    def test_hand_calculated_metrics_and_fixed_baseline(self):
        summary = summarize_predictions(self.frame(),1)
        self.assertEqual(summary['confusion_matrix'],[[1,1],[0,2]])
        self.assertEqual(summary['accuracy'],.75)
        self.assertEqual(summary['errors'],1)
        self.assertAlmostEqual(summary['classification_report']['Chat']['precision'],1)
        self.assertAlmostEqual(summary['classification_report']['Chat']['recall'],.5)
        self.assertAlmostEqual(summary['f1_macro'],((2/3)+.8)/2)
        self.assertEqual(summary['majority_baseline']['accuracy'],.5)

    def test_duplicate_images_rejected(self):
        frame = self.frame();frame.loc[1,'image_id']='a'
        with self.assertRaises(ValueError): summarize_predictions(frame,1)

    def test_invalid_scores_or_decisions_rejected(self):
        for column,value in [('p_chat',float('nan')),('p_chien',1.5),('prediction',0)]:
            frame=self.frame();frame.loc[1,column]=value
            with self.assertRaises(ValueError): summarize_predictions(frame,1)

    def test_tie_uses_first_class(self):
        frame = pd.DataFrame({'image_id':['a'],'target':[0],'prediction':[0],'p_chat':[.5],'p_chien':[.5]})
        self.assertEqual(summarize_predictions(frame,1)['correct'],1)


class SavedFinalResultTests(unittest.TestCase):
    @unittest.skipUnless((__import__('pathlib').Path(__file__).resolve().parents[1]/'results/evaluation/binary_final_v1/test_results/summary.json').is_file(), 'Évaluation officielle non encore exécutée')
    def test_saved_results_cover_test_and_match_recorded_artifacts(self):
        import json
        from pathlib import Path
        from scripts.freeze_evaluation import digest
        root = Path(__file__).resolve().parents[1]
        output = root/'results/evaluation/binary_final_v1/test_results'
        summary = json.loads((output/'summary.json').read_text())
        execution = json.loads((output/'execution.json').read_text())
        predictions = pd.read_csv(output/'predictions.csv')
        annotations = pd.read_csv(root/'results/splits/test.csv')
        self.assertEqual(execution['status'],'completed')
        self.assertEqual(predictions.image_id.tolist(),annotations.image_id.tolist())
        self.assertEqual(predictions.target.tolist(),annotations.target.tolist())
        actual = pd.crosstab(predictions.target,predictions.prediction).reindex(index=[0,1],columns=[0,1],fill_value=0)
        self.assertEqual(actual.to_numpy().tolist(),summary['confusion_matrix'])
        self.assertEqual(int(predictions.target.ne(predictions.prediction).sum()),summary['errors'])
        self.assertEqual(len(predictions),3669)
        for filename, expected in execution['output_sha256'].items():
            self.assertEqual(digest(output/filename),expected,filename)
        self.assertEqual(digest(root/'scripts/evaluate_final.py'),execution['evaluation_script_sha256'])

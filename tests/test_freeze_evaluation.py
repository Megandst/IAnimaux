"""Le contrôle du gel doit détecter une altération sans ouvrir les images."""
import json
from pathlib import Path
import platform
import tempfile
import unittest

from config import MODEL_PATH, PROJECT_ROOT
from scripts.freeze_evaluation import digest, verify_freeze


class FreezeTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory)
        (path/'weights.pt').write_bytes(b'frozen weights')
        manifest = {
            'active_checkpoint': str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            'checkpoint_sha256': digest(MODEL_PATH),
            'frozen_files_sha256': {'weights.pt': digest(path/'weights.pt')},
            'source_files_sha256': {}, 'partition_sha256': {},
            'environment': {'python':platform.python_version(),'packages':{}},
        }
        (path/'manifest.json').write_text(json.dumps(manifest))
        return path

    @unittest.skipUnless(MODEL_PATH.is_file(), 'Checkpoint local nécessaire')
    def test_valid_snapshot_and_tampered_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            verify_freeze(path)
            (path/'weights.pt').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'weights.pt'):
                verify_freeze(path)

    @unittest.skipUnless(MODEL_PATH.is_file(), 'Checkpoint local nécessaire')
    def test_source_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            manifest = json.loads((path/'manifest.json').read_text())
            manifest['source_files_sha256'] = {'ianimaux/model.py':'wrong hash'}
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'ianimaux/model.py'):
                verify_freeze(path)

    @unittest.skipUnless(MODEL_PATH.is_file(), 'Checkpoint local nécessaire')
    def test_partition_drift_is_rejected_without_reading_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            manifest = json.loads((path/'manifest.json').read_text())
            manifest['partition_sha256'] = {'results/splits/test.csv':'wrong hash'}
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'results/splits/test.csv'):
                verify_freeze(path)

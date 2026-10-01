import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "tokenizer_assets", Path(__file__).resolve().parents[1]
    / "experiments/confirmation-20261001/tokenizer_assets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TokenizerAssetsTests(unittest.TestCase):
    def fixture(self, root):
        hub = root / "hub"
        snapshot = hub / "models--facebook--bart-large/snapshots" / module.TOKENIZER_REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps({"model_type": "bart"}))
        (snapshot / "vocab.json").write_text(json.dumps({"example": 0}))
        (snapshot / "merges.txt").write_text("#version: 0.2\n")
        checkpoint = hub / "models--lerobot--xvla-libero/snapshots/checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "policy_preprocessor.json").write_text(json.dumps({"steps": [{
            "registry_name": "tokenizer_processor",
            "config": {"tokenizer_name": module.TOKENIZER_ID}}]}))
        return checkpoint, snapshot

    def test_snapshot_is_explicit_and_does_not_follow_mutable_main_ref(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, snapshot = self.fixture(Path(temporary))
            ref = snapshot.parent.parent / "refs/main"
            ref.parent.mkdir()
            ref.write_text("different-main-revision")
            manifest = module.freeze_tokenizer_assets(checkpoint)
            self.assertEqual(manifest["snapshot"], str(snapshot))
            module.verify_tokenizer_assets(manifest)
            self.assertEqual(manifest["revision"], module.TOKENIZER_REVISION)

    def test_changed_bytes_and_new_optional_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, snapshot = self.fixture(Path(temporary))
            manifest = module.freeze_tokenizer_assets(checkpoint)
            vocab = snapshot / "vocab.json"
            original = vocab.read_bytes()
            vocab.write_text(json.dumps({"different": 0}))
            with self.assertRaises(ValueError):
                module.verify_tokenizer_assets(manifest)
            vocab.write_bytes(original)
            (snapshot / "added_tokens.json").write_text("{}")
            with self.assertRaises(ValueError):
                module.verify_tokenizer_assets(manifest)

    def test_incomplete_or_unexpected_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, snapshot = self.fixture(Path(temporary))
            (snapshot / "merges.txt").unlink()
            with self.assertRaises(ValueError):
                module.freeze_tokenizer_assets(checkpoint)
            (checkpoint / "policy_preprocessor.json").write_text('{"steps":[]}')
            with self.assertRaises(ValueError):
                module.freeze_tokenizer_assets(checkpoint)

    def test_loaded_processor_must_use_explicit_snapshot_not_just_declare_it(self):
        manifest = {"snapshot": "/frozen/snapshot", "model_id": module.TOKENIZER_ID,
                    "revision": module.TOKENIZER_REVISION, "assets": {"vocab.json": "hash"}}
        tokenizer = SimpleNamespace(name_or_path=manifest["snapshot"], vocab_size=1)
        step = SimpleNamespace(input_tokenizer=tokenizer, tokenizer=None,
                               tokenizer_name=manifest["snapshot"])
        processor = SimpleNamespace(steps=[step])
        self.assertTrue(module.verify_loaded_tokenizer(
            processor, manifest)["explicit_snapshot_loaded"])
        tokenizer.name_or_path = module.TOKENIZER_ID
        with self.assertRaises(ValueError):
            module.verify_loaded_tokenizer(processor, manifest)


if __name__ == "__main__":
    unittest.main()

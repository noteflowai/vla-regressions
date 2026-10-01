"""Freeze the selected X-VLA tokenizer's cached immutable snapshot, without imports."""
from hashlib import sha256
import json
from pathlib import Path

TOKENIZER_ID = "facebook/bart-large"
TOKENIZER_REVISION = "cb48c1365bd826bd521f650dc2e0940aee54720c"
TOKENIZER_FILES = (
    "config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
)
REQUIRED_FILES = ("config.json", "vocab.json", "merges.txt")


def manifest_at(snapshot):
    snapshot = Path(snapshot).resolve()
    if (snapshot.name != TOKENIZER_REVISION or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != "models--facebook--bart-large"):
        raise ValueError("Expected the immutable BART tokenizer snapshot")
    if not all((snapshot / name).is_file() for name in REQUIRED_FILES):
        raise ValueError("Incomplete cached tokenizer; do not download during collection")
    if json.loads((snapshot / "config.json").read_text()).get("model_type") != "bart":
        raise ValueError("Unexpected tokenizer model configuration")
    present = [name for name in TOKENIZER_FILES if (snapshot / name).is_file()]
    return {
        "model_id": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
        "snapshot": str(snapshot), "present_files": present,
        "absent_optional_files": [name for name in TOKENIZER_FILES if name not in present],
        "assets": {str(snapshot / name): sha256((snapshot / name).read_bytes()).hexdigest()
                   for name in present},
        "loading": "explicit_local_snapshot_override",
    }


def freeze_tokenizer_assets(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    steps = json.loads((checkpoint / "policy_preprocessor.json").read_text())["steps"]
    tokenizers = [step for step in steps if step.get("registry_name") == "tokenizer_processor"]
    if (len(tokenizers) != 1 or tokenizers[0].get("config", {}).get("tokenizer_name") != TOKENIZER_ID):
        raise ValueError("The selected checkpoint must use the expected BART tokenizer")
    hub = checkpoint.parents[2]
    return manifest_at(hub / "models--facebook--bart-large/snapshots" / TOKENIZER_REVISION)


def verify_tokenizer_assets(manifest):
    if manifest_at(manifest["snapshot"]) != manifest:
        raise ValueError("Tokenizer bytes or optional-file inventory changed")


def verify_loaded_tokenizer(processor, manifest):
    steps = [step for step in processor.steps if hasattr(step, "input_tokenizer")]
    if len(steps) != 1:
        raise ValueError("Exactly one loaded text tokenizer is required")
    step = steps[0]
    tokenizer = step.input_tokenizer
    if (getattr(step, "tokenizer", None) is not None
            or step.tokenizer_name != manifest["snapshot"]
            or tokenizer.name_or_path != manifest["snapshot"]):
        raise ValueError("Actual processor did not load the frozen local tokenizer snapshot")
    return {"model_id": manifest["model_id"], "revision": manifest["revision"],
            "class": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size,
            "asset_files": len(manifest["assets"]), "explicit_snapshot_loaded": True}

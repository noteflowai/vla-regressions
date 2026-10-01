"""Verify the released outcome archive by testing and rebuilding its extracted files."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "reproducibility-manifest.json").read_text())
archive = HERE / "reproducibility.tar.gz"
digest = lambda data: hashlib.sha256(data).hexdigest()
assert digest(archive.read_bytes()) == manifest["archive_sha256"]

with tempfile.TemporaryDirectory(prefix="vla-outcome-rebuild-") as tmp:
    root = Path(tmp)
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers()) == len(manifest["files"])
        assert set(tar.getnames()) == set(manifest["files"])
        for member in tar.getmembers():
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            data = tar.extractfile(member).read()
            assert digest(data) == manifest["files"][member.name]
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    assert all((root / name).is_file() for name in (
        "requirements-analysis.txt", "LICENSE", "NOTICE.md",
        "perstate_eval.py", "run_followup.sh",
    ))
    episode_files = list((root / "results/pilot").glob("*.jsonl"))
    episode_files += list((root / "results/followup").glob("*.jsonl"))
    assert sum(len(p.read_text().splitlines()) for p in episode_files) == 4040
    revision = root / "paper/revision-20261001"
    frozen = json.loads((revision / "figure-provenance.json").read_text())
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                   cwd=root, check=True)
    subprocess.run([sys.executable, str(revision / "build_figures.py")],
                   cwd=root, check=True, stdout=subprocess.DEVNULL)
    rebuilt = json.loads((revision / "figure-provenance.json").read_text())
    assert rebuilt == frozen, "Regenerated figure facts or input identities changed"
    names = ("paired-protocol", "test-resolution", "noise-floor",
             "statewise-pilot", "fresh-confirmation")
    assert all((revision / "figures" / (name + ".pdf")).is_file() for name in names)
    same_figures = all(digest((revision / "figures" / (name + ".pdf")).read_bytes())
                       == digest((HERE / "figures" / (name + ".pdf")).read_bytes())
                       for name in names)

print(json.dumps({
    "archive_sha256": manifest["archive_sha256"],
    "extracted_files": len(manifest["files"]),
    "own_episode_rows": 4040,
    "extracted_analysis_tests_passed": True,
    "regenerated_figure_facts_match": True,
    "figure_pdf_hashes_match_this_checkout": same_figures,
    "native_rollouts_executed": 0,
}))

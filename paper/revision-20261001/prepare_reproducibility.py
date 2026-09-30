"""Package outcome records and analysis separately from the minimal TeX upload."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile
import gzip

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
records = sorted((REPO/"results/pilot").glob("libero_10-*.jsonl"))
records += sorted((REPO/"results/followup").glob("libero_10-t3s3-*.jsonl"))
assert len(records) == 10, "Do not silently expand the released dataset"
assert sum(len(p.read_text().splitlines()) for p in records) == 4040
paths = records + [REPO/"results/vqb/reanalysis.json", REPO/"results/vqb/reanalysis.txt",
                   REPO/"analyze.py", REPO/"reanalyze_vqb.py", REPO/"tests/test_paired_analysis.py",
                   HERE/"build_figures.py", HERE/"figure-provenance.json", HERE/"README.md"]
assert all(p.is_file() for p in paths)
archive = HERE/"reproducibility.tar.gz"
identities = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
    with tarfile.open(fileobj=zipped, mode="w") as tar:
        for path in paths:
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(REPO)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                tar.addfile(info, stream)
with tarfile.open(archive) as tar:
    assert set(tar.getnames()) == set(identities)
    for name, digest in identities.items():
        assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest
(HERE/"reproducibility-manifest.json").write_text(json.dumps({
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "files": identities, "own_episode_rows": 4040, "new_episode_rows": 0,
    "public_data_origin": "https://github.com/jiuyixu25/VLAQuantBench",
    "public_data_commit": "4a2cb7c", "public_data_license": "MIT",
    "public_data_material": "derived reanalysis, not upstream source or model weights",
    "runtime": {"python": sys.version.split()[0]},
    "scope": "Outcome reanalysis and figures; not a bit-exact simulator replay. Local package; no public upload.",
}, indent=2)+"\n")
print(f"Reproducibility archive verified: {len(paths)} files, 4,040 unchanged own outcomes.")

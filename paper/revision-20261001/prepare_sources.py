"""Allowlisted, comment-cleaned upload sources; verify by compiling extracted files."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
FIGURES = ["paired-protocol", "test-resolution", "noise-floor",
           "statewise-pilot", "fresh-confirmation"]
archive = HERE / "arxiv.tar.gz"
with tempfile.TemporaryDirectory(prefix="vla-preprint-") as tmp:
    root = Path(tmp)
    tex = (HERE / "main.tex").read_text().replace(r"\graphicspath{{figures/}}",
                                               r"\graphicspath{{./}}")
    tex = "".join(line for line in tex.splitlines(keepends=True)
                  if not line.lstrip().startswith("%"))
    (root / "main.tex").write_text(tex)
    shutil.copyfile(HERE / ".build/main.bbl", root / "main.bbl")
    shutil.copyfile(HERE / "corl_2026.sty", root / "corl_2026.sty")
    for name in FIGURES:
        shutil.copyfile(HERE / "figures" / (name+".pdf"), root / (name+".pdf"))
    names = ["main.tex", "main.bbl", "corl_2026.sty"] + [n+".pdf" for n in FIGURES]
    identities = {n: {"bytes": (root/n).stat().st_size,
                      "sha256": hashlib.sha256((root/n).read_bytes()).hexdigest()}
                  for n in names}
    with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                 mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name in names:
                info = tar.gettarinfo(str(root/name), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with (root/name).open("rb") as stream:
                    tar.addfile(info, stream)
    # Verify the archive itself, rather than the directory used to construct it.
    extracted = root / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as tar:
        assert tar.getnames() == names
        for member in tar.getmembers():
            assert member.isfile() and Path(member.name).name == member.name
            (extracted/member.name).write_bytes(tar.extractfile(member).read())
    for name in names:
        assert hashlib.sha256((extracted/name).read_bytes()).hexdigest() == identities[name]["sha256"]
    for _ in range(3):
        subprocess.run(["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "main.tex"],
                       cwd=extracted, check=True, stdout=subprocess.DEVNULL)
    local = HERE / "output/pdf/higher-success-new-failures.pdf"
    local_text = subprocess.check_output(["pdftotext", "-layout", str(local), "-"])
    archive_text = subprocess.check_output(["pdftotext", "-layout", str(extracted/"main.pdf"), "-"])
    assert local_text == archive_text, "Upload sources differ from reviewed PDF"
    (HERE / "arxiv-manifest.json").write_text(json.dumps({
        "files": identities, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "reviewed_pdf_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
        "clean_room_compile_passed": True, "text_matches_reviewed_pdf": True,
        "third_party_license_comments_preserved": True,
        "new_rollouts_added": 0, "external_submission_changed": False,
    }, indent=2)+"\n")
print("Source archive: 8 allowlisted files; extracted clean-room build matches reviewed PDF.")

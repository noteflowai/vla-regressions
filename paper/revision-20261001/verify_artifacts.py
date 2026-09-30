"""Check metadata, TeX diagnostics, source identities and preserved workshop receipt."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
pdf = HERE/"output/pdf/higher-success-new-failures.pdf"
info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
assert pages == 9
assert re.search(r"Author:\s+Qiang Guo\n", info)
assert "Research preprint; expanded manuscript" in info
assert "Anonymous Submission" not in info
assert "Title:           Beyond Aggregate Success:" in info
fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
assert "Type 3" not in fonts
for line in fonts.splitlines()[2:]:
    assert line.split()[-5] == "yes", "Unembedded font: "+line
log = (HERE/".build/main.log").read_text()
assert "Overfull" not in log and "undefined" not in log
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(REPO/"paper/latex/main.pdf") == "5f883d848058efee8f5502c86677aab3a274da80e9f4128845a8b6baf121e54c"
assert digest(REPO/"paper/latex/arxiv.tar.gz") == "6d634fbb8667859c885e079478910be611fe1f33051e6bdae44b74bbe738dca5"
assert digest(HERE/"corl_2026.sty") == digest(REPO/"paper/latex/corl_2026.sty")
manifest = json.loads((HERE/"arxiv-manifest.json").read_text())
assert digest(pdf) == manifest["reviewed_pdf_sha256"]
assert digest(HERE/"arxiv.tar.gz") == manifest["archive_sha256"]
assert manifest["clean_room_compile_passed"] and manifest["text_matches_reviewed_pdf"]
assert len(manifest["files"]) == 8
proof = json.loads((HERE/"figure-provenance.json").read_text())
for name, expected in proof["sources"].items():
    assert digest(REPO/name) == expected
assert proof["facts"]["pilot"]["by_discoveries"] == [0, 0, 0, 0]
assert proof["facts"]["bf16_changed_episode_lengths"] == 422
assert proof["facts"]["followup"]["exact_p"] == .0002593994140625
assert .1/proof["facts"]["test_resolution"]["harmonic100"] < 1/32
companion = json.loads((HERE/"reproducibility-manifest.json").read_text())
assert companion["own_episode_rows"] == 4040 and companion["new_episode_rows"] == 0
assert digest(HERE/"reproducibility.tar.gz") == companion["archive_sha256"]
result = dict(pages=pages, figures=5, tables=2, metadata_correct=True,
              all_fonts_embedded=True, no_overfull_or_unresolved_references=True,
              old_submission_preserved=True, archived_data_unchanged=True,
              source_clean_room_passed=True, reproducibility_rows=4040,
              pdf_sha256=digest(pdf), archive_sha256=digest(HERE/"arxiv.tar.gz"),
              visual_review="required separately; renders are not inferred from these checks")
(HERE/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result))

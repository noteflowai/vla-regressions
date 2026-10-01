"""Check metadata, TeX diagnostics and source identities in a clean checkout."""
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
assert 9 <= pages <= 12
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
workshop_snapshot = json.loads((HERE/"workshop-source-snapshot.json").read_text())
for name, expected in workshop_snapshot["files"].items():
    assert digest(REPO/name) == expected, "Workshop source changed: "+name
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
assert proof["facts"]["followup"]["batch_sign_sensitivity_p"] == .0625
assert [(r["fp32_success"], r["bf16_success"])
        for r in proof["facts"]["followup"]["reconstructed_five_lane_batches"]] == [(4, 1), (4, 1), (5, 1), (5, 1)]
assert .1/proof["facts"]["test_resolution"]["harmonic100"] < 1/32
companion = json.loads((HERE/"reproducibility-manifest.json").read_text())
assert companion["own_episode_rows"] == 4040 and companion["new_episode_rows"] == 0
assert digest(HERE/"reproducibility.tar.gz") == companion["archive_sha256"]
audit = json.loads((HERE/"evidence-audit.json").read_text())
assert audit["public_data"]["fresh_reanalysis_equals_published"]
assert audit["public_data"]["jsonl_run_files"] == 354
assert audit["public_data"]["episode_rows"] == 70194
assert audit["public_data"]["baseline_update_comparisons"] == 309
assert digest(REPO/"results/vqb/reanalysis.json") == audit["public_data"]["archived_reanalysis_sha256"]
source = (HERE/"main.tex").read_text()
result = dict(pages=pages, figures=source.count(r"\begin{figure}"),
              tables=source.count(r"\begin{table}"), metadata_correct=True,
              all_fonts_embedded=True, no_overfull_or_unresolved_references=True,
              workshop_sources_preserved=True, archived_data_unchanged=True,
              source_clean_room_passed=True, reproducibility_rows=4040,
              pdf_sha256=digest(pdf), archive_sha256=digest(HERE/"arxiv.tar.gz"),
              pair_independence_validated=False, batch_sign_sensitivity_p=.0625,
              visual_review="required separately; renders are not inferred from these checks")
(HERE/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result))

# Expanded preprint revision, 2026-10-01

**Beyond Aggregate Success: Paired Statewise Evaluation of VLA Policy Updates**
is a signed revision of the workshop submission **Higher Success, New Failures:
Measuring Per-State Regressions in VLA Policy Updates**, using the same archived
outcomes. It is not a second paper or a new external submission.

- Reviewed PDF: `output/pdf/higher-success-new-failures.pdf`.
- Upload sources: `arxiv.tar.gz`; identities and clean-room result:
  `arxiv-manifest.json`.
- Figures: five vector PDFs in `figures/`, rebuilt by `build_figures.py`.
- Data identities and computed figure values: `figure-provenance.json`.
- Outcome/analysis companion: `reproducibility.tar.gz`; verify using
  `reproducibility-manifest.json`. This separate local package is not a claim
  that the data have already been publicly uploaded.

The submitted anonymous five-page baseline under `../latex/` remains the
historical OpenReview attachment. This revision has nine pages: seven main-text
pages, one references page and one reproducibility appendix. It has five figures
and two tables. External arXiv/OpenReview records have not been updated with it.

## What improved

The preprint title reflects the strongest result: one independently confirmed
BF16 loss when aggregate success changes little. The higher-success two-step
variant has only an unconfirmed naive new failure, so the preprint title does
not imply a confirmed loss for a demonstrably higher-success update.

The manuscript separates pilot candidates from fresh confirmation, shows all
100 states and all 20 fresh pairs, and explains why **no possible five-repeat
outcome pattern can pass BY with 100 hypotheses at q=.1**. A best-case 13-repeat
rank-1 crossing is explicitly not a general power recommendation.

The text identifies test assumptions, directional and update-family boundaries,
descriptive bootstrap/posterior scope, the 520-step timeout endpoint, simulated
weight rounding, and missing historical checkpoint/reset/camera digests.
“Changed trajectories” is corrected to the observable **422 changed episode
lengths**. PDF metadata identifies Qiang Guo and a research preprint.

The original numerical conclusions remain: all pilot BY counts are zero; the
fresh BF16 state has 18/20 versus 4/20 successes, 15 harmful versus one beneficial
discordance, and exact one-sided p=.0002593994140625.

## Rebuild

From the repository root:

```bash
python3 paper/revision-20261001/build_figures.py
python3 -m unittest discover -s tests -p test_paired_analysis.py
make -C paper/revision-20261001 all
python3 paper/revision-20261001/prepare_reproducibility.py
```

Figure generation requires Python, NumPy, SciPy and Matplotlib. The PDF build
requires TeX Live with `pdflatex`, `bibtex`, the packages used by the supplied
style, and Poppler's `pdftotext`. The upload archive includes prebuilt bibliography
output and figures; it compiles without Python, raw data or network access.
The source-pack builder extracts and compiles the archive, then requires text
equality with the reviewed PDF.
The companion archive supports outcome reanalysis and figure regeneration.
The separate TeX archive supplies the manuscript build.

VLAQuantBench reanalysis derives from its MIT-licensed public release
<https://github.com/jiuyixu25/VLAQuantBench>, commit `4a2cb7c`. Cite Xu et al.,
arXiv:2609.25376, for that benchmark. Public-data reanalysis is separate from the
newly collected historical X-VLA pilot and follow-up.

## Publication boundary

Resume existing arXiv draft **8144759** after its cs.RO endorsement gate clears.
Use this revision's metadata and archive together; do not reuse the five-page
comments or the previous source hash. Inspect arXiv's generated PDF before
submission. The existing OpenReview URL establishes a workshop submission,
not public publication or acceptance.

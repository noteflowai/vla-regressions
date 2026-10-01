# Expanded preprint revision, 2026-10-01

**Beyond Aggregate Success: Paired Statewise Evaluation of VLA Policy Updates**
is a signed, expanded revision of the same workshop study, originally titled
**Higher Success, New Failures:
Measuring Per-State Regressions in VLA Policy Updates**, using the same archived
outcomes. It is not a second paper or a new external submission.

- Reviewed PDF: `output/pdf/higher-success-new-failures.pdf`.
- Upload sources: `arxiv.tar.gz`; identities and clean-room result:
  `arxiv-manifest.json`.
- Figures: five vector PDFs in `figures/`, rebuilt by `build_figures.py`.
- Data identities and computed figure values: `figure-provenance.json`.
- Outcome/analysis companion: `reproducibility.tar.gz`; verify using
  `reproducibility-manifest.json`. Versioned releases publish the reviewed
  artifacts and checksums; each manifest identifies its own build snapshot.

The anonymous five-page manuscript under `../latex/` remains the workshop
version. The existing OpenReview note was updated on October 1 with its aligned
title and evidence limits, as documented in the repository's submission kit.
This expanded preprint has five figures and three tables; `verification.json`
records the current page count. It is public through GitHub releases and has
not been announced on arXiv.

## What improved

The preprint title reflects the targeted BF16 follow-up when aggregate success
changes little; its paired inference is conditional on the stated assumptions.
The higher-success two-step
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

The readiness revision adds a related-work comparison and the primary McNemar
reference, public artifact links, and an explicit audit of the public-data
matching. Fresh reanalysis from VLAQuantBench's immutable release commit exactly
matches the published derived results: 354 LIBERO files, 70,194 episode rows,
309 baseline/update comparisons and 20 repeated configurations. Two X-VLA files
have extra state labels; their comparisons retain all 190 baseline states and
exclude four additional rows. See `evidence-audit.json`.

The follow-up used four planned batches of five repeats. All four show lower
BF16 success (FP32/BF16: 4/1, 4/1, 5/1, 5/1), but a coarser independent-batch
sign sensitivity test gives p=.0625 under a different null. The episode-level
paired result is conditional on pair independence; it is not a finding robust
to arbitrary within-batch dependence. Batch grouping is reconstructed from the
retained historical scripts, not independently archived execution receipts.

## Rebuild

From the repository root:

```bash
python3 paper/revision-20261001/build_figures.py
python3 -m unittest discover -s tests -p test_paired_analysis.py
make -C paper/revision-20261001 all
python3 paper/revision-20261001/prepare_reproducibility.py
python3 paper/revision-20261001/verify_reproducibility.py
```

Figure generation requires Python, NumPy, SciPy and Matplotlib. The PDF build
requires TeX Live with `pdflatex`, `bibtex`, the packages used by the supplied
style, and Poppler's `pdftotext`. The upload archive includes prebuilt bibliography
output and figures; it compiles without Python, raw data or network access.
The source-pack builder extracts and compiles the archive, then requires text
equality with the reviewed PDF.
The companion archive supports outcome reanalysis and figure regeneration.
It includes pinned numerical dependencies, licence notices, the evidence audit
and the historical scripts needed to inspect the batch/seed plan. Those native
scripts retain historical environment paths and are not supported fresh-rollout
commands. The separate TeX archive supplies the manuscript build.
The extracted-archive check validates every file identity, runs the packaged
analysis tests and regenerates all figure facts without using the original
repository's outcome files. GitHub CI runs this check with pinned numerical
dependencies; it does not load a model or execute native rollouts.

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

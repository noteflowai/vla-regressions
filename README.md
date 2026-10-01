# Paired statewise evaluation of VLA policy updates

Research artifacts for **Beyond Aggregate Success: Paired Statewise Evaluation
of VLA Policy Updates**, by Qiang Guo, Independent Researcher.

This repository contains archived outcomes, paired-analysis code, a signed
expanded preprint and an anonymous workshop manuscript. It does not claim
peer-review acceptance or a completed arXiv announcement.

## Manuscripts and evidence

- [Reviewed preprint revision](https://github.com/noteflowai/vla-regressions/releases/tag/preprint-2026-10-01-r2):
  signed preprint, anonymous workshop PDF, TeX and reproducibility archives,
  with SHA-256 checksums. The
  [initial release](https://github.com/noteflowai/vla-regressions/releases/tag/preprint-2026-10-01)
  retains its original attachments.
- [Signed expanded preprint](paper/revision-20261001/output/pdf/higher-success-new-failures.pdf):
  ten pages, five vector figures and three tables.
- [Preprint TeX source archive](paper/revision-20261001/arxiv.tar.gz) and
  [source manifest](paper/revision-20261001/arxiv-manifest.json).
- [Outcome/analysis reproducibility archive](paper/revision-20261001/reproducibility.tar.gz)
  and [manifest](paper/revision-20261001/reproducibility-manifest.json).
- [Five-page workshop source](paper/latex/main.tex), updated October 1 with the
  same title and explicit historical-provenance limitations.
- [Existing OpenReview submission](https://openreview.net/forum?id=VUuO0hceYF):
  submission 2, restricted to the venue and author; acceptance is unverified.

The signed preprint is a revision of the same workshop study, previously titled
*Higher Success, New Failures*. It is not a second paper. The separately
developed [regression-budget study](https://github.com/noteflowai/vla-update-certification)
has its own prospective protocol and is published as development material,
without completed native update-efficacy claims. It must not be combined with
these archived outcomes.

## What the evidence supports

The exact paired analysis uses Benjamini–Yekutieli correction at q=.1 over 100
states per update. With five paired repeats, no possible outcome pattern passes
that correction. Seven W3 5/5-to-0/5 changes are descriptive candidates.

One pilot-selected BF16 state was tested on 20 fresh pairs: FP32 succeeded in
18/20 and BF16 in 4/20, with 15 harmful and one beneficial discordance
(one-sided exact p=.0002593994140625). That episode-level result assumes
independent paired repeats. The follow-up plan contains four five-lane batches;
all four show lower BF16 success, but a coarser independent-batch sign sensitivity
test gives p=.0625 under a different null. The result is conditional on the
evaluator assumptions and is not robust to arbitrary within-batch dependence.
It does not estimate suite-wide regression incidence. Non-detection does not
certify preservation.

Independent reanalysis of VLAQuantBench's immutable release commit matches the
archived summaries exactly: 354 LIBERO run files, 70,194 episodes, 309
baseline/update comparisons and 20 repeated configurations. The preprint
discloses the common-state matching for two files with additional state labels.

Historical logs lack complete checkpoint provenance and reset/camera snapshots.
Later corrected-pipeline audits cannot retrospectively establish those inputs.
The 422 changed records describe episode lengths, not complete trajectory
differences. Weight rounding stores floating dequantized values and does not
measure integer-kernel acceleration.

A [prospective serial confirmation protocol](experiments/confirmation-20261001/PROTOCOL.md)
fixes one selected state, 80 fresh FP32/BF16 pairs, independent reloads, balanced
randomized order and a separate exact-trajectory FP32 engineering control.
Its paired-test power planner, durable native adapter and CPU controls are
development artifacts, not completed native findings. The reviewed r2
manuscript and its archived outcomes remain the published evidence.
The [first engineering attempt](experiments/confirmation-20261001/engineering-control-001.json)
failed its original all-frame equality rule: two off-prediction camera channel
values differed by one intensity level, while actions, physics and outcomes
matched. The v2 protocol preserves that failure and defines a bounded,
reported exception for intermediate camera rounding; a fresh clean control
is still required before primary collection.

## Reproduce the analysis

Python 3.12 is the validated environment. The numerical dependencies are pinned
in [requirements-analysis.txt](requirements-analysis.txt).

```bash
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
```

The reproducibility archive includes the 4,040 own episode outcomes, public-data
derived summaries, analysis and figure sources with SHA-256 identities, pinned
dependencies and licence notices. The archive's extracted analysis tests and
figure facts can be independently rebuilt:

```bash
python paper/revision-20261001/verify_reproducibility.py
```

Publication state is recorded by the versioned release; manifests identify the
bytes of their build snapshots. Earlier release attachments remain unchanged.

This is outcome reanalysis, not a bit-exact simulator replay. Archived native
scripts are retained for provenance; a fresh rollout study needs its own pinned
checkpoint, processors, reset/scoring audit and resource protocol. No model
weights are included.

Original code is MIT-licensed. Manuscripts and original figures use CC BY 4.0.
Third-party template assets retain their notices; see [NOTICE.md](NOTICE.md).

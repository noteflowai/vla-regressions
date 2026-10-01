# Paired statewise evaluation of VLA policy updates

Research artifacts for **Beyond Aggregate Success: Paired Statewise Evaluation
of VLA Policy Updates**, by Qiang Guo, Independent Researcher.

This repository contains archived outcomes, paired-analysis code, a signed
expanded preprint and an anonymous workshop manuscript. It does not claim
peer-review acceptance or a completed arXiv announcement.

## Manuscripts and evidence

- [Signed expanded preprint](paper/revision-20261001/output/pdf/higher-success-new-failures.pdf):
  nine pages, five vector figures and two tables.
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
developed regression-budget study has its own prospective protocol and must not
be combined with these archived outcomes.

## What the evidence supports

The exact paired analysis uses Benjamini–Yekutieli correction at q=.1 over 100
states per update. With five paired repeats, no possible outcome pattern passes
that correction. Seven W3 5/5-to-0/5 changes are descriptive candidates.

One pilot-selected BF16 state was tested on 20 fresh pairs: FP32 succeeded in
18/20 and BF16 in 4/20, with 15 harmful and one beneficial discordance
(one-sided exact p=.0002593994140625). This supports the selected hypothesis
under the stated evaluator assumptions; it does not estimate suite-wide
regression incidence. Non-detection does not certify preservation.

Historical logs lack complete checkpoint provenance and reset/camera snapshots.
Later corrected-pipeline audits cannot retrospectively establish those inputs.
The 422 changed records describe episode lengths, not complete trajectory
differences. Weight rounding stores floating dequantized values and does not
measure integer-kernel acceleration.

## Reproduce the analysis

Python 3.12 is the validated environment. The numerical dependencies are pinned
in [requirements-analysis.txt](requirements-analysis.txt).

```bash
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
```

The reproducibility archive includes the 4,040 own episode outcomes, public-data
derived summaries, analysis and figure sources with SHA-256 identities. Its
manifests preserve the pre-publication snapshot and therefore still describe the
package as local. Publication does not change those archived bytes.

This is outcome reanalysis, not a bit-exact simulator replay. Archived native
scripts are retained for provenance; a fresh rollout study needs its own pinned
checkpoint, processors, reset/scoring audit and resource protocol. No model
weights are included.

Original code is MIT-licensed. Manuscripts and original figures use CC BY 4.0.
Third-party template assets retain their notices; see [NOTICE.md](NOTICE.md).

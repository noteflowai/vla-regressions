# Manuscript readiness review, 2026-10-01

The expanded preprint is suitable for **subject-area review and an individual
arXiv endorsement request with its stated evidence limits**. This is an author-side
audit, not an independent peer-review verdict or a prediction of acceptance.
It is an empirical evaluation audit and a numerical-update case study, rather
than a general update-safety guarantee or a new statistical method.

## Corrections completed before an invitation

- Reanalysis of the immutable VLAQuantBench release reproduces the existing
  summaries exactly. The paper now reports the actual LIBERO subset: 354 files,
  70,194 episodes, 309 baseline/update comparisons and 20 repeated configurations.
  Two files have extra state labels; all baseline labels remain in the matched
  comparisons, with four additional update rows excluded explicitly.
- The BF16 follow-up covers one state. It has 18/20 FP32 and 4/20 BF16 successes,
  with 15 harmful and one beneficial discordance. The episode-level exact
  p=.0002593994140625 assumes independent paired repeats.
- The four planned five-lane batches have FP32/BF16 counts 4/1, 4/1, 5/1 and 5/1.
  A coarser independent-batch sign sensitivity gives p=.0625 under a different
  null. The preprint now discloses this and avoids a claim robust to arbitrary
  within-batch dependence. No new rollout or missing historical provenance was
  invented to resolve the limitation.
- A comparison table distinguishes this study from LIBERO-CTRL, VLAQuantBench,
  STEP and update-admission work. The paper cites the original McNemar article.
- Public manuscript/data links, pinned numerical dependencies and licence notices
  are included. The extracted outcome archive can run its own analysis tests and
  regenerate the figure facts without using the checkout's original data.
- The build verifier works in a clean checkout and preserves the existing
  workshop sources, without requiring ignored historical PDF files.

The signed PDF has ten pages, five vector figures and three tables. TeX source
archive extraction and clean-room compilation must match the reviewed PDF text.
Visual inspection is recorded separately from numerical/build verification.

## What remains for stronger scientific claims

Historical checkpoint revisions and full reset/camera identities cannot be
retrospectively recovered from these outcomes. A new confirmation should freeze
model, processor and simulator identities, audit reset and success scoring,
use independently seeded single-lane pairs or a prespecified batch-aware
analysis, and fix the sample size or valid sequential rule before collection.
Its resource budget and every attempted rollout should be recorded.

One model and one selected state cannot establish regression prevalence,
cross-model generality or physical safety. A stronger full research paper needs
an additional model/update family, an adequately powered prospective sampling
design, and appropriately scoped uncertainty. More illustrations cannot replace
those experiments. The separate certification study's incomplete native
collection must not be counted as evidence for this preprint.

## Invitation boundary

Use the current signed PDF, not the restricted workshop URL, as review material.
Briefly explain the topic and conditional evidence. Ask a verified eligible
individual to assess cs.RO suitability and consider endorsement; do not ask
them to certify the numerical conclusion. Check the existing request's reply
before another request, contact one person at a time, and keep the private arXiv
link outside public artifacts. This audit sent no invitations.

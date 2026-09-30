# Higher Success, New Failures: current research and submission status

Updated 2026-09-30. This replaces the original development plan; historical
raw results and submission receipts remain preserved.

## Current evidence

The completed X-VLA pilot uses LIBERO-10, 10 tasks × 10 states × 5 repeats
= 500 episodes per variant, a 520-step horizon and absolute control.
A state includes task, initial-state index and scene seed. Old/new variants
share policy seeds within each repeat. Weight rounding is simulated weight-only
quantization, not an integer-kernel acceleration measurement.

The analysis now uses exact paired discordance tests and Benjamini-Yekutieli
at q=.1 per update. BH is secondary and assumption-dependent. The former
independent Fisher calculation was inappropriate for the paired design and
must not be used to support the old seven-discovery claim.

| Comparison with FP32 | Aggregate success | Naive failure/success flips | BY discoveries |
|---|---:|---:|---:|
| BF16 | .926 → .916 | 4 / 3 | 0 |
| W4, group 128 | .926 → .918 | 4 / 2 | 0 |
| W3, group 128 | .926 → .726 | 23 / 2 | 0 |
| Ten → two flow-matching steps | .926 → .932 | 1 / 3 | 0 |
| New policy seeds, same scenes | .926 → .928 | 0 / 3 | 0 |
| New scene seeds, same policy seeds | .926 → .930 | 0 / 2 | 0 |

Seven W3 states have 5/5 → 0/5 successes descriptively. Their minimum paired
one-sided p-value is 1/32; none is significant after the 100-state correction.
No discoveries in five repeats do not establish no statewise loss.

The separately collected BF16 follow-up on the pilot-selected task 3/state 3
uses 20 fresh paired repeats: FP32 18/20, BF16 4/20, 15 harmful and one beneficial
discordance, exact one-sided p=.0002593994. This confirms one selected candidate;
it is not an unbiased estimate of the frequency of BF16 failures across tasks.

The corrected calculations, audit and source hashes are in
`../artifacts/arxiv-acceleration-2026-09-30/`. Old derived JSON files under
`results/pilot/` describe the historical analysis and have not been overwritten.

## Publication

- CoRL 2026 EEAI nonarchival workshop: existing submission #2,
  <https://openreview.net/forum?id=VUuO0hceYF>. The corrected anonymous five-page
  PDF was revised into that same note on 2026-09-30 and downloaded/hash-verified.
  Readers remain restricted; this is a submission, not verified acceptance or
  public publication.
- arXiv: existing draft 8144759 remains incomplete at Start because cs.RO
  endorsement is missing. Its corrected four-file TeX archive has a clean-room
  build and ready metadata. One personal endorsement request was sent on
  September 29. No new outreach was sent during the correction.
- Full follow-up: the finite-suite regression-budget project is in
  `../research/second-paper-options-2026-09-30/`. New synthetic bound comparisons
  are development evidence; new multi-model closed-loop validation remains
  outstanding.

## Research scope

The first paper documents noise and a fresh confirmed candidate loss. It does
not establish a general rollout-saving sequential method or certify an upgrade.
The follow-up investigates that distinct decision problem and explicitly permits
an uncertain outcome. Recent STEP/SAVI, Admission, VLAQuantBench, compressed-policy
failure analysis, benchmark-bug work and LIBERO-MAX must be considered.

RSS 2027 Stage 1 permits preliminary experiments in a six-page extended abstract
(up to five content pages plus one references page), with fully specified
hypotheses and methods. Deadline: December 4, 2026 23:59 AoE, December 5 19:59
Singapore. That preliminary stage is different from the evidence required for
a full archival final paper. No new RSS/TMLR submission has been made.

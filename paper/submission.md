# Submission kit

Everything the account and submission forms ask for, ready to paste. The accounts belong to Qiang Guo:
the author registers, accepts the terms of service and clicks the final "Submit".

## 1. OpenReview profile (do this first: moderation of non-institutional emails takes up to 2 weeks)

Sign up: https://openreview.net/signup

| Field | Value |
|---|---|
| First / last name | Qiang / Guo |
| Preferred email | glay.guo@foxmail.com |
| Career & education history | Position: Independent Researcher; institution name: Independent Researcher; domain: leave empty; start year: current year |
| Personal links | any of: homepage, Google Scholar, DBLP, GitHub, ORCID (moderators need at least one public link that shows the person; ORCID is free: https://orcid.org/register) |
| Expertise | Robot learning; Vision-language-action models; Model quantization; Evaluation methodology |
| Conflicts | none beyond the automatic ones |

Then confirm the email from the foxmail inbox (check spam) and wait for moderation.

## 2. OpenReview submission (CoRL 2026 Workshop EEAI, due 2026-10-12 AoE)

Upload `paper/latex/main.pdf` (anonymous build). Do not upload `main-preprint.pdf`.

- **Title:** Higher Success, New Failures: Measuring Per-State Regressions in VLA Policy Updates
- **Authors:** Qiang Guo (from the profile)
- **Keywords:** vision-language-action models; quantization; evaluation; regression testing;
  LIBERO
- **TL;DR:** One rollout per initial state cannot tell which states a VLA update breaks: rerunning
  an unchanged policy flips up to a quarter of LIBERO states. We give a repeated-rollout,
  FDR-controlled per-state test and apply it to quantized X-VLA.
- **Abstract:** copy from `main.tex` (all results are in; no TODO left).

## 3. arXiv

Register: https://arxiv.org/user/register (the author chooses the username and password and accepts the agreement).

- **Primary category:** cs.RO. **Cross-list:** cs.LG.
- **Title:** as above. **Authors:** Qiang Guo
- **Comments:** 6 pages (4 plus references), 1 figure, 2 tables. Workshop paper.
  (Name the workshop only if its call for papers allows it during review.)
- **License:** the author's choice; CC BY 4.0 is the most common for open preprints.
- **Source:** `make arxiv.tar.gz` in `paper/latex/` (flat directory with main.tex, main.bbl,
  corl_2026.sty, fig_noise_floor.pdf; it compiles with the author names). Upload the tarball,
  check arXiv's generated PDF before "Submit".
- **Endorsement:** a first submission to cs.RO from a non-academic email usually needs an
  endorser. arXiv shows an endorsement code when the author starts the submission; send it to someone
  who has posted to cs.RO (arXiv lists who may endorse on each paper's "Which authors of this
  paper are endorsers?" link). Template:

  > Subject: arXiv endorsement request for cs.RO
  >
  > Dear Dr. …,
  >
  > I am an independent researcher preparing my first arXiv submission in cs.RO, "Higher Success,
  > New Failures: Measuring Per-State Regressions in VLA Policy Updates", which reanalyses
  > VLAQuantBench's LIBERO records and tests per-state regressions of quantized X-VLA. arXiv asks
  > for an endorsement for new submitters. If you are willing, the link is
  > https://arxiv.org/auth/endorse?x=CODE (code CODE). The draft is attached.
  >
  > Thank you for considering it.
  > Qiang Guo

Whether to post on arXiv during the review period is the author's call; the submission PDF stays anonymous
either way.

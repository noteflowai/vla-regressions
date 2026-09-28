# Higher success, new failures: per-state regressions in VLA policy updates

Working title. Literature scan: `~/work/research/paper-vla-regressions-2026-09-28/literature.md` (2026-09-28).

## Claim

When a VLA policy is updated (quantized, sped up, fine-tuned further), papers report the change in
aggregate success. We measure, for each initial state, whether the new policy succeeds less often
than the old one, with a test that accounts for rollout randomness, and show that:

1. a single rollout per state, the usual protocol, reports many "flips" that are rollout noise:
   the old policy against itself already shows them (the noise floor);
2. with repeated rollouts and a per-state test, real regressions remain behind flat or small
   aggregate changes, and they concentrate in particular tasks and states;
3. a sequential design finds them with a fraction of the rollouts of a fixed design.

## What is already published, and how this differs

| Work | What it does | What it does not |
|---|---|---|
| Sawada & Kasahara, arXiv 2609.15940 | pairs LIBERO initial states across perturbation conditions; up to 34.5% of states change outcome | model updates; per-state tests; one rollout per state |
| Yardımcı & Çoğurcu, arXiv 2609.28838 | task collapse under online RL of SmolVLA on LIBERO-10 | per-state; tests; other update types |
| Ma & Wu, arXiv 2609.10873 | paired binomial update-admission gates | a VLA (synthetic task only); per state; sequential |
| Snyder et al., STEP, arXiv 2503.10966 | sequential comparison of aggregate success | which states got worse |
| VLAQuantBench, arXiv 2609.25376 | 94,574 closed-loop quantization episodes, task-clustered intervals, episode records public (MIT) | per-state analysis; repeated rollouts per state |

## Protocol

- A **state** is (task, LIBERO initial state, scene seed). The initial-state file fixes the robot and
  movable objects, but LIBERO places fixtures at random on every hard reset, so the scene seed must
  be fixed too, or a "state" is not the same scene twice (found while building the harness: three
  resets of task 0, state 0 gave three cabinet positions).
- Each state is rolled out *n* times per policy variant. Policy noise seeds depend only on the plan,
  so variants share random numbers (common random numbers). A rerun of the same plan reproduces
  every episode exactly.
- Per state: one-sided Fisher exact test (new worse than old), Benjamini-Hochberg across states;
  Beta posterior P(worse). Aggregate: paired difference with a bootstrap over states.
- **Noise floor**: the old policy against itself with new policy seeds, same states.

## Experiments

| Update | Model | Status |
|---|---|---|
| FP32 → W4 (group 128) weight-only, simulated | X-VLA (`lerobot/xvla-libero`) | done: 0.926 → 0.918, 4+2 naive flips, 0 BH regressions |
| FP32 → W3 (group 128) | X-VLA | queue2 |
| FP32 → BF16 | X-VLA | queue2 |
| 10 → 2 flow-matching steps | X-VLA | queue2 |
| execute 1 → 10 → 50 actions per chunk | SmolVLA (`HuggingFaceVLA/smolvla_libero`) | deferred: smoke test 0/5 on LIBERO-10 task 0 with relative control, 436 s per batch of 5 (about 12 h per variant) |
| noise floor, old vs old (new policy seeds, same scenes) | X-VLA | done: 0.926 → 0.928, 0+3 naive flips, 0 BH regressions |
| scene noise: same policy seeds, new scene seeds | X-VLA | done: 0.926 → 0.930, 0+2 naive flips, 0 BH; 8/500 rollouts differ in outcome (policy seeds: 13/500), 275/500 identical episodes (policy seeds: 174/500). At fixed states both noise sources are small; the hypothesis that scene randomization carries most of VLAQuantBench's seed noise is not supported |
| reanalysis of VLAQuantBench episode records (single rollout per state) | X-VLA, π0.5, π0, OpenVLA-OFT | done: `reanalyze_vqb.py`, `results/vqb/` |
| π0.5 (`lerobot/pi05_libero_finetuned_v044`) | licence accepted; peak RAM too high to run beside the pilot | after the pilot, alone |

Pilot: LIBERO-10, 10 tasks × 10 states × 5 repeats = 500 episodes per variant, about 1.2 h each on
the L40S (LeRobot's 520-step limit; VLAQuantBench uses 900).

## Timeline

- by Oct 5: pilot, noise floor, go/no-go; the remaining X-VLA variants; SmolVLA chunk study.
- Oct 12: 4-page extended abstract, CoRL 2026 workshop "Efficient Foundation Models for Real-Time
  Embodied AI" (non-archival, https://efficient-embodied-ai.github.io/), and arXiv.
- then: full paper with sequential testing and all four LIBERO suites, RA-L (ICRA 2027 transfer
  window closes Dec 31, 2026) or RSS 2027.

## VLAQuantBench reanalysis (2026-09-28, repo at 4a2cb7c)

- Same configuration, seeds 0/1/2, 200 states: between 1 and 25% of states change outcome with
  nothing changed. π0.5 W4A4-ah on LIBERO-Object (success 0.60-0.63): 44-50 of 200 states flip
  between seeds. Near ceiling (0.95-0.99) it is still 1-14 states.
- Runs with the same seed do not share random numbers across configurations (flips at the same
  seed equal flips across seeds), so seed-to-seed flips are the right floor for their comparisons.
- States do differ in difficulty: at 0.61 success, 43 states never succeed in three seeds and 85
  always do; equal per-state rates would give 12 and 46. Per-state effects are real, but one
  rollout cannot find them.
- Example: π0 on LIBERO-10, W8 weights, aggregate 0.485 → 0.475, yet 24 states flip to failure
  and 22 to success; π0.5 LIBERO-10 W4A8, 0.950 → 0.965 with 5 + 8 flips, about the floor seen at
  that success rate.

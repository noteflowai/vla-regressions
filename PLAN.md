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
| FP32 → W8 / W4 (group 128) weight-only, simulated | X-VLA (`lerobot/xvla-libero`) | pilot running: fp32, w4 |
| FP32 → BF16 | X-VLA | planned |
| 10 → 5 flow-matching steps | X-VLA | planned |
| execute 1 → 10 → 50 actions per chunk | SmolVLA (`HuggingFaceVLA/smolvla_libero`) | planned |
| noise floor, old vs old | both | queued after the pilot |
| reanalysis of VLAQuantBench episode records (single rollout per state) | X-VLA, π0.5, π0, OpenVLA-OFT | planned |
| π0.5 (`lerobot/pi05_libero_finetuned_v044`) | needs the PaliGemma licence accepted on HuggingFace | blocked |

Pilot: LIBERO-10, 10 tasks × 10 states × 5 repeats = 500 episodes per variant, about 1.2 h each on
the L40S (LeRobot's 520-step limit; VLAQuantBench uses 900).

## Timeline

- by Oct 5: pilot, noise floor, go/no-go; the remaining X-VLA variants; SmolVLA chunk study.
- Oct 12: 4-page extended abstract, CoRL 2026 workshop "Efficient Foundation Models for Real-Time
  Embodied AI" (non-archival, https://efficient-embodied-ai.github.io/), and arXiv.
- then: full paper with sequential testing and all four LIBERO suites, RA-L (ICRA 2027 transfer
  window closes Dec 31, 2026) or RSS 2027.

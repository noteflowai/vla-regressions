# Higher Success, New Failures: Measuring Per-State Regressions in VLA Policy Updates

Draft for the CoRL 2026 workshop "Efficient Foundation Models for Real-Time Embodied AI"
(4 pages, non-archival, due Oct 12, 2026). Numbers marked TODO come from the pilot.

## Abstract

Quantized, distilled or otherwise accelerated vision-language-action (VLA) policies are judged by
the change in aggregate success rate. A deployment cares about something else too: whether the
new policy fails where the old one succeeded. Comparing the two policies one rollout per initial
state, the usual protocol, cannot answer this. In VLAQuantBench's public LIBERO records, running
the *same* configuration with a different seed changes the outcome of up to 25% of initial states
(50 of 200). We define a state as a LIBERO initial state together with its scene seed, roll out
each state several times per policy with common random numbers, and test each state for
regression with false-discovery control. On X-VLA under weight-only quantization, TODO.

## 1. Introduction

- Updates to deployed VLAs: quantization, fewer denoising steps, longer action chunks, more
  fine-tuning. Reported as Δ success over a benchmark.
- The question an operator asks: which situations did I lose? Backward compatibility of model
  updates (Bansal et al. 2019) for closed-loop control.
- Recent work pairs LIBERO states across conditions with one rollout each (Sawada & Kasahara,
  2609.15940: up to 34.5% of states change outcome). Without a noise floor such counts cannot be
  read: a large part of them happens with no change at all (Section 3).
- Contributions:
  1. the noise floor of single-rollout per-state comparisons, measured on public data from four
     VLAs and four LIBERO suites;
  2. a protocol: states that include the scene seed, repeated rollouts with common random numbers,
     per-state tests with Benjamini-Hochberg control; open-source harness on LeRobot;
  3. per-state regressions of X-VLA under W8/W4 weight quantization, TODO;
  4. (if time) sequential allocation of rollouts to uncertain states.

## 2. Method

- State s = (task, initial state from LIBERO's file, scene seed). LIBERO re-places fixtures on
  every hard reset; without the scene seed, three resets of task 0 / state 0 gave three cabinet
  positions.
- n rollouts per state and policy. Policy noise seeds depend on the plan only, so variants share
  random numbers and a rerun reproduces every episode.
- Per state: one-sided Fisher exact test of H0 "new succeeds at least as often"; BH at q = 0.1.
  Beta(1,1) posteriors for P(p_new < p_old). Aggregate Δ with a bootstrap over states.
- Noise floor: the old policy against itself, same states, new policy seeds.

## 3. How noisy is one rollout per state? (VLAQuantBench reanalysis)

Source: VLAQuantBench episode records (arXiv 2609.25376, MIT), commit 4a2cb7c. 20 configurations
were run with two or three seeds on the same 200 (190 for X-VLA) states.

- Flips between seeds of an unchanged configuration, out of 200 states: 1-8 at 97.5-99.5%
  success, 7-14 at 91-97%, 21-32 at 72-88%, 32-50 at 60-71% (π0.5 W4A4 action head, LIBERO-Object
  and -Spatial; X-VLA 37 of 190 at 66%).
- Seeds do not pair across configurations: two configurations at the same seed disagree as often
  as at different seeds, so these counts are the floor for their cross-configuration comparisons.
- States differ in difficulty: π0.5 W4A4-ah on LIBERO-Object, 43 states fail in all three seeds
  and 85 succeed in all three, against 12 and 46 if every state had the same success rate. Real
  per-state effects exist; single rollouts cannot separate them from noise.
- Consequence for published comparisons: π0 LIBERO-10 W8 moves aggregate success by -1.0 points
  while 24 states flip to failure and 22 to success; at π0's 48% success this is what seed noise
  alone would give. TODO: a figure of flip rate against success rate, updates versus same-config
  repeats, with the binomial curve 2p(1-p).

## 4. Per-state regressions of X-VLA under quantization (LeRobot, LIBERO-10)

- Setup: `lerobot/xvla-libero`, absolute control, 520-step limit, 10 tasks x 10 states x 5
  repeats. FP32 against simulated W8 (per-channel) and W4 (group 128) on all 267 linear layers.
- TODO table: aggregate Δ and CI; naive flips (repeat 0); flips in the fp32-vs-fp32 noise floor;
  BH regressions; per-task breakdown.

## 5. Discussion

- Report per-state regressions next to Δ success when shipping an updated policy; report the
  noise floor next to any per-state flip count.
- Limits: simulation only; LIBERO's states are few per task; weight-only fake quantization.
- Next: π0.5 and SmolVLA, chunk-length and denoising-step updates, sequential testing.

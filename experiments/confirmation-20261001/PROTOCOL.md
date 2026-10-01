# Prospective serial confirmation of the selected X-VLA BF16 state

This is an additional experiment for *Beyond Aggregate Success*, separately
prepared on October 1, 2026. It is not part of the regression-budget paper.
The original v1 document and executable design were published before control
collection. **This v2 revision follows a failed v1 engineering control and
precedes any v2 control or primary outcome.** Preparation and CPU tests are not
native efficacy evidence. The fixed state, 80-pair sample size, alpha, analysis
and primary seed stream are unchanged.

## Question, fixed state, and interpretation

Test whether the official X-VLA BF16 pipeline has lower success probability
than FP32 on **LIBERO-10, task 3, initial state 3, scene seed 10007306**, selected
by the archived pilot. Keep that state even if the result is uninteresting.
Use checkpoint `lerobot/xvla-libero` at immutable revision
`12e8783e996944f5c97e490d37d4c145484ed70a`, absolute control, 10 denoising steps,
30-action chunks and cadence, and a 520-transition collector horizon.

The canonical reset repair, current pinned software, serial sampler, and
immutable model revision define a **new operational evaluator**. This cannot
establish bit-exact reproduction of historical rollouts, whose provenance is
incomplete. BF16 uses the official configuration before model construction,
including BF16 sampler noise. The comparison is between precision pipelines,
not an isolated weight arithmetic intervention with identical noise values.
One selected state cannot establish regression prevalence across a suite.

## Fixed analysis and power

Collect **80 complete pairs**, each with a fresh policy seed shared between its
FP32 and BF16 episodes. The primary one-sided exact paired conditional test
uses alpha .05: among harmful and beneficial discordances, the null harmful
probability is at most .5. There is one predeclared primary hypothesis, no
post-hoc state search, and no statistical early stopping. The result requires
all 80 valid pairs and a clean supervised process exit. Report both success
counts, discordances, paired loss, a conservative exact 95% interval, exact
p-value and rejection decision. The interval subtracts two marginal 97.5%
Clopper-Pearson intervals and uses the Bonferroni union bound; it shares the
independent-pair assumption and may be wider than specialized paired intervals.
Non-rejection is inconclusive and does not certify preservation.

`design.py` analytically integrates over the binomial discordance count. Its
701-point discordance grids give approximately:

| Pairs | Absolute loss | Minimum power on the planning grid |
| ---: | ---: | ---: |
| 20 | .30 | .245 |
| 80 | .30 | .843 |
| 80 | .20 | .526 |
| 160 | .20 | .803 |

These alternatives are planning assumptions, not historical-effect estimates.
Grid minima are not proved continuous worst-case bounds. Power and the
primary test assume independent paired trials. Shared hardware and
`deterministic_algorithms(warn_only=True)` do not guarantee that assumption.
Also report the predeclared coarse sign test on the eight consecutive
ten-pair blocks. It tests a different null and cannot guarantee robustness to
arbitrary dependence. Report it even when it disagrees with the primary test.

## Collection, order, and engineering prerequisite

Every episode uses one synchronous environment lane. Every pair independently
reloads both policies from the same frozen checkpoint and releases each one
before constructing the other. Reset policy queues and seed Torch immediately
before prediction. Predeclare 40 FP32-first and 40 BF16-first pairs, shuffled
using a separate fixed order seed. Each pair creates fresh environments and
records two exact resets before prediction. Bind simulator state, both cameras,
proprioception, prompt, goals and controller to one reset-input identity across
all repeats and both pipelines. Do not reuse historical outcomes.

First collect a separate **two-pair FP32-versus-FP32 independent-reload
engineering control** using a separate seed stream. Require exact simulator
states, proprioception, actions, transitions, terminal arrays and outcomes,
and a clean process exit. Require exact camera inputs at each model prediction
step (0, 30, 60, …), with `n_obs_steps=1` and `n_action_steps=30` explicitly
validated before model construction. The official X-VLA queue consumes one
observation at a refresh and overwrites intermediate observations before the
next refresh. Initial reset cameras remain exact across all sides/repeats.

For recorded camera frames between predictions, permit only uint8
360×360 RGB rounding differences of at most one intensity level in at most
1e-4 of the channel values per frame. Keep and report every such difference;
this is an engineering rendering bound, not tolerance on model input,
physics, actions, or statistical outcomes. Any prediction-input difference,
non-camera difference or larger rendering drift fails the control.

A failure stops the primary launch and remains an
engineering failure, not a BF16 regression. These four control episodes are
excluded from the primary analysis. Control outcomes do not change the fixed
sample size, alpha, state, or hypothesis.

### Why v2 is necessary

The archived v1 attempt stopped after one complete FP32/FP32 pair. Both sides
succeeded at step 287, with identical actions, transitions, simulator states
and proprioception. Across 288 snapshots, the wrist camera differed in just
two uint8 channel values by one intensity level at step 55. That frame is
between model predictions and is overwritten before step 60. The literal v1
all-frame rule nevertheless failed, as specified. Its nonzero exit, raw
evidence and 800.515-second physical charge remain intact; **it is not promoted
to a passed v2 control**. See [the retained audit summary](engineering-control-001.json).
The cause of the two rendering differences is not established; the observation
does not justify blaming model loading, resets, or precision updates.

Future v2 controls require a distinct cohort and full clean execution.
No same-cohort retry or BF16 primary collection was launched after this failure.

## Admission, accounting, and failures

Require the complete source-matched 45-case reset/scoring audit and the existing
five-state X-VLA health evidence, including clean process exit. Record the
checkpoint/config/processor/benchmark hashes, all evaluator sources, package
versions, GPU and driver, actual parameter dtypes and runtime configuration.
Keep official strict-key loading distinct from per-tensor equality verification.

Before construction require at least **18 GiB available host RAM and 28 GiB
free GPU RAM**, with a second check immediately before model construction.
Do not lower these gates or stop other users' jobs. Require disk headroom for
128 MiB per remaining episode plus 8 GiB reserve. This is an admission estimate,
not a proven upper bound; stop if less than 2 GiB remains before a pair.

Use the existing, caller-specified shared 12-hour physical ledger. Never create
a fresh ledger to replenish it. Its canonical location is
`<runtime-root>/runs/native-feasibility-budget-001.json`. Reserve worker wall time plus teardown and
supervision margin under a file lock, and charge actual elapsed time, including
failed controls and model loading. Unresolved reservations remain charged.
Only a clean complete control permits primary admission; estimate the primary
reservation from 80 times its slowest full control-pair duration, with a 50%
margin and 120-second teardown plus 15-second margin. If that estimate exceeds
the remaining shared budget, do not launch.

Persist each attempt before environment construction; save complete raw inputs,
actions before stepping, checker results, environment termination/truncation
and collector truncation separately. Infrastructure errors are never binary
failures. Retain raw artifacts, partial pairs, failed launches and their charged
time. Existing launch records prohibit automatic same-cohort retry. A failed
or incomplete primary cohort yields **no primary inference**. Any redesign
requires a new published protocol and distinct cohort, with old evidence kept.

## Runtime dependency and commands

The adapter reuses the MIT-licensed native kernel in
[vla-update-certification](https://github.com/noteflowai/vla-update-certification/tree/ac10e2e5ef8a0cf1ba4b3c60011cc42110349db2).
Its five core source hashes are enforced before preparation; all further
health-validated sources are bound in the frozen context. The runtime must
already contain the installed pinned simulator, cached model/processors,
complete reset audit, suite inventory and original health receipt. See that
repository's `experiments/NATIVE_COLLECTION_003.md` for its setup and limits.
No weights, health certification or native outcomes are supplied by this adapter.
The pinned upstream health helper currently imports the original first-paper
environment constructor from `/home/dcvuser/work/vla-regressions/perstate_eval.py`;
that file and `quant.py` are additionally hashed. These native commands target
that audited installation layout, not an independently portable fresh install.
The planner and CPU controls run independently of that simulator installation.

CPU-only checks and planning:

```bash
python -m unittest discover -s tests -v
python experiments/confirmation-20261001/design.py --output /tmp/paired-power.json
```

Set `NATIVE_KERNEL_ROOT` to a checkout of the pinned kernel to also execute
the CPU adapter/lifecycle controls. CI does this automatically. Without that
checkout, the three adapter checks are skipped; no native model is imported.

Prepare a distinct control or primary cohort without importing a model:

```bash
/path/to/native/python experiments/confirmation-20261001/run_confirmation.py \
  --runtime-root /path/to/audited/runtime \
  --health-receipt /path/to/xvla-health-receipt.json \
  --budget-ledger /path/to/existing/native-feasibility-budget-001.json \
  --output /path/to/new/control --mode control --prepare-only
```

Remove `--prepare-only` for a bounded control; default wall time is 1,800 seconds.
For primary use `--mode primary --control /path/to/completed/control` and a
distinct `--output`. The runner determines its wall reservation from the
verified control's recorded pair timing. Preparation records readiness but
never schedules a background job to bypass an admission gate.

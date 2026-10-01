# Persistence replay and the next performance decision

This is an engineering measurement, not a policy-update result. The frozen v2
collector and its NPZ writer remain unchanged. The existing failed v1 cohort
remains failed; its retained observations are used only as read-only benchmark
inputs.

## Measurement

`profile_persistence.py` samples eight evenly spaced snapshots, including the
terminal snapshot, from each of the two complete v1 FP32 episodes. It makes
three repetitions per snapshot and shuffles the order of three writers with a
fixed seed. It calls the actual SHA-pinned native writer for the baseline.
Candidate writers retain file fsync, atomic replace and directory fsync.
Input reading and array verification are outside the measured write time.
All 144 writes were checked for field, shape, dtype and array-value equality.

The CPU replay ran while a separate v2 native control was loading. Warm caches,
sampled images and shared disk contention limit extrapolation. Full measurements
and raw images are retained locally; the public JSON is an aggregate report with
sample identities and hashes, not a public raw-data replication package.

| Writer | Median durable write | Mean NPZ size | Sample-based 80-pair storage |
| --- | ---: | ---: | ---: |
| Native compressed NPZ | 50.87 ms | 216,590 bytes | 18.05 GB |
| Deflate level 1 | 24.41 ms | 253,920 bytes | 21.17 GB |
| Stored NPZ | 11.00 ms | 781,218 bytes | 65.12 GB |

Storage assumes 160 episodes, 520 inputs and one terminal snapshot per episode.
It covers NPZ only, excludes other evidence and reserve space, and is a sample
mean rather than a guaranteed admission bound. Stored NPZ alone exceeds the
approximately 63.14 GB free disk measured at the new control's admission.

The two historical episodes together contain 554 intervals ending at a cached
action step and 18 ending at an action-refresh step. Their medians are 0.620 s
and 2.207 s. These intervals combine the preceding simulator advance, evidence
recording, preprocessing and action selection; they are not isolated model
latency or isolated simulator latency.

The native writer's approximately 51 ms is small compared with the cached-action
interval. Halving that writer's median saves approximately 26 ms per snapshot.
This comparison uses separate measurements and cannot establish an end-to-end
speedup. Compression alone is unlikely to make the fixed 80-pair experiment fit
the remaining shared physical budget.

## Next decision

Complete a fresh clean independent-reload control, then use the v5 full-horizon
forecast and the frozen 50% margin to make the primary admission forecast.
Keep all 80 pairs, the single environment lane, both precision pipelines, the
520-step horizon, reset/scoring checks and complete evidence.

Before adopting a faster collector, separately time:

1. Simulator `step`, including rendering and environment observation work.
2. Observation conversion, processor execution and action postprocessing.
3. Policy action selection, distinguishing cached steps from prediction steps;
   account for CUDA synchronization when interpreting wall times.
4. Snapshot encoding, file and directory sync, action/transition persistence.
5. Checkpoint hashing, model construction, weight loading and model release.

Use a new published protocol/context and a distinct engineering cohort for
instrumented or optimized native collection. Instrumentation should preserve
RNG use, model inputs, actions, physics and scoring. Qualify any adopted writer
with the same native control and independently verify complete retained arrays.
Do not reuse a failed cohort or reinterpret the replay as a clean native control.

The v3 introduced `--profile-episodes` for a new control cohort; v4 also
binds the cached text tokenizer's immutable snapshot and file inventory.
Its `phase-profile.json` records every call, inclusive wall/CPU time, count and
error flag. A completed episode requires the expected snapshot, prediction-call,
processor, reset, step and close counts with no failed timed call. CPU checks
verify return and argument identity, exception identity, exact call order and
restoration of both inherited and own attributes on success and error.
The separate v4 attempt collected two instrumented episodes, as recorded below.

## New v2 control and host observation

The separate v2 attempt used its preserved source `d58c0bc`, not the new timers.
One complete independent FP32 pair passed the comparator: both episodes succeeded
at step 287, with all saved arrays, actions and transitions equal, including
intermediate camera frames. The second pair's first load was denied when host
available memory fell to 15,283,400,704 bytes, below the fixed 18 GiB threshold.
The worker exited 1. The full two-pair clean-lifecycle control remains unqualified,
and the primary gate rejects it. Actual charged time is 920.179 seconds; the
canonical 12-hour ledger was finalized, not reset. Its remaining balance at that
point was 10.412 hours.
No BF16 or primary pair was collected.

The host has four logical CPUs. Read-only samples during collection found
23–24 runnable tasks, zero CPU idle and approximately 93% CPU pressure. Other
rendering/encoding tasks were active; their jobs and priorities were unchanged.
The two identical-output episodes took 184.0 and 291.7 seconds inside the episode
kernel. This demonstrates variable host wall time, not a measured policy speedup
or attribution of all overhead to one phase. Collect phase timings under a stable
resource window before adopting an optimization or forecasting the fixed study.

See `engineering-control-v2-001.json` and `cpu-contention-001.json`. Full raw
episodes remain local. The BART tokenizer snapshot was not separately bound in
v2; `tokenizer-assets-audit-001.json` compares the current cached and explicit
snapshot tokenizers on CPU, not historical tokenizers. Vocabulary, parsed backend
and all encoded fields for five strings (including the selected task prompt)
match. This audit constructs no policy and initializes no CUDA context. Future
v4 and later load the explicit snapshot and reject changed files or optional-file inventory.

The latest official LeRobot release checked on October 1 is still v0.6.1:
https://github.com/huggingface/lerobot/releases/tag/v0.6.1.
Its release notes describe direct-device safetensors loading in shared policy
code, but the pinned X-VLA override constructs its instance and loads a CPU state
dictionary before moving it. General release notes do not establish that this
specific loader is already optimized. Any alternate loader needs a separate
prospective design, tensor verification and native control.

## Instrumented v4 result and prospective v5 cleanup

The v4 attempt (`6ef051b`) completed one exact FP32/FP32 pair: both episodes
succeeded at step 287, with all retained observations, actions and transitions
equal and no off-cadence camera differences. Both phase profiles passed their
call-count and error guards. Episode kernel times were 282.162 and 292.689 s.
The next load was denied at 18,586,587,136 available host bytes, below 18 GiB;
worker exit 1 means the full two-pair control remains unqualified.
Charged time was 908.193 s. The finalized ledger then retained 36,575.908 s.
See `engineering-control-v4-001.json`; no BF16 or primary data were collected.

For the second episode, snapshot persistence took 69.702 s inclusive wall time
and 9.060 s process CPU time; simulator steps took 56.575 / 7.553 s, action
selection 50.596 / 8.302 s, and observation conversion 34.607 / 8.249 s.
These are inclusive, partly overlapping timers without added CUDA
synchronization. Their CPU/wall gaps do not isolate scheduler, disk or GPU waits.
They do not establish an end-to-end speedup or justify summing phases as GPU time.

V5 prospectively calls glibc `malloc_trim(0)` in the collector's own process
after model references, garbage collection and the inherited CUDA cleanup.
It records own RSS and global host availability before and after the call.
The API releases unused allocator pages; it does not free live tensors or alter
other processes. Retained allocator pages are a hypothesis, not an established
cause of the resource failures. A fresh-process API check and CPU adapter tests
do not validate cleanup after an actual model reload. A distinct native cohort
is required. The original 18 GiB host and 28 GiB free GPU gates remain fixed.
API reference: https://man7.org/linux/man-pages/man3/malloc_trim.3.html.

V5 also separates the largest observed pair overhead from episode time, scales
the largest observed episode seconds per step to 520 steps on both sides, then
applies 80 pairs and the existing 50% margin. Successful early termination at
287 steps must not understate a possible full-horizon study. This sample-based
forecast is not a worst-case guarantee: precision behavior and shared host load
can differ. Infeasible admission stops collection without shrinking the sample.

## Reproduce the CPU replay

```sh
python experiments/confirmation-20261001/profile_persistence.py \
  --runtime-root /path/to/audited-runtime \
  --episode /path/to/complete-old-episode \
  --episode /path/to/complete-new-episode \
  --output /path/to/new-benchmark-directory \
  --concurrent-native-collection
```

Omit `--concurrent-native-collection` when no native collection overlaps the
benchmark. This command does not construct a policy or run a simulator. It
requires the pinned native writer and numerical NPZ observations; the output
directory must be new and outside the preserved episode directories.

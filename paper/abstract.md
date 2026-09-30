# Higher Success, New Failures: Measuring Per-State Regressions in VLA Policy Updates

Current abstract, synchronized with `latex/main.tex` on 2026-09-30.
The TeX manuscript is authoritative. The earlier outline and its Fisher/BH
claims have been superseded by the paired-analysis correction.

Quantized, distilled or otherwise accelerated vision-language-action (VLA) policies
are judged by the change in their aggregate success rate. A deployment also needs
to know whether the new policy fails where the old one succeeded. Comparing the
two policies with one rollout per initial state, the usual protocol, cannot tell:
in VLAQuantBench's public LIBERO records, rerunning an unchanged configuration
with a new seed changes the outcome of up to a quarter of the initial states.
We define a state as a LIBERO initial state together with its scene seed, roll
out every state several times per policy with common random numbers, and use
exact paired tests with false-discovery control. On X-VLA in LIBERO-10, the
first-rollout old-versus-old floor is 2–3 flips per 100 states. Three-bit weights
concentrate a 20-point aggregate drop on a few tasks; seven states change from
5/5 to 0/5 successes, but the five-repeat screen is underpowered after multiplicity
correction. Four-bit weights and fewer denoising steps produce no detected
statewise regression, which does not certify their absence. BF16 leaves the
aggregate nearly flat but identifies one candidate state; 20 fresh paired
rollouts confirm its loss (18/20 against 4/20, exact p = 0.000259).

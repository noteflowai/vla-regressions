"""Repeated closed-loop rollouts from fixed LIBERO initial states.

LeRobot's evaluator walks through initial states with a stride, so each state is seen once.
To tell a state that got harder from a rollout that was unlucky, every (task, initial state)
here is rolled out several times, each repeat with its own policy seed. The seeds depend only
on the plan (task, batch, base seed), so two policy variants run with the same random numbers
and can be compared pair by pair.

A state is an initial state from LIBERO's file plus a scene seed: the file fixes the robot and
the movable objects, but fixtures (cabinets, stoves) are placed at random each time the scene is
rebuilt, so the seed has to be fixed too.

Writes one JSON line per episode: task, init_state, repeat, success, steps, variant.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# Deterministic cuBLAS, so a rerun of the same plan gives the same episodes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium as gym
import numpy as np
import torch

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import ACTION

import quant


def make_sub_env(suite, suite_name, task_id, init_state, env_cfg, episode_length):
    kwargs = {k: v for k, v in env_cfg.gym_kwargs.items() if k != "task_ids"}
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        camera_name=env_cfg.camera_name,
        episode_length=episode_length,
        episode_index=init_state,
        n_envs=1,
        control_mode=env_cfg.control_mode,
        camera_name_mapping=env_cfg.camera_name_mapping,
        **kwargs,
    )
    # Stay on this initial state on every reset.
    env._reset_stride = 0
    return env


def plan(tasks, states, repeats, batch_size):
    """Batches of (init_state, repeat) per task, in a fixed order."""
    jobs = [(s, r) for r in range(repeats) for s in states]
    for task in tasks:
        for b in range(0, len(jobs), batch_size):
            yield task, b // batch_size, jobs[b : b + batch_size]


def scene_seed(base, task, init_state):
    """Fixtures are placed at random on every hard reset; fix them per state."""
    return base * 10007 + task * 101 + init_state


def run_batch(env, policy, env_pre, env_post, pre, post, n, seeds):
    policy.reset()
    observation, _ = env.reset(seed=seeds)
    max_steps = env.call("_max_episode_steps")[0]
    done = np.zeros(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    steps = np.zeros(n, dtype=int)
    t = 0
    while not done.all() and t < max_steps:
        obs = preprocess_observation(observation)
        obs["task"] = list(env.call("task_description"))
        obs = pre(env_pre(obs))
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = env_post({ACTION: post(action)})[ACTION].to("cpu").numpy()
        observation, _, terminated, truncated, info = env.step(action)
        t += 1
        ended = (terminated | truncated) & ~done
        if ended.any():
            ok = np.asarray(info.get("is_success", np.zeros(n, dtype=bool)), dtype=bool)
            success[ended] = ok[ended]
            steps[ended] = t
            done |= ended
    steps[~done] = t
    return success, steps


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", default="lerobot/xvla-libero")
    p.add_argument("--variant", default="fp32", help="fp32, bf16, w<bits> (weight-only), steps<N> (denoising steps), chunk<N> (actions executed per chunk)")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", default="0-9", help="e.g. 0-9 or 0,3,5")
    p.add_argument("--states", default="0-9", help="initial state indices, e.g. 0-9")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000, help="policy noise seed")
    p.add_argument("--scene-seed", type=int, default=None, help="default: --seed; keep it fixed to compare on the same states")
    p.add_argument("--control-mode", default="absolute")
    p.add_argument("--episode-length", type=int, default=None, help="default: LeRobot's per-suite limit")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    def ids(spec):
        if "-" in spec:
            a, b = spec.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in spec.split(",")]

    tasks, states = ids(args.tasks), ids(args.states)
    if args.scene_seed is None:
        args.scene_seed = args.seed

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
    policy_cfg.pretrained_path = args.policy
    policy_cfg.device = "cuda"
    if args.variant == "bf16":
        policy_cfg.dtype = "bfloat16"
    if args.variant.startswith("steps"):
        policy_cfg.num_denoising_steps = int(args.variant[5:])
    if args.variant.startswith("chunk"):
        policy_cfg.n_action_steps = int(args.variant[5:])
    env_cfg = LiberoEnvConfig(task=args.suite, control_mode=args.control_mode)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    if re.fullmatch(r"w\d", args.variant):
        report = quant.fake_quantize_(policy, bits=int(args.variant[1:]))
        print(json.dumps({"quantized": report}))
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done_keys.add((r["task"], r["init_state"], r["repeat"]))

    with out.open("a") as f:
        for task, batch_ix, jobs in plan(tasks, states, args.repeats, args.batch_size):
            if all((task, s, r) in done_keys for s, r in jobs):
                continue
            # The seed depends on the plan only, never on the variant.
            torch.manual_seed(args.seed * 100003 + task * 1009 + batch_ix)
            np.random.seed((args.seed + task * 7919 + batch_ix) % 2**32)
            env = gym.vector.AsyncVectorEnv(
                [
                    partial(make_sub_env, suite, args.suite, task, s, env_cfg, args.episode_length)
                    for s, _ in jobs
                ],
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            )
            t0 = time.time()
            try:
                seeds = [scene_seed(args.scene_seed, task, s) for s, _ in jobs]
                success, steps = run_batch(env, policy, env_pre, env_post, pre, post, len(jobs), seeds)
            finally:
                env.close()
            for (s, r), ok, n in zip(jobs, success, steps):
                f.write(
                    json.dumps(
                        {
                            "variant": args.variant,
                            "policy": args.policy,
                            "suite": args.suite,
                            "task": task,
                            "init_state": s,
                            "repeat": r,
                            "seed": args.seed,
                            "scene_seed": scene_seed(args.scene_seed, task, s),
                            "success": bool(ok),
                            "steps": int(n),
                        }
                    )
                    + "\n"
                )
            f.flush()
            print(f"task {task} batch {batch_ix}: {int(success.sum())}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

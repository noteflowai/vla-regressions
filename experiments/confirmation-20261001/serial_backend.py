"""First-study precision adapter for the audited native kernel (MIT, Qiang Guo).

Inherited episode/reset/horizon recording comes from vla-update-certification
ac10e2e5ef8a0cf1ba4b3c60011cc42110349db2. The source hashes are pinned in
run_confirmation.py. No shared runtime or installed package is modified.
"""
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import time

from native_collection import (NativeBatchProducer, digest_file, object_digest,
                               bind_state_reset, device_identity)
from native_rollout import LiberoNativeBackend
from paired_collection import PairCache, atomic_json
from precision import configure_precision, verify_precision


class SerialPrecisionBackend(LiberoNativeBackend):
    def load(self, side, context, folder, batch_id):
        if self.policy is not None:
            raise RuntimeError("Release the previous policy before provisioning a side")
        if device_identity() != context["evaluator"]["device_identity"]:
            raise ValueError("Native device or driver changed; do not mix this cohort")
        # These imports/settings are the same ones exercised by the family-health run.
        import run_closedloop_health as h
        from health_supervisor import resource_probe
        from update_variants import configure_variant
        probe = resource_probe()
        atomic_json(folder / f"resource-{batch_id}-{side}-admission.json", probe)
        if not probe["allowed"]:
            raise RuntimeError("Native pipeline loading requires current resource headroom: "
                               + json.dumps(probe, sort_keys=True))
        for name, expected in context["evaluator"]["sources"].items():
            if digest_file(name) != expected:
                raise ValueError("Frozen evaluator source changed: " + name)
        for name, expected in context["evaluator"]["packages"].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError("Frozen dependency version changed: " + name)
        pipeline = context[side + "_pipeline"]
        checkpoint = Path(pipeline["checkpoint"])
        for name, expected in pipeline["assets"].items():
            if digest_file(name) != expected:
                raise ValueError("Frozen processor/config asset changed: " + name)
        weight_path = checkpoint / "model.safetensors"
        stat = weight_path.stat()
        signature = (str(weight_path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns,
                     stat.st_ctime_ns, pipeline["model_weight_sha256"])
        # Hash once per unchanged process-local file signature. No cross-run digest cache.
        if signature != self.verified_checkpoint:
            if digest_file(weight_path) != pipeline["model_weight_sha256"]:
                raise ValueError("Checkpoint bytes differ from the health-validated model")
            self.verified_checkpoint = signature
        probe = resource_probe()
        atomic_json(folder / f"resource-{batch_id}-{side}-construction.json", probe)
        if not probe["allowed"]:
            raise RuntimeError("Resource headroom changed before native model construction")
        h.torch.set_num_threads(2)
        h.torch.cuda.set_per_process_memory_fraction(.4)
        h.torch.backends.cuda.matmul.allow_tf32 = False
        h.torch.backends.cudnn.allow_tf32 = False
        h.torch.backends.cudnn.benchmark = False
        h.torch.backends.cudnn.deterministic = True
        h.torch.use_deterministic_algorithms(True, warn_only=True)
        cfg = h.PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.pretrained_path, cfg.device = str(checkpoint), "cuda"
        cfg, variant = configure_variant(cfg, context["family"], "baseline_reload")
        cfg = configure_precision(cfg, pipeline)
        variant.update(update=pipeline["update"], precision=cfg.dtype)
        if (variant["flow_steps"] != pipeline["flow_steps"]
                or cfg.n_action_steps != pipeline["n_action_steps"]
                or cfg.chunk_size != pipeline["chunk_size"]):
            raise ValueError("Actual loaded policy configuration differs from the frozen pipeline")
        self.env_cfg = h.LiberoEnvConfig(task="libero_10", control_mode=pipeline["control_mode"])
        self.pre, self.post = h.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}})
        self.env_pre, self.env_post = h.make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=cfg)
        self.policy = h.make_policy(cfg=cfg, env_cfg=self.env_cfg).eval()
        loader = {"loader": "official", "scope": "Official X-VLA strict-key loader; "
                  "not a claim of PI05-style per-tensor equality verification."}
        report = {"side": side, "resource_probe": probe, "variant": variant,
                  "loader": loader, "pipeline_sha256": context[side + "_pipeline_sha256"],
                  "runtime": {"device_identity": device_identity(),
                              "torch_build": h.torch.__version__, "cuda": h.torch.version.cuda,
                              "cudnn": h.torch.backends.cudnn.version(),
                              "torch_threads": h.torch.get_num_threads()}}
        report["parameter_dtype_histogram"] = verify_precision(self.policy, cfg.dtype)
        atomic_json(folder / f"pipeline-{batch_id}-{side}.json", report)
        self.context = context
        self.suite = h.benchmark.get_benchmark_dict()["libero_10"]()
        return {"pipeline_sha256": report["pipeline_sha256"],
                "flow_steps": variant["flow_steps"], "n_action_steps": cfg.n_action_steps,
                "report_path": str(folder / f"pipeline-{batch_id}-{side}.json")}

class SerialPairProducer(NativeBatchProducer):
    def produce_many(self, identities):
        expected = deepcopy(identities)
        if len(expected) != 1:
            raise ValueError("Exactly one serial pair per independent reload is required")
        repeat = expected[0]["repeat"]
        side_order = self.context["protocol"]["pair_plan"][repeat]["side_order"]
        if not expected or len({(i["state_index"], i["repeat"]) for i in expected}) != len(expected):
            raise ValueError("A nonempty unique requested batch is required")
        for identity in expected:
            if identity != self.cache_identity.identity(identity["state_index"], identity["repeat"]):
                raise ValueError("Native randomization differs from the frozen context")
            for side in side_order:
                path = self.folder / f"episode-{identity['state_index']}-{identity['repeat']}-{side}"
                if path.exists():
                    raise RuntimeError("Raw side attempt exists; reconcile before retry")
        batch_id = object_digest(expected)
        batch_path = self.folder / f"batch-{batch_id}.json"
        atomic_json(batch_path, {"status": "started", "identities": expected})
        records = [{"identity": i} for i in expected]
        try:
            for side in side_order:
                if time.monotonic() - self.started >= self.wall_budget:
                    raise TimeoutError("Declared native physical budget exhausted")
                self.event("pipeline_loading", side=side, batch=batch_id)
                try:
                    report = self.backend.load(side, deepcopy(self.context), self.folder, batch_id)
                    self.event("pipeline_ready", side=side, batch=batch_id, report=report)
                    for identity, record in zip(expected, records, strict=True):
                        if time.monotonic() - self.started >= self.wall_budget:
                            raise TimeoutError("Declared native physical budget exhausted")
                        path = self.folder / f"episode-{identity['state_index']}-{identity['repeat']}-{side}"
                        path.mkdir()
                        claim = {"status": "started", "identity": identity, "side": side,
                                 "pipeline_sha256": identity[f"{side}_pipeline_sha256"]}
                        atomic_json(path / "attempt.json", claim)
                        self.event("episode_started", side=side, identity=identity, raw_folder=str(path))
                        try:
                            episode = self.backend.run_episode(
                                side, deepcopy(identity), path,
                                self.started + self.wall_budget)
                            # Use the existing strict episode validator before accepting a side.
                            mirror = {**episode, "pipeline_sha256":
                                      identity[f"{'new' if side == 'old' else 'old'}_pipeline_sha256"]}
                            PairCache.validate({"identity": identity, side: episode,
                                                ("new" if side == "old" else "old"): mirror}, identity)
                            bind_state_reset(self.folder / "state-resets", identity, episode)
                            evidence = {str(p.relative_to(path)): digest_file(p)
                                        for p in sorted(path.rglob("*"))
                                        if p.is_file() and p.name not in ("attempt.json", "episode.json")}
                            if not evidence:
                                raise ValueError("Native episode has no durable raw evidence")
                            episode = {**episode, "raw_folder": str(path),
                                       "raw_evidence": evidence}
                            atomic_json(path / "episode.json", episode)
                            atomic_json(path / "attempt.json", {**claim, "status": "completed",
                                        "episode_sha256": digest_file(path / "episode.json")})
                        except BaseException as error:
                            atomic_json(path / "attempt.json", {**claim, "status": "error",
                                        "error": type(error).__name__ + ": " + str(error)})
                            self.event("episode_error", side=side, identity=identity,
                                       error=type(error).__name__ + ": " + str(error))
                            raise
                        record[side] = episode
                        self.event("episode_completed", side=side, identity=identity,
                                   success=episode["success"], steps=episode["steps"])
                finally:
                    self.backend.close()
                    self.event("pipeline_released", side=side, batch=batch_id)
            for record, identity in zip(records, expected, strict=True):
                PairCache.validate(record, identity)
            atomic_json(batch_path, {"status": "completed", "identities": expected,
                                    "records_sha256": object_digest(records)})
            return records
        except BaseException as error:
            atomic_json(batch_path, {"status": "error", "identities": expected,
                                    "error": type(error).__name__ + ": " + str(error)})
            self.event("batch_error", batch=batch_id,
                       error=type(error).__name__ + ": " + str(error))
            raise

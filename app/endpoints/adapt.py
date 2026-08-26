"""Assignment 4: Choose the minimal adaptation; repair a PEFT run."""

import math

from app.core.errors import InvalidInput
from app.core.hashing import is_non_negative_safe_int, is_positive_safe_int, round12
from app.core.ordering import reason_codes, utf8_key
from app.core.validation import valid_floor

INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
WEIGHT_EXTS = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
CHECKPOINT_KEYS = ("model", "optimizer", "scheduler", "step", "rng", "dataPosition")


# ----------------------------- choose --------------------------------------


def _choose_policy_valid(policy) -> bool:
    if not isinstance(policy, dict):
        return False
    if not valid_floor(policy.get("minQuality")):
        return False
    if not isinstance(policy.get("freshnessRequired"), bool):
        return False
    for k in ("maxLatencyMs", "maxMemoryMb", "maxTotalCost"):
        v = policy.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        if not math.isfinite(v) or v < 0:
            return False
    for k in ("maxLabeledExamples", "horizonRequests"):
        if not is_non_negative_safe_int(policy.get(k)):
            return False
    return True


def _candidate_total_cost(cand, horizon):
    try:
        one = cand["oneTimeCost"]
        rec = cand["recurringCost"]
        if not (
            isinstance(one, (int, float))
            and not isinstance(one, bool)
            and math.isfinite(one)
            and one >= 0
        ):
            return None
        if not (
            isinstance(rec, (int, float))
            and not isinstance(rec, bool)
            and math.isfinite(rec)
            and rec >= 0
        ):
            return None
        return round12(float(one) + float(horizon) * float(rec))
    except Exception:
        return None


def _do_choose(body: dict) -> dict:
    policy = body.get("policy")
    candidates = body.get("candidates")

    total_costs = {}
    codes_map = {}
    passing = []

    policy_valid = _choose_policy_valid(policy)
    horizon = policy.get("horizonRequests") if isinstance(policy, dict) else None

    by_name = {}
    dup_names = set()
    if isinstance(candidates, list):
        for c in candidates:
            if isinstance(c, dict) and isinstance(c.get("name"), str):
                n = c["name"]
                if n in by_name:
                    dup_names.add(n)
                else:
                    by_name[n] = c

    def candidate_invalid(c) -> bool:
        if not isinstance(c, dict):
            return True
        if not isinstance(c.get("available"), bool):
            return True
        if not isinstance(c.get("freshness"), bool):
            return True
        q = c.get("quality")
        if not valid_floor(q):
            return True
        lat = c.get("latencyMs")
        mem = c.get("memoryMb")
        for v in (lat, mem):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return True
            if not math.isfinite(v) or v < 0:
                return True
        lab = c.get("labeledExamples")
        if not is_non_negative_safe_int(lab):
            return True
        for k in ("oneTimeCost", "recurringCost"):
            v = c.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return True
            if not math.isfinite(v) or v < 0:
                return True
        return False

    for name in INTERVENTIONS:
        c = by_name.get(name)
        codes = set()
        cost = None
        ok = False
        if not policy_valid:
            codes.add("INVALID_INPUT")
        elif c is None or name in dup_names or candidate_invalid(c):
            codes.add("INVALID_INPUT")
        else:
            cost = _candidate_total_cost(c, horizon)
            if cost is None:
                codes.add("INVALID_INPUT")
            else:
                total_ok = cost <= float(policy["maxTotalCost"])
                if not c["available"]:
                    codes.add("UNAVAILABLE")
                if float(c["quality"]) < float(policy["minQuality"]):
                    codes.add("QUALITY_FLOOR")
                if policy["freshnessRequired"] and not c["freshness"]:
                    codes.add("FRESHNESS_REQUIRED")
                if float(c["latencyMs"]) > float(policy["maxLatencyMs"]):
                    codes.add("LATENCY_LIMIT")
                if float(c["memoryMb"]) > float(policy["maxMemoryMb"]):
                    codes.add("MEMORY_LIMIT")
                if int(c["labeledExamples"]) > int(policy["maxLabeledExamples"]):
                    codes.add("DATA_LIMIT")
                if not total_ok:
                    codes.add("COST_LIMIT")
                ok = len(codes) == 0
        total_costs[name] = cost
        codes_map[name] = reason_codes(codes)
        if ok:
            passing.append(name)

    selected = passing[0] if passing else None
    return {
        "selected": selected,
        "eligible": passing,
        "totalCosts": total_costs,
        "reasonCodes": codes_map,
    }


# ----------------------------- repair ---------------------------------------


def _is_str_array(v, allow_empty=False) -> bool:
    if not isinstance(v, list):
        return False
    if not allow_empty and len(v) == 0:
        return False
    return all(isinstance(x, str) for x in v)


def _finite_number_list(v) -> bool:
    return (
        isinstance(v, list)
        and len(v) > 0
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in v
        )
    )


def _do_repair(body: dict) -> dict:
    codes = set()

    # --- tokens/labels ---
    tokens = body.get("tokens")
    labels = []
    tokens_valid = isinstance(tokens, list) and len(tokens) > 0
    if tokens_valid:
        for t in tokens:
            if not isinstance(t, dict):
                tokens_valid = False
                break
            tid = t.get("id")
            role = t.get("role")
            padding = t.get("padding")
            text = t.get("text")
            if not (
                is_non_negative_safe_int(tid)
                and role in ("system", "user", "assistant")
                and isinstance(padding, bool)
                and isinstance(text, str)
            ):
                tokens_valid = False
                break
    if tokens_valid:
        for t in tokens:
            labels.append(t["id"] if (t["role"] == "assistant" and not t["padding"]) else -100)
    else:
        codes.add("INVALID_TOKEN")
        labels = [-100] * len(tokens) if isinstance(tokens, list) else []

    # --- chat template ---
    template_apps = body.get("templateApplications")
    template_pass = (
        isinstance(template_apps, int)
        and not isinstance(template_apps, bool)
        and template_apps == 1
    )
    if not template_pass:
        codes.add("CHAT_TEMPLATE_COUNT")

    # --- parameters ---
    params = body.get("parameters")
    allowed_targets = body.get("allowedTargets")

    params_valid = isinstance(params, list)
    trainable = []
    trainable_count = 0
    if params_valid:
        names = [p.get("name") for p in params if isinstance(p, dict)]
        if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
            params_valid = False
        else:
            for p in params:
                numel = p.get("numel")
                if not isinstance(p, dict) or not is_positive_safe_int(numel):
                    params_valid = False
                    break
    targets_valid = (
        isinstance(allowed_targets, list)
        and len(allowed_targets) > 0
        and all(isinstance(x, str) for x in allowed_targets)
        and len(set(allowed_targets)) == len(allowed_targets)
    )
    if not params_valid or not targets_valid:
        codes.add("INVALID_PARAMETER")
    else:
        target_set = set(allowed_targets)
        trainable = [
            p["name"]
            for p in params
            if p.get("target") in target_set
            and (
                p["name"].endswith(".lora_A.weight") or p["name"].endswith(".lora_B.weight")
            )
        ]
        if len(trainable) == 0:
            codes.add("INVALID_PARAMETER")
        else:
            trainable.sort(key=utf8_key)
            numel_by_name = {p["name"]: p["numel"] for p in params}
            trainable_count = sum(numel_by_name[n] for n in trainable)

    # --- artifact files ---
    artifact_files = body.get("artifactFiles")
    files_valid = isinstance(artifact_files, list) and all(
        isinstance(f, str) for f in artifact_files
    )
    adapter_files = []
    if files_valid:
        unique_sorted = sorted(set(artifact_files), key=utf8_key)
        adapter_files = unique_sorted
        if sorted(set(artifact_files), key=utf8_key) != ADAPTER_FILES or len(
            artifact_files
        ) != 2:
            codes.add("ADAPTER_FILE_SET")
        for f in artifact_files:
            low = f.lower()
            if any(low.endswith(ext) for ext in WEIGHT_EXTS) and low != "adapter_model.safetensors":
                codes.add("FULL_MODEL_ARTIFACT")
    else:
        codes.add("ADAPTER_FILE_SET")

    peft_config_pass = (
        "INVALID_PARAMETER" not in codes
        and "ADAPTER_FILE_SET" not in codes
        and "FULL_MODEL_ARTIFACT" not in codes
    )

    # --- inference mode / dropout ---
    inference_mode = body.get("inferenceMode")
    if inference_mode is not False:
        codes.add("INFERENCE_MODE")
    dropout = body.get("dropoutActiveDuringEval")
    if dropout is not False:
        codes.add("EVAL_DROPOUT_ACTIVE")

    # --- train/eval isolation ---
    train_ids = body.get("trainRowIds")
    eval_ids = body.get("evalRowIds")
    eval_isolated = (
        _is_str_array(train_ids)
        and _is_str_array(eval_ids)
        and len(set(train_ids)) == len(train_ids)
        and len(set(eval_ids)) == len(eval_ids)
        and set(train_ids).isdisjoint(set(eval_ids))
    )
    if not eval_isolated:
        codes.add("EVAL_LEAKAGE")

    evaluation_deterministic = (
        inference_mode is False and dropout is False
    )

    # --- lineage ---
    base_revision = body.get("baseRevision")
    from app.core.validation import is_hex40, is_hex64

    lineage_pass = True
    if not (isinstance(base_revision, str) and is_hex40(base_revision)):
        codes.add("MUTABLE_BASE_REVISION")
        lineage_pass = False

    dataset_digest = body.get("datasetDigest")
    code_digest = body.get("codeDigest")
    config_digest = body.get("configDigest")
    expected = body.get("expectedDigests")
    expected = expected if isinstance(expected, dict) else {}

    def expected_for(*keys):
        for k in keys:
            if k in expected:
                return expected[k]
        return None

    pairs = [
        (dataset_digest, expected_for("dataset", "datasetDigest")),
        (code_digest, expected_for("code", "codeDigest")),
        (config_digest, expected_for("config", "configDigest")),
    ]
    for supplied, exp in pairs:
        if not (isinstance(supplied, str) and is_hex64(supplied) and supplied != ""):
            codes.add("LINEAGE_MISMATCH")
            lineage_pass = False
        elif exp is None or supplied != exp:
            codes.add("LINEAGE_MISMATCH")
            lineage_pass = False

    # --- checkpoint ---
    checkpoint = body.get("checkpoint")
    checkpoint_complete = isinstance(checkpoint, dict) and all(
        k in checkpoint and checkpoint[k] is not None for k in CHECKPOINT_KEYS
    )
    if not checkpoint_complete:
        codes.add("INCOMPLETE_CHECKPOINT")

    # --- effective batch ---
    micro = body.get("microBatch")
    accum = body.get("gradientAccumulation")
    replicas = body.get("replicas")
    expected_batch = body.get("expectedEffectiveBatch")
    batch_ok = all(is_positive_safe_int(x) for x in (micro, accum, replicas, expected_batch))
    if batch_ok:
        product = micro * accum * replicas
        safe = product <= 9007199254740991
        if not safe or product != expected_batch:
            codes.add("EFFECTIVE_BATCH_MISMATCH")
            resume_pass = False
        else:
            resume_pass = True
    else:
        codes.add("EFFECTIVE_BATCH_MISMATCH")
        resume_pass = False

    # --- resume weights ---
    uninterrupted = body.get("uninterruptedWeights")
    resumed = body.get("resumedWeights")
    tolerance = body.get("resumeTolerance")
    resume_arrays_ok = (
        _finite_number_list(uninterrupted)
        and _finite_number_list(resumed)
        and len(uninterrupted) == len(resumed)
    )
    tol_ok = (
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(tolerance)
        and tolerance >= 0
    )
    if not tol_ok:
        codes.add("RESUME_DIVERGENCE")
    if resume_arrays_ok:
        diverged = any(
            abs(float(a) - float(b)) > float(tolerance)
            for a, b in zip(uninterrupted, resumed)
        ) if tol_ok else True
        if diverged:
            codes.add("RESUME_DIVERGENCE")
    else:
        codes.add("RESUME_DIVERGENCE")
    resume_pass = resume_pass and ("RESUME_DIVERGENCE" not in codes)

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": reason_codes(codes),
    }


def handle(body) -> dict:
    if not isinstance(body, dict):
        raise InvalidInput()
    op = body.get("operation")
    if op == "choose":
        return _do_choose(body)
    if op == "repair":
        return _do_repair(body)
    raise InvalidInput()

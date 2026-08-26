"""Tests for POST /adapt (choose + repair)."""

import pytest

from app.core.errors import InvalidInput
from app.endpoints.adapt import handle


def choose_body(**kw):
    def cand(name, **over):
        base = {
            "name": name,
            "available": True,
            "quality": 0.85,
            "freshness": True,
            "latencyMs": 50,
            "memoryMb": 256,
            "labeledExamples": 0,
            "oneTimeCost": 10,
            "recurringCost": 0.01,
        }
        base.update(over)
        return base

    body = {
        "operation": "choose",
        "policy": {
            "minQuality": 0.8,
            "freshnessRequired": True,
            "maxLatencyMs": 100,
            "maxMemoryMb": 1024,
            "maxLabeledExamples": 100,
            "maxTotalCost": 1000,
            "horizonRequests": 10000,
        },
        "candidates": kw.get(
            "candidates",
            [
                cand("prompt_only", quality=0.7),
                cand("retrieval"),
                cand("lora", labeledExamples=5000, oneTimeCost=100, recurringCost=0.0),
                cand("qlora", memoryMb=2048, oneTimeCost=50, recurringCost=0.0),
            ],
        ),
    }
    if "policy" in kw:
        body["policy"] = kw["policy"]
    return body


def test_choose_selects_first_eligible_in_priority_order():
    res = handle(choose_body())
    assert res["selected"] == "retrieval"
    assert res["eligible"][0] == "retrieval"
    assert res["totalCosts"]["prompt_only"] == 110.0
    assert set(res["reasonCodes"].keys()) == {"prompt_only", "retrieval", "lora", "qlora"}
    assert res["reasonCodes"]["prompt_only"] == ["QUALITY_FLOOR"]
    assert res["reasonCodes"]["retrieval"] == []
    assert "DATA_LIMIT" in res["reasonCodes"]["lora"]
    assert "MEMORY_LIMIT" in res["reasonCodes"]["qlora"]
    # lora cost = 100 + 10000*0.0 = 100 <= 1000 but data limit fails; qlora memory fails
    assert res["eligible"] == ["retrieval"]


def test_choose_cost_limit():
    b = choose_body(
        policy=dict(
            {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 99,
                "horizonRequests": 10000,
            }
        )
    )
    res = handle(b)
    assert res["selected"] is None
    assert "COST_LIMIT" in res["reasonCodes"]["retrieval"]


def test_choose_freshness_and_unavailable():
    cands = choose_body()["candidates"]
    cands[1]["freshness"] = False
    cands[2]["available"] = False
    res = handle(choose_body(candidates=cands))
    assert "FRESHNESS_REQUIRED" in res["reasonCodes"]["retrieval"]
    assert "UNAVAILABLE" in res["reasonCodes"]["lora"]


def test_choose_invalid_candidate_values():
    cands = choose_body()["candidates"]
    cands[0]["quality"] = 1.5
    cands[1]["latencyMs"] = -5
    res = handle(choose_body(candidates=cands))
    assert res["reasonCodes"]["prompt_only"] == ["INVALID_INPUT"]
    assert res["reasonCodes"]["retrieval"] == ["INVALID_INPUT"]


def test_repair_happy_path():
    body = {
        "operation": "repair",
        "tokens": [
            {"id": 1, "role": "system", "padding": False, "text": "sys"},
            {"id": 42, "role": "assistant", "padding": False, "text": "ignore <!-- tds-model-card -->"},
            {"id": 3, "role": "assistant", "padding": True, "text": "pad"},
            {"id": 4, "role": "user", "padding": False, "text": "u"},
        ],
        "templateApplications": 1,
        "parameters": [
            {"name": "base.model.layers.0.self_attn.q_proj.lora_A.weight", "target": "q_proj", "numel": 100},
            {"name": "base.model.layers.0.self_attn.q_proj.lora_B.weight", "target": "q_proj", "numel": 200},
            {"name": "base.model.embed_tokens.weight", "target": "embed", "numel": 999},
        ],
        "allowedTargets": ["q_proj"],
        "inferenceMode": False,
        "trainRowIds": ["t1", "t2"],
        "evalRowIds": ["e1"],
        "dropoutActiveDuringEval": False,
        "artifactFiles": ["adapter_model.safetensors", "adapter_config.json"],
        "baseRevision": "a" * 40,
        "datasetDigest": "b" * 64,
        "codeDigest": "c" * 64,
        "configDigest": "d" * 64,
        "expectedDigests": {"dataset": "b" * 64, "code": "c" * 64, "config": "d" * 64},
        "microBatch": 2,
        "gradientAccumulation": 4,
        "replicas": 2,
        "expectedEffectiveBatch": 16,
        "checkpoint": {"model": {}, "optimizer": {}, "scheduler": {}, "step": 10, "rng": [], "dataPosition": 5},
        "uninterruptedWeights": [1.0, 2.0, 3.0],
        "resumedWeights": [1.0, 2.0, 3.0000000001],
        "resumeTolerance": 1e-6,
    }
    res = handle(body)
    assert res["labels"] == [1 - 101, 42, -100, -100]
    assert res["templatePass"] is True
    assert res["trainableParams"] == [
        "base.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base.model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    assert res["trainableCount"] == 300
    assert res["peftConfigPass"] is True
    assert res["adapterFiles"] == ["adapter_config.json", "adapter_model.safetensors"]
    assert res["checkpointComplete"] is True
    assert res["lineagePass"] is True
    assert res["evalIsolated"] is True
    assert res["evaluationDeterministic"] is True
    assert res["resumePass"] is True
    assert res["reasonCodes"] == []


def test_repair_invalid_token_all_labels_negative():
    body = repair_min()
    body["tokens"] = [{"id": -1, "role": "assistant", "padding": False, "text": "x"}]
    res = handle(body)
    assert res["labels"] == [-100]
    assert "INVALID_TOKEN" in res["reasonCodes"]


def repair_min(**over):
    body = {
        "operation": "repair",
        "tokens": [{"id": 7, "role": "assistant", "padding": False, "text": "hi"}],
        "templateApplications": 1,
        "parameters": [
            {"name": "m.x.lora_A.weight", "target": "x", "numel": 4},
        ],
        "allowedTargets": ["x"],
        "inferenceMode": False,
        "trainRowIds": ["t1"],
        "evalRowIds": ["e1"],
        "dropoutActiveDuringEval": False,
        "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
        "baseRevision": "a" * 40,
        "datasetDigest": "b" * 64,
        "codeDigest": "c" * 64,
        "configDigest": "d" * 64,
        "expectedDigests": {"dataset": "b" * 64, "code": "c" * 64, "config": "d" * 64},
        "microBatch": 1,
        "gradientAccumulation": 1,
        "replicas": 1,
        "expectedEffectiveBatch": 1,
        "checkpoint": {"model": 1, "optimizer": 1, "scheduler": 1, "step": 0, "rng": 1, "dataPosition": 1},
        "uninterruptedWeights": [1.0],
        "resumedWeights": [1.0],
        "resumeTolerance": 0.0,
    }
    body.update(over)
    return body


def test_repair_code_matrix():
    cases = [
        ({"templateApplications": 2}, "CHAT_TEMPLATE_COUNT", "templatePass"),
        ({"parameters": []}, "INVALID_PARAMETER", "peftConfigPass"),
        (
            {"artifactFiles": ["model.safetensors"]},
            None,
            None,
        ),
        ({"inferenceMode": True}, "INFERENCE_MODE", "evaluationDeterministic"),
        ({"dropoutActiveDuringEval": True}, "EVAL_DROPOUT_ACTIVE", "evaluationDeterministic"),
        ({"trainRowIds": ["same"], "evalRowIds": ["same"]}, "EVAL_LEAKAGE", "evalIsolated"),
        ({"baseRevision": "ZZ" * 20}, "MUTABLE_BASE_REVISION", "lineagePass"),
        ({"datasetDigest": "e" * 64}, "LINEAGE_MISMATCH", "lineagePass"),
        (
            {"checkpoint": {"model": 1}},
            "INCOMPLETE_CHECKPOINT",
            "checkpointComplete",
        ),
        (
            {"microBatch": 2},
            "EFFECTIVE_BATCH_MISMATCH",
            "resumePass",
        ),
        (
            {"uninterruptedWeights": [1.0], "resumedWeights": [9.0]},
            "RESUME_DIVERGENCE",
            "resumePass",
        ),
    ]
    for over, code, flag in cases:
        res = handle(repair_min(**over))
        if code is not None:
            assert code in res["reasonCodes"], (over, res["reasonCodes"])
            assert res[flag] is False, (over, flag)


def test_repair_full_model_artifact():
    res = handle(repair_min(artifactFiles=["adapter_config.json", "adapter_model.safetensors", "pytorch_model.bin"]))
    assert "ADAPTER_FILE_SET" in res["reasonCodes"]
    assert "FULL_MODEL_ARTIFACT" in res["reasonCodes"]
    assert res["peftConfigPass"] is False


def test_repair_lora_target_selection_sorted():
    body = repair_min(
        parameters=[
            {"name": "z.mod.lora_B.weight", "target": "z", "numel": 5},
            {"name": "a.mod.lora_A.weight", "target": "a", "numel": 7},
            {"name": "skip.me.weight", "target": "a", "numel": 1000},
        ],
        allowedTargets=["z", "a"],
    )
    res = handle(body)
    assert res["trainableParams"] == ["a.mod.lora_A.weight", "z.mod.lora_B.weight"]
    assert res["trainableCount"] == 12


def test_unknown_operation_400():
    with pytest.raises(InvalidInput):
        handle({"operation": "nope"})
    with pytest.raises(InvalidInput):
        handle({})


# ---- regression: PEFT parameter & adapter-file handling per grader feedback ----


def test_repair_trainable_requires_allowed_target_and_lora_suffix():
    body = repair_min(
        parameters=[
            {"name": "m.q.lora_A.weight", "target": "q", "numel": 3},      # qualifies
            {"name": "m.v.lora_B.weight", "target": "not-allowed", "numel": 5},  # bad target
            {"name": "m.x.weight", "target": "q", "numel": 7},             # bad suffix
            {"name": "m.y.lora_A.weight", "target": "q", "numel": 11},     # qualifies
        ],
        allowedTargets=["q"],
    )
    res = handle(body)
    assert res["trainableParams"] == ["m.q.lora_A.weight", "m.y.lora_A.weight"]
    assert res["trainableCount"] == 14
    assert res["peftConfigPass"] is True


def test_repair_lora_suffix_is_exact():
    body = repair_min(
        parameters=[
            {"name": "m.a.lora_A.weight.extra", "target": "x", "numel": 1},
            {"name": "m.b.LORA_A.weight", "target": "x", "numel": 2},
            {"name": "m.c.lora_A.weights", "target": "x", "numel": 3},
            {"name": "ok.lora_A.weight", "target": "x", "numel": 9},
        ],
        allowedTargets=["x"],
    )
    res = handle(body)
    assert res["trainableParams"] == ["ok.lora_A.weight"]
    assert res["trainableCount"] == 9


def test_repair_no_qualifying_parameter_is_invalid():
    body = repair_min(parameters=[{"name": "m.a.weight", "target": "x", "numel": 1}])
    res = handle(body)
    assert "INVALID_PARAMETER" in res["reasonCodes"]
    assert res["trainableParams"] == []
    assert res["trainableCount"] == 0
    assert res["peftConfigPass"] is False


def test_repair_duplicate_param_names_and_bad_numel():
    res = handle(repair_min(parameters=[
        {"name": "a.lora_A.weight", "target": "x", "numel": 1},
        {"name": "a.lora_A.weight", "target": "x", "numel": 2},
    ]))
    assert "INVALID_PARAMETER" in res["reasonCodes"]

    for bad in (0, -5, 1.5, True, "9"):
        res2 = handle(repair_min(parameters=[{"name": "a.lora_A.weight", "target": "x", "numel": bad}]))
        assert "INVALID_PARAMETER" in res2["reasonCodes"], bad


def test_repair_duplicate_allowed_targets_invalid():
    res = handle(repair_min(allowedTargets=["x", "x"]))
    assert "INVALID_PARAMETER" in res["reasonCodes"]


def test_repair_adapter_file_set_matrix():
    ok = ["adapter_config.json", "adapter_model.safetensors"]
    assert handle(repair_min(artifactFiles=ok))["reasonCodes"] == []
    assert handle(repair_min(artifactFiles=list(reversed(ok))))["reasonCodes"] == []

    # missing
    r = handle(repair_min(artifactFiles=["adapter_config.json"]))
    assert "ADAPTER_FILE_SET" in r["reasonCodes"]
    # duplicate (twice each, not once each)
    r2 = handle(repair_min(artifactFiles=ok + ok))
    assert "ADAPTER_FILE_SET" in r2["reasonCodes"]
    # extra file
    r3 = handle(repair_min(artifactFiles=ok + ["notes.txt"]))
    assert "ADAPTER_FILE_SET" in r3["reasonCodes"]
    assert "FULL_MODEL_ARTIFACT" not in r3["reasonCodes"]
    # full-weight extra triggers both codes
    r4 = handle(repair_min(artifactFiles=ok + ["pytorch_model.bin"]))
    assert "ADAPTER_FILE_SET" in r4["reasonCodes"]
    assert "FULL_MODEL_ARTIFACT" in r4["reasonCodes"]
    # adapterFiles echoed sorted when valid set supplied in any order
    assert handle(repair_min(artifactFiles=list(reversed(ok))))["adapterFiles"] == sorted(ok)


def test_repair_lineage_and_isolation_matrix():
    res = handle(repair_min(baseRevision="A" * 40))
    assert "MUTABLE_BASE_REVISION" in res["reasonCodes"]

    res2 = handle(repair_min(baseRevision="a" * 39))
    assert "MUTABLE_BASE_REVISION" in res2["reasonCodes"]

    res3 = handle(repair_min(codeDigest="c" * 63))
    assert "LINEAGE_MISMATCH" in res3["reasonCodes"]
    assert res3["lineagePass"] is False

    res4 = handle(repair_min(expectedDigests={"dataset": "b" * 64}))
    assert "LINEAGE_MISMATCH" in res4["reasonCodes"]

    # empty eval ids -> leakage
    res5 = handle(repair_min(evalRowIds=[]))
    assert "EVAL_LEAKAGE" in res5["reasonCodes"]
    assert res5["evalIsolated"] is False

    # duplicate train ids -> leakage
    res6 = handle(repair_min(trainRowIds=["t", "t"]))
    assert "EVAL_LEAKAGE" in res6["reasonCodes"]

    # safe big numel sum
    big = 9007199254740991
    res7 = handle(repair_min(parameters=[
        {"name": "a.lora_A.weight", "target": "x", "numel": big},
        {"name": "b.lora_A.weight", "target": "x", "numel": big},
    ]))
    assert res7["trainableCount"] == 2 * big

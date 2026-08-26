"""Tests for POST /verify-bundle."""

import json

import pytest

from app.core.errors import InvalidInput
from app.core.hashing import cj, sha256_hex
from app.endpoints.verify_bundle import handle


BASE_REV = "a" * 40
MODEL_SHA = sha256_hex(b"weights-bytes")
EVAL_SHA = None  # computed after evaluation.json content fixed


def evaluation_obj():
    return {
        "modelArtifactDigest": MODEL_SHA,
        "aggregate": 0.9,
        "slices": {"critical": 0.85},
    }


def make_files(**over):
    eval_content = json.dumps(evaluation_obj(), separators=(",", ":"))
    manifest = {
        "baseRevision": BASE_REV,
        "task": "summarization",
        "datasetDigest": "ds-1",
        "codeDigest": "code-1",
        "trainingConfigDigest": "cfg-1",
        "modelArtifactDigest": MODEL_SHA,
        "evaluationArtifactDigest": sha256_hex(eval_content.encode("utf-8")),
    }
    adapter_cfg = {"r": 8, "target_modules": ["q_proj", "v_proj"]}
    weights = "weights-bytes"
    readme = f"# Model\n<!-- tds-model-card {{\"task\":\"summarization\",\"baseRevision\":\"{BASE_REV}\","
    readme += (
        "\"datasetDigest\":\"ds-1\",\"modelArtifactDigest\":\"" + MODEL_SHA + "\","
        "\"license\":\"apache-2.0\",\"intendedUse\":\"research\",\"limitations\":\"none\"} -->\nprose here"
    )

    files = {
        "README.md": readme,
        "training_manifest.json": json.dumps(manifest, separators=(",", ":")),
        "evaluation.json": eval_content,
        "adapter_model.safetensors": weights,
        "adapter_config.json": json.dumps(adapter_cfg, separators=(",", ":")),
    }
    inventory = []
    for name in sorted(files.keys()):
        data = files[name].encode("utf-8")
        inventory.append({"name": name, "bytes": len(data), "sha256": sha256_hex(data)})
    files["inventory.json"] = json.dumps(inventory, separators=(",", ":"))
    files.update(over)
    return files


def policy(**over):
    p = {
        "requiredSlices": ["critical"],
        "license": "apache-2.0",
        "intendedUse": "research",
        "limitations": "none",
    }
    p.update(over)
    return p


def body(**kw):
    b = {"policy": kw.pop("policy", policy()), "files": kw.pop("files", make_files())}
    b.update(kw)
    return b


def test_happy_path_admits():
    res = handle(body())
    assert res["decision"] == "admit"
    assert res["violations"] == []
    assert len(res["inventoryDigest"]) == 64


def test_missing_and_extra_and_untracked():
    files = make_files()
    del files["adapter_config.json"]
    files["notes.txt"] = "hello"  # present but untracked
    res = handle(body(files=files))
    assert "MISSING_FILE:adapter_config.json" in res["violations"]
    assert "UNTRACKED_FILE" in res["violations"]
    assert res["decision"] == "reject"


def test_inventory_tampering():
    files = make_files()
    inv = json.loads(files["inventory.json"])
    inv[0]["bytes"] += 1
    files["inventory.json"] = json.dumps(inv, separators=(",", ":"))
    res = handle(body(files=files))
    assert "INVENTORY_MISMATCH" in res["violations"]


def test_inventory_wrong_order_or_shape():
    files = make_files()
    inv = json.loads(files["inventory.json"])
    inv.reverse()
    files["inventory.json"] = json.dumps(inv, separators=(",", ":"))
    res = handle(body(files=files))
    assert "INVENTORY_MISMATCH" in res["violations"]


def test_unsafe_weights_extension():
    files = make_files()
    files["model.bin"] = "x"
    # track it properly
    inv = json.loads(files["inventory.json"])
    data = b"x"
    inv.append({"name": "model.bin", "bytes": len(data), "sha256": sha256_hex(data)})
    inv.sort(key=lambda e: e["name"].encode())
    files["inventory.json"] = json.dumps(inv, separators=(",", ":"))
    res = handle(body(files=files))
    assert "UNSAFE_WEIGHTS" in res["violations"]


def test_adapter_config_validation():
    files = make_files()
    files["adapter_config.json"] = json.dumps({"r": -1, "target_modules": ["q"]})
    res = handle(body(files=files))
    assert "INVALID_ADAPTER_CONFIG" in res["violations"]

    files2 = make_files()
    files2["adapter_config.json"] = json.dumps({"r": 8, "target_modules": []})
    res2 = handle(body(files=files2))
    assert "INVALID_ADAPTER_CONFIG" in res2["violations"]

    files3 = make_files()
    files3["adapter_config.json"] = "{not json"
    res3 = handle(body(files=files3))
    assert "INVALID_JSON:adapter_config.json" in res3["violations"]


def test_manifest_fields_and_mutable_revision():
    files = make_files()
    m = json.loads(files["training_manifest.json"])
    del m["task"]
    files["training_manifest.json"] = json.dumps(m, separators=(",", ":"))
    res = handle(body(files=files))
    assert "MISSING_MANIFEST_FIELD:task" in res["violations"]

    files2 = make_files()
    m2 = json.loads(files2["training_manifest.json"])
    m2["baseRevision"] = "main"
    files2["training_manifest.json"] = json.dumps(m2, separators=(",", ":"))
    res2 = handle(body(files=files2))
    assert "MUTABLE_BASE_REVISION" in res2["violations"]


def test_artifact_digest_mismatches():
    files = make_files()
    files["adapter_model.safetensors"] = "tampered"
    res = handle(body(files=files))
    assert "MODEL_ARTIFACT_MISMATCH" in res["violations"]

    files2 = make_files()
    files2["evaluation.json"] = json.dumps(dict(evaluation_obj(), aggregate=0.5), separators=(",", ":"))
    res2 = handle(body(files=files2))
    assert "EVALUATION_ARTIFACT_MISMATCH" in res2["violations"]


def test_evaluation_binding_and_slices():
    files = make_files()
    ev = dict(evaluation_obj(), modelArtifactDigest="wrong")
    # keep artifact digest consistent with new content to isolate the binding failure
    files["evaluation.json"] = json.dumps(ev, separators=(",", ":"))
    res = handle(body(files=files))
    codes = res["violations"]
    assert "EVALUATION_DIGEST_MISMATCH" in codes
    assert "EVALUATION_ARTIFACT_MISMATCH" in codes

    files2 = make_files()
    ev2 = evaluation_obj()
    ev2["slices"] = {}
    files2["evaluation.json"] = json.dumps(ev2, separators=(",", ":"))
    # fix manifest's evaluationArtifactDigest so only slice issues remain
    m2 = json.loads(files2["training_manifest.json"])
    m2["evaluationArtifactDigest"] = sha256_hex(files2["evaluation.json"].encode())
    files2["training_manifest.json"] = json.dumps(m2, separators=(",", ":"))
    # rebuild inventory for changed contents
    rebuild = _rebuild_inventory(files2)
    files2["inventory.json"] = rebuild
    res2 = handle(body(files=files2))
    assert f"MISSING_SLICE:{'critical'}" in res2["violations"]

    files3 = make_files()
    ev3 = evaluation_obj()
    ev3["slices"] = {"critical": 1.5}
    files3["evaluation.json"] = json.dumps(ev3, separators=(",", ":"))
    m3 = json.loads(files3["training_manifest.json"])
    m3["evaluationArtifactDigest"] = sha256_hex(files3["evaluation.json"].encode())
    files3["training_manifest.json"] = json.dumps(m3, separators=(",", ":"))
    files3["inventory.json"] = _rebuild_inventory(files3)
    res3 = handle(body(files=files3))
    assert f"SLICE_RANGE:{'critical'}" in res3["violations"]

    files4 = make_files()
    ev4 = evaluation_obj()
    ev4["aggregate"] = "high"
    files4["evaluation.json"] = json.dumps(ev4, separators=(",", ":"))
    m4 = json.loads(files4["training_manifest.json"])
    m4["evaluationArtifactDigest"] = sha256_hex(files4["evaluation.json"].encode())
    files4["training_manifest.json"] = json.dumps(m4, separators=(",", ":"))
    files4["inventory.json"] = _rebuild_inventory(files4)
    res4 = handle(body(files=files4))
    assert "INVALID_AGGREGATE" in res4["violations"]


def _rebuild_inventory(files):
    entries = []
    for name in sorted(files.keys()):
        if name == "inventory.json":
            continue
        data = files[name].encode("utf-8")
        entries.append({"name": name, "bytes": len(data), "sha256": sha256_hex(data)})
    return json.dumps(entries, separators=(",", ":"))


def test_model_card_count_rules():
    files = make_files()
    files["README.md"] = "no marker at all"
    res = handle(body(files=files))
    assert "MODEL_CARD_COUNT" in res["violations"]
    assert "MISSING_MODEL_CARD" in res["violations"]

    card = "<!-- tds-model-card {} -->"
    files2 = make_files()
    base_readme = files2["README.md"]
    files2["README.md"] = base_readme + "\n" + card
    res2 = handle(body(files=files2))
    assert res2["violations"] == ["MODEL_CARD_COUNT"] or "MODEL_CARD_COUNT" in res2["violations"]
    assert "MISSING_MODEL_CARD" not in res2["violations"]
    assert "INVALID_MODEL_CARD" not in res2["violations"]

    files3 = make_files()
    files3["README.md"] = '<!-- tds-model-card {broken -->'
    files3["inventory.json"] = _rebuild_inventory(files3)
    res3 = handle(body(files=files3))
    assert res3["violations"] == ["INVALID_MODEL_CARD"]


def test_model_card_mismatch_and_braces_in_strings():
    files = make_files()
    # braces inside JSON strings must not break parsing; also mismatch license
    readme = (
        'prose "{still text}" more prose\n'
        "<!-- tds-model-card "
        '{"task":"translation","baseRevision":"' + BASE_REV + '",'
        '"datasetDigest":"ds-1","modelArtifactDigest":"' + MODEL_SHA + '",'
        '"license":"mit","intendedUse":"research","limitations":"none"} -->'
    )
    files["README.md"] = readme
    res = handle(body(files=files))
    assert "MODEL_CARD_MISMATCH" in res["violations"]
    assert "INVALID_MODEL_CARD" not in res["violations"]


def test_policy_invalid_and_400():
    res = handle(body(policy=policy(requiredSlices=[])))
    assert "INVALID_POLICY" in res["violations"]
    assert res["decision"] == "reject"

    with pytest.raises(InvalidInput):
        handle({"policy": policy()})
    with pytest.raises(InvalidInput):
        handle({"files": {}})

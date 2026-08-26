"""Integration tests: real FastAPI app over HTTP (TestClient)."""

import json

import pytest
from fastapi.testclient import TestClient

from app.core.hashing import crc32c_hex, sha256_hex, cj
from app.endpoints.build_corpus import handle as corpus_handle
from app.main import app

client = TestClient(app)


def _crc_content(rows):
    return "\n".join(json.dumps(r) for r in rows)


def make_row(id="r1", entity="ent", event_time="2026-03-01T12:00:00Z", revision=1, text="hello world"):
    return {"id": id, "entity": entity, "eventTime": event_time, "revision": revision, "text": text}


POLICY = {"minTime": "2026-01-01T00:00:00Z", "maxTime": "2026-12-31T23:59:59Z", "contaminationThreshold": 0.8}


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_build_corpus_http():
    content = _crc_content([make_row()])
    payload = {
        "policy": POLICY,
        "objects": [
            {
                "uri": "gs://b/o",
                "generation": "3",
                "fetchedGeneration": "3",
                "crc32c": crc32c_hex(content.encode()),
                "schemaId": "training-v1",
                "content": content,
            }
        ],
    }
    res = client.post("/build-corpus", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {"splits", "rejectedObjects", "rejectedRows", "digests", "lineage"}
    assert set(body["splits"].keys()) == {"train", "validation", "test"}

    # malformed top-level -> exact 400 shape
    res2 = client.post("/build-corpus", json={"objects": []})
    assert res2.status_code == 400
    assert res2.json() == {"error": "INVALID_INPUT"}

    # invalid JSON body -> 400 INVALID_INPUT
    res3 = client.post("/build-corpus", content=b"{nope", headers={"content-type": "application/json"})
    assert res3.status_code == 400
    assert res3.json() == {"error": "INVALID_INPUT"}


def test_bqml_http_flow():
    rows = [
        {
            "id": "t1", "entity": "e", "eventTime": "2026-02-01T00:00:00Z",
            "predictionTime": "2026-02-02T00:00:00Z", "version": 1, "split": "TRAIN",
            "features": {"f": {"value": "v", "availableAt": "2026-01-01T00:00:00Z"}},
        },
        {
            "id": "e1", "entity": "e", "eventTime": "2026-02-03T00:00:00Z",
            "predictionTime": "2026-02-04T00:00:00Z", "version": 1, "split": "EVAL",
            "features": {"f": {"value": "v", "availableAt": "2026-01-01T00:00:00Z"}},
        },
    ]
    sel = {
        "phase": "select",
        "runId": "run-http",
        "forbiddenFeatures": [],
        "numTrialsLimit": 5,
        "rows": rows,
        "trials": [{"trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9}],
    }
    r1 = client.post("/bqml", json=sel)
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["selectedTrialId"] == 1
    assert set(b1.keys()) == {"runId", "selectedTrialId", "trainRowIds", "evalRowIds", "featureNames", "datasetDigest", "reasonCodes"}

    # replay identical
    r2 = client.post("/bqml", json=sel)
    assert r2.json() == b1

    # conflict
    sel_bad = dict(sel, numTrialsLimit=6)
    r3 = client.post("/bqml", json=sel_bad)
    assert r3.status_code == 409
    assert r3.json() == {"error": "RUN_ID_CONFLICT"}

    ev = {
        "phase": "evaluate",
        "runId": "run-http",
        "selectedTrialId": 1,
        "datasetDigest": b1["datasetDigest"],
        "metricFloor": 0.8,
        "requiredSlices": {"critical": 0.5},
        "rows": [{"label": 1, "prediction": 1, "slice": "critical"}],
        "bytesProcessed": 10,
        "maxBytes": 20,
    }
    r4 = client.post("/bqml", json=ev)
    assert r4.status_code == 200
    e = r4.json()
    assert e["decision"] == "admit"
    assert set(e.keys()) == {"runId", "selectedTrialId", "datasetDigest", "testMetric", "criticalSlicePass", "decision", "bytesProcessed", "reasonCodes"}

    # unknown phase -> 400 exact shape
    r5 = client.post("/bqml", json={"phase": "zzz"})
    assert r5.status_code == 400
    assert r5.json() == {"error": "INVALID_INPUT"}


def test_promote_http_idempotent():
    def evaluation(acc):
        return {
            "createdAt": "2026-05-01T00:00:30Z",
            "artifactDigest": f"sha:{acc}",
            "datasetDigest": "d",
            "schemaDigest": "s",
            "accuracy": acc,
            "latencyMs": 10,
            "sizeBytes": 100,
            "slices": {"critical": 0.9},
        }

    payload = {
        "asOf": "2026-05-01T00:01:00Z",
        "championVersion": "1",
        "policy": {
            "datasetDigest": "d",
            "schemaDigest": "s",
            "maxAgeSeconds": 3600,
            "accuracyFloor": 0.5,
            "requiredSlices": {"critical": 0.5},
            "maxLatencyMs": 100,
            "maxSizeBytes": 1000,
            "minImprovement": 0.01,
        },
        "versions": [
            {"version": "1", "artifactDigest": "sha:0.5", "tags": {}, "evaluation": evaluation(0.5)},
            {"version": "2", "artifactDigest": "sha:0.9", "tags": {}, "evaluation": evaluation(0.9)},
        ],
    }
    r1 = client.post("/promote", json=payload)
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["action"] == "promote"
    assert b1["aliasMutation"] == {"alias": "champion", "version": "2"}

    r2 = client.post("/promote", json=payload)
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["action"] == "retain"
    assert b2["aliasMutation"] is None
    assert b2["selectedVersion"] == "2"

    # future evaluation example from spec
    fut = json.loads(json.dumps(payload))
    fut["versions"][1]["evaluation"]["createdAt"] = "2026-05-01T00:01:01Z"
    r3 = client.post("/promote", json=fut)
    codes = r3.json()["failedGates"]["2"]
    assert "FUTURE_EVALUATION" in codes


def test_adapt_http():
    choose = {
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
        "candidates": [
            {"name": "prompt_only", "available": True, "quality": 0.7, "freshness": True,
             "latencyMs": 50, "memoryMb": 256, "labeledExamples": 0, "oneTimeCost": 10, "recurringCost": 0.01},
            {"name": "retrieval", "available": True, "quality": 0.85, "freshness": True,
             "latencyMs": 50, "memoryMb": 256, "labeledExamples": 0, "oneTimeCost": 10, "recurringCost": 0.01},
            {"name": "lora", "available": True, "quality": 0.9, "freshness": True,
             "latencyMs": 50, "memoryMb": 256, "labeledExamples": 5000, "oneTimeCost": 10, "recurringCost": 0.01},
            {"name": "qlora", "available": True, "quality": 0.9, "freshness": True,
             "latencyMs": 50, "memoryMb": 2048, "labeledExamples": 0, "oneTimeCost": 10, "recurringCost": 0.01},
        ],
    }
    r = client.post("/adapt", json=choose)
    assert r.status_code == 200
    b = r.json()
    assert b["selected"] == "retrieval"
    assert b["totalCosts"]["prompt_only"] == 110.0

    bad = client.post("/adapt", json={"operation": "??"})
    assert bad.status_code == 400
    assert bad.json() == {"error": "INVALID_INPUT"}


def test_quantize_http_stateful():
    freeze = {
        "phase": "freeze",
        "freezeId": "fid-1",
        "calibrationDigest": "c1",
        "tokenizerDigest": "t1",
        "allowedUnsupportedReasons": [],
        "candidates": [
            {"name": "int8", "files": {"m.bin": "abcd"}, "loadable": True,
             "calibrationDigest": "c1", "tokenizerDigest": "t1", "unsupportedReason": None},
        ],
    }
    r1 = client.post("/quantize", json=freeze)
    assert r1.status_code == 200
    frozen = r1.json()

    r2 = client.post("/quantize", json=freeze)
    assert r2.json() == frozen

    conflict = client.post("/quantize", json=dict(freeze, calibrationDigest="zz"))
    assert conflict.status_code == 409
    assert conflict.json() == {"error": "FREEZE_ID_CONFLICT"}

    select = {
        "phase": "select",
        "freezeId": "fid-1",
        "candidates": frozen["candidates"],
        "policy": {
            "maxBytes": 1000,
            "aggregateFloor": 0.5,
            "requiredSlices": {"crit": 0.5},
            "maxLatencyMs": 100,
            "candidateOrder": ["int8"],
        },
        "latencies": {"int8": 40},
        "rows": [{"label": 1, "slice": "crit", "predictions": {"int8": 1}}],
    }
    r3 = client.post("/quantize", json=select)
    assert r3.status_code == 200
    s = r3.json()
    assert s["selected"] == "int8"
    assert s["packageManifest"]["name"] == "int8"


def test_pipeline_http_session_isolation():
    def inputs():
        return {
            "generation": "g", "checksum": "c", "canonicalData": "d",
            "prepareCode": "pc", "prepareConfig": "pcf", "trainCode": "tc",
            "trainConfig": "tcf", "runtime": "r", "evaluateCode": "ec",
            "evaluateConfig": "ecf", "schemaDigest": "sd", "publishConfig": "pc2",
        }

    base = {"session": "sess-A", "revision": 1, "inputs": inputs(), "events": []}
    r0 = client.post("/pipeline", json=base)
    assert r0.status_code == 409 is False or r0.status_code == 200
    nodes0 = {n["node"]: n for n in r0.json()["nodes"]}
    k = nodes0["verify_data"]["dependencyDigests"]["cacheKey"]

    # session B must not see A's state even with identical inputs
    base_b = dict(base, session="sess-B")
    rb = client.post("/pipeline", json=base_b).json()
    nodes_b = {n["node"]: n for n in rb["nodes"]}
    assert nodes_b["verify_data"]["action"] == "rerun"

    # run a success in A
    ev = {
        "eventId": "ev1", "revision": 1, "node": "verify_data", "attempt": 1,
        "status": "started", "key": k, "artifactDigest": None, "receiptId": None,
    }
    ev2 = dict(ev, eventId="ev2", status="succeeded", artifactDigest="art")
    ra = client.post("/pipeline", json=dict(base, events=[ev, ev2])).json()
    va = {n["node"]: n for n in ra["nodes"]}["verify_data"]
    assert va["action"] == "reuse"

    # B still isolated
    rb2 = client.post("/pipeline", json=base_b).json()
    vb2 = {n["node"]: n for n in rb2["nodes"]}["verify_data"]
    assert vb2["action"] == "rerun"

    # revision conflict on A
    changed = dict(base, inputs=dict(inputs(), canonicalData="DIFFERENT"))
    rc = client.post("/pipeline", json=changed)
    assert rc.status_code == 409
    assert rc.json() == {"error": "REVISION_CONFLICT"}



def test_verify_bundle_http():
    from app.core.hashing import sha256_hex as _sha

    files = {
        "adapter_model.safetensors": "WBYTES",
        "adapter_config.json": '{"r":16,"target_modules":["q_proj"]}',
    }
    model_sha = _sha(b"WBYTES")
    files["evaluation.json"] = json.dumps(
        {"modelArtifactDigest": model_sha, "aggregate": 0.9, "slices": {"critical": 0.8}},
        separators=(",", ":"),
    )
    files["training_manifest.json"] = json.dumps({
        "baseRevision": "a" * 40, "task": "t", "datasetDigest": "d",
        "codeDigest": "c", "trainingConfigDigest": "cfg",
        "modelArtifactDigest": model_sha,
        "evaluationArtifactDigest": _sha(files["evaluation.json"].encode()),
    }, separators=(",", ":"))
    card = {"task": "t", "baseRevision": "a" * 40, "datasetDigest": "d",
            "modelArtifactDigest": model_sha, "license": "l",
            "intendedUse": "u", "limitations": "x"}
    files["README.md"] = 'p <!-- tds-model-card ' + json.dumps(card, separators=(",", ":")) + ' --> end'

    inv = []
    for name in sorted(files.keys()):
        data = files[name].encode("utf-8")
        inv.append({"name": name, "bytes": len(data), "sha256": _sha(data)})
    files["inventory.json"] = json.dumps(inv, separators=(",", ":"))

    payload = {
        "policy": {"requiredSlices": ["critical"], "license": "l", "intendedUse": "u", "limitations": "x"},
        "files": files,
    }
    r = client.post("/verify-bundle", json=payload)
    assert r.status_code == 200
    b = r.json()
    assert set(b.keys()) == {"decision", "violations", "inventoryDigest"}
    assert b["decision"] == "admit", f"violations={b['violations']}"
    assert b["violations"] == []

    bad = client.post("/verify-bundle", json={"files": {}})
    assert bad.status_code == 400
    assert bad.json() == {"error": "INVALID_INPUT"}

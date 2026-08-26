**1** Build an Immutable, Leakage-Safe Training Corpus (1.5 marks)

### Ask AI

- [How do I canonicalize, split, and hash a version-pinned corpus without leakage?](https://chatgpt.com/?q=How%20do%20I%20canonicalize%2C%20split%2C%20and%20hash%20a%20version-pinned%20corpus%20without%20leakage%3F)

Build a deterministic JSONL corpus service.

**Endpoint:** `POST /build-corpus`. Accept and return `application/json`.

##### Request

```json
{
  "policy": {
    "minTime": "...",
    "maxTime": "...",
    "contaminationThreshold": 0.8
  },
  "objects": [{
    "uri": "gs://bucket/object",
    "generation": "...",
    "fetchedGeneration": "...",
    "crc32c": "...",
    "schemaId": "training-v1",
    "content": "..."
  }]
}
```

- `minTime`, `maxTime`, and every row `eventTime` use `YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)`, where the optional fraction has 1–3 digits. Calendar and offset values must be valid. Offset magnitude is at most `14:00`; hour 14 requires minutes 00.
- `contaminationThreshold` is finite and in `[0,1]`.
- Generations are decimal strings. `crc32c` is 8 lowercase hex digits over the exact UTF-8 `content`.
- Each non-blank JSONL line is an object with exactly `id,entity,eventTime,revision,text`. The four text fields are strings and revision is a non-negative safe integer. Blank lines are ignored, and each file must contain at least one row.

##### Processing rules

1. Reject an object unless its URI matches `gs://bucket/object`, both generations are valid and equal, CRC32C matches, `schemaId` is `training-v1`, and every row is valid.
2. Canonicalize `entity` and `text` with Unicode NFKC, lowercase, trim, and collapse Unicode whitespace to one ASCII space. Normalize `eventTime` to UTC `YYYY-MM-DDTHH:mm:ss.sssZ`.
3. Deduplicate by the JSON tuple `[entity,eventTime,text]`. Keep the highest revision, then the UTF-8-byte-smallest ID; reject every loser as `DUPLICATE`.
4. An invalid policy rejects every retained row as `POLICY_INVALID`. Otherwise, reject times outside the inclusive window as `OUT_OF_WINDOW`.
5. `bucket = firstByte(SHA-256(UTF8(entity))) % 10`: 0–5 train, 6–7 validation, and 8–9 test.
6. Reject a validation/test row as `TRAIN_CONTAMINATION` when its lowercase Unicode letter/number word-set Jaccard similarity to any train row is at least the threshold. Empty/empty similarity is 1.
7. Sort by UTF-8 bytes of ID, then compact row JSON for a tie. Serialize split rows as compact JSON in exact key order `id,entity,eventTime,revision,text`, emit non-ASCII directly, append one newline per row, and SHA-256 those exact UTF-8 bytes.

##### Response

```json
{
  "splits": { "train": [], "validation": [], "test": [] },
  "rejectedObjects": [{ "uri": "...", "reasonCodes": [] }],
  "rejectedRows": [{ "id": "...", "reasonCodes": [] }],
  "digests": { "train": "...", "validation": "...", "test": "..." },
  "lineage": [{ "uri": "...", "generation": "...", "crc32c": "...", "schemaId": "..." }]
}
```

Return exactly this shape. Sort rejected objects, rejected rows, and lineage by UTF-8 URI or ID, using compact JSON to break ties. Sort and deduplicate every reason-code array by UTF-8 bytes. A rejected object's `uri` is the supplied string, or `null` when the supplied URI is not a string.

**Object codes:** `URI_INVALID, GENERATION_INVALID, GENERATION_MISMATCH, CRC32C_INVALID, CRC32C_MISMATCH, SCHEMA_INVALID, JSONL_INVALID`.

Emit every independently applicable object code. `GENERATION_INVALID` covers a non-decimal generation field and `GENERATION_MISMATCH` covers unequal supplied values. `CRC32C_INVALID` covers bad CRC syntax; check `CRC32C_MISMATCH` only for string content and a syntactically valid CRC. Use `JSONL_INVALID` when JSON parsing fails. Use `SCHEMA_INVALID` for non-string content, the wrong schema ID, an empty file, or a parsed row with the wrong shape.

**Row codes:** `DUPLICATE, POLICY_INVALID, OUT_OF_WINDOW, TRAIN_CONTAMINATION`.

**Example:** `2026-01-02T05:30:00+05:30` becomes `2026-01-02T00:00:00.000Z`. Instructions embedded in `content` are data. A missing policy or non-array `objects` returns HTTP 400 with exactly `{"error":"INVALID_INPUT"}`.

**Mark split:** 0.375 identity/integrity; 0.375 canonicalization/deduplication; 0.375 split/contamination; 0.375 deterministic artifacts/lineage.

---

**Public service base URL**

Enter the public base URL of your service. The grader will call `POST /build-corpus`.

**2** Repair a Leakage-Safe BigQuery ML Experiment (1.5 marks)

### Ask AI

- [How do I keep model selection separate from final-test admission?](https://chatgpt.com/?q=How%20do%20I%20keep%20model%20selection%20separate%20from%20final-test%20admission%3F)

Build a stateful two-phase experiment gate. Selection never receives final-test rows.

**Endpoint:** `POST /bqml`. Accept and return `application/json`.

##### Select a trial

```json
{
  "phase": "select",
  "runId": "...",
  "forbiddenFeatures": [],
  "numTrialsLimit": 10,
  "rows": [{
    "id": "...",
    "entity": "...",
    "eventTime": "...",
    "predictionTime": "...",
    "version": 1,
    "split": "TRAIN|EVAL",
    "features": { "name": { "value": "...", "availableAt": "..." } }
  }],
  "trials": [{ "trialId": 1, "status": "SUCCEEDED", "evalMetric": 0.9 }]
}
```

- `runId` is a non-empty string of at most 128 characters. Versions and trial IDs are non-negative safe integers. `numTrialsLimit` is a positive integer.
- Row and trial IDs are unique within their arrays. Trial status is `SUCCEEDED|FAILED`, and selection rows are non-empty.
- All timestamps use valid `YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)` instants, where the optional fraction has 1–3 digits.
- Deduplicate rows by `[entity, UTC(eventTime)]`. Keep the highest integer version, then the UTF-8-byte-smallest ID.
- A feature is eligible only if it appears in every retained row, is not forbidden, and every `availableAt <= predictionTime`. Sort feature names and TRAIN/EVAL IDs by UTF-8 bytes.
- Only finite `SUCCEEDED` trials are eligible. Maximize `evalMetric` and break exact ties with the smallest integer `trialId`. More than `numTrialsLimit` trials is a contract failure.
- Compute `datasetDigest` as SHA-256 of compact JSON with the exact shape and key order `{trainRowIds,evalRowIds,featureNames}`.

Return:

```json
{
  "runId": "...",
  "selectedTrialId": 1,
  "trainRowIds": [],
  "evalRowIds": [],
  "featureNames": [],
  "datasetDigest": "...",
  "reasonCodes": []
}
```

Return exactly these fields. Codes are `INVALID_INPUT, TRIAL_LIMIT_EXCEEDED, NO_SUCCESSFUL_TRIAL`. Any code makes `selectedTrialId` null. A malformed selection also returns a null `datasetDigest`.

Persist the complete response under `runId`. An identical replay returns it unchanged. Reusing the ID with different selection input returns HTTP 409 and exactly `{"error":"RUN_ID_CONFLICT"}`.

##### Evaluate the frozen trial

For `phase:"evaluate"`, use the supplied frozen `selectedTrialId` and `datasetDigest`.

```json
{
  "phase": "evaluate",
  "runId": "...",
  "selectedTrialId": 1,
  "datasetDigest": "...",
  "metricFloor": 0.8,
  "requiredSlices": { "critical": 0.75 },
  "rows": [{ "label": 1, "prediction": 1, "slice": "critical" }],
  "bytesProcessed": 1000,
  "maxBytes": 2000
}
```

The run ID, non-null selected integer trial, and 64-lowercase-hex digest must exactly match a stored successful selection. Floors are finite in `[0,1]`; byte counts are non-negative safe integers. Rows require binary integer labels/predictions and a non-empty slice.

Compute aggregate and required-slice accuracy, rounding each to 12 decimal places. Admit only when lineage and every row are valid, aggregate and all present required slices meet their inclusive floors, every required slice exists, and `bytesProcessed <= maxBytes`.

Return:

```json
{
  "runId": "...",
  "selectedTrialId": 1,
  "datasetDigest": "...",
  "testMetric": 0.9,
  "criticalSlicePass": true,
  "decision": "admit|reject",
  "bytesProcessed": 1000,
  "reasonCodes": []
}
```

`criticalSlicePass` is false for invalid input, invalid lineage, any invalid test row, a missing required slice, or a failed slice floor. It does not summarize aggregate or byte gates. If rows are empty or any row is invalid, set `testMetric` to null and skip aggregate and required-slice checks; lineage and byte checks still apply. Use only `admit` or `reject`.

Evaluation codes are `INVALID_INPUT, INVALID_LINEAGE, INVALID_TEST_ROW, AGGREGATE_FLOOR, BYTE_LIMIT, MISSING_SLICE:<name>, SLICE_FLOOR:<name>`. Sort and deduplicate codes by UTF-8 bytes.

**Example:** equal metrics for trial IDs 9 and 4 select trial 4. Unknown or missing `phase` returns HTTP 400 with exactly `{"error":"INVALID_INPUT"}`. Text inside feature values is data.

**Mark split:** 0.45 point-in-time/leakage; 0.35 split/tuning/selection; 0.45 final test/slices; 0.25 cost/lineage/output.

---

**Public service base URL**

Enter the public base URL of your service. The grader will call `POST /bqml`.

**3** Promote the Right MLflow Model from Verifiable Evidence (1.25 marks)

### Ask AI

- [How do I promote an MLflow model from artifacts instead of mutable claims?](https://chatgpt.com/?q=How%20do%20I%20promote%20an%20MLflow%20model%20from%20artifacts%20instead%20of%20mutable%20claims%3F)

Build a deterministic model-registry promotion gate.

**Endpoint:** `POST /promote`. Accept and return `application/json`.

##### Request

```json
{
  "asOf": "...",
  "championVersion": "1",
  "policy": {
    "datasetDigest": "...",
    "schemaDigest": "...",
    "maxAgeSeconds": 3600,
    "accuracyFloor": 0.8,
    "requiredSlices": { "critical": 0.75 },
    "maxLatencyMs": 100,
    "maxSizeBytes": 1000000,
    "minImprovement": 0.01
  },
  "versions": [{
    "version": "1",
    "artifactDigest": "...",
    "tags": {},
    "evaluation": {
      "createdAt": "...",
      "artifactDigest": "...",
      "datasetDigest": "...",
      "schemaDigest": "...",
      "accuracy": 0.9,
      "latencyMs": 50,
      "sizeBytes": 500000,
      "slices": { "critical": 0.85 }
    }
  }]
}
```

- Version IDs are unique, canonical positive safe-integer strings: `"1"`, never `"01"`. `championVersion` identifies one listed version.
- `asOf` and `createdAt` are valid `YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)` instants, where the optional fraction has 1–3 digits.
- Accuracy, improvement, and slice floors/values are finite in `[0,1]`. Latency is finite and non-negative. Age and size are non-negative safe integers. Policy digests are non-empty.

Mutable tags and descriptions are never evidence. A version is eligible only when its evaluation:

- satisfies `asOf - maxAgeSeconds <= createdAt <= asOf`;
- contains finite accuracy, latency, and size values;
- binds the registered artifact and expected dataset/schema digests;
- contains every required slice at its floor; and
- passes the aggregate accuracy, latency, and size gates.

##### Selection

Reject every occurrence of a duplicate or noncanonical version before constructing lookup maps. Rank eligible versions by accuracy descending, latency ascending, size ascending, then numeric version ascending. If champion evidence is invalid, use `action:"block"` and a null selection. Otherwise, round the challenger's accuracy minus the champion's accuracy to 12 decimal places. Promote only when that value is at least `minImprovement`; otherwise retain the champion.

##### Response

```json
{
  "action": "promote|retain|block",
  "championVersion": "1",
  "selectedVersion": "2",
  "eligibleVersions": ["1", "2"],
  "failedGates": {},
  "aliasMutation": { "alias": "champion", "version": "2" },
  "evidence": {}
}
```

`evidence` is the selected version's complete evaluation object, or null. `failedGates` contains every input version with sorted, unique UTF-8 codes. `aliasMutation` is present only for promotion; otherwise it is null. Replaying after that alias change must retain it.

Gate codes are:

```ruby
INVALID_VERSION, DUPLICATE_VERSION, INVALID_POLICY,
MISSING_EVALUATION, NON_FINITE, METRIC_RANGE, INVALID_TIMESTAMP,
FUTURE_EVALUATION, STALE_EVALUATION,
ARTIFACT_MISMATCH, DATASET_MISMATCH, SCHEMA_MISMATCH,
ACCURACY_FLOOR, LATENCY_LIMIT, SIZE_LIMIT,
MISSING_SLICE:<name>, SLICE_RANGE:<name>, SLICE_FLOOR:<name>
```

**Example:** evidence created one second after `asOf` fails as `FUTURE_EVALUATION`, regardless of its accuracy tag. A missing policy, non-array `versions`, or non-string `championVersion` returns HTTP 400 with exactly `{"error":"INVALID_INPUT"}`.

**Mark split:** 0.35 evidence/lineage; 0.45 gates/winner; 0.25 mutation/idempotency; 0.20 output.

---

**Public service base URL**

Enter the public base URL of your service. The grader will call `POST /promote`.

**4** Choose the Minimal Adaptation and Repair a PEFT Run (2 marks)

### Ask AI

- [How do assistant-only loss masks, LoRA targets, and exact resume state work?](https://chatgpt.com/?q=How%20do%20assistant-only%20loss%20masks%2C%20LoRA%20targets%2C%20and%20exact%20resume%20state%20work%3F)

Build one deterministic endpoint with two operations.

**Endpoint:** `POST /adapt`. Accept and return `application/json`.

##### Choose an intervention

```json
{
  "operation": "choose",
  "policy": {
    "minQuality": 0.8,
    "freshnessRequired": true,
    "maxLatencyMs": 100,
    "maxMemoryMb": 1024,
    "maxLabeledExamples": 100,
    "maxTotalCost": 1000,
    "horizonRequests": 10000
  },
  "candidates": [{
    "name": "prompt_only",
    "available": true,
    "quality": 0.85,
    "freshness": true,
    "latencyMs": 50,
    "memoryMb": 256,
    "labeledExamples": 0,
    "oneTimeCost": 10,
    "recurringCost": 0.01
  }]
}
```

Supply exactly one candidate for each of the four interventions below. Quality is finite in `[0,1]`; ceilings and costs are finite and non-negative; labeled examples and horizon requests are non-negative safe integers. A candidate passes only if it is available and meets every inclusive quality, freshness, latency, memory, labeled-data, and cost gate. Compute `oneTimeCost + horizonRequests * recurringCost`, rounded to 12 decimals.

`prompt_only → retrieval → lora → qlora`

Return exactly `{selected,eligible,totalCosts,reasonCodes}`. Keep `eligible` in published priority order and select its first entry, or null. `totalCosts` and `reasonCodes` contain all four names. Sort and deduplicate each code array by UTF-8 bytes. Codes are `INVALID_INPUT, UNAVAILABLE, QUALITY_FLOOR, FRESHNESS_REQUIRED, LATENCY_LIMIT, MEMORY_LIMIT, DATA_LIMIT, COST_LIMIT`.

##### Repair a PEFT run

```json
{
  "operation": "repair",
  "tokens": [{ "id": 1, "role": "assistant", "padding": false, "text": "..." }],
  "templateApplications": 1,
  "parameters": [{ "name": "...", "target": "...", "numel": 1 }],
  "allowedTargets": [],
  "inferenceMode": false,
  "trainRowIds": [],
  "evalRowIds": [],
  "dropoutActiveDuringEval": false,
  "artifactFiles": [],
  "baseRevision": "...",
  "datasetDigest": "...",
  "codeDigest": "...",
  "configDigest": "...",
  "expectedDigests": {},
  "microBatch": 1,
  "gradientAccumulation": 1,
  "replicas": 1,
  "expectedEffectiveBatch": 1,
  "checkpoint": {},
  "uninterruptedWeights": [],
  "resumedWeights": [],
  "resumeTolerance": 0
}
```

- Tokens are non-empty; IDs are non-negative safe integers; role is `system|user|assistant`; padding is Boolean and text is a string. For a valid list, label an unpadded assistant token with its ID and every other token `-100`. If any token is invalid, all labels are `-100`.
- Require exactly one template application. Parameter names are unique, `numel` is a positive safe integer, and allowed targets are non-empty unique strings. At least one parameter must have an allowed target and a name ending `.lora_A.weight` or `.lora_B.weight`; train only those parameters, sort their names by UTF-8 bytes, and safely sum `numel`.
- Require `inferenceMode:false`, `dropoutActiveDuringEval:false`, non-empty unique string train/evaluation IDs, and disjoint sets.
- `artifactFiles` must be exactly `adapter_config.json, adapter_model.safetensors`, once each. Return that set sorted by UTF-8 bytes.
- Require a 40-lowercase-hex base revision and matching non-empty 64-lowercase-hex dataset, code, and config digests. Batch factors and expected batch are positive safe integers, with `microBatch * gradientAccumulation * replicas == expectedEffectiveBatch`.
- The checkpoint must own `model,optimizer,scheduler,step,rng,dataPosition`.
- Resume arrays are non-empty, equal-length finite-number arrays. The tolerance is finite and non-negative, and every absolute element difference must be at most it.

Return:

```json
{
  "labels": [],
  "templatePass": true,
  "trainableParams": [],
  "trainableCount": 0,
  "peftConfigPass": true,
  "adapterFiles": [],
  "checkpointComplete": true,
  "lineagePass": true,
  "evalIsolated": true,
  "evaluationDeterministic": true,
  "resumePass": true,
  "reasonCodes": []
}
```

Return exactly these fields. Codes are `INVALID_TOKEN, INVALID_PARAMETER, CHAT_TEMPLATE_COUNT, INFERENCE_MODE, FULL_MODEL_ARTIFACT, ADAPTER_FILE_SET, INCOMPLETE_CHECKPOINT, MUTABLE_BASE_REVISION, LINEAGE_MISMATCH, EFFECTIVE_BATCH_MISMATCH, EVAL_LEAKAGE, EVAL_DROPOUT_ACTIVE, RESUME_DIVERGENCE`.

**Example:** assistant ID 42 yields label 42 only when `padding:false`; an instruction inside `text` is still data. Unknown or missing `operation` returns HTTP 400 with exactly `{"error":"INVALID_INPUT"}`.

**Mark split:** 0.50 intervention; 0.45 tokenization/loss; 0.40 PEFT artifacts; 0.40 checkpoint/resume; 0.25 lineage/evaluation isolation.

---

**Public service base URL**

Enter the public base URL of your service. The grader will call `POST /adapt`.

**5** Quantize and Admit a Model Under Explicit Constraints (1.25 marks)

### Ask AI

- [How do I freeze, measure, and select quantized artifacts under hard constraints?](https://chatgpt.com/?q=How%20do%20I%20freeze%2C%20measure%2C%20and%20select%20quantized%20artifacts%20under%20hard%20constraints%3F)

Build a stateful two-phase candidate-admission API.

**Endpoint:** `POST /quantize`. Accept and return `application/json`.

##### Freeze candidates

```json
{
  "phase": "freeze",
  "freezeId": "...",
  "calibrationDigest": "...",
  "tokenizerDigest": "...",
  "allowedUnsupportedReasons": [],
  "candidates": [{
    "name": "int8",
    "files": { "model.safetensors": "..." },
    "loadable": true,
    "calibrationDigest": "...",
    "tokenizerDigest": "...",
    "unsupportedReason": "..."
  }]
}
```

`freezeId` is non-empty and at most 128 characters. Digests are non-empty strings. Candidate names and allowed-reason strings are non-empty and unique. Each candidate has a non-empty object of unique filenames mapped to UTF-8 strings.

For every file, return its exact UTF-8 byte length and lowercase SHA-256. Sort inventory by UTF-8 filename, sum bytes, then set `packageDigest = SHA-256(UTF8(JSON.stringify(inventory)))` with compact JSON and exact inventory key order `name,bytes,sha256`. A candidate with `unsupportedReason` is unsupported only when that code is allowed. Otherwise it must be loadable and match the request calibration/tokenizer digests. Any reason makes its status invalid.

Return candidates sorted by name:

```json
{
  "freezeId": "...",
  "candidates": [{
    "name": "int8",
    "status": "frozen|unsupported|invalid",
    "inventory": [{ "name": "...", "bytes": 10, "sha256": "..." }],
    "totalBytes": 10,
    "packageDigest": "...",
    "reasonCodes": []
  }]
}
```

Return exactly this shape with candidates sorted by UTF-8 name. Codes are `INVALID_INPUT, UNALLOWED_UNSUPPORTED_REASON, NOT_LOADABLE, CALIBRATION_MISMATCH, TOKENIZER_MISMATCH`.

If a candidate's files are invalid, return an empty inventory and null `totalBytes` and `packageDigest`.

Persist the complete response under `freezeId`. Identical replay returns it unchanged. Reuse with different freeze input returns HTTP 409 and exactly `{"error":"FREEZE_ID_CONFLICT"}`.

##### Select a candidate

The grader sends the frozen candidates plus fresh rows containing each label, candidate predictions, and slice.

```json
{
  "phase": "select",
  "freezeId": "...",
  "candidates": [],
  "policy": {
    "maxBytes": 1000000,
    "aggregateFloor": 0.8,
    "requiredSlices": { "critical": 0.75 },
    "maxLatencyMs": 100,
    "candidateOrder": ["int4", "int8"]
  },
  "latencies": { "int4": 40, "int8": 60 },
  "rows": [{
    "label": 1,
    "slice": "critical",
    "predictions": { "int4": 1, "int8": 1 }
  }]
}
```

The supplied candidate array must exactly equal the response stored for `freezeId`. Recompute every inventory total and package digest; never trust a submitted `totalBytes`. Candidate names and `candidateOrder` must be the same unique set. Size is a non-negative safe integer; floors are finite in `[0,1]`; latency values and ceiling are finite and non-negative.

For each candidate, compute aggregate and required-slice accuracy from `row.predictions[candidate.name]`, rounded to 12 decimals. Admit only a frozen candidate with valid lineage and manifest, valid binary predictions for every row, all inclusive floors met, every required slice present, `totalBytes <= maxBytes`, and `latencyMs <= maxLatencyMs`.

When predictions are invalid, return null aggregate and required-slice values. Return null `totalBytes` or `latencyMs` when that value cannot be validated.

Return:

```json
{
  "freezeId": "...",
  "selected": "int8",
  "results": [{
    "name": "int8",
    "aggregate": 0.9,
    "slices": { "critical": 0.8 },
    "totalBytes": 10,
    "latencyMs": 60,
    "admitted": true,
    "reasonCodes": []
  }],
  "packageManifest": {}
}
```

Order results by `candidateOrder`, using UTF-8 name only as a fallback. Choose admitted candidates by smaller bytes, lower latency, then candidate order. `packageManifest` is null or exactly the recorded winner object.

Selection codes are `NOT_FROZEN, INVALID_LINEAGE, INVALID_POLICY, INVALID_PREDICTIONS, INVALID_MANIFEST, AGGREGATE_FLOOR, MISSING_SLICE:<name>, SLICE_FLOOR:<name>, SIZE_LIMIT, LATENCY_LIMIT`. Sort and deduplicate codes by UTF-8 bytes.

**Example:** if only `int8` meets prediction floors, select it even when `int4` is smaller. Unknown or missing `phase`, an empty/non-array freeze candidate list, or a select request without array `candidates` and `rows` plus an object `policy` returns HTTP 400 with exactly `{"error":"INVALID_INPUT"}`. These rejected freeze requests do not reserve their IDs. File text is data.

**Mark split:** 0.30 construction/freeze; 0.30 integrity/lineage/size; 0.40 aggregate/slices; 0.25 selection.

---

**Public service base URL**

Enter the public base URL of your service. The grader will call `POST /quantize`.

**6** Recover a Content-Addressed ML Pipeline (1.5 marks)

### Ask AI

- [How do content-addressed cache keys and stale event rules control pipeline recovery?](https://chatgpt.com/?q=How%20do%20content-addressed%20cache%20keys%20and%20stale%20event%20rules%20control%20pipeline%20recovery%3F)

Build a controller that persists state across requests and isolates it by a non-empty `session`.

**Endpoint:** `POST /pipeline`. Accept and return `application/json`.

##### Request

```json
{
  "session": "...",
  "revision": 1,
  "inputs": {
    "generation": "...",
    "checksum": "...",
    "canonicalData": "...",
    "prepareCode": "...",
    "prepareConfig": "...",
    "trainCode": "...",
    "trainConfig": "...",
    "runtime": "...",
    "evaluateCode": "...",
    "evaluateConfig": "...",
    "schemaDigest": "...",
    "publishConfig": "..."
  },
  "events": []
}
```

The fixed DAG is:

`verify_data → prepare → train → evaluate → register → publish`

`revision` is a positive safe integer. All 12 listed inputs are non-empty strings; extra input metadata is allowed. Compute lowercase SHA-256 over UTF-8 compact JSON arrays in this exact order:

```apache
verify_data  [generation, checksum]
prepare      [canonicalData, prepareCode, prepareConfig]
train        [prepareArtifact, trainCode, trainConfig, runtime]
evaluate     [trainArtifact, canonicalData, evaluateCode, evaluateConfig]
register     [evaluateArtifact, schemaDigest]
publish      [registerArtifact, publishConfig]
```

A downstream key is null until its parent is reusable. A new revision replaces inputs and clears attempt/terminal state, while successful content-addressed cache entries remain. Ignore well-formed events from an older revision. The same revision with any different input, including extra metadata, returns `REVISION_CONFLICT`.

##### Events

```json
{
  "eventId": "...",
  "revision": 1,
  "node": "train",
  "attempt": 1,
  "status": "started|succeeded|retryable_failed|terminal_failed",
  "key": "...",
  "artifactDigest": "...",
  "receiptId": "..."
}
```

- Each event contains exactly the eight listed fields.
- `attempt` is a positive safe integer. A success requires a non-empty artifact digest; every other status requires null.
- Register/publish success requires `receipt:<node>:<key>`; every other event requires a null receipt.
- Process a valid batch in input order. A 409 conflict rolls back the entire batch. Ignored events do not consume their IDs.
- Event IDs are global within a session. An exact replay is ignored; the same ID with different compact canonical JSON conflicts.

Use these transitions for a ready node and its current key:

| Previous stateIncoming eventResult |                                                               |                     |
| ---------------------------------- | ------------------------------------------------------------- | ------------------- |
| none                               | `started`, attempt 1                                          | accept              |
| none                               | completion or attempt > 1                                     | ignore              |
| `started(n)`                       | `succeeded \| retryable_failed \| terminal_failed`, attempt n | accept              |
| `retryable_failed(n)`              | `started`, attempt n+1                                        | accept              |
| non-cached state                   | lower attempt                                                 | ignore              |
| `started` / `retryable_failed`     | other transition                                              | `STATUS_CONFLICT`   |
| succeeded/current cache            | success, different artifact                                   | `EVIDENCE_CONFLICT` |
| succeeded/current cache            | any other new event                                           | `STATUS_CONFLICT`   |
| `terminal_failed`                  | any new valid event                                           | `STATUS_CONFLICT`   |

Ignore a wrong revision, node, or key; an unavailable parent; invalid status, artifact, or receipt; and an invalid attempt. Permanently bind a successful key to its first artifact and event ID.

##### Response

```json
{
  "revision": 1,
  "acceptedEventIds": [],
  "ignoredEventIds": [],
  "nodes": [{
    "node": "verify_data",
    "action": "reuse|rerun|block",
    "reasonCodes": [],
    "dependencyDigests": {},
    "triggeringEventIds": []
  }]
}
```

`dependencyDigests` contains the named inputs plus `cacheKey`. Preserve input order for event IDs and DAG order for nodes. Each node has exactly one reason:

- cached: `reuse / CACHE_HIT`, triggered by its immutable success event;
- ready without cache: `rerun / CACHE_MISS` or `rerun / RETRYABLE_FAILURE`;
- running: `block / RUNNING`, triggered by its start event;
- terminal: `block / TERMINAL_FAILURE`, then descendants use `block / UPSTREAM_TERMINAL`;
- other descendants of a pending node: `block / UPSTREAM_PENDING`.

HTTP 409 returns exactly `{"error":"<code>"}`. Codes are `INVALID_REQUEST, INVALID_EVENT, EVENT_ID_CONFLICT, REVISION_CONFLICT, EVIDENCE_CONFLICT, STATUS_CONFLICT`.

**Example:** `started(1) → retryable_failed(1) → started(2) → succeeded(2)` is valid; success without the first start is ignored. Persist readback for the same session and never share state across sessions.

**Mark split:** 0.40 cache/dependencies; 0.40 event ordering/transitions; 0.35 receipts/terminal; 0.175 immutable atomic evidence; 0.175 session persistence.

---

**Public service base URL**

Enter the public base URL of your service. The grader will call `POST /pipeline`.

**7** Publish a Verifiable Model Bundle and Model Card (1 mark)

### Ask AI

- [How do I make model-card claims independently verifiable from immutable files?](https://chatgpt.com/?q=How%20do%20I%20make%20model-card%20claims%20independently%20verifiable%20from%20immutable%20files%3F)

Build a deterministic verifier for an untrusted UTF-8 model bundle.

**Endpoint:** `POST /verify-bundle`. Accept and return `application/json`.

##### Request

```json
{
  "policy": {
    "requiredSlices": ["critical"],
    "license": "...",
    "intendedUse": "...",
    "limitations": "..."
  },
  "files": { "filename": "UTF-8 string" }
}
```

`requiredSlices` is a non-empty array of unique non-empty strings. The other three policy fields are non-empty strings.

The required files are:

- `README.md`
- `training_manifest.json`
- `evaluation.json`
- `inventory.json`
- `adapter_model.safetensors`
- `adapter_config.json`

##### Verification rules

1. `inventory.json` is a compact JSON array listing every file except itself, with no extra files, sorted by UTF-8 filename. Entries have exact key order `name,bytes,sha256`. Recompute exact UTF-8 bytes and lowercase SHA-256. `inventoryDigest` hashes the exact compact JSON of this recomputed array.
2. Extra files are invalid. Weight extensions `.bin, .pt, .pth, .pkl, .pickle` are unsafe.
3. `adapter_config.json` is an object with a positive safe-integer `r` and a non-empty unique string array `target_modules`. Extra config properties are allowed. This verifies file identity and schema, not framework-level safetensors loadability.
4. The training manifest is an object with an immutable 40-lowercase-hex base revision and non-empty `task`, `datasetDigest`, `codeDigest`, `trainingConfigDigest`, `modelArtifactDigest`, and `evaluationArtifactDigest`.
5. Recompute the last two digests from `adapter_model.safetensors` and the exact bytes of `evaluation.json`.
6. `evaluation.json` is an object that binds that model digest. Its aggregate and every required slice are finite in `[0,1]`. Extra evaluation properties and non-required slices are allowed.

##### Model card

`README.md` must contain exactly one marker with the literal delimiters shown:

```html
<!-- tds-model-card {"task":"...", ...} -->
```

Parse the entire payload between the marker prefix and `-->`; braces inside JSON strings are ordinary characters. The parsed value must be an object. Its `task`, `baseRevision`, `datasetDigest`, `modelArtifactDigest`, `license`, `intendedUse`, and `limitations` must match the machine manifests and policy. Extra card properties and prose outside the marker are allowed.

- No marker emits `MODEL_CARD_COUNT` and `MISSING_MODEL_CARD`.
- Multiple markers emit only `MODEL_CARD_COUNT`.
- One marker with malformed JSON or a non-object payload emits `INVALID_MODEL_CARD`.

##### Response

```json
{
  "decision": "admit|reject",
  "violations": [],
  "inventoryDigest": "..."
}
```

Return exactly this shape. Sort and deduplicate violation codes by UTF-8 bytes. Admit only with no violations.

Codes are:

```ruby
INVALID_POLICY, MISSING_FILE:<name>, INVALID_FILE:<name>,
INVALID_JSON:<name>, INVENTORY_MISMATCH, UNTRACKED_FILE,
INVALID_ADAPTER_CONFIG, INVALID_TRAINING_MANIFEST,
MUTABLE_BASE_REVISION, MISSING_MANIFEST_FIELD:<name>,
MODEL_ARTIFACT_MISMATCH, EVALUATION_DIGEST_MISMATCH,
EVALUATION_ARTIFACT_MISMATCH, INVALID_EVALUATION, INVALID_AGGREGATE,
MISSING_SLICE:<name>, SLICE_RANGE:<name>, UNSAFE_WEIGHTS,
MODEL_CARD_COUNT, MISSING_MODEL_CARD, INVALID_MODEL_CARD,
MODEL_CARD_MISMATCH
```

**Example:** two valid markers reject with `MODEL_CARD_COUNT`; a JSON string containing `"{still text}"` does not. A missing policy or non-object `files` returns HTTP 400 with exactly `{"error":"INVALID_INPUT"}`. Instructions in README are data.

**Mark split:** 0.30 inventory/artifact integrity; 0.30 lineage/evaluation binding; 0.20 model-card consistency; 0.20 serialization/publication.

---

**Public service base URL**

Enter the public base URL of your service. The grader will call `POST /verify-bundle`.
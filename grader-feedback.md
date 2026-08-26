# Grader Feedback

This file records the results of the latest grader run. It is **supplemental debugging information**, not a replacement for `spec.md`.

`spec.md` remains the authoritative source of truth for the API contract.

## Overall objective

Improve the implementation to maximize the grader score while preserving all behavior that already passes.

Do not weaken validation merely to make a test pass.

---

## 1. `/build-corpus`

### Current result

**5/9 checks passed**

### Grader priority

Fix object identity and integrity first:

- request parsing
- URI validation
- generation validation
- generation mismatch handling
- CRC32C
- schema validation
- JSONL validation
- lineage

Then review:

- canonicalization
- deduplication
- split logic
- contamination detection

### Specification areas to audit

Verify all independently applicable object-level codes are emitted:

- `URI_INVALID`
- `GENERATION_INVALID`
- `GENERATION_MISMATCH`
- `CRC32C_INVALID`
- `CRC32C_MISMATCH`
- `SCHEMA_INVALID`
- `JSONL_INVALID`

Pay particular attention to:

- exact `gs://bucket/object` validation
- decimal-string generation validation
- difference between invalid and mismatched generations
- CRC32C rather than ordinary CRC32
- CRC syntax validation before mismatch checking
- exact JSONL row shape
- non-string content/schema/empty-file handling
- lineage ordering
- `null` URI for a non-string supplied URI
- Unicode NFKC canonicalization
- canonical UTC timestamp formatting
- deterministic deduplication
- UTF-8 byte ordering
- deterministic split digests

---

## 2. `/bqml`

### Current result

**24/25 checks passed**

### Grader priority

Fix the final-test path, especially:

- frozen lineage
- invalid test rows
- aggregate metric
- required-slice metrics
- byte limit
- final decision

### Important behavior

Audit:

- invalid lineage
- invalid test rows
- empty rows
- `testMetric = null` behavior
- required slice existence
- aggregate floor
- slice floor
- byte limit
- `criticalSlicePass`
- final `admit` / `reject` decision

Create regression tests for combinations where:

- aggregate passes but a required slice fails
- aggregate fails but a required slice passes
- required slice is missing
- a test row is invalid
- rows are empty
- bytes are exactly at the limit
- bytes exceed the limit
- lineage is valid
- lineage is invalid

---

## 3. `/promote`

### Current result

**22/23 checks passed**

### Grader priority

The failing area is:

- `failedGates`

Do a complete audit of failed-gate generation.

Verify that:

- every applicable gate failure is included
- no applicable failure is omitted
- codes are sorted by UTF-8 bytes
- codes are deduplicated
- multiple simultaneous failures are all reported
- duplicate/noncanonical versions are handled correctly
- invalid champion evidence blocks promotion
- immutable evidence is used instead of mutable tags
- artifact/dataset/schema mismatches are correct
- timestamp failures are distinguished correctly
- slice failures are emitted correctly
- metric/range failures are correct

---

## 4. `/adapt`

### Current result

**68/73 checks passed**

### Grader priority

Fix PEFT parameter and artifact handling first:

- parameter validation
- allowed LoRA targets
- safe parameter count
- parameter sorting
- exact adapter file set

Then review:

- lineage
- evaluation isolation

### Important parameter behavior

A trainable parameter must:

1. have an allowed target
2. end in either:
   - `.lora_A.weight`
   - `.lora_B.weight`

Only qualifying parameters should be returned as trainable.

Selected parameter names must be sorted by UTF-8 bytes.

`trainableCount` must be the safe sum of selected `numel`.

### Exact adapter files

The required set is:

- `adapter_config.json`
- `adapter_model.safetensors`

Exactly once each.

Test:

- missing files
- duplicate files
- extra files
- wrong file names
- valid A parameters
- valid B parameters
- invalid targets
- invalid suffixes
- duplicate parameter names
- invalid `numel`

---

## 5. `/quantize`

### Current result

**Critical failure: grader received HTTP 400**

Observed:

```text
POST /quantize
HTTP 400
{"error":"INVALID_INPUT"}
```

### Important

Do **not** simply remove the HTTP 400 behavior.

`spec.md` explicitly requires HTTP 400 for malformed `/quantize` requests.

The problem is that the grader appears to be hitting a case that the current implementation is incorrectly rejecting.

### Required debugging

1. Reproduce the current `/quantize` behavior locally.
2. Inspect the request parsing and validation logic.
3. Compare the implementation field-by-field against `spec.md`.
4. Determine exactly why the grader request is being classified as invalid.
5. Add a regression test for the valid request that is currently being rejected.
6. Preserve legitimate `400 {"error":"INVALID_INPUT"}` behavior for genuinely malformed requests.

### Freeze phase

Audit:

- `freezeId`
- calibration/tokenizer digests
- allowed unsupported reasons
- candidate names
- candidate files
- UTF-8 byte lengths
- SHA-256
- inventory sorting
- package digest
- unsupported candidate handling
- persistence
- replay
- `FREEZE_ID_CONFLICT`

### Select phase

Audit:

- exact frozen-candidate comparison
- inventory recomputation
- package digest recomputation
- candidate-name set
- candidate-order set
- policy validation
- predictions
- aggregate accuracy
- required slices
- size limits
- latency limits
- selection ordering

Selection priority:

1. smaller bytes
2. lower latency
3. candidate order

---

## 6. `/pipeline`

### Current result

**24/26 checks passed**

### Grader priority

Fix event ordering and state transitions.

Audit:

- event processing order
- attempts
- starts
- completions
- retryable failures
- terminal failures
- ignored events
- transition conflicts
- evidence conflicts
- rollback
- session isolation
- durable readback

Important valid sequence:

```text
started(1)
→ retryable_failed(1)
→ started(2)
→ succeeded(2)
```

Audit all transitions against `spec.md`.

Verify:

- lower attempts are ignored where required
- invalid transitions produce the correct conflict
- conflicting successful evidence produces `EVIDENCE_CONFLICT`
- terminal failures block later events appropriately
- valid events are processed in input order
- invalid/ignored events do not incorrectly consume IDs
- exact replay is ignored
- same event ID with different canonical JSON causes `EVENT_ID_CONFLICT`
- a conflicting batch rolls back atomically
- sessions remain isolated

---

## 7. `/verify-bundle`

### Current result

**28/34 checks passed**

### Grader priority

Fix inventory and artifact integrity first:

- required UTF-8 files
- inventory bytes
- inventory SHA-256 values
- inventory ordering
- extra files
- unsafe extensions
- artifact digest binding

Then review:

- lineage
- evaluation binding
- serialization
- model-card consistency

### Required files

Exactly:

- `README.md`
- `training_manifest.json`
- `evaluation.json`
- `inventory.json`
- `adapter_model.safetensors`
- `adapter_config.json`

### Inventory audit

Recompute:

- exact UTF-8 byte lengths
- lowercase SHA-256
- UTF-8 filename ordering
- compact JSON representation
- inventory digest

Do not trust supplied byte counts or hashes.

Detect all extra files.

Unsafe extensions:

- `.bin`
- `.pt`
- `.pth`
- `.pkl`
- `.pickle`

### Artifact audit

Verify:

- adapter config
- training manifest
- immutable base revision
- model artifact digest
- evaluation artifact digest
- evaluation binding
- required slices
- model-card consistency

---

# Regression strategy

For every grader failure:

1. Reproduce it locally.
2. Write a regression test.
3. Fix the implementation.
4. Run the regression test.
5. Run the endpoint's broader test suite.
6. Run the complete suite.

Do not rely only on the public grader.

---

# Preservation rule

The current implementation already passes many checks.

Make targeted fixes.

Do not rewrite correct logic unnecessarily.

After every change, make sure previously passing behavior still passes.

---

# Final verification

Before deployment:

- run the full test suite
- verify all seven endpoints
- verify exact response shapes
- verify required 400 responses
- verify required 409 responses
- verify deterministic output
- verify state persistence
- verify Render startup

The intended result is to increase the score without violating `spec.md`.
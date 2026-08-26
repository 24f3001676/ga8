"""Assignment 7: Deterministic model-bundle and model-card verifier."""

import json
import math

from app.core.errors import InvalidInput
from app.core.hashing import cj, is_non_negative_safe_int, sha256_hex
from app.core.ordering import reason_codes, utf8_key
from app.core.validation import is_hex40

REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]

UNSAFE_EXTS = (".bin", ".pt", ".pth", ".pkl", ".pickle")
MANIFEST_STRING_FIELDS = [
    "task",
    "datasetDigest",
    "codeDigest",
    "trainingConfigDigest",
    "modelArtifactDigest",
    "evaluationArtifactDigest",
]
MARKER_PREFIX = "<!-- tds-model-card"


def _parse_json_file(content: str):
    try:
        return json.loads(content)
    except Exception:
        return "__PARSE_ERROR__"


def _inventory_recomputed(files: dict):
    """Build the recomputed inventory array from supplied files (minus inventory.json)."""
    entries = []
    for name, content in files.items():
        if name == "inventory.json":
            continue
        if not isinstance(content, str):
            return None
        data = content.encode("utf-8")
        entries.append({"name": name, "bytes": len(data), "sha256": sha256_hex(data)})
    entries.sort(key=lambda e: utf8_key(e["name"]))
    return entries


def _extract_markers(text: str):
    """Return list of payload strings between marker prefix and next '-->'."""
    payloads = []
    idx = 0
    while True:
        start = text.find(MARKER_PREFIX, idx)
        if start == -1:
            break
        payload_start = start + len(MARKER_PREFIX)
        end = text.find("-->", payload_start)
        if end == -1:
            payloads.append(text[payload_start:])
            break
        payloads.append(text[payload_start:end])
        idx = end + 3
    return payloads


def handle(body) -> dict:
    if not isinstance(body, dict):
        raise InvalidInput()
    policy = body.get("policy")
    if not isinstance(policy, dict):
        raise InvalidInput()
    files = body.get("files")
    if not isinstance(files, dict):
        raise InvalidInput()

    violations = set()

    # ---- policy ----
    policy_ok = True
    slices_required = policy.get("requiredSlices")
    if (
        not isinstance(slices_required, list)
        or len(slices_required) == 0
        or any(not isinstance(s, str) or s == "" for s in slices_required)
        or len(set(slices_required)) != len(slices_required)
    ):
        policy_ok = False
    for k in ("license", "intendedUse", "limitations"):
        v = policy.get(k)
        if not isinstance(v, str) or v == "":
            policy_ok = False
    if not policy_ok:
        violations.add("INVALID_POLICY")

    # ---- files presence / types ----
    file_contents = {}
    for name in REQUIRED_FILES:
        if name not in files:
            violations.add(f"MISSING_FILE:{name}")
            continue
        val = files[name]
        if not isinstance(val, str):
            violations.add(f"INVALID_FILE:{name}")
        else:
            file_contents[name] = val
    for name, val in files.items():
        if not isinstance(val, str):
            if name not in REQUIRED_FILES:
                violations.add(f"INVALID_FILE:{name}")

    # ---- unsafe weights ----
    for name in files.keys():
        low = name.lower() if isinstance(name, str) else ""
        if any(low.endswith(ext) for ext in UNSAFE_EXTS):
            violations.add("UNSAFE_WEIGHTS")

    # ---- inventory ----
    recomputed = None
    inv_digest_out = None
    if "inventory.json" in file_contents:
        parsed_inv = _parse_json_file(file_contents["inventory.json"])
        if parsed_inv == "__PARSE_ERROR__":
            violations.add("INVALID_JSON:inventory.json")
        elif not isinstance(parsed_inv, list):
            violations.add("INVENTORY_MISMATCH")
        else:
            recomputed = _inventory_recomputed(files)
            if recomputed is None:
                violations.add("INVENTORY_MISMATCH")
            else:
                inv_digest_out = sha256_hex(cj(recomputed).encode("utf-8"))
                expected_shape = [
                    {"name": e["name"], "bytes": e["bytes"], "sha256": e["sha256"]}
                    for e in recomputed
                ]
                listed_names = []
                shape_ok = True
                for entry in parsed_inv:
                    if not isinstance(entry, dict) or set(entry.keys()) != {"name", "bytes", "sha256"}:
                        shape_ok = False
                        break
                    if not isinstance(entry["name"], str):
                        shape_ok = False
                        break
                    listed_names.append(entry["name"])
                if not shape_ok:
                    violations.add("INVENTORY_MISMATCH")
                else:
                    if listed_names != sorted(listed_names, key=utf8_key):
                        violations.add("INVENTORY_MISMATCH")
                    if sorted(listed_names, key=utf8_key) != sorted(
                        [e["name"] for e in recomputed], key=utf8_key
                    ):
                        violations.add("INVENTORY_MISMATCH")
                    elif parsed_inv != expected_shape:
                        violations.add("INVENTORY_MISMATCH")
                    tracked = set(listed_names)
                    for fname in files.keys():
                        if fname != "inventory.json" and fname not in tracked:
                            violations.add("UNTRACKED_FILE")

    # ---- adapter config ----
    if "adapter_config.json" in file_contents:
        parsed_cfg = _parse_json_file(file_contents["adapter_config.json"])
        if parsed_cfg == "__PARSE_ERROR__":
            violations.add("INVALID_JSON:adapter_config.json")
        elif not isinstance(parsed_cfg, dict):
            violations.add("INVALID_ADAPTER_CONFIG")
        else:
            r_val = parsed_cfg.get("r")
            targets = parsed_cfg.get("target_modules")
            cfg_ok = is_non_negative_safe_int(r_val) and r_val >= 1
            cfg_ok = cfg_ok and isinstance(targets, list) and len(targets) > 0
            cfg_ok = cfg_ok and all(isinstance(t, str) for t in targets)
            cfg_ok = cfg_ok and len(set(targets)) == len(targets) if isinstance(targets, list) else False
            if not cfg_ok:
                violations.add("INVALID_ADAPTER_CONFIG")

    # ---- training manifest ----
    manifest = None
    manifest_parse_failed = False
    if "training_manifest.json" in file_contents:
        parsed_man = _parse_json_file(file_contents["training_manifest.json"])
        if parsed_man == "__PARSE_ERROR__":
            violations.add("INVALID_JSON:training_manifest.json")
            manifest_parse_failed = True
        elif not isinstance(parsed_man, dict):
            violations.add("INVALID_TRAINING_MANIFEST")
        else:
            manifest = parsed_man
            base_rev = manifest.get("baseRevision")
            if base_rev is None:
                violations.add("MISSING_MANIFEST_FIELD:baseRevision")
            elif not (isinstance(base_rev, str) and is_hex40(base_rev)):
                violations.add("MUTABLE_BASE_REVISION")
            for field in MANIFEST_STRING_FIELDS:
                v = manifest.get(field)
                if v is None and field not in manifest:
                    violations.add(f"MISSING_MANIFEST_FIELD:{field}")
                elif not isinstance(v, str) or v == "":
                    violations.add("INVALID_TRAINING_MANIFEST")

    # ---- artifact digest binding ----
    model_digest = None
    eval_bytes_digest = None
    if "adapter_model.safetensors" in file_contents:
        model_digest = sha256_hex(file_contents["adapter_model.safetensors"].encode("utf-8"))
    if "evaluation.json" in file_contents:
        eval_bytes_digest = sha256_hex(file_contents["evaluation.json"].encode("utf-8"))

    evaluation = None
    if "evaluation.json" in file_contents:
        parsed_eval = _parse_json_file(file_contents["evaluation.json"])
        if parsed_eval == "__PARSE_ERROR__":
            violations.add("INVALID_JSON:evaluation.json")
        elif not isinstance(parsed_eval, dict):
            violations.add("INVALID_EVALUATION")
        else:
            evaluation = parsed_eval

    def finite01(v) -> bool:
        return (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            and 0.0 <= float(v) <= 1.0
        )

    if manifest is not None and not manifest_parse_failed:
        if manifest.get("modelArtifactDigest") != model_digest:
            violations.add("MODEL_ARTIFACT_MISMATCH")
        if manifest.get("evaluationArtifactDigest") != eval_bytes_digest:
            violations.add("EVALUATION_ARTIFACT_MISMATCH")

    if evaluation is not None:
        bound = evaluation.get("modelArtifactDigest")
        if bound is None or bound != model_digest:
            violations.add("EVALUATION_DIGEST_MISMATCH")
        aggregate = evaluation.get("aggregate")
        if aggregate is None and "aggregate" not in evaluation:
            violations.add("INVALID_AGGREGATE")
        elif not finite01(aggregate):
            violations.add("INVALID_AGGREGATE")
        slices_obj = evaluation.get("slices")
        slices_dict = slices_obj if isinstance(slices_obj, dict) else {}
        for s in slices_required:
            if s not in slices_dict:
                violations.add(f"MISSING_SLICE:{s}")
            elif not finite01(slices_dict[s]):
                violations.add(f"SLICE_RANGE:{s}")

    # ---- model card ----
    card_payloads = []
    if "README.md" in file_contents:
        card_payloads = _extract_markers(file_contents["README.md"])

    card = None
    if len(card_payloads) == 0:
        violations.add("MODEL_CARD_COUNT")
        violations.add("MISSING_MODEL_CARD")
    elif len(card_payloads) > 1:
        violations.add("MODEL_CARD_COUNT")
    else:
        raw = card_payloads[0]
        try:
            parsed_card = json.loads(raw.strip())
        except Exception:
            parsed_card = None
        if not isinstance(parsed_card, dict):
            violations.add("INVALID_MODEL_CARD")
        else:
            card = parsed_card

    if card is not None and manifest is not None and not manifest_parse_failed:
        mismatch = False
        if card.get("task") != manifest.get("task"):
            mismatch = True
        if card.get("baseRevision") != manifest.get("baseRevision"):
            mismatch = True
        if card.get("datasetDigest") != manifest.get("datasetDigest"):
            mismatch = True
        if card.get("modelArtifactDigest") != manifest.get("modelArtifactDigest"):
            mismatch = True
        if policy_ok:
            if card.get("license") != policy.get("license"):
                mismatch = True
            if card.get("intendedUse") != policy.get("intendedUse"):
                mismatch = True
            if card.get("limitations") != policy.get("limitations"):
                mismatch = True
        if mismatch:
            violations.add("MODEL_CARD_MISMATCH")

    decision = "admit" if not violations else "reject"
    return {
        "decision": decision,
        "violations": reason_codes(violations),
        "inventoryDigest": inv_digest_out,
    }

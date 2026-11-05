"""Eight balanced JSON field-order rewrites with exactly preserved attributes.

These are serialization perturbations, not inferred new preference modes. Every
field occupies each token-block position twice across the eight candidates.
"""

from __future__ import annotations

import json


ATTRIBUTE_KEYS = ("Color", "Material", "Design features", "Clothing Fashion Style")


def build_order_rewrites(canonical_text: str) -> list[dict]:
    payload = json.loads(canonical_text)
    if set(payload) != set(ATTRIBUTE_KEYS):
        raise ValueError(f"Expected exactly the four preference attributes: {ATTRIBUTE_KEYS}")
    keys = list(ATTRIBUTE_KEYS)
    reverse = list(reversed(keys))
    orders = [keys[i:] + keys[:i] for i in range(4)]
    orders.extend(reverse[i:] + reverse[:i] for i in range(4))
    rewrites = []
    for rewrite_id, order in enumerate(orders):
        text = json.dumps({key: payload[key] for key in order}, ensure_ascii=False, separators=(",", ": "))
        if json.loads(text) != payload:
            raise ValueError("A rewrite changed the original preference values")
        rewrites.append({"rewrite_id": rewrite_id, "key_order": order, "preference_text": text})
    if len({row["preference_text"] for row in rewrites}) != 8:
        raise ValueError("Expected eight distinct serializations")
    return rewrites


def build_history_rewrites(category_preference: dict, item_info: dict) -> list[dict]:
    """Use co-occurring attributes of eight past items as fixed hypotheses.

    This specializes an aggregate history into item-level modes; it is not
    meaning-preserving paraphrasing. No generated image or target item enters
    this builder. Fail rather than fabricate missing or duplicate modes.
    """
    rewrites, seen = [], set()
    for source_id in category_preference.get("iiids", []):
        elements = item_info.get(int(source_id), {}).get("elements", {})
        if any(not isinstance(elements.get(key), str) or not elements[key].strip()
               for key in ATTRIBUTE_KEYS):
            continue
        payload = {key: elements[key].strip() for key in ATTRIBUTE_KEYS}
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ": "))
        if text in seen:
            continue
        seen.add(text)
        rewrites.append({"rewrite_id": len(rewrites), "key_order": list(ATTRIBUTE_KEYS),
                         "preference_text": text, "source_history_item_id": int(source_id)})
        if len(rewrites) == 8:
            return rewrites
    raise ValueError(f"Need eight distinct complete historical attribute modes; found {len(rewrites)}")

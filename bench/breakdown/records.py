"""Metadata-only extraction of a selected layer's existing nested timers."""
from __future__ import annotations


def parse_profile_layers(spec: str, *, count: int) -> frozenset[int]:
    selected: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(value) for value in part.split("-", 1))
            if start > stop:
                raise ValueError("profile layer range must be ascending")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(part))
    if any(index < 0 or index >= int(count) for index in selected):
        raise ValueError(f"profile layers must lie in [0, {int(count)})")
    return frozenset(selected)


def nested_stage_records(nested: dict, layer_idx: int, selected: frozenset[int]) -> dict:
    if int(layer_idx) not in selected:
        return {}
    return {
        output: {str(name): round(float(value), 6)
                 for name, value in nested.get(source, {}).items()
                 if isinstance(value, (int, float))}
        for output, source in (("decode_attention_stages", "decode_attention"),
                               ("decode_mlp_stages", "mlp"))
    }

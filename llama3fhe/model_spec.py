"""Static model shapes the Couplet packing is instantiated for.

Everything in the Prefill / Decode graphs derives its layouts from one of
these specs: the feature-major (FM) hidden carriers, the attention pair
layout and the MLP intermediate width.  The production (paper) model is
Llama-3-8B; Llama-3.2-1B is the small-model comparison target
(``head_dim 64 < seq_len 128`` exercises the repeated-C / Delta-grid design).
Llama-3.2-3B needs a separate, semantics-preserving padded checkpoint: its
native 3072-dimensional hidden state and 24 Q heads do not fit the current
power-of-two feature-major carrier geometry.  The padded runtime checkpoint
uses 4096 physical hidden channels and 32 physical Q heads; the extra channels
and heads are arranged so that its logical 3B outputs are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RopeScaling:
    """HF ``rope_scaling`` of type ``llama3`` (Llama 3.1 / 3.2)."""

    factor: float = 32.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float
    rms_eps: float = 1e-5
    rope_scaling: RopeScaling | None = None
    tie_word_embeddings: bool = False
    #: Original model shape before a semantics-preserving physical padding.
    #: Ordinary specs set these to their physical values in ``__post_init__``.
    source_hidden_dim: int | None = None
    source_query_heads: int | None = None

    def __post_init__(self) -> None:
        source_hidden = int(self.hidden_dim if self.source_hidden_dim is None else self.source_hidden_dim)
        source_heads = int(self.query_heads if self.source_query_heads is None else self.source_query_heads)
        object.__setattr__(self, "source_hidden_dim", source_hidden)
        object.__setattr__(self, "source_query_heads", source_heads)
        if source_hidden <= 0 or source_heads <= 0:
            raise ValueError(f"{self.name}: source hidden dimension and query heads must be positive.")
        if int(self.query_heads) * int(self.head_dim) != int(self.hidden_dim):
            raise ValueError(
                f"{self.name}: query_heads*head_dim={self.query_heads * self.head_dim} "
                f"!= hidden_dim={self.hidden_dim}."
            )
        if int(self.query_heads) % int(self.key_value_heads):
            raise ValueError(f"{self.name}: query_heads must be a multiple of key_value_heads.")
        if source_heads * int(self.head_dim) != source_hidden:
            raise ValueError(f"{self.name}: source_query_heads*head_dim != source_hidden_dim.")
        if source_hidden > int(self.hidden_dim) or source_heads > int(self.query_heads):
            raise ValueError(f"{self.name}: source geometry cannot exceed physical geometry.")
        if source_heads % int(self.key_value_heads):
            raise ValueError(f"{self.name}: source query heads must be a multiple of KV heads.")

    def physical_query_head(self, source_head: int) -> int:
        """Map an original Q head to the padded head in its GQA group.

        Padded heads are inserted at the end of each KV group.  Ordinary
        models have an identity mapping.  The 3B mapping uses three live
        heads per source group and one unused physical head per group.
        """

        source_head = int(source_head)
        if not 0 <= source_head < int(self.source_query_heads):
            raise IndexError(f"source Q head {source_head} is outside [0, {self.source_query_heads}).")
        source_group = int(self.source_query_heads) // int(self.key_value_heads)
        physical_group = int(self.query_heads) // int(self.key_value_heads)
        return (source_head // source_group) * physical_group + source_head % source_group

    @property
    def kv_dim(self) -> int:
        return int(self.key_value_heads) * int(self.head_dim)

    @property
    def gqa_ratio(self) -> int:
        return int(self.query_heads) // int(self.key_value_heads)

    @property
    def query_pair_count(self) -> int:
        return int(self.query_heads) // 2

    def fm_cipher_count(self, *, seq_len: int = 128, slots: int = 32768) -> int:
        """Feature-major hidden ciphertexts for one ``seq_len`` prompt."""

        tokens_per_cipher = int(slots) // int(self.hidden_dim)
        if int(seq_len) % tokens_per_cipher:
            raise ValueError(
                f"{self.name}: seq_len={seq_len} is not a multiple of {tokens_per_cipher} tokens per ciphertext."
            )
        return int(seq_len) // tokens_per_cipher

    def prefill_layouts(self, *, seq_len: int = 128, slots: int = 32768):
        """``(FeatureMajorPrefillLayout, AttentionPairLayout)`` for this model."""

        from llama3fhe.layouts.attention import AttentionPairLayout
        from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

        feature = FeatureMajorPrefillLayout(
            seq_len=int(seq_len), hidden_dim=int(self.hidden_dim), slots=int(slots)
        )
        attention = AttentionPairLayout(
            seq_len=int(seq_len),
            head_dim=int(self.head_dim),
            query_heads=int(self.query_heads),
            key_value_heads=int(self.key_value_heads),
            slots=int(slots),
        )
        return feature, attention

    def layer_weight_shapes(self) -> dict[str, tuple[int, ...]]:
        """HF ``model.layers.N.*`` tensor shapes (out, in)."""

        hidden = int(self.hidden_dim)
        inter = int(self.intermediate_dim)
        kv = self.kv_dim
        return {
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
            "self_attn.q_proj.weight": (hidden, hidden),
            "self_attn.k_proj.weight": (kv, hidden),
            "self_attn.v_proj.weight": (kv, hidden),
            "self_attn.o_proj.weight": (hidden, hidden),
            "mlp.gate_proj.weight": (inter, hidden),
            "mlp.up_proj.weight": (inter, hidden),
            "mlp.down_proj.weight": (hidden, inter),
        }


LLAMA3_8B = ModelSpec(
    name="llama3-8b",
    hidden_dim=4096,
    intermediate_dim=14336,
    num_layers=32,
    query_heads=32,
    key_value_heads=8,
    head_dim=128,
    vocab_size=128256,
    rope_theta=500000.0,
    rms_eps=1e-5,
    rope_scaling=None,
    tie_word_embeddings=False,
)

LLAMA32_1B = ModelSpec(
    name="llama3.2-1b",
    hidden_dim=2048,
    intermediate_dim=8192,
    num_layers=16,
    query_heads=32,
    key_value_heads=8,
    head_dim=64,
    vocab_size=128256,
    rope_theta=500000.0,
    rms_eps=1e-5,
    rope_scaling=RopeScaling(
        factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ),
    tie_word_embeddings=True,
)

# The physical checkpoint is created by tools/convert/llama32/prepare_llama32_3b_padded.py.
# Its RMSNorm epsilon is scaled by 3072/4096; its folded norm weights compensate
# for the larger physical mean-square denominator.  This spec must never be
# applied directly to the official 3072-wide Hugging Face checkpoint.
LLAMA32_3B_PADDED = ModelSpec(
    name="llama3.2-3b-padded",
    hidden_dim=4096,
    intermediate_dim=8192,
    num_layers=28,
    query_heads=32,
    key_value_heads=8,
    head_dim=128,
    vocab_size=128256,
    rope_theta=500000.0,
    rms_eps=1e-5 * (3072 / 4096),
    rope_scaling=RopeScaling(
        factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ),
    tie_word_embeddings=False,
    source_hidden_dim=3072,
    source_query_heads=24,
)

MODEL_SPECS: dict[str, ModelSpec] = {
    LLAMA3_8B.name: LLAMA3_8B,
    "llama3-8b": LLAMA3_8B,
    "8b": LLAMA3_8B,
    LLAMA32_1B.name: LLAMA32_1B,
    "llama3.2-1b": LLAMA32_1B,
    "llama32-1b": LLAMA32_1B,
    "1b": LLAMA32_1B,
    LLAMA32_3B_PADDED.name: LLAMA32_3B_PADDED,
    "llama32-3b-padded": LLAMA32_3B_PADDED,
    "3b-padded": LLAMA32_3B_PADDED,
}


def model_spec_by_name(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[str(name).lower()]
    except KeyError as exc:
        raise KeyError(
            f"unknown model spec {name!r}; choose from {sorted(set(MODEL_SPECS))}"
        ) from exc


def model_spec_from_hf_config(config: dict, *, name: str | None = None) -> ModelSpec:
    """Build a spec from a HF ``config.json`` dict (checks the known shapes)."""

    scaling = config.get("rope_scaling")
    rope_scaling = None
    if scaling and str(scaling.get("rope_type", scaling.get("type", ""))).lower() == "llama3":
        rope_scaling = RopeScaling(
            factor=float(scaling["factor"]),
            low_freq_factor=float(scaling["low_freq_factor"]),
            high_freq_factor=float(scaling["high_freq_factor"]),
            original_max_position_embeddings=int(scaling["original_max_position_embeddings"]),
        )
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    return ModelSpec(
        name=str(name or config.get("_name_or_path", "hf-model")),
        hidden_dim=hidden,
        intermediate_dim=int(config["intermediate_size"]),
        num_layers=int(config["num_hidden_layers"]),
        query_heads=heads,
        key_value_heads=int(config.get("num_key_value_heads", heads)),
        head_dim=int(config.get("head_dim", hidden // heads)),
        vocab_size=int(config["vocab_size"]),
        rope_theta=float(config.get("rope_theta", 10000.0)),
        rms_eps=float(config.get("rms_norm_eps", 1e-5)),
        rope_scaling=rope_scaling,
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        source_hidden_dim=(
            None if not isinstance(config.get("odin_padding"), dict)
            else config["odin_padding"].get("source_hidden_dim")
        ),
        source_query_heads=(
            None if not isinstance(config.get("odin_padding"), dict)
            else config["odin_padding"].get("source_query_heads")
        ),
    )


__all__ = [
    "LLAMA32_1B",
    "LLAMA32_3B_PADDED",
    "LLAMA3_8B",
    "MODEL_SPECS",
    "ModelSpec",
    "RopeScaling",
    "model_spec_by_name",
    "model_spec_from_hf_config",
]

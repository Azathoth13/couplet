"""Slot layout of activations produced by the coefficient projections.

A tensor of F features over 128 tokens is a batch of 2 * F / 512 real slot
ciphertexts ordered (shard 0 lo, shard 0 hi, shard 1 lo, ...). Shard s holds
features 512 s ... 512 s + 511; lo holds tokens 0..63 and hi tokens 64..127.
Inside one ciphertext, feature r of token t sits at slot bitrev9(r) * 64 + bitrev6(t % 64)
(the HalfBoot output of coefficient r + 512 t).

For a head of 128 channels at local position j (r = 128 j + c):
slot = (bitrev7(c) * 4 + bitrev2(j)) * 64 + lane.
The 256-slot "block" index bitrev7(c) therefore enumerates channels.
"""
from __future__ import annotations

import math

import numpy as np

SLOTS = 1 << 15
LANES = 64
WIDTH = 512           # features per ciphertext
TOKENS = 128
HEAD_DIM = 128
BLOCK = 256           # slots per channel block (4 head phases x 64 lanes)


def bitrev(value, bits):
    value = np.asarray(value)
    out = np.zeros_like(value)
    for b in range(bits):
        out |= ((value >> b) & 1) << (bits - 1 - b)
    return out


LANE_TOKEN = bitrev(np.arange(LANES), 6)            # token (mod 64) held by lane p
FEATURE_SLOT = bitrev(np.arange(WIDTH), 9) * LANES  # base slot of feature r


def slot_of(r, t):
    """Slot of feature r (0..511) for token t (0..63) inside one ciphertext."""
    return FEATURE_SLOT[np.asarray(r)] + bitrev(np.asarray(t), 6)


_SLOT_INDEX = (FEATURE_SLOT[None, :] + bitrev(np.arange(LANES), 6)[:, None])   # [64 tokens, 512]


def pack(values):
    """[128, F] -> [2F/512, SLOTS] slot vectors in batch order."""
    values = np.asarray(values, dtype=np.float64)
    tokens, features = values.shape
    assert tokens == TOKENS and features % WIDTH == 0
    out = np.zeros((2 * features // WIDTH, SLOTS))
    for s in range(features // WIDTH):
        for h in range(2):
            out[2 * s + h, _SLOT_INDEX] = values[64 * h:64 * h + 64, WIDTH * s:WIDTH * s + WIDTH]
    return out


def unpack(slots):
    """[2F/512, SLOTS] -> [128, F]."""
    slots = np.asarray(slots)
    shards = slots.shape[0] // 2
    out = np.zeros((TOKENS, shards * WIDTH))
    for s in range(shards):
        for h in range(2):
            out[64 * h:64 * h + 64, WIDTH * s:WIDTH * s + WIDTH] = slots[2 * s + h][_SLOT_INDEX]
    return out


def feature_vector(per_feature, shard):
    """A public per-feature vector for one shard, replicated over every token lane."""
    v = np.zeros(SLOTS)
    for r in range(WIDTH):
        v[FEATURE_SLOT[r]:FEATURE_SLOT[r] + LANES] = per_feature[WIDTH * shard + r]
    return v


# ---- head geometry ----------------------------------------------------------------------
def rope_frequencies(head_dim, theta, rope_scaling=None):
    """Per-pair RoPE angular frequencies ``theta ** (-2 i / head_dim)``; ``rope_scaling`` is the
    HF ``llama3`` dict (factor, low_freq_factor, high_freq_factor, original_max_position_embeddings)
    used by Llama-3.1/3.2, applied exactly as transformers' ``_compute_llama3_parameters``."""
    inv = np.asarray(float(theta) ** (-2.0 * np.arange(head_dim // 2) / head_dim), dtype=np.float64)
    if not rope_scaling:
        return inv
    factor = float(rope_scaling["factor"])
    low, high = float(rope_scaling["low_freq_factor"]), float(rope_scaling["high_freq_factor"])
    old = float(rope_scaling["original_max_position_embeddings"])
    wavelen = 2.0 * np.pi / inv
    scaled = np.where(wavelen > old / low, inv / factor, inv)
    smooth = (old / wavelen - low) / (high - low)
    smoothed = (1.0 - smooth) * inv / factor + smooth * inv
    medium = ~(wavelen < old / high) & ~(wavelen > old / low)
    return np.where(medium, smoothed, scaled)


class HeadGeometry:
    """Slot geometry of one attention head family inside the 512-feature shard layout.

    Feature ``r = head_dim * j + c`` (local head ``j`` of a shard, channel ``c``) sits at base slot
    ``bitrev9(r) * 64 = (bitrev(c) * hps + bitrev(j)) * 64``: channel blocks of ``block = hps * 64``
    slots enumerate the channels, the ``hps = 512 / head_dim`` phases of a block enumerate the
    shard's heads. Llama-3-8B: head_dim 128, 4 heads per shard, 256-slot blocks, 128 blocks.
    Llama-3.2-1B: head_dim 64, 8 heads per shard, 512-slot blocks, 64 blocks.

    Score ciphertexts hold ``[key block][phase][lane]`` for one Q item (shard, query half); with
    ``head_dim >= 128`` both key halves share one ciphertext (``kh_per_ct = 2``), with head_dim 64
    each key half has its own (``kh_per_ct = 1``). ``items = 2 * shards`` Q items of ``block`` slots
    form the row ("D") layout period ``item_period`` (4096 for both models).
    """

    def __init__(self, head_dim=128, heads=32, kv_heads=8):
        head_dim, heads, kv_heads = int(head_dim), int(heads), int(kv_heads)
        if WIDTH % head_dim or head_dim % LANES or heads % kv_heads:
            raise ValueError("head_dim must divide 512 and be a multiple of 64; kv_heads must divide heads")
        if (heads * head_dim) % WIDTH or (kv_heads * head_dim) % WIDTH:
            raise ValueError("Q and KV widths must be multiples of 512 features")
        self.head_dim, self.heads, self.kv_heads = head_dim, heads, kv_heads
        self.gqa = heads // kv_heads
        self.hps = WIDTH // head_dim                     # heads per shard
        self.block = self.hps * LANES                    # slots per channel block
        self.channel_bits = int(math.log2(head_dim))
        self.phase_bits = int(math.log2(self.hps))
        self.phase = bitrev(np.arange(self.hps), self.phase_bits)   # slot phase of local head j
        self.shards = heads * head_dim // WIDTH          # Q shards
        self.kv_shards = kv_heads * head_dim // WIDTH
        self.items = 2 * self.shards                     # Q items (shard, query half)
        # Row reductions require a power-of-two slot period.  Native 3B has six
        # Q shards (twelve items); leave four empty items in each 4096-slot row
        # period, while keeping the actual Q/score batches at six/twelve.
        self.item_capacity = 1 << (self.items - 1).bit_length()
        self.item_period = self.item_capacity * self.block
        self.kh_per_ct = max(1, head_dim // 64)          # key halves per score ciphertext
        if self.kh_per_ct > 2:
            raise ValueError("head_dim > 128 is not supported")
        if SLOTS % self.item_period:
            raise ValueError("Q items must tile the slots")
        slot = np.arange(SLOTS)
        self.slot_block = slot // self.block
        self.slot_phase = (slot // LANES) % self.hps
        self.slot_lane = slot % LANES
        self.channel_of_block = bitrev(np.arange(head_dim), self.channel_bits)
        self.slot_channel = self.channel_of_block[self.slot_block]
        self.slot_local = self.phase[self.slot_phase]    # local head of a slot (bitrev is an involution)
        if self.gqa % self.hps and self.hps % self.gqa and (heads, kv_heads, head_dim) != (24, 8, 128):
            raise ValueError("GQA groups must align with the shard heads")

    def kv_head(self, shard, local):
        """KV head read by local head ``local`` of Q shard ``shard``."""
        return (self.hps * int(shard) + int(local)) // self.gqa

    def kv_head_of_slot(self, shard):
        return (self.hps * int(shard) + self.slot_local) // self.gqa

    @property
    def entries(self):
        """Private score ciphertexts: (shard, query half, key halves held)."""
        out = []
        for s in range(self.shards):
            if self.kh_per_ct == 2:
                out += [(s, 0, (0, 1)), (s, 1, (0, 1))]
            else:
                out += [(s, 0, (0,)), (s, 1, (0,)), (s, 1, (1,))]
        return tuple(out)

    @staticmethod
    def item(entry):
        s, qh, _ = entry
        return 2 * s + qh

    @staticmethod
    def kh_offset(entry, kh):
        """First key block of key half ``kh`` inside the entry's score ciphertext."""
        return 64 * entry[2].index(kh)


def head_rms_gamma(gamma, geometry):
    """Per-slot QK-Norm gamma.

    Channel ``c`` of every local head in a 512-feature shard sits in block
    ``bitrev(c)``.  Qwen's gamma is one vector of length ``head_dim``, shared
    by every head, so the same slot vector multiplies Q and K.
    """

    gamma = np.asarray(gamma, dtype=np.float64).reshape(int(geometry.head_dim))
    return gamma[geometry.slot_channel]

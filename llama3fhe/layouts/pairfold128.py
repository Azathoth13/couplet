from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _real_array(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values)
    if np.iscomplexobj(values):
        raise TypeError(f"{name} must be real, got {values.dtype}.")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    return values


def _is_pow2(value: int) -> bool:
    return value >= 1 and value & (value - 1) == 0


@dataclass(frozen=True)
class PairFoldLayout:
    """Decode-native pair/lane-interleaved folded layout (general geometry).

    One cache ciphertext stores ``key_value_heads / key_value_groups`` KV heads
    as ``pair_blocks`` head-pair blocks (two heads per pair on the two lanes),
    ``fold_dim = head_dim / 2`` folded (complex) features and ``seq_len``
    tokens.  The temporary Decode QKV carrier keeps the same pair/lane
    ownership while multiplexing K, V and the ``gqa_ratio`` query offsets of
    every group into ``2 + gqa_ratio`` residues per group.

    Every index below is derived from the geometry; for the production
    Llama-3-8B shape (``PairFold128Layout``) they reduce to the historical
    constants (page block 16384, channel row 32, residue stride 16, ...).
    Keeping this separate from :class:`AttentionPairLayout` prevents legacy
    Prefill C-form assumptions from leaking into the Decode path.
    """

    seq_len: int = 128
    head_dim: int = 128
    query_heads: int = 32
    key_value_heads: int = 8
    slots: int = 32768
    native_token_lanes: int = 8
    key_value_groups: int = 2

    def __post_init__(self) -> None:
        seq_len, head_dim = int(self.seq_len), int(self.head_dim)
        q_heads, kv_heads = int(self.query_heads), int(self.key_value_heads)
        slots, groups = int(self.slots), int(self.key_value_groups)
        lanes = int(self.native_token_lanes)
        if not (_is_pow2(seq_len) and _is_pow2(head_dim) and _is_pow2(slots)):
            raise ValueError("PairFold requires power-of-two seq_len, head_dim and slots.")
        if head_dim < 2:
            raise ValueError("PairFold folds head_dim/2 complex features; head_dim must be >= 2.")
        if kv_heads <= 0 or q_heads % kv_heads:
            raise ValueError(
                f"query_heads={q_heads} must be a multiple of key_value_heads={kv_heads}."
            )
        if groups <= 0 or kv_heads % groups:
            raise ValueError(
                f"key_value_heads={kv_heads} must split into key_value_groups={groups}."
            )
        if (kv_heads // groups) % self.head_lanes:
            raise ValueError(
                f"{kv_heads // groups} KV heads per group do not form {self.head_lanes}-lane pairs."
            )
        if lanes * self.residue_slots != seq_len:
            raise ValueError(
                f"native_token_lanes={lanes} must equal seq_len / residue_slots = "
                f"{seq_len} / {self.residue_slots}."
            )
        if self.pair_blocks * self.page_block > slots:
            raise ValueError(
                f"{self.pair_blocks} pair pages of {self.page_block} slots exceed {slots} slots."
            )
        if self.pair_blocks * self.fold_dim * self.channel_row > self.carrier_dimension:
            raise ValueError("carrier channels exceed the carrier dimension.")
        if 2 * seq_len * groups * self.pair_blocks * self.gqa_ratio > slots:
            raise ValueError("score pairs exceed the ciphertext.")
        if slots % self.score_payload_slots:
            raise ValueError("compact score payload must divide the ciphertext.")

    @property
    def fold_dim(self) -> int:
        return int(self.head_dim) // 2

    @property
    def gqa_ratio(self) -> int:
        return int(self.query_heads) // int(self.key_value_heads)

    @property
    def pair_blocks(self) -> int:
        return self.key_value_heads_per_group // self.head_lanes

    @property
    def head_lanes(self) -> int:
        return 2

    @property
    def key_value_heads_per_group(self) -> int:
        return int(self.key_value_heads) // int(self.key_value_groups)

    @property
    def carrier_dimension(self) -> int:
        return int(self.slots) // int(self.native_token_lanes)

    @property
    def residues_per_group(self) -> int:
        """K, V and the GQA query offsets of one group."""
        return 2 + self.gqa_ratio

    @property
    def active_residue_count(self) -> int:
        return int(self.key_value_groups) * self.residues_per_group

    @property
    def residue_slots(self) -> int:
        """Residue stride of a carrier channel row (next power of two of the active residues)."""
        count = self.active_residue_count
        return 1 << (count - 1).bit_length() if count > 1 else 1

    @property
    def channel_row(self) -> int:
        """Carrier channels per folded feature (residue slots x lanes)."""
        return self.residue_slots * self.head_lanes

    @property
    def page_block(self) -> int:
        """Slots of one head-pair page: fold_dim rows of seq_len lane-interleaved tokens."""
        return self.head_lanes * self.fold_dim * int(self.seq_len)

    @property
    def score_payload_slots(self) -> int:
        return self.head_lanes * int(self.seq_len) * int(self.head_dim)

    @property
    def score_payload_repetitions(self) -> int:
        return int(self.slots) // self.score_payload_slots

    @property
    def active_channel_count(self) -> int:
        return self.pair_blocks * self.fold_dim * self.active_residue_count * self.head_lanes

    @property
    def query_residue_count(self) -> int:
        return int(self.key_value_groups) * self.gqa_ratio

    @property
    def cache_cipher_count(self) -> int:
        return 2 * int(self.key_value_groups)

    def _require_group(self, group: int) -> int:
        group = int(group)
        if not 0 <= group < int(self.key_value_groups):
            raise IndexError(f"group={group} outside [0, {self.key_value_groups}).")
        return group

    def _require_pair(self, pair: int) -> int:
        pair = int(pair)
        if not 0 <= pair < self.pair_blocks:
            raise IndexError(f"pair={pair} outside [0, {self.pair_blocks}).")
        return pair

    def _require_lane(self, lane: int) -> int:
        lane = int(lane)
        if not 0 <= lane < self.head_lanes:
            raise IndexError(f"lane={lane} outside [0, {self.head_lanes}).")
        return lane

    def _require_feature(self, feature: int) -> int:
        feature = int(feature)
        if not 0 <= feature < self.fold_dim:
            raise IndexError(f"feature={feature} outside [0, {self.fold_dim}).")
        return feature

    def _require_query_offset(self, query_offset: int) -> int:
        query_offset = int(query_offset)
        if not 0 <= query_offset < self.gqa_ratio:
            raise IndexError(
                f"query_offset={query_offset} outside [0, {self.gqa_ratio})."
            )
        return query_offset

    def _require_token(self, token: int) -> int:
        token = int(token)
        if not 0 <= token < int(self.seq_len):
            raise IndexError(f"token={token} outside [0, {self.seq_len}).")
        return token

    def split_local_head(self, local_head: int) -> tuple[int, int]:
        local_head = int(local_head)
        if not 0 <= local_head < self.key_value_heads_per_group:
            raise IndexError(
                f"local_head={local_head} outside "
                f"[0, {self.key_value_heads_per_group})."
            )
        return divmod(local_head, self.head_lanes)

    def local_head(self, pair: int, lane: int) -> int:
        return self._require_pair(pair) * self.head_lanes + self._require_lane(lane)

    def key_value_head(self, group: int, pair: int, lane: int) -> int:
        group = self._require_group(group)
        return group * self.key_value_heads_per_group + self.local_head(pair, lane)

    def query_head(
        self, group: int, pair: int, lane: int, query_offset: int
    ) -> int:
        query_offset = self._require_query_offset(query_offset)
        return self.key_value_head(group, pair, lane) * self.gqa_ratio + query_offset

    def key_residue(self, group: int) -> int:
        return self.residues_per_group * self._require_group(group)

    def value_residue(self, group: int) -> int:
        return self.key_residue(group) + 1

    def query_residue(self, group: int, query_offset: int) -> int:
        group = self._require_group(group)
        query_offset = self._require_query_offset(query_offset)
        return self.residues_per_group * group + 2 + query_offset

    def query_cipher_index(self, group: int, query_offset: int) -> int:
        group = self._require_group(group)
        query_offset = self._require_query_offset(query_offset)
        return group * self.gqa_ratio + query_offset

    def score_pair_index(self, group: int, pair: int, query_offset: int) -> int:
        group = self._require_group(group)
        pair = self._require_pair(pair)
        query_offset = self._require_query_offset(query_offset)
        return (self.pair_blocks * group + pair) * self.gqa_ratio + query_offset

    def page_slot(self, pair: int, feature: int, token: int, lane: int) -> int:
        pair = self._require_pair(pair)
        feature = self._require_feature(feature)
        token = self._require_token(token)
        lane = self._require_lane(lane)
        return (
            pair * self.page_block
            + self.head_lanes * (feature * int(self.seq_len) + token)
            + lane
        )

    def carrier_channel(
        self, pair: int, feature: int, residue: int, lane: int
    ) -> int:
        pair = self._require_pair(pair)
        feature = self._require_feature(feature)
        lane = self._require_lane(lane)
        residue = int(residue)
        if not 0 <= residue < self.residue_slots:
            raise IndexError(f"residue={residue} outside [0, {self.residue_slots}).")
        return (
            pair * self.fold_dim * self.channel_row
            + self.channel_row * feature
            + self.head_lanes * residue
            + lane
        )

    def carrier_slot(
        self,
        pair: int,
        feature: int,
        residue: int,
        lane: int,
        *,
        token_lane: int = 0,
    ) -> int:
        token_lane = int(token_lane)
        if not 0 <= token_lane < int(self.native_token_lanes):
            raise IndexError(
                f"token_lane={token_lane} outside "
                f"[0, {self.native_token_lanes})."
            )
        return (
            self.carrier_channel(pair, feature, residue, lane)
            * int(self.native_token_lanes)
            + token_lane
        )

    def score_slot(
        self,
        group: int,
        pair: int,
        query_offset: int,
        token: int,
        lane: int,
    ) -> int:
        pair_index = self.score_pair_index(group, pair, query_offset)
        token = self._require_token(token)
        lane = self._require_lane(lane)
        return self.head_lanes * (pair_index * int(self.seq_len) + token) + lane

    def output_slot(
        self,
        group: int,
        pair: int,
        query_offset: int,
        feature: int,
        lane: int,
    ) -> int:
        return self.carrier_slot(
            pair,
            feature,
            self.query_residue(group, query_offset),
            lane,
        )

    def carrier_to_page_rotation(
        self, residue: int, lane: int, token: int
    ) -> int:
        residue = int(residue)
        if not 0 <= residue < self.active_residue_count:
            raise IndexError(
                f"active residue={residue} outside [0, {self.active_residue_count})."
            )
        lane = self._require_lane(lane)
        token = self._require_token(token)
        lanes = int(self.native_token_lanes)
        # carrier_slot(pair, f, residue, lane) - page_slot(pair, f, token, lane)
        return self.head_lanes * lanes * residue + (lanes - 1) * lane - self.head_lanes * token

    def page_to_score_rotation(
        self, group: int, pair: int, query_offset: int
    ) -> int:
        pair = self._require_pair(pair)
        pair_index = self.score_pair_index(group, pair, query_offset)
        return pair * self.page_block - self.head_lanes * int(self.seq_len) * pair_index

    def score_to_page_rotation(
        self, group: int, pair: int, query_offset: int
    ) -> int:
        return -self.page_to_score_rotation(group, pair, query_offset)

    def page_to_output_rotation(
        self, group: int, query_offset: int, lane: int
    ) -> int:
        residue = self.query_residue(group, query_offset)
        lanes = int(self.native_token_lanes)
        return -(self.head_lanes * lanes * residue + (lanes - 1) * self._require_lane(lane))

    def active_carrier_channels(self) -> tuple[int, ...]:
        channels: list[int] = []
        for group in range(int(self.key_value_groups)):
            residues = (
                self.key_residue(group),
                self.value_residue(group),
                *(self.query_residue(group, r) for r in range(self.gqa_ratio)),
            )
            for pair in range(self.pair_blocks):
                for feature in range(self.fold_dim):
                    for residue in residues:
                        for lane in range(self.head_lanes):
                            channels.append(
                                self.carrier_channel(pair, feature, residue, lane)
                            )
        return tuple(channels)

    def token_rotations(self) -> tuple[int, ...]:
        """PairFold token-axis rotations; lane interleaving gives stride two."""

        return tuple(
            self.head_lanes * (1 << bit) for bit in range(int(self.seq_len).bit_length() - 1)
        )

    def feature_rotations(self) -> tuple[int, ...]:
        """PairFold folded-feature rotations; one row is 2*128 slots."""

        row = self.head_lanes * int(self.seq_len)
        return tuple(row * (1 << bit) for bit in range(self.fold_dim.bit_length() - 1))

    def _one_query(self, query: np.ndarray, *, name: str) -> np.ndarray:
        query = _real_array(query, name=name)
        shape = (int(self.query_heads), int(self.head_dim))
        if query.shape == shape:
            return query
        if query.shape == (1, *shape):
            return query[0]
        if query.shape in {(shape[0] * shape[1],), (1, shape[0] * shape[1])}:
            return query.reshape(shape)
        raise ValueError(
            f"{name} must contain one token of shape {shape}, got {query.shape}."
        )

    def _one_kv(self, values: np.ndarray, *, name: str) -> np.ndarray:
        values = _real_array(values, name=name)
        shape = (int(self.key_value_heads), int(self.head_dim))
        if values.shape == shape:
            return values
        if values.shape == (1, *shape):
            return values[0]
        if values.shape in {(shape[0] * shape[1],), (1, shape[0] * shape[1])}:
            return values.reshape(shape)
        raise ValueError(
            f"{name} must contain one token of shape {shape}, got {values.shape}."
        )

    def _kv_page(self, values: np.ndarray, *, name: str) -> np.ndarray:
        values = _real_array(values, name=name)
        flat_width = int(self.key_value_heads) * int(self.head_dim)
        if values.ndim == 2 and values.shape[1] == flat_width:
            values = values.reshape(
                values.shape[0], int(self.key_value_heads), int(self.head_dim)
            )
        if values.ndim != 3 or values.shape[1:] != (
            int(self.key_value_heads),
            int(self.head_dim),
        ):
            raise ValueError(
                f"{name} must have shape [tokens, {self.key_value_heads}, "
                f"{self.head_dim}], got {values.shape}."
            )
        if not 0 <= int(values.shape[0]) <= int(self.seq_len):
            raise ValueError(
                f"{name} token count {values.shape[0]} exceeds {self.seq_len}."
            )
        return values

    @staticmethod
    def _complex_dtype(dtype) -> np.dtype:
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.complexfloating):
            raise TypeError(f"PairFold payloads require a complex dtype, got {dtype}.")
        return dtype

    def _one_cipher_payload(self, packed: np.ndarray, *, name: str) -> np.ndarray:
        packed = np.asarray(packed)
        if packed.shape == (1, int(self.slots)):
            return packed[0]
        if packed.shape == (int(self.slots),):
            return packed
        raise ValueError(
            f"{name} must have shape {(1, self.slots)} or {(self.slots,)}, "
            f"got {packed.shape}."
        )

    def pack_qkv_carrier(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        *,
        dtype=np.complex128,
    ) -> np.ndarray:
        """Pack one unrotated Decode Q/K/V token into one sparse carrier."""

        query = self._one_query(query, name="query")
        key = self._one_kv(key, name="key")
        value = self._one_kv(value, name="value")
        packed = np.zeros((1, int(self.slots)), dtype=self._complex_dtype(dtype))
        half = self.fold_dim
        for group in range(int(self.key_value_groups)):
            for pair in range(self.pair_blocks):
                for lane in range(self.head_lanes):
                    kv_head = self.key_value_head(group, pair, lane)
                    for feature in range(half):
                        packed[
                            0,
                            self.carrier_slot(
                                pair, feature, self.key_residue(group), lane
                            ),
                        ] = complex(
                            key[kv_head, feature], -key[kv_head, feature + half]
                        )
                        packed[
                            0,
                            self.carrier_slot(
                                pair, feature, self.value_residue(group), lane
                            ),
                        ] = complex(
                            value[kv_head, feature], value[kv_head, feature + half]
                        )
                        for query_offset in range(self.gqa_ratio):
                            query_head = self.query_head(
                                group, pair, lane, query_offset
                            )
                            packed[
                                0,
                                self.carrier_slot(
                                    pair,
                                    feature,
                                    self.query_residue(group, query_offset),
                                    lane,
                                ),
                            ] = 0.5 * complex(
                                query[query_head, feature],
                                query[query_head, feature + half],
                            )
        return packed

    def unpack_qkv_carrier(
        self, packed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        packed = self._one_cipher_payload(packed, name="PairFold QKV carrier")
        real_dtype = packed.real.dtype
        query = np.zeros(
            (int(self.query_heads), int(self.head_dim)), dtype=real_dtype
        )
        key = np.zeros(
            (int(self.key_value_heads), int(self.head_dim)), dtype=real_dtype
        )
        value = np.zeros_like(key)
        half = self.fold_dim
        for group in range(int(self.key_value_groups)):
            for pair in range(self.pair_blocks):
                for lane in range(self.head_lanes):
                    kv_head = self.key_value_head(group, pair, lane)
                    for feature in range(half):
                        key_fold = packed[
                            self.carrier_slot(
                                pair, feature, self.key_residue(group), lane
                            )
                        ]
                        value_fold = packed[
                            self.carrier_slot(
                                pair, feature, self.value_residue(group), lane
                            )
                        ]
                        key[kv_head, feature] = key_fold.real
                        key[kv_head, feature + half] = -key_fold.imag
                        value[kv_head, feature] = value_fold.real
                        value[kv_head, feature + half] = value_fold.imag
                        for query_offset in range(self.gqa_ratio):
                            query_head = self.query_head(
                                group, pair, lane, query_offset
                            )
                            query_fold = packed[
                                self.carrier_slot(
                                    pair,
                                    feature,
                                    self.query_residue(group, query_offset),
                                    lane,
                                )
                            ]
                            query[query_head, feature] = 2.0 * query_fold.real
                            query[query_head, feature + half] = 2.0 * query_fold.imag
        return query, key, value

    def pack_query(self, query: np.ndarray, *, dtype=np.complex128) -> np.ndarray:
        """Pack eight extracted Q groups at PairFold token column zero."""

        query = self._one_query(query, name="query")
        packed = np.zeros(
            (self.query_residue_count, int(self.slots)),
            dtype=self._complex_dtype(dtype),
        )
        half = self.fold_dim
        for group in range(int(self.key_value_groups)):
            for query_offset in range(self.gqa_ratio):
                cipher = self.query_cipher_index(group, query_offset)
                for pair in range(self.pair_blocks):
                    for lane in range(self.head_lanes):
                        query_head = self.query_head(
                            group, pair, lane, query_offset
                        )
                        for feature in range(half):
                            packed[
                                cipher, self.page_slot(pair, feature, 0, lane)
                            ] = 0.5 * complex(
                                query[query_head, feature],
                                query[query_head, feature + half],
                            )
        return packed

    def unpack_query(self, packed: np.ndarray) -> np.ndarray:
        packed = np.asarray(packed)
        expected = (self.query_residue_count, int(self.slots))
        if packed.shape != expected:
            raise ValueError(f"PairFold Q must have shape {expected}, got {packed.shape}.")
        query = np.zeros(
            (int(self.query_heads), int(self.head_dim)), dtype=packed.real.dtype
        )
        half = self.fold_dim
        for group in range(int(self.key_value_groups)):
            for query_offset in range(self.gqa_ratio):
                cipher = self.query_cipher_index(group, query_offset)
                for pair in range(self.pair_blocks):
                    for lane in range(self.head_lanes):
                        query_head = self.query_head(
                            group, pair, lane, query_offset
                        )
                        for feature in range(half):
                            folded = packed[
                                cipher, self.page_slot(pair, feature, 0, lane)
                            ]
                            query[query_head, feature] = 2.0 * folded.real
                            query[query_head, feature + half] = 2.0 * folded.imag
        return query

    def _pack_kv(
        self, values: np.ndarray, *, key: bool, dtype=np.complex128
    ) -> np.ndarray:
        name = "key" if key else "value"
        values = self._kv_page(values, name=name)
        packed = np.zeros(
            (int(self.key_value_groups), int(self.slots)),
            dtype=self._complex_dtype(dtype),
        )
        half = self.fold_dim
        for token in range(int(values.shape[0])):
            for group in range(int(self.key_value_groups)):
                for pair in range(self.pair_blocks):
                    for lane in range(self.head_lanes):
                        head = self.key_value_head(group, pair, lane)
                        for feature in range(half):
                            imaginary = values[token, head, feature + half]
                            if key:
                                imaginary = -imaginary
                            packed[
                                group,
                                self.page_slot(pair, feature, token, lane),
                            ] = complex(values[token, head, feature], imaginary)
        return packed

    def pack_key(self, key: np.ndarray, *, dtype=np.complex128) -> np.ndarray:
        return self._pack_kv(key, key=True, dtype=dtype)

    def pack_value(self, value: np.ndarray, *, dtype=np.complex128) -> np.ndarray:
        return self._pack_kv(value, key=False, dtype=dtype)

    def _unpack_kv(
        self, packed: np.ndarray, *, key: bool, occupied: int | None
    ) -> np.ndarray:
        packed = np.asarray(packed)
        expected = (int(self.key_value_groups), int(self.slots))
        if packed.shape != expected:
            raise ValueError(
                f"PairFold {'key' if key else 'value'} must have shape "
                f"{expected}, got {packed.shape}."
            )
        occupied = int(self.seq_len if occupied is None else occupied)
        if not 0 <= occupied <= int(self.seq_len):
            raise ValueError(f"occupied={occupied} outside [0, {self.seq_len}].")
        values = np.zeros(
            (occupied, int(self.key_value_heads), int(self.head_dim)),
            dtype=packed.real.dtype,
        )
        half = self.fold_dim
        for token in range(occupied):
            for group in range(int(self.key_value_groups)):
                for pair in range(self.pair_blocks):
                    for lane in range(self.head_lanes):
                        head = self.key_value_head(group, pair, lane)
                        for feature in range(half):
                            folded = packed[
                                group,
                                self.page_slot(pair, feature, token, lane),
                            ]
                            values[token, head, feature] = folded.real
                            values[token, head, feature + half] = (
                                -folded.imag if key else folded.imag
                            )
        return values

    def unpack_key(
        self, packed: np.ndarray, *, occupied: int | None = None
    ) -> np.ndarray:
        return self._unpack_kv(packed, key=True, occupied=occupied)

    def unpack_value(
        self, packed: np.ndarray, *, occupied: int | None = None
    ) -> np.ndarray:
        return self._unpack_kv(packed, key=False, occupied=occupied)

    def pack_scores(self, scores: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        scores = _real_array(scores, name="scores")
        expected = (int(self.query_heads), int(self.seq_len))
        if scores.shape != expected:
            raise ValueError(f"scores must have shape {expected}, got {scores.shape}.")
        packed = np.zeros((1, int(self.slots)), dtype=np.dtype(dtype))
        for group in range(int(self.key_value_groups)):
            for pair in range(self.pair_blocks):
                for lane in range(self.head_lanes):
                    for query_offset in range(self.gqa_ratio):
                        head = self.query_head(group, pair, lane, query_offset)
                        for token in range(int(self.seq_len)):
                            packed[
                                0,
                                self.score_slot(
                                    group, pair, query_offset, token, lane
                                ),
                            ] = scores[head, token]
        base = packed[:, : self.score_payload_slots]
        return np.tile(base, (1, self.score_payload_repetitions))

    def unpack_scores(self, packed: np.ndarray) -> np.ndarray:
        packed = self._one_cipher_payload(packed, name="PairFold scores")
        scores = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=packed.real.dtype
        )
        for group in range(int(self.key_value_groups)):
            for pair in range(self.pair_blocks):
                for lane in range(self.head_lanes):
                    for query_offset in range(self.gqa_ratio):
                        head = self.query_head(group, pair, lane, query_offset)
                        for token in range(int(self.seq_len)):
                            scores[head, token] = packed[
                                self.score_slot(
                                    group, pair, query_offset, token, lane
                                )
                            ].real
        return scores

    def score_prefix_mask(
        self, occupied: int, *, dtype=np.float64
    ) -> np.ndarray:
        occupied = int(occupied)
        if not 0 <= occupied <= int(self.seq_len):
            raise ValueError(f"occupied={occupied} outside [0, {self.seq_len}].")
        mask = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=np.float64
        )
        mask[:, :occupied] = 1.0
        return self.pack_scores(mask, dtype=dtype)

    def score_token0_mask(self, *, dtype=np.float64) -> np.ndarray:
        mask = np.zeros(
            (int(self.query_heads), int(self.seq_len)), dtype=np.float64
        )
        mask[:, 0] = 1.0
        return self.pack_scores(mask, dtype=dtype)

    def pack_output(self, output: np.ndarray, *, dtype=np.complex128) -> np.ndarray:
        """Pack one logical attention output into Q-residue carrier positions."""

        output = self._one_query(output, name="output")
        packed = np.zeros((1, int(self.slots)), dtype=self._complex_dtype(dtype))
        half = self.fold_dim
        for group in range(int(self.key_value_groups)):
            for pair in range(self.pair_blocks):
                for lane in range(self.head_lanes):
                    for query_offset in range(self.gqa_ratio):
                        head = self.query_head(group, pair, lane, query_offset)
                        for feature in range(half):
                            packed[
                                0,
                                self.output_slot(
                                    group,
                                    pair,
                                    query_offset,
                                    feature,
                                    lane,
                                ),
                            ] = complex(
                                output[head, feature],
                                output[head, feature + half],
                            )
        return packed

    def unpack_output(self, packed: np.ndarray) -> np.ndarray:
        packed = self._one_cipher_payload(packed, name="PairFold output")
        output = np.zeros(
            (int(self.query_heads), int(self.head_dim)), dtype=packed.real.dtype
        )
        half = self.fold_dim
        for group in range(int(self.key_value_groups)):
            for pair in range(self.pair_blocks):
                for lane in range(self.head_lanes):
                    for query_offset in range(self.gqa_ratio):
                        head = self.query_head(group, pair, lane, query_offset)
                        for feature in range(half):
                            folded = packed[
                                self.output_slot(
                                    group,
                                    pair,
                                    query_offset,
                                    feature,
                                    lane,
                                )
                            ]
                            output[head, feature] = folded.real
                            output[head, feature + half] = folded.imag
        return output


@dataclass(frozen=True)
class PairFold128Layout(PairFoldLayout):
    """The production Llama-3-8B PairFold layout (128 tokens, 4 KV heads per cache ciphertext).

    Same indices as :class:`PairFoldLayout`; the constructor additionally pins the
    historical shape so the 8B Decode path cannot drift.
    """

    def __post_init__(self) -> None:
        expected = {
            "seq_len": 128,
            "head_dim": 128,
            "query_heads": 32,
            "key_value_heads": 8,
            "slots": 32768,
            "native_token_lanes": 8,
            "key_value_groups": 2,
        }
        for name, wanted in expected.items():
            actual = int(getattr(self, name))
            if actual != wanted:
                raise ValueError(
                    f"PairFold128 requires {name}={wanted}, got {actual}."
                )
        super().__post_init__()


def pairfold_layout_for(
    *,
    head_dim: int,
    query_heads: int,
    key_value_heads: int,
    seq_len: int = 128,
    slots: int = 32768,
    key_value_groups: int | None = None,
) -> PairFoldLayout:
    """PairFold layout of a model geometry.

    ``key_value_groups`` defaults to the fewest cache ciphertexts whose pair pages
    still fit ``slots`` (Llama-3-8B: 2 groups of 4 heads; Llama-3.2-1B, head_dim 64:
    1 group of 8 heads, i.e. 2 cache ciphertexts instead of 4).  The 8B geometry
    returns the pinned :class:`PairFold128Layout`.
    """

    head_dim, q_heads, kv_heads = int(head_dim), int(query_heads), int(key_value_heads)
    seq_len, slots = int(seq_len), int(slots)
    if kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError(f"query_heads={q_heads} must be a multiple of key_value_heads={kv_heads}.")
    gqa = q_heads // kv_heads
    if key_value_groups is None:
        groups = None
        for candidate in range(1, kv_heads + 1):
            if kv_heads % candidate or (kv_heads // candidate) % 2:
                continue
            page = 2 * (head_dim // 2) * seq_len
            residues = candidate * (2 + gqa)
            residue_slots = 1 << (residues - 1).bit_length() if residues > 1 else 1
            if (kv_heads // candidate) // 2 * page <= slots and seq_len % residue_slots == 0:
                groups = candidate
                break
        if groups is None:
            raise ValueError("no PairFold grouping fits the geometry.")
    else:
        groups = int(key_value_groups)
    residues = groups * (2 + gqa)
    residue_slots = 1 << (residues - 1).bit_length() if residues > 1 else 1
    if seq_len % residue_slots:
        raise ValueError(f"seq_len={seq_len} is not a multiple of residue_slots={residue_slots}.")
    kwargs = dict(
        seq_len=seq_len, head_dim=head_dim, query_heads=q_heads, key_value_heads=kv_heads,
        slots=slots, native_token_lanes=seq_len // residue_slots, key_value_groups=groups,
    )
    if kwargs == dict(seq_len=128, head_dim=128, query_heads=32, key_value_heads=8, slots=32768,
                      native_token_lanes=8, key_value_groups=2):
        return PairFold128Layout(**kwargs)
    return PairFoldLayout(**kwargs)


__all__ = ["PairFoldLayout", "PairFold128Layout", "pairfold_layout_for"]

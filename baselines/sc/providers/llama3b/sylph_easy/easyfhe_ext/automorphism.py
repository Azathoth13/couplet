"""Pure arithmetic for the slot rotation associated with an RLWE automorphism."""


def rotation_for_automorphism(g, n):
    """Slot rotation step s with 5**s == g (mod 2N); EasyFHE keys rotations this way."""
    m = 2 * n
    g %= m
    value = 1
    for s in range(n // 2):
        if value == g:
            return s
        value = value * 5 % m
    raise ValueError(f"automorphism {g} is not a power of 5 mod {m}")

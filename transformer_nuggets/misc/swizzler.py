def swizzle_functor(mask_bits, least_sig_bits, shift_distance):
    """Creates a functor that applies the swizzle pattern."""
    assert least_sig_bits >= 0
    assert mask_bits >= 0
    assert abs(shift_distance) >= mask_bits

    bit_mask = (1 << mask_bits) - 1
    yy_mask = bit_mask << (least_sig_bits + max(0, shift_distance))
    # zz_mask = bit_mask << (least_sig_bits - min(0, shift_distance))

    def swizzler(index):
        return index ^ ((index & yy_mask) >> shift_distance)

    return swizzler

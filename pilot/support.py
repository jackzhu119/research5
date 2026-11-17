"""Structural local support of the pinned AnySleep, with all global paths fixed.

This propagates Boolean dependency (not an empirical receptive field) through
the exact Conv/odd-left-pad/pool/upsample/crop/skip/pooling topology. It assumes
eval-mode BatchNorm, fixed attention alpha, and fixed preprocessing statistics.
"""
import numpy as np


def conv_same(mask, kernel):
    left = (kernel - 1) // 2
    right = kernel - 1 - left
    padded = np.pad(mask, (left, right))
    out = np.zeros_like(mask)
    for k in range(kernel):
        out |= padded[k:k + len(mask)]
    return out


def anysleep_local_support(mask, depth=12, epoch_samples=3840):
    mask = np.asarray(mask, dtype=bool)
    skips = []
    for _ in range(depth):
        mask = conv_same(mask, 9)
        if len(mask) % 2:
            mask = np.pad(mask, (1, 0))
        skips.append(mask)
        mask = mask.reshape(-1, 2).any(axis=1)
    mask = conv_same(mask, 9)
    for skip in reversed(skips):
        mask = conv_same(np.repeat(mask, 2), 2)
        diff = max(0, len(mask) - len(skip))
        start = (diff + 1) // 2
        mask = mask[start:start + len(skip)]
        mask = conv_same(mask | skip, 9)
    n = len(mask) // epoch_samples
    return mask[:n * epoch_samples].reshape(n, epoch_samples).any(axis=1)


def selftest():
    # Even kernel uses right padding: input[5] affects output[4] and output[5].
    m = np.zeros(10, bool)
    m[5] = True
    assert np.flatnonzero(conv_same(m, 2)).tolist() == [4, 5]
    assert not anysleep_local_support(np.zeros(3840 * 64, bool)).any()
    assert anysleep_local_support(np.ones(3840 * 64, bool)).all()
    print('STRUCTURAL_SUPPORT_SELFTEST_OK; no EEG inference')


if __name__ == '__main__':
    selftest()

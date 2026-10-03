"""Fit-local caches for row-wise, frozen network predictions.

No replay sampling or RNG calls occur here. Keep inference batch dimensions
equal to the fitting batches (including padding the last cache chunk) to avoid
unnecessarily changing the matrix multiplication kernel. These caches must
never hold predictions from a network that changes during the owning fit.
"""

import numpy as np
import torch


def build_frozen_cache(row_count, batch_size, train_steps, predict_rows, *, device):
    """Return a host float32 cache, or None when building it would cost more.

The validated production backend is CPU. Other devices retain uncached
inference until separately checked for numerical equivalence.
"""
    if not row_count or batch_size == 0 or torch.device(device).type != "cpu":
        return None
    batch_size = row_count if batch_size < 0 else min(batch_size, row_count)
    if (row_count + batch_size - 1) // batch_size >= train_steps:
        return None
    result = None
    with torch.no_grad():
        for start in range(0, row_count, batch_size):
            stop = min(start + batch_size, row_count)
            indices = slice(start, stop)
            if stop - start < batch_size:
                # Repeat populated rows, never read uninitialised storage.
                indices = np.arange(start, start + batch_size, dtype=np.intp) % row_count
            values = predict_rows(indices).detach().cpu().numpy()
            if result is None:
                result = np.empty((row_count, *values.shape[1:]), dtype=np.float32)
            result[start:stop] = values[:stop-start]
    return result


def cached_tensor(cache, indices, device):
    return torch.as_tensor(cache[indices], dtype=torch.float32, device=device)

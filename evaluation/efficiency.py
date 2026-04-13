"""
evaluation/efficiency.py
========================
Measures computational efficiency metrics:
  - Model size in MB
  - Training time (recorded during training)
  - Inference time per sample (mean over N runs)
  - Peak memory usage during inference
"""

import time
import os
import numpy as np
import psutil
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def get_model_size_mb(model) -> float:
    """
    Estimate model size from parameter count (float32 = 4 bytes).
    Returns size in MB.
    """
    total_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    size_mb = (total_params * 4) / (1024 ** 2)
    return round(size_mb, 3)


def measure_inference_time(
    model,
    X_sample: np.ndarray,
    n_runs: int = 100,
    batch_size: int = 1,
) -> dict:
    """
    Measure per-sample inference latency.

    Args:
        model:    Trained Keras model
        X_sample: A small batch of sequences, shape (N, T, F)
        n_runs:   Number of timing iterations
        batch_size: Inference batch size

    Returns:
        Dict with mean, std, min, max times in milliseconds
    """
    # Warm up
    _ = model.predict(X_sample[:batch_size], verbose=0)

    times = []
    for _ in range(n_runs):
        idx = np.random.randint(0, len(X_sample) - batch_size)
        batch = X_sample[idx: idx + batch_size]
        t0 = time.perf_counter()
        _ = model.predict(batch, verbose=0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)   # ms

    result = {
        "mean_ms":   round(np.mean(times), 3),
        "std_ms":    round(np.std(times), 3),
        "min_ms":    round(np.min(times), 3),
        "max_ms":    round(np.max(times), 3),
        "per_sample_mean_ms": round(np.mean(times) / batch_size, 3),
    }
    logger.info(f"  Inference: mean={result['mean_ms']}ms | "
                f"per-sample={result['per_sample_mean_ms']}ms")
    return result


def measure_peak_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    return round(mem_mb, 2)


def full_efficiency_report(
    models_dict: dict,
    X_test: np.ndarray,
    train_times: dict,
) -> dict:
    """
    Run full efficiency analysis for all models.

    Args:
        models_dict:  {model_name: keras_model}
        X_test:       Test sequences for latency measurement
        train_times:  {model_name: training_seconds}

    Returns:
        Nested dict of efficiency metrics per model
    """
    report = {}

    for name, model in models_dict.items():
        logger.info(f"\n─── Efficiency: {name} ───────────────────────────")
        size_mb  = get_model_size_mb(model)
        inf_time = measure_inference_time(model, X_test, n_runs=50, batch_size=1)
        mem_mb   = measure_peak_memory_mb()
        train_s  = train_times.get(name, None)

        report[name] = {
            "size_mb":            size_mb,
            "train_time_s":       round(train_s, 1) if train_s else "N/A",
            "train_time_min":     round(train_s / 60, 2) if train_s else "N/A",
            "inference_ms":       inf_time,
            "peak_memory_mb":     mem_mb,
        }

        logger.info(f"  Size:          {size_mb} MB")
        logger.info(f"  Training time: {round(train_s/60, 2) if train_s else 'N/A'} min")
        logger.info(f"  Inference:     {inf_time['per_sample_mean_ms']} ms/sample")
        logger.info(f"  Peak memory:   {mem_mb} MB")

    return report
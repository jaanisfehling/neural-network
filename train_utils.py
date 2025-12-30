import numpy as np

from network import *


def epoch_run(
    model: Model, loss_fn, optimizer, dataloader: DataLoader, final_metric_fn: callable, update_weights: bool
) -> tuple[float, float]:
    total_n = 0
    losses = 0.0
    final_metric_agg = 0.0
    for x_batch, y_batch in dataloader:
        y_pred = model.forward(x_batch)
        loss = loss_fn.forward(y_pred, y_batch)
        if update_weights:
            grad_loss = loss_fn.backward()
            model.backward(grad_loss)
            optimizer.step()
        total_n += y_batch.shape[0]
        losses += loss * y_batch.shape[0]
        final_metric_agg += final_metric_fn(y_pred, y_batch)
    return losses / total_n, final_metric_agg / total_n


def accuracy(y_pred, y_true):
    y_hat = (y_pred >= 0.5).astype(int)
    correct_n = np.sum(y_hat == y_true)
    return correct_n


def mean_euclidean_error(y_pred, y_true):
    return np.sum(np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1)))

import numpy as np
from tqdm import tqdm

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


def generate_param_cfgs(parameters: dict[list]) -> dict:
    cfgs = []
    for lr in parameters["learning_rates"]:
        for wd in parameters["weight_decays"]:
            for h1 in parameters["hidden_units_layer_1"]:
                for h2 in parameters["hidden_units_layer_2"]:
                    for h3 in parameters["hidden_units_layer_3"]:
                        for bs in parameters["batch_sizes"]:
                            for ah in parameters["activations_hidden_layer"]:
                                for ao in parameters["activations_output_layer"]:
                                    for lf in parameters["loss_functions"]:
                                        for op in parameters["optimizers"]:
                                            cfgs.append(
                                                {
                                                    "learning_rate": lr,
                                                    "weight_decay": wd,
                                                    "input_size": parameters["input_size"],
                                                    "hidden_units_layer_1": h1,
                                                    "hidden_units_layer_2": h2,
                                                    "hidden_units_layer_3": h3,
                                                    "batch_size": bs,
                                                    "output_size": parameters["output_size"],
                                                    "activation_hidden_layer": ah,
                                                    "activation_output_layer": ao,
                                                    "loss_function": lf,
                                                    "optimizer": op,
                                                }
                                            )
    return cfgs


def create_model(cfg: dict) -> Model:
    layers = [
        LinearLayer(cfg["input_size"], cfg["hidden_units_layer_1"]),
        cfg["activation_hidden_layer"](),
    ]
    last_layer_units = cfg["hidden_units_layer_1"]
    if cfg["hidden_units_layer_2"]:
        layers.extend(
            [
                LinearLayer(cfg["hidden_units_layer_1"], cfg["hidden_units_layer_2"]),
                cfg["activation_hidden_layer"](),
            ]
        )
        last_layer_units = cfg["hidden_units_layer_2"]
        if cfg["hidden_units_layer_3"]:
            layers.extend(
                [
                    LinearLayer(cfg["hidden_units_layer_2"], cfg["hidden_units_layer_3"]),
                    cfg["activation_hidden_layer"](),
                ]
            )
            last_layer_units = cfg["hidden_units_layer_3"]
    layers.extend(
        [
            LinearLayer(last_layer_units, cfg["output_size"]),
            cfg["activation_output_layer"](),
        ]
    )
    return Model(*layers)


def train_model(
    model: Model,
    loss_fn,
    optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    final_metric_fn: callable,
    max_epochs: int = 200,
    min_epochs: int = 20,
) -> tuple[Model, float, float]:
    patience = 5
    epochs_since_improve = 0
    best_val_loss = float("inf")
    best_val_result = None
    best_num_of_epochs = 0
    for i in range(1, max_epochs + 1):
        epoch_run(model, loss_fn, optimizer, train_dataloader, final_metric_fn, update_weights=True)
        if val_dataloader is not None:
            val_loss, val_result = epoch_run(
                model, loss_fn, optimizer, val_dataloader, final_metric_fn, update_weights=False
            )
            # Min epochs before early stopping can kick in
            if i > min_epochs:
                # Early stopping using loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_since_improve = 0
                    # Save best result according to final metric
                    best_val_result = val_result
                    # Also save the number of epochs needed
                    best_num_of_epochs = i
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= patience:
                        break

    return (model, best_val_result, best_num_of_epochs)


def cross_validate(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: dict,
    final_metric_fn: callable,
    k: int = 5,
) -> tuple[float, float]:
    fold_results = []
    fold_epochs = []
    X_folds = np.array_split(X, k)
    Y_folds = np.array_split(Y, k)

    for vl_idx in range(k):
        XVl = X_folds[vl_idx]
        YVl = Y_folds[vl_idx]
        XTr = np.vstack([X_folds[i] for i in range(k) if i != vl_idx])
        YTr = np.vstack([Y_folds[i] for i in range(k) if i != vl_idx])

        XTr_dl = DataLoader(Dataset(XTr, YTr), batch_size=cfg["batch_size"], shuffle=True)
        XVl_dl = DataLoader(Dataset(XVl, YVl), batch_size=cfg["batch_size"], shuffle=False)

        model = create_model(cfg)
        loss_fn = cfg["loss_function"]()
        optimizer = cfg["optimizer"](model, learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

        # Average the number of epochs needed, so we do not need early stopping when retraining on full data later
        _, val_result, epochs = train_model(model, loss_fn, optimizer, XTr_dl, XVl_dl, final_metric_fn)
        fold_results.append(val_result)
        fold_epochs.append(epochs)

    return (np.mean(fold_results), int(np.mean(fold_epochs)))


def grid_search_cross_validate(
    X: np.ndarray,
    Y: np.ndarray,
    parameter_cfgs: list[dict],
    final_metric_fn: callable,
    k: int = 5,
) -> list[tuple[dict, float]]:
    results = []
    for cfg in tqdm(parameter_cfgs):
        result, epochs = cross_validate(X, Y, cfg, final_metric_fn, k=k)
        results.append((cfg, result, epochs))
    return results

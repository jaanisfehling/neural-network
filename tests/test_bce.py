import numpy as np

from network.bce import BCELoss


def test_bce_forward_single():
    loss_fn = BCELoss()
    y = np.array([1.0, 0.0, 1.0, 0.0])
    y_hat = np.array([0.9, 0.2, 0.7, 0.1])
    loss = loss_fn.forward(y_hat, y)
    expected_loss = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))
    assert np.isclose(loss, expected_loss)


def test_bce_forward_batch():
    loss_fn = BCELoss()
    y = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    y_hat = np.array([[0.9, 0.2, 0.7, 0.1], [0.3, 0.8, 0.4, 0.6]])
    loss = loss_fn.forward(y_hat, y)
    expected_loss = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))
    assert np.isclose(loss, expected_loss)


def test_bce_backward_single():
    loss_fn = BCELoss()
    y = np.array([1.0, 0.0, 1.0, 0.0])
    y_hat = np.array([0.9, 0.2, 0.7, 0.1])
    loss_fn.forward(y_hat, y)
    gradient = loss_fn.backward()
    N = y.size
    expected_gradient = -((y / y_hat) - ((1.0 - y) / (1.0 - y_hat))) / N
    assert np.allclose(gradient, expected_gradient)


def test_bce_backward_batch():
    loss_fn = BCELoss()
    y = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    y_hat = np.array([[0.9, 0.2, 0.7, 0.1], [0.3, 0.8, 0.4, 0.6]])
    loss_fn.forward(y_hat, y)
    gradient = loss_fn.backward()
    N = y.size
    expected_gradient = -((y / y_hat) - ((1.0 - y) / (1.0 - y_hat))) / N
    assert np.allclose(gradient, expected_gradient)

import numpy as np

from network.sigmoid import Sigmoid


def test_forward_single():
    layer = Sigmoid()
    x = np.array([1.0, 2.0, 3.0, 0.0, -2.0])
    y = layer.forward(x)

    expected_output = 1.0 / (1.0 + np.exp(-x))
    assert np.allclose(y, expected_output)


def test_forward_batch():
    layer = Sigmoid()
    x = np.array([[1.0, -1.0, 0.0], [-2.0, 3.0, 4.0]])
    y = layer.forward(x)

    expected_output = 1.0 / (1.0 + np.exp(-x))
    assert np.allclose(y, expected_output)


def test_backward_single():
    layer = Sigmoid()
    x = np.array([1.0, -1.0, 0.0, 2.0])
    layer.forward(x)

    upstream_gradient = np.array([0.5, 0.5, 0.5, 0.5])
    dx = layer.backward(upstream_gradient)

    s = 1.0 / (1.0 + np.exp(-x))
    expected_gradient = upstream_gradient * s * (1.0 - s)
    assert np.allclose(dx, expected_gradient)


def test_backward_batch():
    layer = Sigmoid()
    x = np.array([[1.0, -1.0, 0.0], [-2.0, 3.0, 4.0]])
    layer.forward(x)

    upstream_gradient = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    dx = layer.backward(upstream_gradient)

    s = 1.0 / (1.0 + np.exp(-x))
    expected_gradient = upstream_gradient * s * (1.0 - s)
    assert np.allclose(dx, expected_gradient)

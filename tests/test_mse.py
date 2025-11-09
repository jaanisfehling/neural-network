from network.mse import MSELoss
import numpy as np

def test_mse_forward_single():
    loss_fn = MSELoss()
    y = np.array([3.0, -0.5, 2.0, 7.0])
    y_hat = np.array([2.5, 0.0, 2.0, 8.0])
    loss = loss_fn.forward(y_hat, y)
    expected_loss = 0.375
    assert np.isclose(loss, expected_loss)

def test_mse_forward_batch():
    loss_fn = MSELoss()
    y = np.array([[3.0, -0.5, 2.0, 7.0],
                  [4.0, 0.0, 2.0, 8.0]])
    y_hat = np.array([[2.5, 0.0, 2.0, 8.0],
                      [3.5, -0.5, 2.0, 7.0]])
    loss = loss_fn.forward(y_hat, y)
    expected_loss = 0.375
    assert np.isclose(loss, expected_loss)

def test_mse_backward_single():
    loss_fn = MSELoss()
    y = np.array([3.0, -0.5, 2.0, 7.0])
    y_hat = np.array([2.5, 0.0, 2.0, 8.0])
    loss_fn.forward(y_hat, y)
    gradient = loss_fn.backward()
    expected_gradient = np.array([-0.25, 0.25, 0.0, 0.5])
    assert np.allclose(gradient, expected_gradient)

def test_mse_backward_batch():
    loss_fn = MSELoss()
    y = np.array([[3.0, -0.5, 2.0, 7.0],
                  [4.0, 0.0, 2.0, 8.0]])
    y_hat = np.array([[2.5, 0.0, 2.0, 8.0],
                      [3.5, -0.5, 2.0, 7.0]])
    loss_fn.forward(y_hat, y)
    gradient = loss_fn.backward()
    expected_gradient = np.array([[-0.25, 0.25, 0.0, 0.5],
                                  [ -0.25, -0.25, 0.0, -0.5]])
    assert np.allclose(gradient, expected_gradient)


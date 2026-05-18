from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dim: int = 128
    seed: int = 0


class NumpyMLPRegressor:
    """Small one-hidden-layer MLP regressor trained with Adam."""

    def __init__(self, config: MLPConfig):
        rng = np.random.default_rng(config.seed)
        scale1 = np.sqrt(2.0 / max(config.input_dim, 1))
        scale2 = np.sqrt(2.0 / max(config.hidden_dim, 1))
        self.params: Dict[str, np.ndarray] = {
            "w1": rng.normal(0.0, scale1, size=(config.input_dim, config.hidden_dim)),
            "b1": np.zeros(config.hidden_dim, dtype=np.float64),
            "w2": rng.normal(0.0, scale2, size=(config.hidden_dim, 1)),
            "b2": np.zeros(1, dtype=np.float64),
        }
        self.adam_m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.adam_v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step_count = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hidden_pre = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(hidden_pre, 0.0)
        pred = hidden @ self.params["w2"] + self.params["b2"]
        return pred[:, 0], hidden

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred, _ = self.forward(x)
        return pred

    def train_batch(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        pred, hidden = self.forward(x)
        diff = pred - y
        loss = float(np.mean(diff ** 2))
        grad_pred = (2.0 / len(x)) * diff[:, None]

        grads = {}
        grads["w2"] = hidden.T @ grad_pred
        grads["b2"] = grad_pred.sum(axis=0)
        grad_hidden = grad_pred @ self.params["w2"].T
        grad_hidden[hidden <= 0.0] = 0.0
        grads["w1"] = x.T @ grad_hidden
        grads["b1"] = grad_hidden.sum(axis=0)

        self._adam_update(grads, lr)
        return loss

    def _adam_update(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self.step_count += 1
        for key, grad in grads.items():
            self.adam_m[key] = beta1 * self.adam_m[key] + (1.0 - beta1) * grad
            self.adam_v[key] = beta2 * self.adam_v[key] + (1.0 - beta2) * (grad ** 2)
            m_hat = self.adam_m[key] / (1.0 - beta1 ** self.step_count)
            v_hat = self.adam_v[key] / (1.0 - beta2 ** self.step_count)
            self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self.params.items()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.params = {key: np.asarray(value, dtype=np.float64).copy() for key, value in state.items()}
        self.adam_m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.adam_v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step_count = 0

"""Roboflow model access via inference-sdk HTTP client."""

from __future__ import annotations

import os
import sys
from typing import Any

try:
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient
except ImportError:  # pragma: no cover
    InferenceConfiguration = None
    InferenceHTTPClient = None

# Point this at a self-hosted Roboflow inference server for deployment
# (e.g. ROBOFLOW_INFERENCE_URL=http://localhost:9001 after
#  `docker run -p 9001:9001 roboflow/roboflow-inference-server-gpu`).
# The model then stays loaded in local GPU/CPU memory and per-frame latency
# drops from ~300 ms (hosted) to ~20-50 ms.
ROBOFLOW_DETECT_URL = os.environ.get("ROBOFLOW_INFERENCE_URL", "").strip() or "https://detect.roboflow.com"
_client: InferenceHTTPClient | None = None


class RoboflowConfigError(RuntimeError):
    """Raised when Roboflow inference cannot run."""


def inference_available() -> bool:
    return InferenceHTTPClient is not None


def require_roboflow_api_key() -> str:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RoboflowConfigError(
            "ROBOFLOW_API_KEY is not set. Get a free API key at https://app.roboflow.com "
            "and add it to backend/.env as ROBOFLOW_API_KEY=your_key_here"
        )
    if not inference_available():
        raise RoboflowConfigError(
            "inference-sdk is not installed for this Python interpreter "
            f"({sys.executable}). Run: py -3.12 -m pip install inference-sdk"
        )
    return api_key


class _RemoteRoboflowModel:
    def __init__(self, client: InferenceHTTPClient, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    def infer(self, image: Any, confidence: float = 0.25, **kwargs) -> list[dict]:
        del kwargs
        config = InferenceConfiguration(confidence_threshold=confidence)
        with self._client.use_configuration(config):
            result = self._client.infer(
                inference_input=image,
                model_id=self._model_id,
            )
        return _normalize_inference_result(result)


def _normalize_inference_result(result: Any) -> list[dict]:
    if isinstance(result, list):
        if not result:
            return [{"predictions": []}]
        if isinstance(result[0], dict):
            return result
        return [{"predictions": result}]
    if isinstance(result, dict):
        return [result]
    return [{"predictions": []}]


def _get_client(api_key: str) -> InferenceHTTPClient:
    global _client
    if _client is None:
        _client = InferenceHTTPClient(api_url=ROBOFLOW_DETECT_URL, api_key=api_key)
    return _client


def get_model(*, model_id: str, api_key: str | None = None) -> _RemoteRoboflowModel:
    key = api_key or require_roboflow_api_key()
    return _RemoteRoboflowModel(_get_client(key), model_id)

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
from urllib import request

import numpy as np


@dataclass
class LLMAnalystConfig:
    enabled: bool = False
    endpoint: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: int = 20
    blend_weight: float = 0.25

    @classmethod
    def from_env(cls) -> "LLMAnalystConfig":
        enabled = os.getenv("LLM_ANALYST_ENABLED", "0").lower() in {"1", "true", "yes"}
        return cls(
            enabled=enabled,
            endpoint=os.getenv("LLM_ANALYST_ENDPOINT", "").strip(),
            api_key=os.getenv("LLM_ANALYST_API_KEY", "").strip(),
            model=os.getenv("LLM_ANALYST_MODEL", "").strip(),
            timeout_seconds=int(os.getenv("LLM_ANALYST_TIMEOUT", "20")),
            blend_weight=float(np.clip(float(os.getenv("LLM_ANALYST_BLEND_WEIGHT", "0.25")), 0.0, 1.0)),
        )


class LLMTechnicalAnalyst:
    """OpenAI-compatible technical analyst signal adapter.

    Expected response body (either direct or in `content`) should be JSON object:
    {"SYMBOL": score, ...} where score is in [-1, 1].
    """

    def __init__(self, config: LLMAnalystConfig):
        self.config = config

    def get_signal(self, symbols: Iterable[str], market_snapshot: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        symbols = list(symbols)
        if not self.config.enabled:
            return {symbol: 0.0 for symbol in symbols}
        if not self.config.endpoint or not self.config.model:
            raise ValueError("LLM analyst is enabled but endpoint/model are not configured.")

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a technical analyst. Return ONLY compact JSON mapping symbols to scores in [-1,1].",
                },
                {
                    "role": "user",
                    "content": json.dumps({"symbols": symbols, "snapshot": market_snapshot}, separators=(",", ":")),
                },
            ],
            "temperature": 0,
        }

        req = request.Request(
            self.config.endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")

        parsed = json.loads(raw)
        content = self._extract_content(parsed)
        scores = json.loads(content) if isinstance(content, str) else content

        signal = {symbol: 0.0 for symbol in symbols}
        for symbol in symbols:
            if symbol in scores:
                signal[symbol] = float(np.clip(scores[symbol], -1.0, 1.0))
        return signal

    @staticmethod
    def _extract_content(parsed_response):
        if isinstance(parsed_response, dict) and "choices" in parsed_response:
            choice = parsed_response["choices"][0]
            message = choice.get("message", {})
            return message.get("content", "{}")
        return parsed_response


def blend_actions(model_actions: np.ndarray, llm_scores: Dict[str, float], symbols: Iterable[str], weight: float) -> np.ndarray:
    symbols = list(symbols)
    model_actions = np.asarray(model_actions, dtype=np.float64)
    llm_vector = np.array([llm_scores.get(symbol, 0.0) for symbol in symbols], dtype=np.float64)
    weight = float(np.clip(weight, 0.0, 1.0))
    blended = (1 - weight) * model_actions + weight * llm_vector
    return np.clip(blended, -1.0, 1.0)

from __future__ import annotations

from typing import Any, Literal


TemperatureModel = Literal["hot_linear", "hot_quadratic"]
TEMPERATURE_MODELS: frozenset[str] = frozenset({"hot_linear", "hot_quadratic"})
RACS_TEMPERATURE_EPSILON_FLOOR = 1e-6


def evaluate_temperature_response(
    temperature: Any,
    beta: Any,
    reference_temperature: float,
    *,
    model: TemperatureModel,
    xp: Any,
) -> Any:
    """Evaluate a multiplicative hot-temperature response for an array backend."""
    delta_temperature = xp.maximum(temperature - reference_temperature, 0.0)
    if model == "hot_linear":
        response = 1.0 - beta * delta_temperature
    elif model == "hot_quadratic":
        response = 1.0 - xp.square(beta * delta_temperature)
    else:
        raise ValueError(f"Unknown temperature model: {model!r}.")
    return xp.maximum(response, RACS_TEMPERATURE_EPSILON_FLOOR)

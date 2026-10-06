from typing import Any, Optional


class RuntimeConfig:
    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cuda:0",
        gradient_accumulation_steps: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.params = {} if params is None else dict(params)
        configured_accumulation = self.params.get("gradient_accumulation_steps", 1)
        if gradient_accumulation_steps is not None:
            configured_accumulation = gradient_accumulation_steps
        self.gradient_accumulation_steps = max(1, int(configured_accumulation))


class RunnerResult:
    def __init__(
        self,
        predictions: Any,
        targets: Any,
        metrics: Optional[dict[str, float]] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.predictions = predictions
        self.targets = targets
        self.metrics = {} if metrics is None else metrics
        self.details = {} if details is None else details

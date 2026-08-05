import gc
import logging
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Generator(ABC):
    def __init__(self, config: dict[str, str | dict[str, str]]) -> None:
        self.stopped: bool = True
        try:
            self.model_name = config["model"]
            self.backend = config["backend"]
        except Exception as e:
            logger.error(f"Got Exception {e} while setting generator config.")
            raise

    @abstractmethod
    async def generate(
        self, prompt: str, request_id: str, max_tokens: int | None = None
    ) -> str: ...

    @abstractmethod
    def _backend_cleanup(self) -> None: ...

    def cleanup(self) -> None:
        self.stopped = True
        self._backend_cleanup()
        gc.collect()

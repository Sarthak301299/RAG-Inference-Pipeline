import logging

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TextPrompt
from vllm.inputs.preprocess import InputPreprocessor
from vllm.sampling_params import StructuredOutputsParams

from src.generation.backend import Generator
from src.generation.schemas import make_output_schema

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VLLMGenerator(Generator):
    def __init__(self, config: dict[str, str | dict[str, str]]) -> None:
        try:
            super().__init__(config)
            if not isinstance(self.model_name, str):
                raise TypeError("Model name must be a string.")
            if not isinstance(self.backend, str):
                raise TypeError("Backend name must be a str.")
            local_conf = config["vllm"]
            if not isinstance(local_conf, dict):
                raise TypeError("Local VLLM configuration must be a dict.")
            self.temperature = float(local_conf["temperature"])
            self.max_num_seqs = int(local_conf["max_num_seqs"])
            self.max_num_batched_tokens = int(local_conf["max_num_batched_tokens"])
            self.max_tokens = int(local_conf["max_tokens"])
            self.top_p = float(local_conf["top_p"])
            self.max_thought_process_words = int(
                local_conf["max_thought_process_words"]
            )
            self.max_answer_words = int(local_conf["max_answer_words"])
        except Exception as e:
            logger.error(f"Got Exception {e} while parsing vllm configs")
            raise
        try:
            # Lower utilization limit for local testing on a single 8GB GPU.
            self.engineargs = AsyncEngineArgs(gpu_memory_utilization=0.6)
            self.engineargs.model = self.model_name
            self.engineargs.max_num_seqs = self.max_num_seqs
            self.engineargs.max_num_batched_tokens = self.max_num_batched_tokens
            self.vllmengine = AsyncLLMEngine.from_engine_args(self.engineargs)
            self.output_schema = StructuredOutputsParams(
                json=make_output_schema(
                    max_thought_process_len=self.max_thought_process_words * 5,
                    max_answer_len=self.max_answer_words * 5,
                ).model_json_schema()
            )
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                structured_outputs=self.output_schema,
            )
            self.preprocessor = InputPreprocessor(
                vllm_config=self.vllmengine.vllm_config
            )
        except Exception as e:
            logger.error(
                f"Got Exception {e} while parsing initializing Async LLM Engine"
            )
            raise
        self.stopped = False

    async def generate(self, prompt: str, request_id: str) -> str:
        if self.stopped:
            raise RuntimeError("Generator is stopped.")
        try:
            results_generator = self.vllmengine.generate(
                prompt=self.preprocessor.preprocess(prompt=TextPrompt(prompt=prompt)),
                sampling_params=self.sampling_params,
                request_id=request_id,
            )
            output = ""
            async for result in results_generator:
                output = result.outputs[0].text
        except Exception as e:
            logger.error(f"Got Exception {e} during output generation.")
            raise
        return output

    def _backend_cleanup(self) -> None:
        self.vllmengine.shutdown()

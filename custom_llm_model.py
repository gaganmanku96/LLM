from typing import Any, List, Mapping, Optional

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class LLAMA_MODEL:
    def __init__(self) -> None:
        # Make sure the model path is correct for your system!
        self.llm = LlamaCpp(
            model_path="./llama-2-13b-chat.ggmlv3.q5_1",
            input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        return self.llm(prompt)


class CustomLLM(LLM):
    llama_model = LLAMA_MODEL()

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.llama_model.generate(prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "custom"}

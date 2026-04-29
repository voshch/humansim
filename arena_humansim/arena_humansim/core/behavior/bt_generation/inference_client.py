import os
from abc import ABC, abstractmethod
from typing import Any


class InferenceClient(ABC):
    @abstractmethod
    def generate(self, contents: str | list[str], system_instruction: str | list[str], response_json_schema: dict[str, Any] | None = None) -> str | None:
        raise NotImplementedError()


class GoogleGenAIClient(InferenceClient):
    def __init__(self, model: str):
        from google import genai

        if "GEMINI_API_KEY" not in os.environ:
            raise OSError("GEMINI_API_KEY environment variable not set!")

        self.inference_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def generate(self, contents: str | list[str], system_instruction: str | list[str], response_json_schema: dict[str, Any] | None = None) -> str | None:
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        res = self.inference_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=GenerateContentConfig(system_instruction=system_instruction, thinking_config=ThinkingConfig(include_thoughts=False, thinking_level="low"), response_json_schema=response_json_schema, response_mime_type="application/json" if response_json_schema is not None else None, temperature=0.0),
        )
        answer = res.text

        return answer


class OpenAIClient(InferenceClient):
    def __init__(self, model: str):
        from openai import OpenAI

        if "OPENAI_API_KEY" not in os.environ:
            raise OSError("OPENAI_API_KEY environment variable not set!")

        self.inference_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def generate(self, contents: str | list[str], system_instruction: str | list[str], response_json_schema: dict[str, Any] | None = None) -> str | None:
        # TODO: Handle structured output with `response_json_schema`
        response = self.inference_client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": str(system_instruction)}, {"role": "user", "content": str(contents)}])

        return response.choices[0].message.content

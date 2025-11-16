import os
import json

from tau_bench.model_utils.api.datapoint import Datapoint
from tau_bench.model_utils.model.chat import ChatModel, Message
from tau_bench.model_utils.model.completion import approx_cost_for_datapoint, approx_prompt_str
from tau_bench.model_utils.model.general_model import wrap_temperature
from tau_bench.model_utils.model.utils import approx_num_tokens


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
API_KEY_ENV_VAR = "GROQ_API_KEY"

# You may tune these maps as needed:
PRICE_PER_INPUT_TOKEN_MAP = {
    "openai/gpt-oss-20b": 0.2 / 1000000,
    "llama-3.3-70b-versatile": 0.25 / 1000000,
}
INPUT_PRICE_PER_TOKEN_FALLBACK = 1 / 1000000

CAPABILITY_SCORE_MAP = {
    "openai/gpt-oss-20b": 0.5,
    "llama-3.3-70b-versatile": 0.45,
}
CAPABILITY_SCORE_FALLBACK = 0.3

LATENCY_MS_PER_OUTPUT_TOKEN_MAP = {
    "openai/gpt-oss-20b": 0.4,
}
LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK = 0.6

MAX_CONTEXT_LENGTH_MAP = {
    "openai/gpt-oss-20b": 32768,
    "llama-3.3-70b-versatile": 4096,
}
MAX_CONTEXT_LENGTH_FALLBACK = 32768


class GroqModel(ChatModel):
    """
    A Groq-backed implementation of ChatModel,
    matching the interface and behavior of the OpenAI version.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.0):
        from openai import OpenAI  # Groq uses OpenAI-compatible client

        self.model = model or DEFAULT_GROQ_MODEL

        if api_key is None:
            api_key = os.getenv(API_KEY_ENV_VAR)
            if not api_key:
                raise ValueError(f"{API_KEY_ENV_VAR} environment variable is not set")

        # Groq: set `base_url` to their OpenAI-compatible endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.temperature = temperature

    def generate_message(
        self,
        messages: list[Message],
        force_json: bool,
        temperature: float | None = None,
    ) -> Message:
        if temperature is None:
            temperature = self.temperature

        msgs = self.build_generate_message_state(messages)
        res = self.client.chat.completions.create(
            model=self.model,
            input=msgs,
            temperature=wrap_temperature(temperature),
            response_format={"type": "json_object" if force_json else "text"},
        )
        return self.handle_generate_message_response(
            prompt=msgs,
            content=res.choices[0].message.content,
            force_json=force_json,
        )

    def get_approx_cost(self, dp: Datapoint) -> float:
        cost_per_token = PRICE_PER_INPUT_TOKEN_MAP.get(
            self.model, INPUT_PRICE_PER_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=cost_per_token)

    def get_latency(self, dp: Datapoint) -> float:
        latency_per_output_token = LATENCY_MS_PER_OUTPUT_TOKEN_MAP.get(
            self.model, LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=latency_per_output_token)

    def get_capability(self) -> float:
        return CAPABILITY_SCORE_MAP.get(self.model, CAPABILITY_SCORE_FALLBACK)

    def supports_dp(self, dp: Datapoint) -> bool:
        prompt = approx_prompt_str(dp)
        return approx_num_tokens(prompt) <= MAX_CONTEXT_LENGTH_MAP.get(
            self.model, MAX_CONTEXT_LENGTH_FALLBACK
        )

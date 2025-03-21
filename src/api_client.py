import base64
from functools import lru_cache
from typing import Dict, Optional

import anthropic
import google.generativeai as genai
from openai import AsyncOpenAI

from .config import (
    AVAILABLE_MODELS,
    CAPTION_GENERATION,
    CLAUDE_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    QA_EVALUATION,
    QA_GENERATION,
)


@lru_cache()
def get_required_providers() -> set:
    """Returns a list of providers for configured models"""
    providers = set()
    all_models = (
        CAPTION_GENERATION.get("models", [])
        + QA_GENERATION["system1"].get("models", [])
        + QA_GENERATION["system2"].get("models", [])
        + QA_EVALUATION["system1"].get("models", [])
        + QA_EVALUATION["system2"].get("models", [])
    )

    for model in all_models:
        if model in AVAILABLE_MODELS:
            providers.add(AVAILABLE_MODELS[model]["provider"])
    return providers


class APIClients:
    """API client management class"""

    _instance = None
    _clients: Dict[str, Optional[object]] = {
        "openai": None,
        "google": None,
        "anthropic": None,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_clients()
        return cls._instance

    def _initialize_clients(self):
        """필요한 API 클라이언트만 초기화"""
        required_providers = get_required_providers()

        if "openai" in required_providers and OPENAI_API_KEY:
            self._clients["openai"] = AsyncOpenAI(api_key=OPENAI_API_KEY)

        if "google" in required_providers and GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            self._clients["google"] = genai

        if "anthropic" in required_providers and CLAUDE_API_KEY:
            self._clients["anthropic"] = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def get_client(self, provider: str) -> Optional[object]:
        """특정 제공자의 API 클라이언트 반환"""
        return self._clients.get(provider)


# Init global client instance
api_clients = APIClients()


def get_model_config(model_name):
    """Get model configuration"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    return AVAILABLE_MODELS[model_name]


async def generate_vision_response(provider, image_path, prompt, model_name):
    """Generate image-based response"""
    try:
        model_config = get_model_config(model_name)
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            if provider == "openai":
                response = await api_clients.get_client(
                    "openai"
                ).chat.completions.create(
                    model=model_config["name"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=model_config.get("max_tokens", 1000),
                )
                return response.choices[0].message.content
            elif provider == "google":
                model = api_clients.get_client("google").GenerativeModel(
                    model_name=model_config["name"]
                )
                response = model.generate_content(
                    [{"mime_type": "image/jpeg", "data": base64_image}, prompt]
                )
                return response.text
            elif provider == "anthropic":
                message = api_clients.get_client("anthropic").messages.create(
                    model=model_config["name"],
                    max_tokens=model_config.get("max_tokens", 1000),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                return message.content[0].text
    except Exception as e:
        print(f"Error generating {provider} Vision API response: {e}")
        return None


async def generate_response(provider, prompt, model_name=None):
    """Generate text-based response"""
    try:
        model_config = get_model_config(model_name)
        if provider == "openai":
            if model_name == "o1-mini":
                response = await api_clients.get_client(
                    "openai"
                ).chat.completions.create(
                    model=model_config["name"],
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            response = await api_clients.get_client("openai").chat.completions.create(
                model=model_config["name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=model_config.get("max_tokens", 1000),
            )
            return response.choices[0].message.content
        elif provider == "google":
            model = api_clients.get_client("google").GenerativeModel(
                model_name=model_config["name"]
            )
            response = model.generate_content(prompt)
            return response.text
        elif provider == "anthropic":
            message = api_clients.get_client("anthropic").messages.create(
                model=model_config["name"],
                max_tokens=model_config.get("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
    except Exception as e:
        print(f"Error generating {provider} API response: {e}")
        return None


async def get_model_response(
    model_name: str, prompt: str, image_path: str = None
) -> Optional[str]:
    """Generate model response"""
    try:
        model_config = get_model_config(model_name)
        provider = model_config["provider"]

        if image_path:
            if provider in ["openai", "google", "anthropic"]:
                return await generate_vision_response(
                    provider, image_path, prompt, model_name
                )
        else:
            if provider in ["openai", "google", "anthropic"]:
                return await generate_response(provider, prompt, model_name)
        return None

    except Exception as e:
        print(f"Error generating model response: {e}")
        return None

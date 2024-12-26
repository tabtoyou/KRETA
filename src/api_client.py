import base64
from functools import lru_cache
from typing import Dict, Optional
from openai import OpenAI
import anthropic
import google.generativeai as genai
from .config import (
    OPENAI_API_KEY, 
    GOOGLE_API_KEY, 
    CLAUDE_API_KEY, 
    AVAILABLE_MODELS,
    CAPTION_GENERATION,
    QA_GENERATION,
    QA_EVALUATION
)

@lru_cache()
def get_required_providers() -> set:
    """설정된 모델들의 provider 목록을 반환"""
    providers = set()
    all_models = (
        CAPTION_GENERATION.get('models', []) +
        QA_GENERATION['system1'].get('models', []) +
        QA_GENERATION['system2'].get('models', []) +
        QA_EVALUATION['system1'].get('models', []) +
        QA_EVALUATION['system2'].get('models', [])
    )
    
    for model in all_models:
        if model in AVAILABLE_MODELS:
            providers.add(AVAILABLE_MODELS[model]['provider'])
    return providers

class APIClients:
    """API 클라이언트 관리 클래스"""
    _instance = None
    _clients: Dict[str, Optional[object]] = {
        'openai': None,
        'google': None,
        'anthropic': None
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_clients()
        return cls._instance

    def _initialize_clients(self):
        """필요한 API 클라이언트만 초기화"""
        required_providers = get_required_providers()
        
        if 'openai' in required_providers and OPENAI_API_KEY:
            self._clients['openai'] = OpenAI(api_key=OPENAI_API_KEY)
            
        if 'google' in required_providers and GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            self._clients['google'] = genai
            
        if 'anthropic' in required_providers and CLAUDE_API_KEY:
            self._clients['anthropic'] = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def get_client(self, provider: str) -> Optional[object]:
        """특정 제공자의 API 클라이언트 반환"""
        return self._clients.get(provider)

# 글로벌 클라이언트 인스턴스
api_clients = APIClients()

def get_model_config(model_name):
    """모델 설정 가져오기"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    return AVAILABLE_MODELS[model_name]

def generate_gpt_vision_response(image_path, prompt, model_name="gpt-4o-mini"):
    """OpenAI Vision API를 사용하여 이미지 기반 응답 생성"""
    try:
        model_config = get_model_config(model_name)
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = api_clients.get_client('openai').chat.completions.create(
                model=model_config['name'],
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            }
                        }
                    ]
                }],
                max_tokens=model_config.get('max_tokens', 1000),
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"이미지 캡션 생성 중 오류 발생: {e}")
        return None

def generate_gpt_response(prompt, model_name="gpt-4o-mini"):
    """OpenAI API를 사용하여 텍스트 기반 응답 생성"""
    try:
        model_config = get_model_config(model_name)
        response = api_clients.get_client('openai').chat.completions.create(
            model=model_config['name'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=model_config.get('max_tokens', 1000),
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT 응답 생성 중 오류 발생: {e}")
        return None

def generate_gemini_vision_response(image_path, prompt, model_name="gemini-2.0-flash-exp"):
    """Gemini Vision API를 사용하여 이미지 기반 응답 생성"""
    try:
        model_config = get_model_config(model_name)
        model = api_clients.get_client('google').GenerativeModel(model_name=model_config['name'])
        
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            
        response = model.generate_content([
            {
                'mime_type': 'image/jpeg',
                'data': base64.b64encode(image_data).decode('utf-8')
            },
            prompt
        ])
        
        return response.text
    except Exception as e:
        print(f"Gemini Vision 응답 생성 중 오류 발생: {e}")
        return None

def generate_gemini_response(prompt, model_name="gemini-2.0-flash-exp"):
    """Gemini API를 사용하여 텍스트 기반 응답 생성"""
    try:
        model_config = get_model_config(model_name)
        model = api_clients.get_client('google').GenerativeModel(model_name=model_config['name'])
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini 응답 생성 중 오류 발생: {e}")
        return None

def generate_sonnet_vision_response(image_path, prompt, model_name="claude-3-5-sonnet-latest"):
    """Claude Sonnet Vision API를 사용하여 이미지 기반 응답 생성"""
    try:
        model_config = get_model_config(model_name)
        
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            
        message = api_clients.get_client('anthropic').messages.create(
            model=model_config['name'],
            max_tokens=model_config.get('max_tokens', 1000),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(image_data).decode('utf-8')
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        return message.content[0].text
    except Exception as e:
        print(f"Claude Sonnet Vision 응답 생성 중 오류 발생: {e}")
        return None

def generate_sonnet_response(prompt, model_name="claude-3-5-sonnet-latest"):
    """Claude Sonnet API를 사용하여 텍스트 기반 응답 생성"""
    try:
        model_config = get_model_config(model_name)
        message = api_clients.get_client('anthropic').messages.create(
            model=model_config['name'],
            max_tokens=model_config.get('max_tokens', 1000),
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Claude Sonnet 응답 생성 중 오류 발생: {e}")
        return None

def get_model_response(model_name: str, prompt: str, image_path: str = None) -> Optional[str]:
    """모델별 응답 생성 함수
    
    Args:
        model_name: 사용할 모델 이름
        prompt: 프롬프트 텍스트
        image_path: 이미지 경로 (이미지 관련 요청시에만 사용)
    
    Returns:
        생성된 응답 텍스트 또는 None
    """
    try:
        model_config = get_model_config(model_name)
        provider = model_config['provider']
        
        # 이미지가 포함된 요청인 경우
        if image_path:
            if provider == 'openai':
                return generate_gpt_vision_response(image_path, prompt, model_name)
            elif provider == 'google':
                return generate_gemini_vision_response(image_path, prompt, model_name)
            elif provider == 'anthropic':
                return generate_sonnet_vision_response(image_path, prompt, model_name)
        # 텍스트만 있는 요청인 경우
        else:
            if provider == 'openai':
                return generate_gpt_response(prompt, model_name)
            elif provider == 'google':
                return generate_gemini_response(prompt, model_name)
            elif provider == 'anthropic':
                return generate_sonnet_response(prompt, model_name)
                
        return None
        
    except Exception as e:
        print(f"모델 응답 생성 중 오류 발생: {e}")
        return None
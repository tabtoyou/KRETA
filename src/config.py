import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

# Language setting
LANGUAGE = 'Korean'

AVAILABLE_MODELS = {
    'gpt-4o': {
        'provider': 'openai',
        'name': 'gpt-4o-2024-11-20',
        'max_tokens': 1000,
    },
    'gpt-4o-mini': {
        'provider': 'openai',
        'name': 'gpt-4o-mini-2024-07-18',
        'max_tokens': 1000,
    },
    'o1-mini': {
        'provider': 'openai',
        'name': 'o1-mini-2024-09-12',
    },
    'gemini-2.0-flash-exp': {
        'provider': 'google',
        'name': 'gemini-2.0-flash-exp',
    },
    'claude-3-sonnet': {
        'provider': 'anthropic',
        'name': 'claude-3-5-sonnet-latest',
        'max_tokens': 1000,
    }
}

CAPTION_GENERATION = {
    'models': ['gpt-4o-mini', 'gemini-2.0-flash-exp'],  
}

QA_GENERATION = {
    'system1': {
        'models': ['gpt-4o-mini', 'gemini-2.0-flash-exp'],  
        'candidates_per_model': 2, 
    },
    'system2': {
        'models': ['o1-mini', 'gemini-2.0-flash-exp'], 
        'candidates_per_model': 2,  
    }
}

QA_EVALUATION = {
    'system1': {
        'models': ['gpt-4o-mini', 'gemini-2.0-flash-exp'],  
        'num_to_select': 1, 
    },
    'system2': {
        'models': ['gpt-4o', 'gemini-2.0-flash-exp'],
        'num_to_select': 1,
    }
}
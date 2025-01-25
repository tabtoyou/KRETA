# KoTextVQA
Text-Rich VQA for Non-English Language in Real-World Contexts

## Settings
conda 혹은 virtualenv 이후
```
pip install -r requirements.txt
```

`.env` 생성 후 아래 key 값 입력
```
OPENAI_API_KEY=xxx
GOOGLE_API_KEY=xxx
CLAUDE_API_KEY=xxx
```

## Usage
VQA 데이터 생성
```
python main.py
```

VQA 데이터 확인 및 개선
```
streamlit run eval.py
```
import json

from .config import LANGUAGE

# Caption generation prompt
CAPTION_PROMPT = """이미지에서 텍스트를 포함한 시각적 요소들에 대해 최대한 상세하게 설명해주세요. 출력한 상세설명만 읽고도 이미지를 파악할 수 있도록 작성해 주세요.

1. 텍스트 외의 시각적 요소들
   - 전반적인 장면/문서의 구성과 분위기
   - 주요 사물 혹은 객체들의 특징
   - 배경 및 시공간의 세부 특징

2. 텍스트와 이미지 요소 간의 관계성
    - 추출한 텍스트 정보와 이미지 요소 간의 시각적/의미적 연관성 분석
    - 각 텍스트 정보가 이미지 내에서 전달하는 정보가 무엇인지 분석

3. 이미지 내 텍스트 정보
   3.1 이미지에 포함된 **모든 텍스트**를 출력
   3.2 앞서 출력한 모든 텍스트 정보를 json과 같이 구조화된 형태로 재구성해 출력"""


# QA generation prompt
SYSTEM1_QA_TEMPLATE = """다음은 이미지 내 텍스트 정보를 인식하고 이해하는 능력을 평가하기 위한 질의응답 생성 요청입니다.
아래 <이미지 설명>을 바탕으로, 이미지 속 텍스트를 정확히 파악하고 그 의미를 이해함으로써만 정답에 도달할 수 있는 단답형 질문과 답변을 총 {num_questions}개 생성해주세요.
<이미지 설명>으로 여러 캡션이 주어질 수 있습니다. 여러 개일 경우 각각은 다른 모델이 같은 이미지를 보고 캡셔닝한 결과로, 공통적으로 언급한 부분이 신뢰도가 높습니다.

<문제 설계 조건>
- 이미지에 포함된 텍스트를 인식하지 않고는 답변할 수 없을 것
- 외부 지식 없이, 이미지에 나타난 텍스트 정보만으로 정답을 추론할 수 있을 것
- 이미지 내 텍스트를 이해해야 정답을 도출할 수 있을 것
- 객관적이고 명확한 근거를 바탕으로 하나의 정답이 도출될 수 있도록 할 것
- 각 답변은 한 단어나 짧은 구 형태로 제시할 것

<예시1>
이미지에 '문화센터 가는 길'이라는 텍스트와 왼쪽 화살표가 있고, '운동장 가는 길'이라는 텍스트와 오른쪽 화살표가 있을 때,
{{
    "question": "문화센터로 가려면 사진을 찍고 있는 사람을 기준으로 어느 방향으로 가야하나요?",
    "answer": "왼쪽"
}}

<예시2>
문서 이미지 내에 인증서 관련 내용과 '중소벤처기업부'라는 텍스트가 도장과 함께 있을 때,
{{
    "question": "이 문서를 발급한 기관은 어디인가요?",
    "answer": "중소벤처기업부"
}}  


<이미지 설명>
{image_caption}

아래와 같은 JSON 형식으로 출력해주세요:
{{
    "qa_list": [
        {{"question": "이미지 텍스트 인식 기반 질문1", "answer": "정답1"}},
        {{"question": "이미지 텍스트 인식 기반 질문2", "answer": "정답2"}},
        {{"question": "이미지 텍스트 인식 기반 질문3", "answer": "정답3"}}
    ]
}}
"""

SYSTEM2_QA_TEMPLATE = """다음은 이미지 내 텍스트 정보를 기반으로 심화된 추론 능력을 평가하기 위한 질의응답 생성 요청입니다.
사람 또는 AI가 이미지만 보고 문제를 풀 수 있는지 평가하기 위한 질의응답 {num_questions}개를 생성해 주세요.
특히 이미지 내 텍스트를 스스로 인식 및 이해하고 **추론하는 능력**이 중요하기 때문에, 이를 평가할 수 있는 질문을 생성해주세요.

문제 설계 조건:
- <이미지 설명>의 언어를 사용하는 국가와 관련된 외부 지식이나 문화적/역사적 배경지식 등을 통한 추론이 필요한 문제를 생성
- <이미지 설명>으로 여러 캡션이 주어질 수 있습니다. 여러 개일 경우 각각은 다른 모델이 같은 이미지를 보고 캡셔닝한 결과로, 공통적으로 언급한 부분이 신뢰도가 높습니다.
- 단순한 텍스트 인식에 그치지 않고, 텍스트를 토대로 깊은 의미의 이해나 수학적/논리적 추론이 필요한 문제를 생성 (기간, 수치 계산 등)
- 사용법과 관련된 정보가 포함된 이미지일 경우, 다음 스텝에 대한 질문 등을 생성
- 단순히 이미지 내 정보만으로는 답하기 어렵게 구성하고, 최종적인 정답은 한 단어나 짧은 구 형태로 제시
- 일반적으로 이미지를 보고 사람들이 궁금해 할 부분들을 질문으로 작성
- 객관적이고 명확한 근거를 바탕으로 모호하지 않은 질문과 정답을 생성
- 정답을 도출하기 위해 필요한 추론 과정을 "reasoning" 필드에 명시

<예시1>
이미지에 '손기정 문화센터 가는 길' 이라는 표지판이 있을 때, 
{{
    "question": "이미지 속에 표시된 문화센터는 한 역사적 인물을 기념하기 위해 지어졌습니다. 이 인물과 관련된 스포츠는 무엇일까요?", 
    "reasoning": "이미지 속에는 '손기정 문화센터'가 적힌 표지판이 있다. 손기정 선수는 1936 베를린 올림픽의 마라톤 우승자로, 해당 인물과 관련된 스포츠는 마라톤이다.",
    "answer": "마라톤"
}}

<예시2>
이미지에 '관악로14길'이라는 거리 이름이 있을 때,
{{
    "question": "이미지 속 거리의 행정구역은 어디인가요?",
    "reasoning": "'관악로14길'과 같은 거리 이름은 대한민국 서울의 관악구와 관련이 있을 가능성이 높다. 일반적으로 거리 이름에 지역명이 포함되므로, 이 정보는 행정구역을 추론하는 데 유용하다.",
    "answer": "서울시 관악구"
}}

<예시3>
이미지에 '점심 특선 세트 제공 시간: 11:00 am - 14:30 pm'이라는 시간 정보가 있을 때,
{{
    "question": "점심 특선 세트의 제공 시간은 몇 시간인가요?",
    "reasoning": "점심 특선 세트는 '11:00 am - 14:30 pm'로 명시되어 있어, 이를 계산하면 3시간 30분의 시간 차이가 생긴다.",
    "answer": "3시간 30분"
}}

아래의 JSON 형식으로 출력해주세요:
{{
    "qa_list": [
        {{
            "question": "추론 필요한 질문1",
            "reasoning": "정답 도출을 위한 추론 과정1",
            "answer": "정답1"
        }},
        {{
            "question": "추론 필요한 질문2",
            "reasoning": "정답 도출을 위한 추론 과정2",
            "answer": "정답2"
        }}
]
}}
 
<이미지 설명>
{image_caption}
"""

# Evaluation prompt
SYSTEM1_EVAL_TEMPLATE = """이미지와 관련된 질의응답 <QA 후보>를 평가하고, 순위를 결정해 주세요.

- 아래 <응답 형식>에 맞춰 reasoning에 판단 근거를 작성하고, ranking에 순위를 답하세요.
- 각 QA가 <평가 기준>의 5개 항목에 대해 각각 0~5점으로 평가하여 총점 25점 중 몇 점인지 **엄격하고 비판적으로 평가**해 reasoning에 기록해 주세요. 그리고 각 QA의 점수를 합산하여 최종 순위를 결정하세요.

<평가 기준>
Text Utilization: 이미지 내 텍스트 인식이 필수적이며 텍스트 정보를 얼마나 활용하는가? (0~5점)
Clarity: 질문이 구체적이고 명확하며, 답변이 정확하고 명료한가? (0~5점)
Factfulness: 이미지를 고려했을 때 답변이 사실에 기반하여 정확하고 신뢰할 수 있는가? (0~5점)
Naturalness: 질문과 답변이 자연스럽고 어색하지 않은가? (0~5점)
Alignment: 질문의 내용과 의도가 이미지의 정보 전달 의도에 얼마나 잘 부합하는가? (0~5점)

<응답 형식>
{{"reasoning": "순위 판단 근거", "ranking": [1위 질문 번호, 2위 질문 번호, 3위 질문 번호, ..., n위 질문 번호]}}

<응답 예시>
{{"reasoning": "QA 1:\n- Text Utilization: 3점\n- Clarity: 4점\n ... \n\n총합\n- QA 1: 20점\n- QA 2: 23점 ...", "ranking": [2, 3, 1, ..., n]}}

<QA 후보>
{qa_candidates}
"""

SYSTEM2_EVAL_TEMPLATE = """이미지와 관련된 질의응답 <QA 후보>를 평가하고, 순위를 결정해 주세요.

- 아래 <응답 형식>에 맞춰 reasoning에 판단 근거를 작성하고, ranking에 순위를 답하세요.
- 각 QA가 <평가 기준>의 5개 항목에 대해 0~5점으로 평가하여 총점 35점 중 몇 점인지 **엄격하고 비판적으로 평가**해 reasoning에 기록해 주세요. 그리고 각 QA의 점수를 합산하여 최종 순위를 결정하세요.

<평가 기준>
Text Utilization: 이미지 내 텍스트 인식이 필수적이며 텍스트 정보를 얼마나 활용하는가? (0~5점)
Clarity: 질문이 구체적이고 명확하며, 답변이 정확하고 명료한가? (0~5점)
Factfulness: 이미지를 고려했을 때 답변이 사실에 기반하여 정확하고 신뢰할 수 있는가? (0~5점)
Naturalness: 질문과 답변이 자연스럽고 어색하지 않은가? (0~5점)
Alignment: 질문의 내용과 의도가 이미지의 정보 전달 의도에 얼마나 잘 부합하는가? (0~5점)
Complexity: 질문이 한 번 이상의 추론을 요구하며, 이미지 내 정보와 관련된 일반 상식을 묻는 등 난이도가 쉽지 않고 적절한가? (0~5점)
Coherence: 추론 과정이 논리적이고 단계적으로 명확한가? (0~5점)

<응답 형식>
{{"reasoning": "순위 판단 근거", "ranking": [1위 질문 번호, 2위 질문 번호, 3위 질문 번호, ..., n위 질문 번호]}}

<응답 예시>
{{"reasoning": "QA 1:\n- Text Utilization: 3점\n- Clarity: 4점\n ... \n\n총합\n- QA 1: 20점\n- QA 2: 23점 ...", "ranking": [2, 3, 1, ..., n]}}

<QA 후보>
{qa_candidates}
"""

# Similar options generation prompt
HARD_NEGATIVE_OPTIONS_PROMPT = """주어진 이미지와 이미지에 대한 <질문>과 <정답>을 참고하여 <정답>과 유사한 오답 옵션을 json 형식으로 출력해 주세요:

- 정답과 유사하지만 정답이 아닌 3개의 옵션을 생성
- 형태나 주제가 기존 정답과 비슷해 구별하기 어려운 Hard Negatives로 작성
- 실제 정답과 혼동될 수 있도록 설계

<질문>
{question}

<정답>
{correct_answer}

아래와 같은 JSON 형식으로 출력해 주세요:
{{
    "options": ["hard negative 1", "hard negative 2", "hard negative 3"],
}}
"""

# Image type and domain classification prompt
DOMAIN_AND_TYPE_PROMPT = """이미지에 대해 자세히 설명한 <이미지 설명> 정보를 참고하여 아래 2가지 작업들을 수행해 후 json 형식으로 출력해 주세요:

### 1. 이미지 타입 분류
이미지의 타입을 아래 카테고리 중 하나 이상을 선택해서 영어로 작성. 아래 카테고리에 포함되지 않는 경우 비어있는 [""]로 출력
- 보고서 (Report)
- 시험지 (Test_Paper)
- 종이신문 (Newspaper)
- 안내서/매뉴얼 (Manual)
- 책 페이지 (Book_Page)
- 잡지 (Magazine)
- 브로슈어 (Brochure)
- 책표지 (Book_Cover)
- 그림책/만화 (Illustrated_Books_and_Comics)
- 차트 (Chart_and_Plot)
- 표 (Table)
- 다이어그램 (Diagram)
- 인포그래픽 (Infographic)
- 포스터 (Poster)
- 배너 (Banner)
- 메뉴판 (Menu)
- 포장 라벨 (Packaging_Label)
- 광고 전단 (Flyer)
- 안내판 (Signage)
- 상점 간판 (Store_Sign)
- 상품 상세 이미지 (Product_Detail)
- 공공 표지판 (Public_Signs)
- 도로 표지판 (Street_Signs)
- 벽화 (Mural_and_Graffiti)
- 모바일 화면 (Mobile_Screenshot)
- PC 화면 (PC_Screenshot)
- 슬라이드 (Presentation_Slides)
- 영상 표지 (Video_Thumbnail)
- 영상 장면 (Video_Scene)
- 영수증 (Receipts_and_Invoices)
- 계약서 (Contracts_Documents)
- 수료증/인증서 (Certificates)
- 손글씨 (Handwriting)
- 티켓/승선권 (Tickets_and_Boarding_Passes)

### 2. 도메인 분류
이미지의 용도나 목적을 나타내는 구체적인 도메인을 아래 카테고리 중 하나로 영어로 작성
- 공공/행정 (Public_and_Administration)
- 법률/규제 (Legal_and_Regulations)
- 경제/금융 (Economics_and_Finance)
- 기업/비즈니스 (Corporate_and_Business)
- 마케팅/광고 (Marketing_and_Advertising)
- 교육/학술 (Education_and_Academia)
- 의료/보건 (Medical_and_Healthcare)
- 교통/물류 (Transportation_and_Logistics)
- 관광/여행 (Travel_and_Tourism)
- 유통/상거래 (Retail_and_Commerce)
- 외식/숙박 (Hospitality_and_Food_Service)
- 오락/미디어 (Entertainment_and_Media)
- 과학/기술 (Science_and_Technology)
- 인문/예술 (Arts_and_Humanities)
- 개인/생활 (Personal_and_Lifestyle)

<이미지 설명>
{image_caption}

아래와 같은 JSON 형식으로 출력해 주세요:
{{
    "img_type": ["image type"],
    "domain": ["domain"]
}}
"""


def format_qa_generation_prompt(
    system_type: str, image_caption: str, num_questions: int
) -> str:
    """Format QA generation prompt"""
    template = SYSTEM1_QA_TEMPLATE if system_type == "system1" else SYSTEM2_QA_TEMPLATE
    return template.format(
        image_caption=image_caption, num_questions=num_questions, language=LANGUAGE
    )


def format_type_domain_generation_prompt(image_caption: str) -> str:
    """Format type and domain generation prompt"""
    template = DOMAIN_AND_TYPE_PROMPT
    return template.format(image_caption=image_caption)


def format_qa_evaluation_prompt(system_type: str, qa_candidates: list) -> str:
    """Format QA evaluation prompt"""
    template = (
        SYSTEM1_EVAL_TEMPLATE if system_type == "system1" else SYSTEM2_EVAL_TEMPLATE
    )

    # Convert QA candidates list to string
    qa_list_str = json.dumps(qa_candidates, ensure_ascii=False, indent=2)

    # Split lines for adding numbers
    lines = qa_list_str.split("\n")

    # Add numbers to the start of each item in the "qa_list" array
    current_qa = 0
    numbered_lines = []
    for line in lines:
        if '"question"' in line:  # Start of a new QA item
            current_qa += 1
            numbered_lines.append(f"\nQA {current_qa}:")
        numbered_lines.append(line)

    qa_list_str = "\n".join(numbered_lines)

    print(qa_list_str)

    return template.format(qa_candidates=qa_list_str, language=LANGUAGE)

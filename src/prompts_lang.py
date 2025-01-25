import json
from .config import LANGUAGE

# Caption generation prompt
CAPTION_PROMPT = """You possess advanced expertise in the detailed analysis and interpretation of text contained within images. Please describe all visual elements in the image, including text, step by step in detail, according to the requirements outlined below. Please respond in {language}.

1. Text Information
   - List ALL text present in the image

2. Visual Elements (excluding text)
   - Overall scene/document composition and atmosphere
   - Key objects and their characteristics
   - Background and contextual details

3. Text-Image Relationship
    - Analyze visual and semantic connections between text and image elements
    - Explain the purpose and context of each text element within the image"""

# QA generation prompts
SYSTEM1_QA_TEMPLATE = """This is a request to generate question-answer pairs that evaluate the ability to recognize and understand text in images.
Please generate all questions and answers in {language}.

Based on the <Image Description> below, please create {num_questions} short-answer questions where the answers can only be reached by accurately identifying and understanding the text in the image.
If multiple captions are provided in <Image Description>, they are different models' interpretations of the same image. Information mentioned consistently across captions is likely to be more reliable.

<Question Design Requirements>
- Questions must require text recognition to be answered
- Answers should be derivable solely from text information in the image without external knowledge
- Questions should test understanding of the text content, not just recognition
- Questions should have clear, objective evidence leading to a single correct answer
- Answers should be in the form of a single word or short phrase

<Example 1>
When the image shows text "Cultural Center" with a left arrow and "Playground" with a right arrow:
{{
    "question": "Which direction should one go to reach the Cultural Center from the photographer's position?",
    "answer": "left"
}}

<Example 2>
When the image shows a certificate with "Ministry of SMEs and Startups" text and an official seal:
{{
    "question": "Which organization issued this document?",
    "answer": "Ministry of SMEs and Startups"
}}

<Image Description>
{image_caption}

Please output in the following JSON format:
{{
    "qa_list": [
        {{"question": "text-based question 1", "answer": "answer 1"}},
        {{"question": "text-based question 2", "answer": "answer 2"}},
        {{"question": "text-based question 3", "answer": "answer 3"}}
    ]
}}
"""

SYSTEM2_QA_TEMPLATE = """This is a request to generate question-answer pairs that evaluate advanced reasoning abilities based on text information in images.
Please generate all questions, reasoning, and answers in {language}.

Please create {num_questions} QA pairs that assess how well a human or AI can solve problems by looking at the image alone.
Focus on evaluating the ability to recognize, understand, and **reason with** the text in the image.

Question Design Requirements:
- Create questions that require reasoning using cultural/historical background knowledge relevant to the language and region shown in the image
- If multiple captions are provided, they are different models' interpretations of the same image. Information mentioned consistently is more reliable
- Questions should require deep understanding or mathematical/logical reasoning based on the text (e.g., date calculations, numerical analysis)
- For instructional images, include questions about next steps or procedures
- Make questions challenging beyond simple information retrieval, but keep answers concise
- Avoid using direct text from the image as answers
- Ensure questions and answers have clear, objective reasoning
- Include the reasoning process in the "reasoning" field

<Example 1>
When the image shows a sign for "Son Kee-jung Cultural Center":
{{
    "question": "What sport is associated with the historical figure this cultural center commemorates?",
    "reasoning": "The sign shows 'Son Kee-jung Cultural Center'. Son Kee-jung was the marathon gold medalist at the 1936 Berlin Olympics, so the sport associated with him is marathon.",
    "answer": "marathon"
}}

<Example 2>
When the image shows a street name "Gwanak-ro 14-gil":
{{
    "question": "What administrative district is the street in?",
    "reasoning": "'Gwanak-ro 14-gil' is a street name in Seoul, South Korea. Generally, street names include the district name, so this information is useful for determining the administrative district.",
    "answer": "Seoul, South Korea"
}}

<Example 3>
When the image shows a time information "Lunch Special Set Service Time: 11:00 am - 14:30 pm":
{{
    "question": "How long is the Lunch Special Set Service Time?",
    "reasoning": "The Lunch Special Set Service Time is '11:00 am - 14:30 pm', which results in a 3.5-hour time difference.",
    "answer": "3.5 hours"
}}

Please output in the following JSON format:
{{
    "qa_list": [
        {{
            "question": "reasoning-based question 1",
            "reasoning": "reasoning process 1",
            "answer": "answer 1"
        }},
        {{
            "question": "reasoning-based question 2",
            "reasoning": "reasoning process 2",
            "answer": "answer 2"
        }}
    ]
}}

<Image Description>
{image_caption}
"""

# Evaluation prompts
SYSTEM1_EVAL_TEMPLATE = """Please evaluate and rank the following question-answer pairs based on the given image.

Please select the most appropriate QA pairs according to these criteria:

Evaluation Criteria:
1. Does it require and accurately utilize text recognition from the image?
2. Is the answer clear, objective, and verifiable?
3. Is the difficulty level appropriate (neither too easy nor too difficult)?
4. Is the question specific and clearly written?
5. Does it effectively integrate both image and text information?

Please rank the QA candidates according to the above criteria. Respond with only the ranking in the format below.

<Response Format>
{{"ranking": [1st question number, 2nd question number, 3rd question number, ..., nth question number]}}

<Example Response>
{{"ranking": [2, 3, 1, ..., n]}}

<QA Candidates>
{qa_candidates}
"""

SYSTEM2_EVAL_TEMPLATE = """Please rank the reasoning-based QA pairs based on the given image according to the following criteria.

<Evaluation Criteria>
1. Does it require higher-order reasoning based on text in the image?
2. Is the reasoning process logical and clearly structured?
3. Is the difficulty level appropriately challenging?
4. Are the questions specific and answers accurate?
5. Does it effectively integrate both image and text information?

Please respond with only the ranking in the format below.

<Response Format>
{{"ranking": [1st question number, 2nd question number, 3rd question number, ..., nth question number]}}

<Example Response>
{{"ranking": [2, 3, 1, ..., n]}}

<QA Candidates>
{qa_candidates}
"""

# Hard negative options generation prompt
HARD_NEGATIVE_OPTIONS_PROMPT = """Based on the given image and the <Question> and <Answer>, please generate similar but incorrect options in JSON format:

- Generate 3 options that are similar to but different from the correct answer
- Create hard negatives that are difficult to distinguish from the correct answer
- Design options that could potentially be confused with the actual answer

<Question>
{question}

<Answer>
{correct_answer}

Please output in the following JSON format:
{{
    "options": ["hard negative 1", "hard negative 2", "hard negative 3"],
}}
"""

# Image type and domain classification prompt
DOMAIN_AND_TYPE_PROMPT = """Please analyze the given image and perform the following 2 tasks:

### 1. Image Type Classification
Select one or more categories from the list below. If none apply, output an empty [""]
- Report
- Test_Paper
- Newspaper
- Manual
- Book_Page
- Magazine
- Brochure
- Book_Cover
- Illustrated_Books_and_Comics
- Chart_and_Plot
- Table
- Diagram
- Infographic
- Poster
- Banner
- Menu
- Packaging_Label
- Flyer
- Signage
- Store_Sign
- Product_Detail
- Public_Signs
- Street_Signs
- Mural_and_Graffiti
- Mobile_Screenshot
- PC_Screenshot
- Presentation_Slides
- Video_Thumbnail
- Video_Scene
- Receipts_and_Invoices
- Contracts_Documents
- Certificates
- Handwriting
- Tickets_and_Boarding_Passes

### 2. Domain Classification
Select one category that best represents the image's purpose or context
- Public_and_Administration
- Legal_and_Regulations
- Economics_and_Finance
- Corporate_and_Business
- Marketing_and_Advertising
- Education_and_Academia
- Medical_and_Healthcare
- Transportation_and_Logistics
- Travel_and_Tourism
- Retail_and_Commerce
- Hospitality_and_Food_Service
- Entertainment_and_Media
- Science_and_Technology
- Arts_and_Humanities
- Personal_and_Lifestyle

Please output in the following JSON format:
{{
    "img_type": ["image type"],
    "domain": ["domain"]
}}
"""

def format_qa_generation_prompt(system_type: str, image_caption: str, num_questions: int) -> str:
    """Format QA generation prompt"""
    template = SYSTEM1_QA_TEMPLATE if system_type == 'system1' else SYSTEM2_QA_TEMPLATE
    return template.format(
        image_caption=image_caption,
        num_questions=num_questions,
        language=LANGUAGE
    )

def format_qa_evaluation_prompt(system_type: str, qa_candidates: list) -> str:
    """Format QA evaluation prompt"""
    template = SYSTEM1_EVAL_TEMPLATE if system_type == 'system1' else SYSTEM2_EVAL_TEMPLATE
    
    # Convert QA candidates list to string
    qa_list_str = json.dumps(qa_candidates, ensure_ascii=False, indent=2)
    
    return template.format(
        qa_candidates=qa_list_str,
        language=LANGUAGE
    ) 
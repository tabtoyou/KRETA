import os
import json
from tqdm import tqdm
import sys
import json
import sys
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset

if len(sys.argv) == 3:
    MODEL = sys.argv[1]
    SETTING = sys.argv[2]
else:
    print("Usage: python script.py [MODEL] [SETTING], default: python infer/infer_gpt.py gpt-4o-mini default")
    MODEL = 'gpt-4o-mini'
    SETTING = 'default'

API_KEY = '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'
WORKERS = 20
NUM = 2577

prompt_config = {
    'default': "Please select the correct answer from the options above. The last line of your response should be of the following format: 'Answer: LETTER' (without quotes) where LETTER is one of options.",
}

def parse_options(doc):
    options = []
    option_letters = ["A", "B", "C", "D"]
    for letter in option_letters:
        if letter in doc:
            options.append(doc[letter])
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(doc)
    question = f"Question: {question}\nOptions:\n{parsed_options}\n{prompt_config[SETTING]}"
    return question

def kotextvqa_doc_to_text(doc):
    question = construct_prompt(doc)
    return question

def vision_kotextvqa_doc_to_visual(doc):
    return doc['image']

def load_model(model_name="GPT4", base_url="", api_key="", model="gpt-4o-mini"):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    return model_components

# Function to send request with image and text
def request_with_image(text, image, timeout=60, max_tokens=300, base_url="https://api.openai.com/v1", api_key="123", model="gpt-4o-mini"):
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    base64_image = image
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response.choices[0].message.content

def infer(prompt, image, max_tokens=4096, use_vllm=False, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)

    try:
        response = request_with_image(prompt, image, max_tokens=max_tokens, base_url=base_url, api_key=api_key, model=model)
    except Exception as e:
        print(f"Error: {e}") 
        response = {"error": str(e)}
    
    return response


def process_prompt(data, model_components):
    prompt = kotextvqa_doc_to_text(data)
    image = vision_kotextvqa_doc_to_visual(data)

    return infer(prompt, image, max_tokens=4096, **model_components), data

def run_and_save():
    def save_results_to_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output
                # Exclude image fields but keep topic_difficulty and image_type
                data = {k: v for k, v in data.items() if not (k.startswith('image') and k != 'image_type')}
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
    
    dataset = load_dataset('tabtoyou/KoTextVQA', split='test')
    model_components = load_model(model_name='GPT4O-MINI', base_url="https://api.openai.com/v1", api_key=API_KEY, model=MODEL)
    
    def process_and_save_part(part_data, part_name, model_components):
        print(f"Begin processing {part_name}")
        results = []
        output_path = f"./output/{MODEL}_{part_name}.jsonl"

        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    data = json.loads(line)
                    results.append((data['response'], data))
            print(f"Loaded existing results for {part_name}")
        else:
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(process_prompt, data, model_components) for data in part_data]
                for future in tqdm(futures, desc=f"Processing {part_name}"):
                    result, data = future.result()
                    results.append((result, data))

            save_results_to_file(results, output_path)

        return output_path

    temp_files = []
    temp_files.append(process_and_save_part(dataset, SETTING, model_components))


def main():
    run_and_save()


if __name__ == '__main__':  
    main()
import os
from paddleocr import PaddleOCR
from src.utils import (
    filter_images, 
    aggregate_votes,
    parse_ranking_response,
)
from src.api_client import get_model_response
import json
import time
import argparse
import pandas as pd
from PIL import Image, ExifTags
import io
import asyncio
from src.config import (
    CAPTION_GENERATION, 
    QA_GENERATION, 
    QA_EVALUATION,
    LANGUAGE
)
from src.prompts import CAPTION_PROMPT, format_qa_generation_prompt, format_qa_evaluation_prompt, HARD_NEGATIVE_OPTIONS_PROMPT, DOMAIN_AND_TYPE_PROMPT
from typing import Dict, List, Optional

class TVQAGenerator:
    def __init__(self):
        self.ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang=LANGUAGE.lower(),
            use_gpu=True,
            det=True,
            rec=True,
            show_log=False
        )
        self.collected_data = []

    async def generate_image_caption(self, image_path):
        """Generate detailed image caption"""
        caption_tasks = [
            get_model_response(model_name, CAPTION_PROMPT, image_path)
            for model_name in CAPTION_GENERATION['models']
        ]
        captions = await asyncio.gather(*caption_tasks)
        formatted_captions = [f'<Caption {i + 1}>\n{caption}' for i, caption in enumerate(captions) if caption]
        return '\n\n'.join(formatted_captions)

    async def generate_qa_candidates(self, image_caption: str) -> Dict[str, Dict[str, List[dict]]]:
        """Generate QA candidates"""
        qa_candidates = {
            'system1': {'qa_list': []},
            'system2': {'qa_list': []}
        }
        
        # System1 QA generation
        system1_tasks = [
            get_model_response(
                model_name, 
                format_qa_generation_prompt('system1', image_caption, QA_GENERATION['system1']['candidates_per_model'])
            )
            for model_name in QA_GENERATION['system1']['models']
        ]
        system1_responses = await asyncio.gather(*system1_tasks)
        
        for response in system1_responses:
            if response:
                try:
                    qa_list = json.loads(response.replace('```json', '').replace('```', '').replace('\n','').strip())['qa_list']
                    qa_candidates['system1']['qa_list'].extend(qa_list)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    continue
        
        # System2 QA generation
        system2_tasks = [
            get_model_response(
                model_name, 
                format_qa_generation_prompt('system2', image_caption, QA_GENERATION['system2']['candidates_per_model'])
            )
            for model_name in QA_GENERATION['system2']['models']
        ]
        system2_responses = await asyncio.gather(*system2_tasks)
        
        for response in system2_responses:
            if response:
                try:
                    qa_list = json.loads(response.replace('```json', '').replace('```', '').replace('\n','').strip())['qa_list']
                    qa_candidates['system2']['qa_list'].extend(qa_list)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    continue
                
        return qa_candidates

    async def multi_models_evaluation(self, qa_candidates: Dict, image_path: str) -> Dict[str, Optional[int]]:
        """Perform evaluation using multiple models"""
        evaluations = {
            'system1': {},
            'system2': {}
        }
        
        # Perform evaluation for each system
        for system_type in ['system1', 'system2']:
            prompt = format_qa_evaluation_prompt(
                system_type, 
                qa_candidates['system1']['qa_list'] if system_type == 'system1' else qa_candidates['system2']['qa_list']
            )
            eval_tasks = [
                get_model_response(model_name, prompt, image_path)
                for model_name in QA_EVALUATION[system_type]['models']
            ]
            responses = await asyncio.gather(*eval_tasks)

            print("multi_models_evaluation: ", responses)
            
            for model_name, response in zip(QA_EVALUATION[system_type]['models'], responses):
                if response:
                    evaluations[system_type][model_name] = parse_ranking_response(response)
        
        print('evaluation 1: ', evaluations['system1'])
        print('evaluation 2: ', evaluations['system2'])
        
        # Aggregate voting results
        return {
            'system1': aggregate_votes(evaluations['system1'], QA_EVALUATION['system1']['num_to_select']),
            'system2': aggregate_votes(evaluations['system2'], QA_EVALUATION['system1']['num_to_select'])
        }

    async def generate_options_and_type(self, qa_candidates: Dict, evaluated_qa: Dict, image_path: str) -> Dict:
        """Generate options and type"""
        # Generate image type and domain information once
        type_domain_response = await get_model_response(
            QA_EVALUATION['system2']['models'][0], 
            DOMAIN_AND_TYPE_PROMPT, 
            image_path
        )
        
        try:
            parsed_type_domain = json.loads(type_domain_response.replace('```json', '').replace('```', '').replace('\n','').strip())
            img_type = parsed_type_domain.get('img_type', [""])
            domain = parsed_type_domain.get('domain', [""])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error occurred during type/domain generation: {str(e)}")
            img_type = [""]
            domain = [""]

        # Add new fields to qa_candidates structure
        for system_type in ['system1', 'system2']:
            qa_candidates[system_type].update({
                'options': [""],
                'img_type': img_type,
                'domain': domain
            })
            
            if evaluated_qa[system_type] is not None:
                selected_indices = [evaluated_qa[system_type] - 1] if isinstance(evaluated_qa[system_type], int) else \
                                 [idx - 1 for idx in evaluated_qa[system_type]]
                
                option_tasks = []
                for idx in selected_indices:
                    qa = qa_candidates[system_type]['qa_list'][idx]
                    prompt = HARD_NEGATIVE_OPTIONS_PROMPT.format(
                        question=qa['question'],
                        correct_answer=qa['answer']
                    )
                    option_tasks.append(
                        get_model_response(QA_EVALUATION['system2']['models'][0], prompt, image_path)
                    )
                
                option_responses = await asyncio.gather(*option_tasks)
                
                for response in option_responses:
                    try:
                        parsed_response = json.loads(response.replace('```json', '').replace('```', '').replace('\n','').strip())
                        qa_candidates[system_type]['options'] = parsed_response.get('options', [""])
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error occurred during options generation: {str(e)}")
        
        return qa_candidates

    def save_final_qa_dataset(self, evaluated_qa, qa_candidates, image_path, image_caption):
        """Collect QA data"""
        # Convert image to bytes
        with Image.open(image_path) as img:
            original_format = img.format or 'JPEG'

            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(img._getexif().items())

                if orientation in exif:
                    if exif[orientation] == 3:
                        img = img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img = img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=original_format)
            img_byte_arr = img_byte_arr.getvalue()
        
        # Process System1 and System2 data
        for system_type in ['system1', 'system2']:
            if evaluated_qa[system_type] is not None:
                selected_indices = [evaluated_qa[system_type] - 1] if isinstance(evaluated_qa[system_type], int) else \
                                 [idx - 1 for idx in evaluated_qa[system_type]]
                selected_qa_list = [qa_candidates[system_type]['qa_list'][idx] for idx in selected_indices]
                
                self.collected_data.append({
                    'image': img_byte_arr,
                    'detailed_caption': image_caption,
                    'candidates': qa_candidates[system_type]['qa_list'],
                    'selected_qa': selected_qa_list,
                    'system': int(system_type[-1]),
                    'options': qa_candidates[system_type]['options'],
                    'img_type': qa_candidates[system_type]['img_type'],
                    'domain': qa_candidates[system_type]['domain'],
                })
        
        return True

    async def generate_qa_from_image_directory(self, dir_path, output_dir, save_interval=20):
        """Generate QA dataset from image directory"""
        filtered_results = filter_images(dir_path, self.ocr_model)
        os.makedirs(output_dir, exist_ok=True)
        
        processed_dir = os.path.join(os.path.dirname(dir_path), 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        processed_count = 0
        
        for item in filtered_results:
            print(f"\n진행 상황: {processed_count + 1}/{len(filtered_results)} 이미지 처리 중...")
            try:
                image_caption = await self.generate_image_caption(item['image_path'])
                print("image_caption: ", image_caption)

                qa_candidates = await self.generate_qa_candidates(image_caption)
                print("qa_candidates: ", qa_candidates)
                evaluated_qa = await self.multi_models_evaluation(qa_candidates, item['image_path'])
                print("evaluated_qa: ", evaluated_qa)
                
                qa_candidates = await self.generate_options_and_type(
                    qa_candidates,
                    evaluated_qa,
                    item['image_path']
                )
                
                
                self.save_final_qa_dataset(
                    evaluated_qa, 
                    qa_candidates, 
                    item['image_path'],
                    image_caption
                )
                
                image_filename = os.path.basename(item['image_path'])
                processed_path = os.path.join(processed_dir, image_filename)

                # 파일 이동 및 원본 삭제
                try:
                    os.rename(item['image_path'], processed_path)
                except Exception as move_error:
                    print(f"Error moving file to processed directory: {str(move_error)}")
                    continue
                
                processed_count += 1
                
                if processed_count % save_interval == 0:
                    temp_df = pd.DataFrame(self.collected_data)
                    temp_output_path = os.path.join(output_dir, f'KoTextVQA_temp_{processed_count}.parquet')
                    temp_df.to_parquet(temp_output_path, compression='gzip')
                    print(f"Intermediate save completed: {processed_count} data items processed.")
                    
            except Exception as e:
                print(f"Error occurred while processing image ({item['image_path']}): {str(e)}")
                continue

        if self.collected_data:
            df = pd.DataFrame(self.collected_data)
            output_path = os.path.join(output_dir, f'KoTextVQA_{time.strftime("%m%d%H%M")}.parquet')
            df.to_parquet(output_path, compression='gzip')
            print(f"Total {len(self.collected_data)} data items have been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TVQA Generator')
    parser.add_argument('-d', '--input_directory', type=str, required=False, default='./data/images')
    parser.add_argument('-r', '--output_directory', type=str, required=False, default='./results')
    parser.add_argument('-s', '--save_interval', type=int, required=False, default=2,
                      help='중간 저장 간격 (처리된 이미지 수 기준)')
    args = parser.parse_args()
    
    generator = TVQAGenerator()
    
    # 비동기 실행
    asyncio.run(generator.generate_qa_from_image_directory(
        args.input_directory, 
        args.output_directory,
        args.save_interval
    ))

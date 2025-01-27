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
from tqdm import tqdm
import gc 
import logging
from datetime import datetime

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
        
        # 로그 설정 수정
        log_filename = f'log_{datetime.now().strftime("%m%d")}.txt'
        log_file = open(log_filename, 'a', encoding='utf-8')
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 핸들러에 즉시 플러시 설정
        file_handler.flush = lambda: True

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

            self.logger.info(f"multi_models_evaluation: {responses}")
            
            for model_name, response in zip(QA_EVALUATION[system_type]['models'], responses):
                if response:
                    evaluations[system_type][model_name] = parse_ranking_response(response)
        
        self.logger.info(f'evaluation 1: {evaluations["system1"]}')
        self.logger.info(f'evaluation 2: {evaluations["system2"]}')
        
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

    async def generate_qa_from_image_directory(self, dir_path, output_dir, batch_size=30):
        """Generate QA dataset from image directory"""
        # filtered_results = filter_images(dir_path, self.ocr_model)  # 주석 처리
        os.makedirs(output_dir, exist_ok=True)
        
        processed_dir = os.path.join(os.path.dirname(dir_path), 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # 디렉토리에서 이미지 파일 목록 가져오기
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        processed_count = 0
        batch_count = 0
        
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_count += 1
            batch_files = image_files[i:i + batch_size]
            batch_success = 0
            
            for image_file in batch_files:
                image_path = os.path.join(dir_path, image_file)
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Starting Image Processing [{processed_count + 1}/{len(image_files)}]")
                self.logger.info(f"Image Path: {image_path}")
                self.logger.info(f"{'-'*50}")
                
                try:
                    image_caption = await self.generate_image_caption(image_path)
                    self.logger.info(f"[Image {processed_count + 1}] Caption Generation Result:\n{image_caption}")
                    
                    qa_candidates = await self.generate_qa_candidates(image_caption)
                    self.logger.info(f"[Image {processed_count + 1}] QA Candidates Generation Result:\n{qa_candidates}")
                    
                    evaluated_qa = await self.multi_models_evaluation(qa_candidates, image_path)
                    self.logger.info(f"[Image {processed_count + 1}] QA Evaluation Result:\n{evaluated_qa}")
                    
                    qa_candidates = await self.generate_options_and_type(
                        qa_candidates,
                        evaluated_qa,
                        image_path
                    )
                    
                    self.save_final_qa_dataset(
                        evaluated_qa, 
                        qa_candidates, 
                        image_path,
                        image_caption
                    )
                    
                    image_filename = os.path.basename(image_path)
                    processed_path = os.path.join(processed_dir, image_filename)

                    try:
                        os.rename(image_path, processed_path)
                    except Exception as move_error:
                        print(f"Error moving file to processed directory: {str(move_error)}")
                        continue
                    
                    processed_count += 1
                    batch_success += 1
                    
                except Exception as e:
                    print(f"Error occurred while processing image ({image_path}): {str(e)}")
                    continue

            # 배치 처리 후 데이터 저장
            if self.collected_data:
                try:
                    df = pd.DataFrame(self.collected_data)
                    output_path = os.path.join(
                        output_dir, 
                        f'KoTextVQA_batch_{batch_count}_{time.strftime("%m%d%H%M")}_{batch_success}items.parquet'
                    )
                    df.to_parquet(output_path, compression='gzip')
                    print(f"Batch {batch_count} processed and saved: {batch_success}/{len(batch_files)} items successful")
                    
                    # 메모리 정리
                    self.collected_data.clear()
                    del df
                    gc.collect()  # 명시적 가비지 컬렉션
                    
                except Exception as save_error:
                    print(f"Error saving batch {batch_count}: {str(save_error)}")
                    # 에러가 발생해도 메모리는 정리
                    self.collected_data.clear()
                    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TVQA Generator')
    parser.add_argument('-d', '--input_directory', type=str, required=False, default='./data/images')
    parser.add_argument('-r', '--output_directory', type=str, required=False, default='./results')
    parser.add_argument('-s', '--save_batch', type=int, required=False, default=30)
    args = parser.parse_args()
    
    generator = TVQAGenerator()
    
    # 비동기 실행
    asyncio.run(generator.generate_qa_from_image_directory(
        args.input_directory, 
        args.output_directory,
        args.save_batch
    ))

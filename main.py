import os
from paddleocr import PaddleOCR
from src.utils import (
    filter_images, 
    aggregate_votes,
    parse_ranking_response,
)
from src.api_client import get_model_response
import json
import argparse
import pandas as pd
from PIL import Image
import io
from src.config import CAPTION_GENERATION, QA_GENERATION, QA_EVALUATION
from src.prompts import CAPTION_PROMPT, format_qa_generation_prompt, format_qa_evaluation_prompt, OPTIONS_AND_TYPE_PROMPT
from typing import Dict, List, Optional

class TVQAGenerator:
    def __init__(self):
        self.ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang='korean',
            use_gpu=True,
            det=True,
            rec=True,
            show_log=False
        )

    def generate_image_caption(self, image_path):
        """이미지 기반 상세 캡션 생성"""
        captions = [
            get_model_response(model_name, CAPTION_PROMPT, image_path)
            for model_name in CAPTION_GENERATION['models']
        ]
        formatted_captions = [f'<Caption {i + 1}>\n{caption}' for i, caption in enumerate(captions)]

        return '\n\n'.join(formatted_captions)

    def generate_qa_candidates(self, image_caption: str) -> Dict[str, Dict[str, List[dict]]]:
        """QA 후보 생성"""
        qa_candidates = {
            'system1': {'qa_list': []},
            'system2': {'qa_list': []}
        }
        
        # System1 QA 생성
        for model_name in QA_GENERATION['system1']['models']:
            system1_prompt = format_qa_generation_prompt('system1', image_caption, QA_GENERATION['system1']['candidates_per_model'])
            response = get_model_response(model_name, system1_prompt)
            try:
                qa_list = json.loads(response.replace('```json','').replace('```',''))['qa_list']
                qa_candidates['system1']['qa_list'].extend(qa_list)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류 ({model_name}): {str(e)}")
                print(f"문제가 있는 응답: {response[:200]}...")  # 처음 200자만 출력
                continue
        
        # System2 QA 생성
        for model_name in QA_GENERATION['system2']['models']:
            system2_prompt = format_qa_generation_prompt('system2', image_caption, QA_GENERATION['system2']['candidates_per_model'])
            response = get_model_response(model_name, system2_prompt)
            if response:
                try:
                    qa_list = json.loads(response.replace('```json','').replace('```',''))['qa_list']
                    qa_candidates['system2']['qa_list'].extend(qa_list)
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류 ({model_name}):", str(e))
                    continue
                
        return qa_candidates

    def multi_models_evaluation(self, qa_candidates: Dict, image_path: str) -> Dict[str, Optional[int]]:
        evaluations = {
            'system1': {},
            'system2': {}
        }
        
        # 각 시스템별 평가 수행
        for system_type in ['system1', 'system2']:
            prompt = format_qa_evaluation_prompt(system_type, qa_candidates['system1']['qa_list'] if system_type == 'system1' else qa_candidates['system2']['qa_list'])
            for model_name in QA_EVALUATION[system_type]['models']:
                response = get_model_response(model_name, prompt, image_path)
                if response:
                    evaluations[system_type][model_name] = parse_ranking_response(response)
        
        print('evaluation 1: ', evaluations['system1'])
        print('evaluation 2: ', evaluations['system2'])
        
        # 투표 결과 집계
        return {
            'system1': aggregate_votes(evaluations['system1'], QA_EVALUATION['system1']['num_to_select']),
            'system2': aggregate_votes(evaluations['system2'], QA_EVALUATION['system1']['num_to_select'])
        }

    def save_final_qa_dataset(self, evaluated_qa, qa_candidates, image_path, image_caption, output_dir):
        """최종 QA 데이터셋을 parquet로 저장"""
        final_data = []
        
        # 이미지를 바이트로 변환
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format)
            img_byte_arr = img_byte_arr.getvalue()
        
        # System1과 System2 데이터 처리
        for system_type in ['system1', 'system2']:
            if evaluated_qa[system_type] is not None:
                selected_indices = [evaluated_qa[system_type] - 1] if isinstance(evaluated_qa[system_type], int) else \
                                 [idx - 1 for idx in evaluated_qa[system_type]]
                selected_qa_list = [qa_candidates[system_type]['qa_list'][idx] for idx in selected_indices]
                
                final_data.append({
                    'image': img_byte_arr,
                    'detailed_caption': image_caption,
                    'candidates': qa_candidates[system_type]['qa_list'],
                    'selected_qa': selected_qa_list,
                    'system': int(system_type[-1]),
                    'options': qa_candidates[system_type]['options'],
                    'img_type': qa_candidates[system_type]['img_type'],
                    'domain': qa_candidates[system_type]['domain']
                })
        
        # DataFrame 생성 및 parquet 저장
        if final_data:
            df = pd.DataFrame(final_data)
            output_path = os.path.join(output_dir, 'qa_dataset.parquet')
            df.to_parquet(output_path, compression='gzip')
            return True
        
        return False

    def generate_options_and_type(self, qa_candidates: Dict, evaluated_qa: Dict, image_path: str) -> Dict:
        """선택된 QA에 대해 유사한 오답 옵션들과 이미지 타입, 도메인 정보를 생성"""
        # qa_candidates의 기존 구조를 유지하면서 새로운 필드 추가
        for system_type in ['system1', 'system2']:
            qa_candidates[system_type].update({
                'options': [""],
                'img_type': [""],
                'domain': [""]
            })
            
            if evaluated_qa[system_type] is not None:
                selected_indices = [evaluated_qa[system_type] - 1] if isinstance(evaluated_qa[system_type], int) else \
                                 [idx - 1 for idx in evaluated_qa[system_type]]
                
                for idx in selected_indices:
                    qa = qa_candidates[system_type]['qa_list'][idx]
                    prompt = OPTIONS_AND_TYPE_PROMPT.format(
                        question=qa['question'],
                        correct_answer=qa['answer']
                    )
                    
                    response = get_model_response(QA_EVALUATION['system2']['models'][0], prompt, image_path)
                    try:
                        parsed_response = json.loads(response.replace('```json','').replace('```',''))
                        qa_candidates[system_type].update({
                            'options': parsed_response.get('options', [""]),
                            'img_type': parsed_response.get('img_type', [""]),
                            'domain': parsed_response.get('domain', [""])
                        })
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"옵션/타입/도메인 생성 중 오류 발생: {str(e)}")
        
        return qa_candidates

    def generate_qa_from_image_directory(self, dir_path, output_dir):
        """디렉토리 내 이미지들을 필터링하고 QA 데이터셋을 생성"""
        filtered_results = filter_images(dir_path, self.ocr_model)
        
        for item in filtered_results:
            # image_caption = self.generate_image_caption(item['image_path'])
            # print("image_caption: ", image_caption)
            image_caption = """**1. 이미지 내 텍스트 정보**

*   **제목:** 한국의 국가별 반도체 수출 비중
*   **(자료=2020년 기준 무역협회)**
*   **필리핀 3.0**
*   **대만 5.2**
*   **기타 10.6**
*   **미국 7.7**
*   **중국 41.1%**
*   **단위: %**
*   **베트남 11.6**
*   **홍콩 20.8**

**2. 텍스트 외의 시각적 요소들**

*   **전반적인 구성:** 파이 차트 형태로, 원형 그래프를 통해 국가별 반도체 수출 비중을 시각적으로 나타냅니다. 배경은 단색의 연한 회색입니다.
*   **주요 사물:**
    *   **파이 차트:** 여러 색상으로 구분된 원형 그래프이며, 각 영역은 특정 국가의 반도체 수출 비중을 나타냅니다. 각 영역 옆에는 해당 국가명과 수치가 표시되어 있습니다.
    *   **반도체 칩:** 이미지 하단 우측에 반도체 칩 그림이 있습니다. 칩은 흰색과 회색의 선으로 단순하게 표현되어 있으며, 그 위에 작은 태극기 깃발이 꽂혀 있습니다.
    *   **색상:** 각 국가별로 파이 차트 조각이 색상으로 구분되어 있으며, 밝고 채도가 높은 색상이 사용되었습니다.
*   **분위기:** 통계 데이터를 시각화하여 보여주는 이미지로, 정보 전달에 초점이 맞춰져 있으며, 전체적으로 깔끔하고 정돈된 느낌입니다."""

            qa_candidates = self.generate_qa_candidates(image_caption)
            evaluated_qa = self.multi_models_evaluation(qa_candidates, item['image_path'])
            print("evaluated_qa: ", evaluated_qa)
            
            # 선택된 QA에 대해 유사 옵션 생성
            qa_candidates = self.generate_options_and_type(
                qa_candidates,
                evaluated_qa,
                item['image_path']
            )
            
            # 최종 데이터셋 저장
            self.save_final_qa_dataset(
                evaluated_qa, 
                qa_candidates, 
                item['image_path'],
                image_caption,
                output_dir
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TVQA Generator')
    parser.add_argument('-d', '--input_directory', type=str, required=False, default='./data/images')
    parser.add_argument('-r', '--output_directory', type=str, required=False, default='./results')
    args = parser.parse_args()

    generator = TVQAGenerator()
    input_directory = args.input_directory
    output_directory = args.output_directory
    
    generator.generate_qa_from_image_directory(input_directory, output_directory)

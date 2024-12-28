import cv2
import numpy as np
import shutil
import os
import json

def check_resolution(image_path):
    """이미지 해상도 체크 (최소 단축 384px)"""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            return False
        
        height, width, _ = img.shape
        return min(height, width) > 384
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {image_path}")
        print(f"오류 내용: {str(e)}")
        return False

def check_letters_and_extract_text(image_path, ocr_model):
    """OCR로 텍스트 길이 체크 (10~1000자) 및 텍스트 추출"""
    try:
        result = ocr_model.ocr(image_path)[0]
        if not result:
            print(f"OCR 처리 실패: {image_path}")
            return False, ""
        
        valid_texts = [
            {
                'text': line[1][0],
                'box': line[0],
                'center_y': (line[0][0][1] + line[0][2][1]) / 2
            }
            for line in result
            if line[1][1] >= 0.85
        ]
        
        total_letters = sum(len(text['text']) for text in valid_texts)
        if not (10 <= total_letters <= 1000) or not valid_texts:
            return False, ""
            
        extracted_texts = []
        current_line_texts = []
        # 인식된 텍스트의 평균 높이를 계산하여 줄 바꿈 기준으로 사용
        line_height = sum(text['box'][2][1] - text['box'][0][1] for text in valid_texts) / len(valid_texts)
        current_line_y = valid_texts[0]['center_y']
        
        for text_info in valid_texts:
            # 텍스트 높이가 평균 높이의 절반보다 크면 새로운 라인으로 간주
            if abs(text_info['center_y'] - current_line_y) > line_height/2:
                extracted_texts.append("".join(current_line_texts))
                extracted_texts.append("\n")
                current_line_texts = []
                current_line_y = text_info['center_y']
            
            current_line_texts.append(text_info['text'])
        
        if current_line_texts:
            extracted_texts.append("".join(current_line_texts))
        
        return True, "".join(extracted_texts)
    except Exception as e:
        print(f"OCR 처리 중 오류 발생: {image_path}")
        print(f"오류 내용: {str(e)}")
        return False, ""

def filter_images(dir_path, ocr_model):
    """이미지 필터링 및 새 디렉토리에 복사"""
    try:
        if not os.path.exists(dir_path):
            print(f"디렉토리를 찾을 수 없습니다: {dir_path}")
            return []

        new_dir_path = os.path.join(os.path.dirname(dir_path), 'target_images')
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)

        filtered_results = []
        image_list = os.listdir(dir_path)
        print(f"처리 중: {dir_path}...")
        
        for idx, image in enumerate(image_list):
            try:
                print(f"{idx+1}/{len(image_list)}")
                image_path = os.path.join(dir_path, image)
                
                if not os.path.isfile(image_path):
                    continue

                if check_resolution(image_path):
                    is_valid, extracted_text = check_letters_and_extract_text(image_path, ocr_model)
                    if is_valid:
                        new_image_path = os.path.join(new_dir_path, image)
                        shutil.copy2(image_path, new_image_path)
                        filtered_results.append({
                            'image_path': new_image_path,
                            'extracted_text': extracted_text
                        })
            
            except Exception as e:
                print(f"이미지 처리 중 오류 발생: {image}")
                print(f"오류 내용: {str(e)}")
                continue
        
        print(f"필터링된 이미지가 저장된 경로: {new_dir_path}")
        return filtered_results    
    except Exception as e:
        print(f"처리 중 오류 발생")
        print(f"오류 내용: {str(e)}")
        return []

def aggregate_votes(eval_results, num_to_select):
    """모델들의 순위를 가중치를 적용하여 집계"""
    if not eval_results:
        return None
        
    vote_counts = {}
    for model_results in eval_results.values():
        if not model_results:
            continue
            
        for i, rank in enumerate(model_results):
            if rank not in vote_counts:
                vote_counts[rank] = 0
            vote_counts[rank] += (len(model_results) - i)
    
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    return [qa for qa, _ in sorted_votes[:num_to_select]] if vote_counts else None

def parse_ranking_response(response):
    """평가 응답에서 순위 정보 추출"""
    try:
        result = json.loads(response.replace('```json','').replace('```',''))
        return result.get('ranking', [])
    except Exception as e:
        print(f"순위 응답 파싱 중 오류 발생: {e}")
        print(response)
        return []
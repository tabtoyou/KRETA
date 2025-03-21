import json
import os
import shutil

import cv2
import numpy as np


def check_resolution(image_path):
    """Check image resolution (minimum dimension: 384px)"""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return False

        height, width, _ = img.shape
        return min(height, width) > 384
    except Exception as e:
        print(f"Error processing image: {image_path}")
        print(f"Error details: {str(e)}")
        return False


def check_letters_and_extract_text(image_path, ocr_model):
    """Check text length using OCR (10-1000 characters) and extract text"""
    try:
        result = ocr_model.ocr(image_path)[0]
        if not result:
            print(f"OCR processing failed: {image_path}")
            return False, ""

        valid_texts = [
            {
                "text": line[1][0],
                "box": line[0],
                "center_y": (line[0][0][1] + line[0][2][1]) / 2,
            }
            for line in result
            if line[1][1] >= 0.85
        ]

        total_letters = sum(len(text["text"]) for text in valid_texts)
        if not (10 <= total_letters <= 1000) or not valid_texts:
            return False, ""

        extracted_texts = []
        current_line_texts = []
        # 인식된 텍스트의 평균 높이를 계산하여 줄 바꿈 기준으로 사용
        line_height = sum(
            text["box"][2][1] - text["box"][0][1] for text in valid_texts
        ) / len(valid_texts)
        current_line_y = valid_texts[0]["center_y"]

        for text_info in valid_texts:
            # 텍스트 높이가 평균 높이의 절반보다 크면 새로운 라인으로 간주
            if abs(text_info["center_y"] - current_line_y) > line_height / 2:
                extracted_texts.append("".join(current_line_texts))
                extracted_texts.append("\n")
                current_line_texts = []
                current_line_y = text_info["center_y"]

            current_line_texts.append(text_info["text"])

        if current_line_texts:
            extracted_texts.append("".join(current_line_texts))

        return True, "".join(extracted_texts)
    except Exception as e:
        print(f"Error during OCR processing: {image_path}")
        print(f"Error details: {str(e)}")
        return False, ""


def filter_images(source_dir_path, ocr_model):
    """Filter images and copy to new directory"""
    try:
        if not os.path.exists(source_dir_path):
            print(f"Directory not found: {source_dir_path}")
            return []

        new_dir_path = os.path.join(os.path.dirname(source_dir_path), "valid_images")
        invalid_dir_path = os.path.join(
            os.path.dirname(source_dir_path), "invalid_images"
        )

        for path in [new_dir_path, invalid_dir_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        filtered_results = []
        image_list = [
            f
            for f in os.listdir(source_dir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"Processing: {source_dir_path}...")

        for idx, image in enumerate(image_list):
            try:
                print(f"\r{idx+1}/{len(image_list)}", end="")
                image_path = os.path.join(source_dir_path, image)

                if not os.path.isfile(image_path):
                    continue

                is_valid_resolution = check_resolution(image_path)
                if is_valid_resolution:
                    is_valid, extracted_text = check_letters_and_extract_text(
                        image_path, ocr_model
                    )
                    if is_valid:
                        new_image_path = os.path.join(new_dir_path, image)
                        shutil.copy2(image_path, new_image_path)
                        filtered_results.append(
                            {
                                "image_path": new_image_path,
                                "extracted_text": extracted_text,
                            }
                        )
                    else:
                        invalid_image_path = os.path.join(invalid_dir_path, image)
                        shutil.copy2(image_path, invalid_image_path)
                else:
                    invalid_image_path = os.path.join(invalid_dir_path, image)
                    shutil.copy2(image_path, invalid_image_path)

            except Exception as e:
                print(f"Error processing image: {image}")
                print(f"Error details: {str(e)}")
                continue

        return filtered_results
    except Exception as e:
        print("Error during processing")
        print(f"Error details: {str(e)}")
        return []


def aggregate_votes(eval_results, num_to_select):
    """Aggregate model rankings with weights"""
    if not eval_results:
        return None

    vote_counts = {}
    for model_results in eval_results.values():
        if not model_results:
            continue

        for i, rank in enumerate(model_results):
            if rank not in vote_counts:
                vote_counts[rank] = 0
            vote_counts[rank] += len(model_results) - i

    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    return [qa for qa, _ in sorted_votes[:num_to_select]] if vote_counts else None


def parse_ranking_response(response):
    """Extract ranking information from evaluation response"""
    try:
        response_cleaned = (
            response.replace("```json", "").replace("```", "").replace("\n", "").strip()
        )
        result = json.loads(response_cleaned)
        return result.get("ranking", [])
    except Exception as e:
        print(f"Error parsing ranking response: {e}")
        return []

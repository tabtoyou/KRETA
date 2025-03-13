from paddleocr import PaddleOCR
from src.utils import filter_images
from src.config import LANGUAGE
import argparse
import os
import logging
from datetime import datetime

class ImageFilter:
    def __init__(self):
        self.ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang=LANGUAGE.lower(),
            use_gpu=True,
            det=True,
            rec=True,
            show_log=False
        )
        
        # 로그 설정
        log_filename = f'filter_log_{datetime.now().strftime("%m%d")}.txt'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def process_directory(self, input_directory):
        """Process and filter images in the directory"""
        self.logger.info(f"Starting image filtering process for: {input_directory}")
        
        try:
            filtered_results = filter_images(input_directory, self.ocr_model)
            
            # 결과 로깅
            total_images = len([f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            valid_images = len(filtered_results)
            
            self.logger.info(f"Filtering completed:")
            self.logger.info(f"Total images processed: {total_images}")
            self.logger.info(f"Valid images: {valid_images}")
            self.logger.info(f"Invalid images: {total_images - valid_images}")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error during filtering process: {str(e)}")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Filter for TVQA Generator')
    parser.add_argument('-d', '--input_directory', type=str, required=False, default='./data/images')
    args = parser.parse_args()

    image_filter = ImageFilter()
    image_filter.process_directory(args.input_directory)

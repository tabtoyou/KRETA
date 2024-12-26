import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from paddleocr import PaddleOCR

def has_korean_text(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='korean')
    result = ocr.ocr(image_path, cls=True)
    
    if not result or not result[0]:
        return False
        
    # PaddleOCR의 결과 구조가 [pages[lines[words]]] 형태임
    for line in result[0]:  # 첫 번째 페이지의 결과만 사용
        if line is None:
            continue
        text = line[1][0]  # OCR 결과에서 텍스트 부분 추출
        # 한글 유니코드 범위 체크 (가-힣)
        if any(('\uAC00' <= char <= '\uD7A3') for char in text):
            return True
    return False

def download_posters(base_url, save_dir='raw_posters', max_pages=44):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 이미 다운로드한 URL을 추적하기 위한 세트
    downloaded_urls = set()
    
    # 1페이지부터 max_pages까지 순차적으로 크롤링
    for page in range(1, max_pages + 1):
        try:
            page_url = f"{base_url}?page={page}"
            print(f"현재 페이지: {page}/{max_pages}")
            
            response = requests.get(page_url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 이미지 다운로드 링크 찾기
            download_links = soup.find_all('a', string='다운로드')
            
            for link in download_links:
                try:
                    # 이미지 제목 가져오기 (상위 요소에서)
                    title_element = link.find_previous(text=True)
                    if title_element and title_element.strip():
                        title = title_element.strip()
                    else:
                        title = f"image_{len(downloaded_urls)}"
                    
                    # 실제 이미지 URL 가져오기
                    img_url = link.get('href')
                    if not img_url:
                        continue
                        
                    if img_url not in downloaded_urls:
                        img_data = requests.get(img_url, headers=headers).content
                        
                        # 파일명에서 사용할 수 없는 문자 제거
                        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        if not safe_title:
                            safe_title = f"image_{len(downloaded_urls)}"
                        file_name = f'{safe_title}.jpg'
                        file_path = os.path.join(save_dir, file_name)
                        
                        with open(file_path, 'wb') as f:
                            f.write(img_data)
                        downloaded_urls.add(img_url)
                        print(f'이미지 다운로드 완료: {file_name}')
                        
                except Exception as e:
                    print(f'이미지 다운로드 실패: {e}')
            
        except Exception as e:
            print(f'페이지 {page} 처리 실패: {e}')
            continue

    print(f'총 {len(downloaded_urls)}개의 이미지 다운로드 완료')

def process_images_with_ocr(raw_dir='raw_posters', processed_dir='korean_posters'):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    ocr = PaddleOCR(use_angle_cls=True, lang='korean')
    
    for filename in os.listdir(raw_dir):
        if filename.endswith(('.jpg', '.png')):
            input_path = os.path.join(raw_dir, filename)
            
            if has_korean_text(input_path):
                output_path = os.path.join(processed_dir, filename)
                os.rename(input_path, output_path)
                print(f'한글 텍스트 발견, 이동 완료: {filename}')
            else:
                print(f'한글 텍스트 없음: {filename}')

# 크롤링 실행
url = "https://cjnews.cj.net/medialibrary/cgv/"
download_posters(url)
process_images_with_ocr()
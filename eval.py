import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import glob
import signal
import time
import numpy as np

IMG_TYPE_LIST = ["Report", "Test_Paper", "Newspaper", "Manual", "Magazine", "Brochure", "Book_Cover", "Illustrated_Books_and_Comics", 
                 "Chart_and_Plot", "Table", "Diagram", "Infographic", "Poster", "Banner", "Menu", "Packaging_Label", "Flyer", "Signage", "Store_Sign",
                 "Product_Detail", "Public_Signs", "Street_Signs", "Mural_and_Graffiti", "Mobile_Screenshot", "PC_Screenshot", "Presentation_Slides", 
                 "Video_Thumbnail", "Video_Scene", "Receipts_and_Invoices", "Contracts_Documents", "Certificates", "Handwriting", 
                 "Tickets_and_Boarding_Passes"]

DOMAIN_LIST = ["Public_and_Administration", "Legal_and_Regulations", "Economics_and_Finance", "Corporate_and_Business", 
               "Marketing_and_Advertising", "Education_and_Academia", "Medical_and_Healthcare", "Transportation_and_Logistics", 
               "Travel_and_Tourism", "Retail_and_Commerce", "Hospitality_and_Food_Service", "Entertainment_and_Media", "Science_and_Technology", "Arts_and_Humanities", "Personal_and_Lifestyle"]

def run_app(dataset_path: str):
    # 전체 폭 차지
    st.set_page_config(layout="wide")

    # Rerun 시에는 캐시된 dataframe을 사용
    if "df" not in st.session_state:
        st.session_state.df = pd.read_parquet(dataset_path)
        st.session_state.df["domain"] = st.session_state.df["domain"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        st.session_state.df["img_type"] = st.session_state.df["img_type"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        st.session_state.df["options"] = st.session_state.df["options"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df = st.session_state.df

    # session state 초기화 (page index, selected qa)
    if "index" not in st.session_state:
        st.session_state.index = 0

    if "selected_qa_dict" not in st.session_state:
        st.session_state.selected_qa_dict = {}

    # 저장 후 종료
    def save_df_and_stop():
        for idx, sel_question in st.session_state.selected_qa_dict.items():
            row = df.iloc[idx]
            # selected_qa에 반영
            for candidate in row["candidates"]:
                if candidate["question"] == sel_question:
                    row["selected_qa"] = [candidate]
                    break
            df.iloc[idx] = row

        df.to_parquet(dataset_path)
        st.markdown("저장되었습니다. 브라우저를 닫아주세요.")
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    # previous, next button
    col1, col2, col3 = st.columns([1,4,1])
    with col1:
        if st.button("Previous"):
            if st.session_state.index > 0:
                st.session_state.index -= 1
    with col3:
        if st.button("Next"):
            if st.session_state.index < len(df) - 1:
                st.session_state.index += 1

    current_index = st.session_state.index
    row = df.iloc[current_index]

    # 2열 레이아웃(왼쪽: 이미지, 오른쪽: QA와 수정 UI)
    col_left, col_right = st.columns([1,3])

    with col_left:
        st.subheader(f"{current_index + 1}번 이미지 <System: {row['system']}>")
        if isinstance(row["image"], bytes):
            image = Image.open(io.BytesIO(row["image"]))
            st.image(image, caption="참고 이미지")

    with col_right:
        # CSS 스타일 제거 (더 이상 필요하지 않음)
        st.markdown(
            """
            <style>
            .card {
                background-color: #f9f9f9;
                border: 1px solid #000;
                border-radius: 8px;
                padding: 16px;
                margin: 8px;
                color: #333;
                height: 250px;
                overflow-y: auto;
            }
            .selected-qa {
                background-color: #8bc0c4;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # container를 사용하여 스크롤 가능한 영역 생성
        with st.container(height=800):
            if current_index not in st.session_state.selected_qa_dict:
                st.session_state.selected_qa_dict[current_index] = row["selected_qa"][0]["question"]

            selected_qa_question = st.session_state.selected_qa_dict[current_index]

            candidates = row["candidates"]
            num_columns = min(4, len(candidates))
            cols = st.columns(num_columns)

            # 1) 카드 + Select 버튼 UI
            for i, candidate in enumerate(candidates):
                col_idx = i % num_columns
                with cols[col_idx]:
                    selected_css = "selected-qa" if candidate["question"] == selected_qa_question else ""
                    reasoning_html = f"<br><strong>Reasoning:</strong> {candidate['reasoning']}" if candidate.get("reasoning") else ""
                    card_html = f"""
                        <div class="card {selected_css}">
                            <strong>Question:</strong> {candidate["question"]}<br>
                            <strong>Answer:</strong> {candidate["answer"]}{reasoning_html}
                        </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    if st.button("Select this QA", key=f"select_btn_{current_index}_{i}"):
                        st.session_state.selected_qa_dict[current_index] = candidate["question"]
                        st.rerun()

            # 2) 선택된 QA 수정 UI
            current_candidate = next((c for c in candidates if c["question"] == selected_qa_question), None)
            if current_candidate:
                qa_col_left, qa_col_right = st.columns(2)
                
                # 왼쪽 열: Question/Answer/Reasoning
                with qa_col_left:
                    temp_question = st.text_input("Question", value=current_candidate["question"])
                    temp_answer = st.text_input("Answer", value=current_candidate["answer"])
                    temp_reasoning = st.text_input("Reasoning", value=current_candidate.get("reasoning", ""))
                
                # 오른쪽 열: Options
                with qa_col_right:
                    temp_option_1 = st.text_input("Option 1", value=row["options"][0])
                    temp_option_2 = st.text_input("Option 2", value=row["options"][1])
                    temp_option_3 = st.text_input("Option 3", value=row["options"][2])

                if st.button("Update this QA"):
                    current_candidate["question"] = temp_question
                    current_candidate["answer"] = temp_answer
                    current_candidate["reasoning"] = temp_reasoning
                    row["options"] = [temp_option_1, temp_option_2, temp_option_3]
                    # question이 selected_qa를 구분하는 key이므로 업데이트
                    st.session_state.selected_qa_dict[current_index] = temp_question
                    # DataFrame 반영
                    df.iloc[current_index] = row
                    st.rerun()

            # 3) 이미지 타입, 도메인 수정
            col_img_type, col_domain = st.columns(2)

            def update_img_type():
                row['img_type'] = st.session_state[f"img_type_{current_index}"]
                df.iloc[current_index] = row

            with col_img_type:
                default_img_types = row["img_type"]
                st.multiselect(
                    "Select image types",
                    options=IMG_TYPE_LIST,
                    default=default_img_types,
                    key=f"img_type_{current_index}",
                    on_change=update_img_type
                )

            def update_domain():
                row['domain'] = st.session_state[f"domain_{current_index}"]
                df.iloc[current_index] = row

            with col_domain:
                default_domains = row["domain"]
                st.multiselect(
                    "Select domains",
                    options=DOMAIN_LIST,
                    default=default_domains,
                    key=f"domain_{current_index}",
                    on_change=update_domain
                )

            # 4) 저장 후 종료 버튼
            if st.button("저장 후 종료"):
                save_df_and_stop()

def main():
    # 이미 선택된 파일이 있으면 바로 실행
    if "selected_file" in st.session_state:
        run_app(st.session_state.selected_file)
        return

    # 없으면 파일 선택 UI (results 폴더 내 parquet 파일 목록)
    st.title("Select a dataset to load")

    parquet_files = glob.glob(os.path.join("results", "*.parquet"))

    if not parquet_files:
        st.write("results 폴더에 parquet 파일이 없습니다.")
        return

    selected_file = st.selectbox("Choose a parquet file", parquet_files)

    if st.button("Load selected file"):
        st.session_state.selected_file = selected_file
        st.rerun()

if __name__ == '__main__':
    main()

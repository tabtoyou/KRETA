import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import glob
import signal
import time
import numpy as np
from src.api_client import generate_response
from src.prompts import HARD_NEGATIVE_OPTIONS_PROMPT
import asyncio

IMG_TYPE_LIST = ["Report", "Test_Paper", "Newspaper", "Manual", "Book_Page", "Magazine", "Brochure", "Book_Cover", "Illustrated_Books_and_Comics", 
                 "Chart_and_Plot", "Table", "Diagram", "Infographic", "Poster", "Banner", "Menu", "Packaging_Label", "Flyer", "Signage", "Store_Sign",
                 "Product_Detail", "Public_Signs", "Street_Signs", "Mural_and_Graffiti", "Mobile_Screenshot", "PC_Screenshot", "Presentation_Slides", 
                 "Video_Thumbnail", "Video_Scene", "Receipts_and_Invoices", "Contracts_Documents", "Certificates", "Handwriting", 
                 "Tickets_and_Boarding_Passes", ""]

DOMAIN_LIST = ["Public_and_Administration", "Legal_and_Regulations", "Economics_and_Finance", "Corporate_and_Business", 
               "Marketing_and_Advertising", "Education_and_Academia", "Medical_and_Healthcare", "Transportation_and_Logistics", 
               "Travel_and_Tourism", "Retail_and_Commerce", "Hospitality_and_Food_Service", "Entertainment_and_Media", "Science_and_Technology", "Arts_and_Humanities", "Personal_and_Lifestyle", ""]

# run_app 함수를 async로 변경
async def run_app(dataset_path: str):
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

        # 수정된 내용이 있는지 확인
        if st.session_state.selected_qa_dict:
            # 새로운 파일명 생성 (원본파일명_MMDDHHMM.parquet)
            base_path = dataset_path.split('.')[0]
            new_path = f"{base_path}_savebutton_{time.strftime('%m%d%H%M')}.parquet"

            df.to_parquet(new_path)
            st.markdown(f"새로운 파일에 저장되었습니다: {new_path}")
        else:
            st.markdown("수정된 내용이 없습니다.")
        
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.rerun()

    def save_df_next():
        for idx, sel_question in st.session_state.selected_qa_dict.items():
            row = df.iloc[idx]
            for candidate in row["candidates"]:
                if candidate["question"] == sel_question:
                    row["selected_qa"] = [candidate]
                    break
            df.iloc[idx] = row

        if st.session_state.selected_qa_dict:
            base_path = dataset_path.split('.')[0]
            new_path = f"{base_path}_nextbutton.parquet"
            df.to_parquet(new_path)
            st.success(f"저장되었습니다: {new_path}")

    # 현재 행 삭제 함수
    def delete_current_row():
        if st.session_state.index in st.session_state.selected_qa_dict:
            del st.session_state.selected_qa_dict[st.session_state.index]
        
        # df를 session_state에서 직접 수정
        st.session_state.df = st.session_state.df.drop(index=st.session_state.df.index[st.session_state.index]).reset_index(drop=True)
        
        if st.session_state.index >= len(st.session_state.df):
            st.session_state.index = len(st.session_state.df) - 1
        st.rerun()

    # --------------------------------------------------------------------------------
    # (1) 현재 화면의 QA 정보를 업데이트하는 함수
    # --------------------------------------------------------------------------------
    def update_current_qa():
        """
        UI에서 입력된 Question / Answer / Reasoning / Options 값을
        현재 row 및 DataFrame에 반영한다.
        """
        current_index = st.session_state.index
        row = df.iloc[current_index]
        
        # 선택된 QA candidate 찾기
        selected_qa_question = st.session_state.selected_qa_dict[current_index]
        current_candidate = next((c for c in row["candidates"] if c["question"] == selected_qa_question), None)
        
        if current_candidate:
            # 입력받은 값들
            temp_question = st.session_state.get(f"temp_question_{current_index}", current_candidate["question"])
            temp_answer = st.session_state.get(f"temp_answer_{current_index}", current_candidate["answer"])
            temp_reasoning = st.session_state.get(f"temp_reasoning_{current_index}", current_candidate.get("reasoning", ""))
            temp_option_1 = st.session_state.get(f"temp_option_1_{current_index}", row["options"][0])
            temp_option_2 = st.session_state.get(f"temp_option_2_{current_index}", row["options"][1])
            temp_option_3 = st.session_state.get(f"temp_option_3_{current_index}", row["options"][2])
            
            # current_candidate 업데이트
            current_candidate["question"] = temp_question
            current_candidate["answer"] = temp_answer
            current_candidate["reasoning"] = temp_reasoning
            
            # options 업데이트
            row["options"] = [temp_option_1, temp_option_2, temp_option_3]
            
            # question 변경시 selected_qa_dict도 갱신
            st.session_state.selected_qa_dict[current_index] = temp_question
            
            # DataFrame 반영
            df.iloc[current_index] = row

    # --------------------------------------------------------------------------------
    # Previous / Next / Delete 버튼
    # --------------------------------------------------------------------------------
    col1, col2, col3 = st.columns([1.3,4.7,0.4])
    with col1:
        if st.button("Previous"):
            if st.session_state.index > 0:
                # 이전으로 이동
                st.session_state.index -= 1
                st.rerun()
    with col2:
        if st.button("Next"):
            update_current_qa()
            save_df_next()
            if st.session_state.index < len(df) - 1:
                st.session_state.index += 1
            st.rerun()
    with col3:
        if st.button("🗑️ Delete"):
            if st.session_state.index < len(df):
                delete_current_row()

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
        # CSS 스타일 (카드 UI)
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

        with st.container():
            if current_index not in st.session_state.selected_qa_dict:
                st.session_state.selected_qa_dict[current_index] = row["selected_qa"][0]["question"]

            selected_qa_question = st.session_state.selected_qa_dict[current_index]
            candidates = row["candidates"]
            num_columns = min(4, len(candidates))
            cols = st.columns(num_columns)

            # 1) QA 카드 표시 + Select 버튼
            for i, candidate in enumerate(candidates):
                col_idx = i % num_columns
                with cols[col_idx]:
                    selected_css = "selected-qa" if candidate["question"] == selected_qa_question else ""
                    reasoning_html = f"<br><strong>Reasoning:</strong> {candidate['reasoning']}" if candidate.get("reasoning") else ""
                    card_html = f"""
                        <div class="card {selected_css}">
                            <strong>Question:</strong> {candidate["question"]}<br><br>
                            <strong>Answer:</strong> {candidate["answer"]}<br>{reasoning_html}
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
                    # text_input에서 key를 사용하여, 값이 바뀌면 session_state에 저장
                    st.text_input(
                        "Question",
                        value=current_candidate["question"],
                        key=f"temp_question_{current_index}"
                    )
                    st.text_input(
                        "Answer",
                        value=current_candidate["answer"],
                        key=f"temp_answer_{current_index}"
                    )
                    st.text_input(
                        "Reasoning",
                        value=current_candidate.get("reasoning", ""),
                        key=f"temp_reasoning_{current_index}"
                    )
                
                # 오른쪽 열: Options
                with qa_col_right:
                    st.text_input(
                        "Option 1",
                        value=row["options"][0],
                        key=f"temp_option_1_{current_index}"
                    )
                    st.text_input(
                        "Option 2",
                        value=row["options"][1],
                        key=f"temp_option_2_{current_index}"
                    )
                    st.text_input(
                        "Option 3",
                        value=row["options"][2],
                        key=f"temp_option_3_{current_index}"
                    )

                    # "Update Options" 버튼 - 선택한 QA를 참고해서 새로운 options 만들고 저장
                    if st.button("🤖 Update Options"):
                        import json
                        
                        prompt = HARD_NEGATIVE_OPTIONS_PROMPT.format(
                            question=current_candidate["question"],
                            correct_answer=current_candidate["answer"]
                        )
                        
                        # 직접 await 사용
                        response = await generate_response('openai', prompt, 'gpt-4o-mini')
                        
                        if response:
                            try:
                                # JSON 응답 파싱
                                response_json = json.loads(response.replace('```json','').replace('```',''))
                                new_options = response_json.get("options", [])
                                print(new_options)
                                
                                if len(new_options) == 3:
                                    all_options = [
                                        new_options[0],
                                        new_options[1],
                                        new_options[2],
                                        current_candidate["answer"]
                                    ]
                                    
                                    # DataFrame에 반영
                                    row["options"] = all_options
                                    df.iloc[current_index] = row
                                    
                                    # 성공 메시지 표시
                                    st.success("옵션이 업데이트되었습니다!")
                                    st.rerun()
                                else:
                                    st.error("충분한 옵션이 생성되지 않았습니다.")
                            except json.JSONDecodeError as e:
                                st.error("응답 형식이 올바르지 않습니다.")
                            except Exception as e:
                                st.error(f"오류가 발생했습니다: {str(e)}")

            # 3) 이미지 타입, 도메인 수정
            col_img_type, col_domain = st.columns(2)

            # 유효하지 않은 이미지 타입 확인
            invalid_types = [t for t in row["img_type"] if t not in IMG_TYPE_LIST]
            if invalid_types:
                st.markdown(
                    f"""
                    <script>
                        alert("유효하지 않은 이미지 타입이 있습니다: {', '.join(invalid_types)}");
                    </script>
                    """,
                    unsafe_allow_html=True
                )

            def update_img_type():
                current_img_types = st.session_state[f"img_type_{current_index}"]
                row['img_type'] = current_img_types
                df.iloc[current_index] = row

            with col_img_type:
                default_img_types = [t for t in row["img_type"] if t in IMG_TYPE_LIST]
                st.multiselect(
                    "Select image types",
                    options=IMG_TYPE_LIST,
                    default=default_img_types,
                    key=f"img_type_{current_index}",
                    on_change=update_img_type
                )

            # 유효하지 않은 도메인 확인
            invalid_domains = [d for d in row["domain"] if d not in DOMAIN_LIST]
            if invalid_domains:
                st.markdown(
                    f"""
                    <script>
                        alert("유효하지 않은 도메인이 있습니다: {', '.join(invalid_domains)}");
                    </script>
                    """,
                    unsafe_allow_html=True
                )

            def update_domain():
                current_domains = st.session_state[f"domain_{current_index}"]
                row['domain'] = current_domains
                df.iloc[current_index] = row

            with col_domain:
                default_domains = [d for d in row["domain"] if d in DOMAIN_LIST]
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

async def main():
    if "selected_file" in st.session_state:
        await run_app(st.session_state.selected_file)
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
    asyncio.run(main())

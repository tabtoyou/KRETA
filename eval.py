import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import glob
import signal
import time

def run_app(dataset_path: str):
    # Rerun 시에는 캐시된 dataframe을 사용
    if "df" not in st.session_state:
        st.session_state.df = pd.read_parquet(dataset_path)
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

    st.title("Evaluate QA")

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

    st.subheader(f"{current_index + 1}번 이미지 <System: {row['system']}>")

    if isinstance(row["image"], bytes):
        image = Image.open(io.BytesIO(row["image"]))
        st.image(image, caption="참고 이미지")

    # selected qa 초기화
    if current_index not in st.session_state.selected_qa_dict:
        st.session_state.selected_qa_dict[current_index] = row["selected_qa"][0]["question"]

    selected_qa_question = st.session_state.selected_qa_dict[current_index]

    # CSS for card
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
            height: 200px;
            overflow-y: auto;
        }
        .selected-qa {
            background-color: #8bc0c4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    candidates = row["candidates"]
    num_columns = min(3, len(candidates))
    cols = st.columns(num_columns)

    # card + select button UI
    for i, candidate in enumerate(candidates):
        col_idx = i % num_columns
        with cols[col_idx]:
            selected_css = "selected-qa" if candidate["question"] == selected_qa_question else ""
            card_html = f"""
                <div class="card {selected_css}">
                    <strong>Question:</strong> {candidate["question"]}<br>
                    <strong>Answer:</strong> {candidate["answer"]}<br>
                    {"<strong>Reasoning:</strong> " + candidate["reasoning"] if candidate.get("reasoning") else ""}
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            if st.button("Select this QA", key=f"select_btn_{current_index}_{i}"):
                st.session_state.selected_qa_dict[current_index] = candidate["question"]
                st.rerun()

    st.markdown("---")

    # QA 내용 수정 UI
    current_candidate = next((c for c in candidates if c["question"] == selected_qa_question), None)
    if current_candidate:
        temp_question = st.text_area("Question", value=current_candidate["question"])
        temp_answer = st.text_area("Answer", value=current_candidate["answer"])
        temp_reasoning = st.text_area("Reasoning", value=current_candidate.get("reasoning", ""))

        if st.button("Update thisQA"):
            current_candidate["question"] = temp_question
            current_candidate["answer"] = temp_answer
            current_candidate["reasoning"] = temp_reasoning

            # question이 selected_qa를 구분하는 key이므로 업데이트
            st.session_state.selected_qa_dict[current_index] = temp_question

            # dataframe에 반영
            df.iloc[current_index] = row
            st.rerun()

    st.markdown("---")

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
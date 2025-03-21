import asyncio
import glob
import io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from src.api_client import generate_response
from src.prompts import HARD_NEGATIVE_OPTIONS_PROMPT

IMG_TYPE_LIST = [
    "Report",
    "Test_Paper",
    "Newspaper",
    "Manual",
    "Book_Page",
    "Magazine",
    "Brochure",
    "Book_Cover",
    "Illustrated_Books_and_Comics",
    "Chart_and_Plot",
    "Table",
    "Diagram",
    "Infographic",
    "Poster",
    "Banner",
    "Menu",
    "Packaging_Label",
    "Flyer",
    "Signage",
    "Store_Sign",
    "Product_Detail",
    "Public_Signs",
    "Street_Signs",
    "Mural_and_Graffiti",
    "Mobile_Screenshot",
    "PC_Screenshot",
    "Presentation_Slides",
    "Video_Thumbnail",
    "Video_Scene",
    "Receipts_and_Invoices",
    "Contracts_Documents",
    "Certificates",
    "Handwriting",
    "Tickets_and_Boarding_Passes",
    "",
]

DOMAIN_LIST = [
    "Public_and_Administration",
    "Legal_and_Regulations",
    "Economics_and_Finance",
    "Corporate_and_Business",
    "Marketing_and_Advertising",
    "Education_and_Academia",
    "Medical_and_Healthcare",
    "Transportation_and_Logistics",
    "Travel_and_Tourism",
    "Retail_and_Commerce",
    "Hospitality_and_Food_Service",
    "Entertainment_and_Media",
    "Science_and_Technology",
    "Arts_and_Humanities",
    "Personal_and_Lifestyle",
    "",
]


# Change run_app function to async
async def run_app(dataset_path: str):
    # Take full width
    st.set_page_config(layout="wide")

    # Use cached dataframe on rerun
    if "df" not in st.session_state:
        st.session_state.df = pd.read_parquet(dataset_path)
        st.session_state.df["domain"] = st.session_state.df["domain"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        st.session_state.df["img_type"] = st.session_state.df["img_type"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        st.session_state.df["options"] = st.session_state.df["options"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
    df = st.session_state.df

    # Initialize session state (page index, selected qa)
    if "index" not in st.session_state:
        st.session_state.index = 0

    if "selected_qa_dict" not in st.session_state:
        st.session_state.selected_qa_dict = {}

    # Save and exit
    def save_df_and_stop():
        for idx, sel_question in st.session_state.selected_qa_dict.items():
            row = df.iloc[idx]
            # Update selected_qa
            for candidate in row["candidates"]:
                if candidate["question"] == sel_question:
                    row["selected_qa"] = [candidate]
                    break
            df.iloc[idx] = row

        # Check if there are any modifications
        if st.session_state.selected_qa_dict:
            # Create new filename (original_filename_MMDDHHMM.parquet)
            base_path = dataset_path.split(".")[0]
            new_path = f"{base_path}_savebutton_{time.strftime('%m%d%H%M')}.parquet"

            df.to_parquet(new_path)
            st.markdown(f"Saved to new file: {new_path}")
        else:
            st.markdown("No changes to save.")

        # Reset session state
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
            base_path = dataset_path.split(".")[0]
            new_path = f"{base_path}_nextbutton.parquet"
            df.to_parquet(new_path)
            st.success(f"Saved to: {new_path}")

    # Function to delete current row
    def delete_current_row():
        if st.session_state.index in st.session_state.selected_qa_dict:
            del st.session_state.selected_qa_dict[st.session_state.index]

        # Directly modify df in session_state
        st.session_state.df = st.session_state.df.drop(
            index=st.session_state.df.index[st.session_state.index]
        ).reset_index(drop=True)

        if st.session_state.index >= len(st.session_state.df):
            st.session_state.index = len(st.session_state.df) - 1
        st.rerun()

    # --------------------------------------------------------------------------------
    # (1) Function to update QA information on the current screen
    # --------------------------------------------------------------------------------
    def update_current_qa():
        """
        Reflects the Question / Answer / Reasoning / Options values
        input from the UI to the current row and DataFrame.
        """
        current_index = st.session_state.index
        row = df.iloc[current_index]

        # Find selected QA candidate
        selected_qa_question = st.session_state.selected_qa_dict[current_index]
        current_candidate = next(
            (c for c in row["candidates"] if c["question"] == selected_qa_question),
            None,
        )

        if current_candidate:
            # Input values
            temp_question = st.session_state.get(
                f"temp_question_{current_index}", current_candidate["question"]
            )
            temp_answer = st.session_state.get(
                f"temp_answer_{current_index}", current_candidate["answer"]
            )
            temp_reasoning = st.session_state.get(
                f"temp_reasoning_{current_index}",
                current_candidate.get("reasoning", ""),
            )
            temp_option_1 = st.session_state.get(
                f"temp_option_1_{current_index}", row["options"][0]
            )
            temp_option_2 = st.session_state.get(
                f"temp_option_2_{current_index}", row["options"][1]
            )
            temp_option_3 = st.session_state.get(
                f"temp_option_3_{current_index}", row["options"][2]
            )

            # Update current_candidate
            current_candidate["question"] = temp_question
            current_candidate["answer"] = temp_answer
            current_candidate["reasoning"] = temp_reasoning

            # Update options
            row["options"] = [temp_option_1, temp_option_2, temp_option_3]

            # Update selected_qa_dict when question changes
            st.session_state.selected_qa_dict[current_index] = temp_question

            # Update DataFrame
            df.iloc[current_index] = row

    # --------------------------------------------------------------------------------
    # Previous / Next / Delete buttons
    # --------------------------------------------------------------------------------
    col1, col2, col3 = st.columns([1.3, 4.7, 0.4])
    with col1:
        if st.button("Previous"):
            if st.session_state.index > 0:
                # Move to previous
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

    # 2-column layout (left: image, right: QA and edit UI)
    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader(f"Image #{current_index + 1} <System: {row['system']}>")
        if isinstance(row["image"], bytes):
            image = Image.open(io.BytesIO(row["image"]))
            st.image(image, caption="Reference Image")

    with col_right:
        # CSS style (card UI)
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
            unsafe_allow_html=True,
        )

        with st.container():
            if current_index not in st.session_state.selected_qa_dict:
                st.session_state.selected_qa_dict[current_index] = row["selected_qa"][
                    0
                ]["question"]

            selected_qa_question = st.session_state.selected_qa_dict[current_index]
            candidates = row["candidates"]
            num_columns = min(4, len(candidates))
            cols = st.columns(num_columns)

            # 1) Display QA cards + Select button
            for i, candidate in enumerate(candidates):
                col_idx = i % num_columns
                with cols[col_idx]:
                    selected_css = (
                        "selected-qa"
                        if candidate["question"] == selected_qa_question
                        else ""
                    )
                    reasoning_html = (
                        f"<br><strong>Reasoning:</strong> {candidate['reasoning']}"
                        if candidate.get("reasoning")
                        else ""
                    )
                    card_html = f"""
                        <div class="card {selected_css}">
                            <strong>Question:</strong> {candidate["question"]}<br><br>
                            <strong>Answer:</strong> {candidate["answer"]}<br>{reasoning_html}
                        </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    if st.button(
                        "Select this QA", key=f"select_btn_{current_index}_{i}"
                    ):
                        st.session_state.selected_qa_dict[current_index] = candidate[
                            "question"
                        ]
                        st.rerun()

            # 2) Selected QA edit UI
            current_candidate = next(
                (c for c in candidates if c["question"] == selected_qa_question), None
            )
            if current_candidate:
                qa_col_left, qa_col_right = st.columns(2)

                # Left column: Question/Answer/Reasoning
                with qa_col_left:
                    # Use key in text_input to save values to session_state when changed
                    st.text_input(
                        "Question",
                        value=current_candidate["question"],
                        key=f"temp_question_{current_index}",
                    )
                    st.text_input(
                        "Answer",
                        value=current_candidate["answer"],
                        key=f"temp_answer_{current_index}",
                    )
                    st.text_input(
                        "Reasoning",
                        value=current_candidate.get("reasoning", ""),
                        key=f"temp_reasoning_{current_index}",
                    )

                # Right column: Options
                with qa_col_right:
                    st.text_input(
                        "Option 1",
                        value=row["options"][0],
                        key=f"temp_option_1_{current_index}",
                    )
                    st.text_input(
                        "Option 2",
                        value=row["options"][1],
                        key=f"temp_option_2_{current_index}",
                    )
                    st.text_input(
                        "Option 3",
                        value=row["options"][2],
                        key=f"temp_option_3_{current_index}",
                    )

                    # "Update Options" button - Create and save new options based on selected QA
                    if st.button("🤖 Update Options"):
                        import json

                        prompt = HARD_NEGATIVE_OPTIONS_PROMPT.format(
                            question=current_candidate["question"],
                            correct_answer=current_candidate["answer"],
                        )

                        # Use await directly
                        response = await generate_response(
                            "openai", prompt, "gpt-4o-mini"
                        )

                        if response:
                            try:
                                # Parse JSON response
                                response_json = json.loads(
                                    response.replace("```json", "").replace("```", "")
                                )
                                new_options = response_json.get("options", [])
                                print(new_options)

                                if len(new_options) == 3:
                                    all_options = [
                                        new_options[0],
                                        new_options[1],
                                        new_options[2],
                                        current_candidate["answer"],
                                    ]

                                    # Update DataFrame
                                    row["options"] = all_options
                                    df.iloc[current_index] = row

                                    # Display success message
                                    st.success("Options have been updated!")
                                    st.rerun()
                                else:
                                    st.error("Not enough options were generated.")
                            except json.JSONDecodeError:
                                st.error("Response format is incorrect.")
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")

            # 3) Edit image type and domain
            col_img_type, col_domain = st.columns(2)

            # Check for invalid image types
            invalid_types = [t for t in row["img_type"] if t not in IMG_TYPE_LIST]
            if invalid_types:
                st.markdown(
                    f"""
                    <script>
                        alert("Invalid image types found: {', '.join(invalid_types)}");
                    </script>
                    """,
                    unsafe_allow_html=True,
                )

            def update_img_type():
                current_img_types = st.session_state[f"img_type_{current_index}"]
                row["img_type"] = current_img_types
                df.iloc[current_index] = row

            with col_img_type:
                default_img_types = [t for t in row["img_type"] if t in IMG_TYPE_LIST]
                st.multiselect(
                    "Select image types",
                    options=IMG_TYPE_LIST,
                    default=default_img_types,
                    key=f"img_type_{current_index}",
                    on_change=update_img_type,
                )

            # Check for invalid domains
            invalid_domains = [d for d in row["domain"] if d not in DOMAIN_LIST]
            if invalid_domains:
                st.markdown(
                    f"""
                    <script>
                        alert("Invalid domains found: {', '.join(invalid_domains)}");
                    </script>
                    """,
                    unsafe_allow_html=True,
                )

            def update_domain():
                current_domains = st.session_state[f"domain_{current_index}"]
                row["domain"] = current_domains
                df.iloc[current_index] = row

            with col_domain:
                default_domains = [d for d in row["domain"] if d in DOMAIN_LIST]
                st.multiselect(
                    "Select domains",
                    options=DOMAIN_LIST,
                    default=default_domains,
                    key=f"domain_{current_index}",
                    on_change=update_domain,
                )

            # 4) Save and exit button
            if st.button("Save and Exit"):
                save_df_and_stop()


async def main():
    if "selected_file" in st.session_state:
        await run_app(st.session_state.selected_file)
        return

    # If not present, show file selection UI (list of parquet files in results folder)
    st.title("Select a dataset to load")

    parquet_files = glob.glob(os.path.join("results", "*.parquet"))

    if not parquet_files:
        st.write("No parquet files found in the results folder.")
        return

    selected_file = st.selectbox("Choose a parquet file", parquet_files)

    if st.button("Load selected file"):
        st.session_state.selected_file = selected_file
        st.rerun()


if __name__ == "__main__":
    asyncio.run(main())

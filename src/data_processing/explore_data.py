import pandas as pd
import gradio as gr
from sqlalchemy import create_engine, text
import re
import os
from sqlalchemy.dialects.postgresql.base import PGDialect

def get_from_cnfg(key, config_file):
    import yaml
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config.get(key, None)

def setup_db_connection(config_file):
    DB_URL = get_from_cnfg("cd_url", config_file)

    def fake_get_server_version_info(self, connection):
        version_str = connection.execute(text("SELECT version()")).scalar()
        match = re.search(r'v(\d+)\.(\d+)\.(\d+)', version_str)
        if match:
            return tuple(map(int, match.groups()))
        return (13, 0, 0)

    PGDialect._get_server_version_info = fake_get_server_version_info

    return create_engine(DB_URL)

def fetch_data(engine):
    query = """
    select distinct a.uid, a.question_id, a.question, a.question_category, a.actual_answer, a.model_name, a.model_answer, a.image_link,
    b.evaluated_by_model, b.total_score, b.severity_classification 
    from model_responses_r2 a
    join model_response_evaluation_r2 b on a.uid=b.uid and a.question_id=b.question_id and a.model_name=b.model_id
    """
    return pd.read_sql(query, con=engine)

def transform_image_path(image_link, source_folder):
    if not image_link:
        return None

    pattern = r'(\/p\d+\/p\d+\/s\d+\/[a-f0-9-]+\.jpg)'
    match = re.search(pattern, image_link)

    if match:
        relative_path = match.group(1)
        return source_folder + relative_path
    return None

def get_dropdown_values(df, column):
    # Convert all values to strings
    return sorted([str(x) for x in df[column].unique().tolist()])

def update_model_dropdown(question_category, df):
    if not question_category:
        return [], None
    filtered_df = df[df['question_category'] == question_category]
    models = get_dropdown_values(filtered_df, 'model_name')
    return models, models[0] if models else None

def update_question_dropdown(question_category, model_name, df):
    if not question_category or not model_name:
        return [], None
    filtered_df = df[(df['question_category'] == question_category) & (df['model_name'] == model_name)]
    question_ids = get_dropdown_values(filtered_df, 'question_id')
    return question_ids, question_ids[0] if question_ids else None

def display_question_data(question_category, model_name, question_id, df, source_folder):
    if not question_category or not model_name or not question_id:
        return "No data found", "No data found", "No data found", "No data found", None

    # Handle both string and numeric question_id
    if isinstance(question_id, (int, float)):
        question_id = str(question_id)

    filtered_df = df[
        (df['question_category'] == question_category) &
        (df['model_name'] == model_name) &
        (df['question_id'].astype(str) == question_id)
        ]

    if filtered_df.empty:
        return "No data found", "No data found", "No data found", "No data found", None

    row = filtered_df.iloc[0]

    question = row['question']
    actual_answer = row['actual_answer']
    model_answer = row['model_answer']
    score_info = f"Evaluated by: {row['evaluated_by_model']}\nTotal Score: {row['total_score']}\nSeverity: {row['severity_classification']}"

    image_path = transform_image_path(row['image_link'], source_folder)

    return question, actual_answer, model_answer, score_info, image_path

def create_gradio_app(config_file, source_folder):
    engine = setup_db_connection(config_file)
    df = fetch_data(engine)

    # Convert all IDs to strings for consistency
    df['question_id'] = df['question_id'].astype(str)

    categories = get_dropdown_values(df, 'question_category')
    initial_category = categories[0] if categories else None

    initial_models = []
    initial_questions = []

    if initial_category:
        filtered_df = df[df['question_category'] == initial_category]
        initial_models = get_dropdown_values(filtered_df, 'model_name')

    initial_model = initial_models[0] if initial_models else None

    if initial_category and initial_model:
        filtered_df = df[(df['question_category'] == initial_category) & (df['model_name'] == initial_model)]
        initial_questions = get_dropdown_values(filtered_df, 'question_id')

    initial_question = initial_questions[0] if initial_questions else None

    with gr.Blocks(title="MIMIC Data Explorer") as app:
        gr.Markdown("# MIMIC Model Data Explorer")

        with gr.Row():
            with gr.Column(scale=1):
                category_dropdown = gr.Dropdown(
                    choices=categories,
                    value=initial_category,
                    label="Question Category",
                    allow_custom_value=True
                )

                model_dropdown = gr.Dropdown(
                    choices=initial_models,
                    value=initial_model,
                    label="Model Name",
                    allow_custom_value=True
                )

                question_dropdown = gr.Dropdown(
                    choices=initial_questions,
                    value=initial_question,
                    label="Question ID",
                    allow_custom_value=True
                )

            with gr.Column(scale=2):
                question_display = gr.Textbox(label="Question", lines=2)
                actual_answer_display = gr.Textbox(label="Ground Truth Answer", lines=4)
                model_answer_display = gr.Textbox(label="Model Answer", lines=8)
                evaluation_display = gr.Textbox(label="Evaluation Metrics", lines=3)
                image_display = gr.Image(label="X-ray Image (if available)")

        def handle_category_change(question_category):
            if not question_category:
                return [], None, [], None, "", "", "", "", None

            # Update model dropdown
            models, default_model = update_model_dropdown(question_category, df)
            if not default_model:
                return models, None, [], None, "", "", "", "", None

            # Update question dropdown
            question_ids, default_question = update_question_dropdown(question_category, default_model, df)
            if not default_question:
                return models, default_model, question_ids, None, "", "", "", "", None

            # Get display data
            question, actual, model, eval_info, image = display_question_data(
                question_category, default_model, default_question, df, source_folder
            )

            return models, default_model, question_ids, default_question, question, actual, model, eval_info, image

        def handle_model_change(question_category, model_name):
            if not question_category or not model_name:
                return [], None, "", "", "", "", None

            # Update question dropdown based on selected model
            question_ids, default_question = update_question_dropdown(question_category, model_name, df)
            if not default_question:
                return question_ids, None, "", "", "", "", None

            # Get display data for the first question
            question, actual, model, eval_info, image = display_question_data(
                question_category, model_name, default_question, df, source_folder
            )

            return question_ids, default_question, question, actual, model, eval_info, image

        # Setup event handlers
        category_dropdown.change(
            fn=handle_category_change,
            inputs=category_dropdown,
            outputs=[
                model_dropdown, model_dropdown,
                question_dropdown, question_dropdown,
                question_display, actual_answer_display, model_answer_display, evaluation_display, image_display
            ]
        )

        model_dropdown.change(
            fn=handle_model_change,
            inputs=[category_dropdown, model_dropdown],
            outputs=[
                question_dropdown, question_dropdown,
                question_display, actual_answer_display, model_answer_display, evaluation_display, image_display
            ]
        )

        question_dropdown.change(
            fn=display_question_data,
            inputs=[category_dropdown, model_dropdown, question_dropdown, gr.State(df), gr.State(source_folder)],
            outputs=[question_display, actual_answer_display, model_answer_display, evaluation_display, image_display]
        )

        # Initialize display with initial values if they exist
        if initial_category and initial_model and initial_question:
            initial_data = display_question_data(initial_category, initial_model, initial_question, df, source_folder)
            question_display.value = initial_data[0]
            actual_answer_display.value = initial_data[1]
            model_answer_display.value = initial_data[2]
            evaluation_display.value = initial_data[3]
            image_display.value = initial_data[4]

    return app

if __name__ == "__main__":
    config_file = "/Users/bineshkumar/Documents/config.yaml"
    source_folder = "/Users/bineshkumar/Documents/mimic-cxr-jpg/2.1.0/files"

    app = create_gradio_app(config_file, source_folder)

    # Add source_folder as an allowed path to fix the image error
    app.launch(allowed_paths=[source_folder])
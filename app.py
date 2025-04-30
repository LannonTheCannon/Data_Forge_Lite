from node_template import BaseNode, ThemeNode, QuestionNode, TerminalNode
import numpy as np
import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import json
from code_editor import code_editor
import plotly.graph_objects as go
import plotly.io as pio
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import ManualLayout, RadialLayout, TreeLayout

# Node template (your custom classes)
from node_template import BaseNode, ThemeNode, QuestionNode, TerminalNode

# Agents
from ai_data_science_team.agents import DataCleaningAgent, FeatureEngineeringAgent

# LLM
from langchain_openai import ChatOpenAI
    # Create the ChatOpenAI instance NOW



st.sidebar.header('Enter your OpenAI API Key')

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = None

st.session_state['OPENAI_API_KEY'] = st.sidebar.text_input(
    "API KEY",
    type="password",
    help="Your OpenAI API key is required for the app to function."
)

# Test OpenAI API Key
if st.session_state['OPENAI_API_KEY']:
    # Set the API key for OPENAI


    # Test the API key (optional)
    try:
        llm = ChatOpenAI(
        model='gpt-4o-mini',
        openai_api_key=st.session_state['OPENAI_API_KEY']
        )
        client = openai.OpenAI(
        api_key=st.session_state['OPENAI_API_KEY']
)
        # Example: Fetch models to validate the key
        # models = client.models.list()
        # st.success("API Key is valid!")

    except Exception as e:
        st.error(f"Invalid API key: {e}")
else:
    st.info("Please enter your OpenAI API Key to proceed!")
    st.stop()

# OpenAI Client
# client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Config
# st.set_page_config(page_title="DataForge: Clean, Engineer, Map", layout="wide")

# Colors for mindmap
COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

# Initialize Agents
data_cleaning_agent = DataCleaningAgent(model=llm, n_samples=50, log=False)
feature_engineering_agent = FeatureEngineeringAgent(model=llm, n_samples=50, log=False)

# -------------- Session State Initialization -------------- #

if "curr_state" not in st.session_state:
    dataset_label = st.session_state.get("dataset_name", "Dataset")
    root_theme = ThemeNode(
        node_id="S0",
        label="ROOT",
        full_question="Overview of the dataset",
        category="Meta",
        node_type="theme",
        parent_id=None,
        metadata={"content": dataset_label}
    )
    st.session_state.mindmap_nodes = {"S0": root_theme}
    st.session_state.curr_state = StreamlitFlowState(nodes=[root_theme.to_streamlit_node()], edges=[])

for key in [
    "chart_path", "df", "df_preview", "df_summary", "metadata_string",
    "saved_charts", "DATA_RAW", "plots", "dataframes", "msg_index",
    "clicked_questions", "dataset_name"
]:
    if key not in st.session_state:
        if key == "df":
            # seed with an empty DataFrame so .columns always exists
            st.session_state[key] = pd.DataFrame()
        elif key in ["chart_path", "df_preview", "df_summary", "metadata_string", "DATA_RAW"]:
            st.session_state[key] = None
        else:
            st.session_state[key] = []
            
if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()

if "seen_embeddings" not in st.session_state:
    st.session_state.seen_embeddings = []

# -------------- Utility Functions -------------- #

def generate_root_summary_question(metadata_string: str) -> str:
    if not metadata_string:
        return "Overview of the dataset"
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a data summarizer."},
                {"role": "user", "content": f"Dataset metadata:\n{metadata_string}\n\nSummarize in one sentence."}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip().split("\n")[0]
    except Exception:
        return "Overview of the dataset"


def get_assistant_interpretation(user_input, metadata, valid_columns):
    column_names = ', '.join(valid_columns)

    prompt = f"""
You are a visualization interpreter.

Your job is to rephrase the user's request into a **precise and code-compatible** instruction. Use this format:

→ "Create a [chart type] of the `[y_column]` on the y-axis ([aggregation]) and the `[x_column]` in the x-axis and make the chart [color]."

---

Rules:
- DO NOT invent or guess column names. Use ONLY from this list:
  {column_names}
- NEVER say "average salary in USD" — instead say: "`salary_in_usd` on the y-axis (avg)"
- Keep aggregation words like "avg", "sum", or "count" OUTSIDE of the column name.
- Keep axis mappings clear and exact.
- Mention the color explicitly at the end.
- Avoid words like “visualize” or “illustrate.” Just say "Create a bar chart..."

---

📥 USER QUERY:
{user_input}

📊 METADATA:
{metadata}

✍️ Respond with just one sentence using the format shown above.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites data visualization queries into precise and code-friendly instructions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        return response.choices[0].message.content

    except Exception as e:
        st.warning(f"Error in get_assistant_interpretation: {e}")
        return "Could not interpret user request."

def display_chat_history():
    if "chat_artifacts" not in st.session_state:
        st.session_state["chat_artifacts"] = {}

    for i, msg in enumerate(msgs.messages):
        role_label = "User" if msg.type == "human" else "Assistant"
        with st.chat_message(msg.type):
            st.markdown(f"**{role_label}:** {msg.content}")

            if i in st.session_state["chat_artifacts"]:
                for j, artifact in enumerate(st.session_state["chat_artifacts"][i]):
                    unique_key = f"msg_{i}_artifact_{j}"
                    editor_key = f"editor_code_{unique_key}"
                    output_key = f"output_chart_{unique_key}"

                    with st.expander(f"\U0001F4CE {artifact['title']}", expanded=True):
                        tabs = st.tabs(["📊 Output", "📋 Data Preview", "💻 Code"])

                        # --- Code Tab First, to capture edits and trigger updates ---
                        with tabs[0]:
                            output_obj = st.session_state.get(output_key, artifact.get("data"))
                            render_type = artifact.get("render_type")

                            if isinstance(output_obj, dict) and "data" in output_obj and "layout" in output_obj:
                                output_obj = pio.from_json(json.dumps(output_obj))

                            if render_type == "plotly":
                                st.plotly_chart(
                                    output_obj,
                                    use_container_width=True,
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                        "displaylogo": False
                                    },
                                    key=f"plotly_{output_key}"
                                )
                            elif render_type == "dataframe":
                                st.dataframe(output_obj, key=f"df_{output_key}")
                            else:
                                st.write(output_obj)
                        with tabs[1]:
                            df_preview = artifact.get("data_preview")
                            if df_preview is not None:
                                st.write("### Data‑Wrangler Output")
                                st.dataframe(df_preview, use_container_width=True)
                            else:
                                st.write("_No data preview available._")

                        with tabs[2]:
                            code_before = st.session_state.get(editor_key, artifact.get("code", ""))
                            editor_response = code_editor(
                                code=code_before,
                                lang="python",
                                theme="dracula",
                                height=300,
                                # buttons=[
                                #     {
                                #         "name": "Run",
                                #         "feather": "Play",
                                #         "primary": True,
                                #         "hasText": True,
                                #         "showWithIcon": True,
                                #         "commands": ["submit"],
                                #         "style": {"bottom": "0.44rem", "right": "0.4rem"}
                                #     }
                                # ],
                                key=f"code_editor_{unique_key}"
                            )

                            new_code = editor_response.get("text", "").strip()


                            #
                            #
                            # # Only run if the code has changed
                            # if new_code and new_code != st.session_state.get(editor_key):
                            #     try:
                            #         exec_globals = {
                            #             "df": st.session_state.df,
                            #             "pd": pd,
                            #             "np": np,
                            #             "sns": sns,
                            #             "go": go,
                            #             "plt": plt,
                            #             "pio": pio,
                            #             "st": st,
                            #             "json": json
                            #         }
                            #         exec_locals = {}
                            #         exec(new_code, exec_globals, exec_locals)
                            #
                            #         output_obj = exec_locals.get("fig") or \
                            #                      exec_locals.get("output") or \
                            #                      exec_locals.get("fig_dict")
                            #
                            #         if isinstance(output_obj, dict) and "data" in output_obj and "layout" in output_obj:
                            #             output_obj = pio.from_json(json.dumps(output_obj))
                            #
                            #         artifact["data"] = output_obj
                            #         artifact["render_type"] = "plotly" if isinstance(output_obj, go.Figure) else "dataframe"
                            #         st.session_state[editor_key] = new_code
                            #         st.session_state[output_key] = output_obj
                            #
                            #     except Exception as e:
                            #         st.error(f"Error executing code: {e}")


# -------------- Page Layouts -------------- #

PAGE_OPTIONS = ['Data Upload', 'Mind Mapping', "Data Analyst Agent"]
page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

# -------------- Main -------------- #

if __name__ == "__main__":

    if page == 'Data Upload':
        st.title('🧹 DataForge Upload + Transformation')

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        # If a new file uploaded → process it
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # st.success("✅ File uploaded successfully!")

                dataset_name = uploaded_file.name.rsplit('.', 1)[0]

                # Cleaning
                with st.spinner('🧹 Cleaning Data...'):
                    data_cleaning_agent.invoke_agent(
                        data_raw=df,
                        user_instructions="Use default cleaning steps."
                    )
                    df_cleaned = data_cleaning_agent.get_data_cleaned()

                # Feature Engineering
                with st.spinner('🛠️ Engineering Features...'):
                    feature_engineering_agent.invoke_agent(
                        data_raw=df_cleaned,
                        user_instructions="Use default feature engineering steps."
                    )
                    df_final = feature_engineering_agent.get_data_engineered()

                # Save everything into session_state
                st.session_state.df = df_final
                st.session_state.DATA_RAW = df_final
                st.session_state.df_preview = df_final.head()

                st.session_state.df_uploaded_raw = df  # <-- new: raw uploaded file
                st.session_state.df_cleaned = df_cleaned
                st.session_state.df_final = df_final
                st.session_state.cleaning_code = data_cleaning_agent.get_data_cleaner_function()
                st.session_state.feature_engineering_code = feature_engineering_agent.get_feature_engineer_function()

                numeric_summary = df_final.describe()
                # categorical_summary = df_final.describe(include=['object', 'category', 'bool'])

                #############
                numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df_final.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

                # Protect against weird columns
                safe_categorical_cols = []
                cat_cardinalities = {}
                top_cats = {}

                for col in categorical_cols:
                    try:
                        nunique = int(df_final[col].nunique())
                        top_values = df_final[col].value_counts().head(3).to_dict()

                        cat_cardinalities[col] = nunique
                        top_cats[col] = top_values
                        safe_categorical_cols.append(col)

                    except Exception as e:
                        st.warning(f"Skipping column `{col}` due to error: {e}")

                # Now use only the safe columns
                st.session_state.df_summary = numeric_summary
                st.session_state.metadata_string = (
                    f"Columns: {list(df_final.columns)}\n"
                    f"Numeric columns: {numeric_cols}\n"
                    f"Categorical columns: {safe_categorical_cols} (cardinalities: {cat_cardinalities})\n"
                    f"Top categories: {top_cats}\n"
                    f"Row count: {len(df_final)}"
                )
                # st.write(st.session_state.metadata_string)

                root_question = generate_root_summary_question(st.session_state.metadata_string)
                if st.session_state.curr_state.nodes:
                    root_node = st.session_state.curr_state.nodes[0]
                    root_node.data["full_question"] = root_question
                    root_node.data["content"] = dataset_name
                    root_node.data["short_label"] = "ROOT"

            except Exception as e:
                st.error(f"Something went wrong during upload: {e}")

        # --- Now, if data already exists in session_state, show it ---
        if "df_uploaded_raw" in st.session_state:
            tabs = st.tabs([
                "📂 Raw Uploaded Data",
                "🧹 Cleaned Data",
                "🛠️ Feature Engineered Data",
                "📜 Cleaning Agent Code",
                "📜 Feature Engineering Code"
            ])

            with tabs[0]:
                st.subheader("STEP 1) Raw Uploaded Data Preview")
                st.dataframe(st.session_state.df_uploaded_raw.head())

            with tabs[1]:
                st.subheader("STEP 2) Cleaned Data Preview")
                st.dataframe(st.session_state.df_cleaned.head())

            with tabs[2]:
                st.subheader("STEP 3) Final Feature Engineered Data")
                st.dataframe(st.session_state.df_final.head())

                csv = st.session_state.df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Dataset",
                    data=csv,
                    file_name="dataforge_cleaned_dataset.csv",
                    mime="text/csv"
                )

            with tabs[3]:
                st.subheader("STEP 4) Data Cleaning Agent - Generated Code")
                st.code(st.session_state.cleaning_code, language='python')

            with tabs[4]:
                st.subheader("STEP 5) Feature Engineering Agent - Generated Code")
                st.code(st.session_state.feature_engineering_code, language='python')


    elif page == 'Mind Mapping':
        st.title('🧠 Mind Mapping + Agentic Exploration')

        if st.session_state.get("dataset_name"):
            root_node = st.session_state.curr_state.nodes[0]
            if root_node.data["content"] != st.session_state["dataset_name"]:
                root_node.data["content"] = st.session_state["dataset_name"]

        col1, col2 = st.columns([3, 1])

        with col2:
            if st.button("🔄 Reset Mind Map"):
                dataset_label = st.session_state.get("dataset_name", "Dataset")
                new_root = StreamlitFlowNode(
                    "S0",
                    (0, 0),
                    {
                        "section_path": "S0",
                        "short_label": "ROOT",
                        "full_question": st.session_state.metadata_string,
                        "content": dataset_label
                    },
                    "input",
                    "right",
                    style={"backgroundColor": COLOR_PALETTE[0]}
                )
                st.session_state.curr_state = StreamlitFlowState(nodes=[new_root], edges=[])
                st.session_state.expanded_nodes = set()
                st.session_state.clicked_questions = []
                st.rerun()

        # Render mind map
        st.session_state.curr_state = streamlit_flow(
            "mind_map",
            st.session_state.curr_state,
            layout=TreeLayout(direction="right"),
            fit_view=True,
            height=550,
            get_node_on_click=True,
            enable_node_menu=True,
            enable_edge_menu=True,
            show_minimap=False
        )

        # Node click event
        clicked_node_id = st.session_state.curr_state.selected_id

        if clicked_node_id and clicked_node_id not in st.session_state.expanded_nodes:
            clicked_obj = st.session_state.mindmap_nodes.get(clicked_node_id)

            if clicked_obj:
                # ✨ Log clicked node if not already recorded
                already_logged = any(q["section"] == clicked_obj.node_id for q in st.session_state.clicked_questions)
                if not already_logged:
                    st.session_state.clicked_questions.append({
                        "section": clicked_obj.node_id,
                        "short_label": clicked_obj.label,
                        "node_type": clicked_obj.node_type,
                        "full_question": clicked_obj.full_question
                    })

                # ✅ Expand children if needed
                if clicked_obj.can_expand() and clicked_node_id not in st.session_state.expanded_nodes:
                    children = clicked_obj.get_children(openai_client=client, metadata_string=st.session_state.metadata_string) or []

                    for child in children:
                        st.session_state.mindmap_nodes[child.node_id] = child
                        st.session_state.curr_state.nodes.append(child.to_streamlit_node())
                        st.session_state.curr_state.edges.append(
                            StreamlitFlowEdge(
                                f"{clicked_obj.node_id}-{child.node_id}",
                                clicked_obj.node_id,
                                child.node_id,
                                animated=True
                            )
                        )
                    clicked_obj.mark_expanded()
                    st.session_state.expanded_nodes.add(clicked_node_id)

                st.rerun()
        # --- Always show the Clicked Questions Table, even if empty
        st.write("## 🧠 Your Exploration Path")

        if st.session_state.clicked_questions:
            df_log = pd.DataFrame(st.session_state.clicked_questions)
            df_log = df_log[["section", "short_label", "node_type", "full_question"]]
            st.dataframe(df_log, use_container_width=True)
        else:
            st.info("Start clicking nodes on the mind map to populate your exploration path!")

    elif page=='Data Analyst Agent':
        st.subheader('Pandas Data Analyst Mode')
        msgs = StreamlitChatMessageHistory(key="pandas_data_analyst_messages")
        if len(msgs.messages) == 0:
            pass
            # msgs.add_ai_message("IMPORTANT: For best results use this formula -> Create a [chart] of the [field] on the y-axis (aggregation) and the [field] on the x-axis and make the chart [color].")
        if 'pandas_data_analyst' not in st.session_state:
            model = ChatOpenAI(model='gpt-4.1-mini',
                               api_key=st.session_state['OPENAI_API_KEY'])
            st.session_state.pandas_data_analyst = PandasDataAnalyst(
                model=model,
                data_wrangling_agent=DataWranglingAgent(model=model,
                                                        log=True,
                                                        n_samples=100),
                data_visualization_agent=DataVisualizationAgent(
                    model=model,
                    log=True,
                    log_path="logs",
                    overwrite=False,  # ✅ Ensures every chart gets a separate file
                    n_samples=100,
                    bypass_recommended_steps=False)
            )
        question = st.chat_input('Ask a question about your dataset!')
        interpretation = get_assistant_interpretation(
            question,
            st.session_state['metadata_string'],
            st.session_state.df.columns  # ✅ pass real column names
        )
        # print(interpretation)

        if question:
            msgs.add_user_message(question)
            with st.spinner("Thinking..."):
                try:
                    st.session_state.pandas_data_analyst.invoke_agent(
                        user_instructions=question,
                        data_raw=st.session_state["DATA_RAW"]
                    )
                    result = st.session_state.pandas_data_analyst.get_response()
                    route = result.get("routing_preprocessor_decision", "")
                    ai_msg = "Here's what I found:"
                    msgs.add_ai_message(ai_msg)
                    msg_index = len(msgs.messages) - 1
                    if "chat_artifacts" not in st.session_state:
                        st.session_state["chat_artifacts"] = {}
                    st.session_state["chat_artifacts"][msg_index] = []
                    if route == "chart" and not result.get("plotly_error", False):
                        plot_obj = pio.from_json(json.dumps(result["plotly_graph"]))
                        st.session_state.plots.append(plot_obj)
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        viz_code = result.get('data_visualization_function', "")
                        wrangle_code = result.get('data_wrangler_function', "")
                        df_wrangled  = result.get('data_wrangled')

                        # Combine both functions into one code block
                        combined_code = f"{wrangle_code}\n\n{viz_code}\n\n# Runtime Execution\noutput = data_visualization(data_wrangler([df]))"

                        st.session_state["chat_artifacts"][msg_index].append({
                            "title": "Chart",
                            "render_type": "plotly",
                            "data": plot_obj,
                            "code": combined_code,
                            "data_preview": df_wrangled
                        })
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        # print(result['data_visualization_function'])

                    elif route == "table":
                        df = result.get("data_wrangled")
                        if df is not None:
                            st.session_state.dataframes.append(df)
                            st.session_state["chat_artifacts"][msg_index].append({
                                "title": "Table",
                                "render_type": "dataframe",
                                "data": df,
                                'code': result.get('data_wrangler_function')
                            })
                except Exception as e:
                    error_msg = f"Error: {e}"
                    msgs.add_ai_message(error_msg)
        display_chat_history()
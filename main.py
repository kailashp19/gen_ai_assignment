import asyncio
import streamlit as st
import yaml

st.set_page_config(layout='wide')
# Page title
st.sidebar.title("Evaluation Framework")

library_select = []
# st.sidebar.title("Framework Tools")

genre = st.sidebar.radio(
    ":rainbow[Framework Tools]",
    ["RAGAS", "Trulens", "DeepEval"]
)

def categorize_data(json_data):
    string_values = {}
    numeric_values = {}

    for key, value in json_data.items():
        if isinstance(value, (int, float)):
            numeric_values[key] = value
        else:
            string_values[key] = value

    return string_values, numeric_values

with st.container():
    # Prompt input
    col1, col2 = st.columns([13, 13])
    with col1:
        userinput = st.text_input("Prompt", placeholder="Enter prompt here")

    # Usecase dropdown
    with col2:
        usecase = st.selectbox("Usecase", options=["RAG", "txt2SQL", "codegen", "text_summarization"])
    
    if usecase == "RAG":
        groundtruth = st.text_area("GroundTruth", placeholder="Enter groundtruth context here")
        ragcontext = st.text_area("Context", placeholder="Enter RAG context here")
        answer = st.text_input("Response", placeholder="Enter LLM generated response here")
    else:
        answer = st.text_input("Response", placeholder="Enter LLM generated response here")
        groundtruth = ""
        ragcontext = ""


if st.button("Show Evaluation Metrics", icon="📈",type="primary"):
    from Ragaseval import RAGAS
    from trulensfile import Trulenslib
    from deepevalfile import DeepEval

    # Reading nested data from a YAML file
    with open('metrics.yaml', 'r') as file:
        nested_data = yaml.safe_load(file)

    # Bottom section: Evaluation metrics
    st.subheader("Evaluation Metrics", divider=True)

    usecase_rules = nested_data[usecase]
    rules = []
    if usecase == "RAG" and genre in list(usecase_rules.keys()):
        if groundtruth == "":
            rules = usecase_rules[genre]['WithoutGT']
        else:
            rules = usecase_rules[genre]['WithGT']
    else:
        if genre in list(usecase_rules.keys()):
            rules = usecase_rules[genre]
        else:
            rules = []
            st.warning("no metrics are available for this library")

    # st.multiselect(f"{usecase} Metrics:", rules)
    if len(rules) !=0:
        # cols = st.columns(len(rules))
        # i=0
        # while i< len(rules):
        #     with cols[i]:
        #         check = st.checkbox(rules[i])
        #         i=i+1

        # HTML template for representation
        html_content = """
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 8px; width: 100%; margin: auto;">
            <ul style="list-style-type: square; font-size: 16px; color: #555;">
        """
        for metric in rules:
            html_content += f"        <li><h5 style='color: #333;'>{metric}</h5></li>\n"
        html_content += """
        </div>
        """

        # Display the HTML content
        st.markdown(html_content, unsafe_allow_html=True)

        obj = eval(genre)(rules, userinput, ragcontext, answer, groundtruth)
        report = obj.call_funcs()
        print("report",report)
        # Visualize the JSON data
        st.subheader("Evaluation Visualizer", divider=True)

        # Load and categorize data
        string_values, numeric_values = categorize_data(report)

        # Display string-based values
        # for key, value in string_values.items():
        #     col1str, _= st.columns([6,5])
        #     with col1str:
        #         st.markdown(f"<div style='border:1px solid #ccc; padding:10px; margin:5px; border-radius:5px;'><strong>{key}:</strong> {value}</div>", unsafe_allow_html=True)
        # Display numeric-based values
        for key, value in numeric_values.items():
            col1s, _= st.columns([6,5])
            with col1s:
                st.slider(f"{key}", min_value=0.0, max_value=1.0, value=float(value))


#         # Metrics buttons
#         with col3:
#             st.markdown("### Metrics")
#             metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
#             with metrics_col1:
#                 st.button("⚙")
#             with metrics_col2:
#                 st.button("📈")
#             with metrics_col3:
#                 st.button("🔄")
#             with metrics_col4:
#                 st.button("✖")

# # Divider
# st.divider()

# # Bottom section: Evaluation metrics
# st.subheader("Evaluation Metrics")

# # Faithfulness, Context Relevancy, and Hallucination
# col4, col5 = st.columns([2, 2])
# with col4:
#     faithfulness = st.slider("Faithfulness (%)", min_value=0, max_value=100, value=85)
#     context_relevancy = st.slider("Context Relevancy (%)", min_value=0, max_value=100, value=90)

# with col5:
#     hallucination = st.radio("Hallucination", options=["True", "False"], index=1)

# # Additional evaluation space
# st.text_area("Additional Notes", placeholder="Enter additional evaluation notes here")
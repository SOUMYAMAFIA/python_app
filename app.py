import streamlit as st
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Define your Azure OpenAI deployment details
azure_openai_api_key = "df9D3simI5RalfedvfQBPnuUg1PLTZcQuTStmUpu8BQ0EynYFAaLJQQJ99ALACfhMk5XJ3w3AAAAACOG4lEB"
azure_openai_api_base = "https://kanan-m49gpffn-swedencentral.cognitiveservices.azure.com/"
azure_openai_deployment_name = "gpt-4o-mini"
azure_openai_api_version = "2024-08-01-preview"

def analyze_sales_data(df):
    llm = AzureChatOpenAI(
        deployment_name=azure_openai_deployment_name,
        openai_api_base=azure_openai_api_base,
        openai_api_key=azure_openai_api_key,
        openai_api_version=azure_openai_api_version,
        temperature=0.7,
        max_tokens=1024,
    )
    prompt = f"""
    You are Sales Bot. An expert in sales reporting and Analytics.
    Given the below DataFrame as string, your job is to provide with:
        * `Detailed Summary as a paragraph`
        * `Key Highlights in the form of Bullet Points`
    Note: 
        The Summary and Key Highlights are to help design Marketing Strategy. 
        Ensure to mention which products did better/worse compared to last year. Which products Need Immediate Attention.
        Add any information you see in the data you think is relevant.

    # Provided Sales Dataset:
    ---
    {str(df)}
    ---
    """
    response = llm.invoke(prompt)
    return response.content

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")

    with st.spinner("Analyzing sales data..."):
        summary = analyze_sales_data(df.copy())  # Avoid modifying the original DataFrame

    # Create a side pane for the DataFrame
    with st.sidebar:
        st.subheader("Uploaded Dataframe")
        st.dataframe(df)

    st.subheader("Sales Bot Analysis")
    st.write(summary)

    # Chat window for data exploration
    with st.sidebar:
        st.subheader("Explore Your Data (Type your query below)")

        user_query = st.text_input("", key="user_query")

        if user_query:
            agent = create_pandas_dataframe_agent(
                AzureChatOpenAI(
                    deployment_name=azure_openai_deployment_name,
                    openai_api_base=azure_openai_api_base,
                    openai_api_key=azure_openai_api_key,
                    openai_api_version=azure_openai_api_version,
                    temperature=0,
                ),
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
            )
            answer = agent.invoke(user_query)
            query_results = answer["output"]

            # Display the query and results
            st.write(query_results)

else:
    st.info("Upload an Excel file to perform sales analysis.")

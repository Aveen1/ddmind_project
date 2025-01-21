from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import os
import openai
import json
from typing import List, Dict


# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define Pydantic model for structured response parsing
class AnalysisRecommendation(BaseModel):
    analysis_types: List[str]
    filters: Dict[str, List[str]]  # Change to dictionary with keys as filter names and values as lists of filter values
    value_columns: List[str]
    time_periods: Dict[str, List[str]]  # Change to dictionary with keys like 'start_date', 'end_date', etc.

def analyze_data_with_langchain(df):
    """Uses LangChain to structure the ChatGPT prompt and retrieve recommendations."""
    # Create a LangChain prompt
    prompt_template = PromptTemplate(
        input_variables=["dataset_preview"],
        template=(
            "Analyze the following dataset and provide structured analysis recommendations in JSON format. "
            "The JSON object should include the following keys: "
            "'analysis_types', 'filters', 'value_columns', and 'time_periods'.\n"
            "Make sure to return a valid JSON object with the exact keys mentioned above.\n"
            "Dataset preview:\n{dataset_preview}"
        )
    )

    #Format the prompt
    prompt = prompt_template.format(dataset_preview=df.head().to_string())

    #Initialize the ChatOpenAI model
    chat = ChatOpenAI(model="gpt-4o", temperature=0)

    # Use SystemMessage and HumanMessage for the input
    messages = [
        SystemMessage(content="You are an expert data analyst."),
        HumanMessage(content=prompt)
    ]

    # Get the response
    response = chat(messages)
    return response.content

def parse_chatgpt_response(response, df_columns):
    """Parses ChatGPT's response to extract analysis options using Pydantic."""
    try:
        # Log the raw response for debugging
        #st.write("### Raw Response from ChatGPT:")
        #st.write(response)

        # Check if the response is empty or not a valid JSON
        if not response.strip():
            st.error("Received empty response")
            return [], [], [], []

        # Try to load the response as JSON
        response_dict = json.loads(response)

        # Parse the response as a structured recommendation
        parsed_response = AnalysisRecommendation.parse_obj(response_dict)

        # Validate that filters and value columns exist in the dataset
        filters = {key: [col for col in value if col in df_columns] for key, value in parsed_response.filters.items()}
        value_columns = [col for col in parsed_response.value_columns if col in df_columns]

        return parsed_response.analysis_types, filters, value_columns, parsed_response.time_periods
    except json.JSONDecodeError as e:
        #st.error(f"Error decoding JSON response: {e}")
        return [], [], [], []
    except ValidationError as e:
        st.error(f"Error parsing ChatGPT response: {e}")
        return [], [], [], []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return [], [], [], []

def main():
    st.title("DDMind")

    # Step 1: Upload Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception:
            df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data:")
        st.write(df)

        # Step 2: Clean the data
        df_cleaned = df.drop_duplicates().fillna(method='ffill').fillna(method='bfill')
        st.write("### Cleaned Data:")
        st.write(df_cleaned)

        # Step 3: Use LangChain and ChatGPT for analysis suggestions
        st.write("### Recommendations from DDMind:")
        analysis_response = analyze_data_with_langchain(df_cleaned)
        st.write(analysis_response)

        # Step 4: Parse ChatGPT response using Pydantic
        analysis_types, filters, value_columns, time_periods = parse_chatgpt_response(analysis_response, df_cleaned.columns)
        
        st.write("### Extracted Analysis Types:")
        st.write(analysis_types)
        st.write("### Extracted Filters:")
        st.write(filters)
        st.write("### Extracted Value Columns:")
        st.write(value_columns)
        st.write("### Extracted Time Periods:")
        st.write(time_periods)

        #if not filters or not value_columns:
            #st.error("No valid filters or value columns extracted. Ensure the recommendations align with the dataset.")
            #return

        # Step 5: Dropdowns for user to choose analysis
        analysis_type = st.selectbox("Select Analysis Type", analysis_types)
        filter_column = st.selectbox("Select Filter Column", filters)
        value_column = st.selectbox("Select Value Column", value_columns)
        time_period = st.selectbox("Select Time Period", time_periods)

        # Step 6: Perform analysis based on selected options
        if filter_column in df_cleaned.columns and value_column in df_cleaned.columns:
            try:
                st.write(f"Performing {analysis_type} on {value_column} filtered by {filter_column} ({time_period})...")
                analysis_result = df_cleaned.groupby(filter_column)[value_column].sum()
                st.write("### Analysis Result:")
                st.write(analysis_result)
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
        #else:
            #st.error("Selected filter or value column does not exist in the dataset. Please choose valid columns.")

if __name__ == "__main__":
    main()

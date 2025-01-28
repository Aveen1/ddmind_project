from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st
import pandas as pd
import os
import openai
import json
from io import BytesIO
from datetime import datetime
import numpy as np

#Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_data_with_langchain(df):
    """Uses LangChain to structure the ChatGPT prompt and retrieve recommendations."""
    prompt_template = PromptTemplate(
        input_variables=["dataset_preview", "columns"],
        template=(
            "Analyze the following dataset with columns: {columns}\n\n"
            "Dataset preview:\n{dataset_preview}\n\n"
            "Provide analysis recommendations in JSON format with the following structure:\n"
            "{{\n"
            "  \"analysis_types\": [list of appropriate analysis types],\n"
            "  \"filters\": [column names good for filtering],\n"
            "  \"value_columns\": [numeric columns good for analysis],\n"
            "  \"time_periods\": [\"Yearly\", \"Quarterly\", \"Monthly\"],\n"
            "  \"date_columns\": [column names that contain dates],\n"
            "  \"recommendations\": \"explanation points\"\n"
            "}}\n\n"
            "Make sure to add Cohort, Retention, Segmentation Analysis in the analysis list as well"
            "Ensure the response is valid JSON and uses the exact keys specified above."
            "Put each recommendation on a new line and elaborate the recommendations."
        )
    )

    #Format the prompt
    prompt = prompt_template.format(
        dataset_preview=df.head().to_string(),
        columns=df.columns.tolist()
    )

    try:
        #Initialize the ChatOpenAI model
        chat = ChatOpenAI(model="gpt-4", temperature=0)

        #Use SystemMessage and HumanMessage for the input
        messages = [
            SystemMessage(content="You are an expert data analyst. Always provide responses in valid JSON format."),
            HumanMessage(content=prompt)
        ]

        #Get the response
        response = chat(messages)

        #Try to parse the JSON response
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            #If JSON parsing fails, try to clean the response
            cleaned_response = response.content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            return json.loads(cleaned_response)

    except Exception as e:
        #Provide a fallback response if the API call fails
        date_columns = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' \
                       or (isinstance(df[col].dtype, object) and 'date' in col.lower())]
        return {
            "analysis_types": ["Basic Analysis"],
            "filters": df.select_dtypes(include=['object']).columns.tolist(),
            "value_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "time_periods": ["Monthly", "Quarterly", "Yearly"],
            "date_columns": date_columns,
            "recommendations": f"Unable to generate detailed recommendations. Using basic analysis options. Error: {str(e)}"
        }

def process_time_period(df, date_column, time_period):
    """Process the dataframe according to the selected time period."""
    try:
        #Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])

        #Create period labels based on selected time period
        if time_period == "Yearly":
            df['period'] = df[date_column].dt.year.astype(str)
        elif time_period == "Quarterly":
            df['period'] = df[date_column].dt.to_period('Q').astype(str)
        elif time_period == "Monthly":
            df['period'] = df[date_column].dt.strftime('%Y-%m')

        return df
    except Exception as e:
        st.error(f"Error processing time period: {e}")
        return df

def to_excel_download_link(sum_df, avg_df, count_df, filename="analysis_result.xlsx"):
    """Generates a link to download the dataframes as an Excel file with separate sheets."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sum_df.to_excel(writer, index=True, sheet_name="Sum")
        avg_df.to_excel(writer, index=True, sheet_name="Average")
        count_df.to_excel(writer, index=True, sheet_name="Count")
    excel_data = output.getvalue()
    return excel_data

def generate_recommendations_from_file(file_content):
    """Send analysis result file content to GPT for generating recommendations."""
    try:
        chat = ChatOpenAI(model="gpt-4", temperature=0)
        prompt = (
            "Analyze the provided analysis results and provide recommendations. Focus on key insights and actionable business strategies. "
            "Look for trends across different time periods.Summarize key trends or patterns (e.g., growth, decline, stability) in numerical terms."
            "Highlight noteworthy anomalies or outliers (e.g., sudden surges, significant declines)."
            "Explain the business implications of these changes (e.g., market diversification, customer retention challenges)."
            "Make sure insights are actionable, concise, and logically derived from the data."

        )
        messages = [
            SystemMessage(content="You are a data analysis expert."),
            HumanMessage(content=prompt + f"\n\n{file_content}")
        ]
        response = chat(messages)
        return response.content
    except Exception as e:
        return f"Error generating recommendations: {e}"
    

def calculate_growth(df):
    """Calculate year-over-year percentage growth."""
    return df.pct_change(axis=1) * 100

def calculate_concentration(df):
    """Calculate concentration (percentage of total) for each cell."""
    return df.div(df.sum(axis=0), axis=1) * 100

def to_excel_download_link(analysis_dfs, filename="analysis_result.xlsx"):
    """Generates a link to download all analysis dataframes as an Excel file with separate sheets."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in analysis_dfs.items():
            df.to_excel(writer, index=True, sheet_name=sheet_name)
    excel_data = output.getvalue()
    return excel_data

def main():
    st.title("DDMind")

#Initialize session state for tracking analysis state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    #Step 1: Upload Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("### Extracted Data:")
            st.write(df)

            #Step 2: Clean the data
            df_cleaned = df.drop_duplicates().fillna(method='ffill').fillna(method='bfill')

            #Step 3: Get analysis recommendations from ChatGPT
            with st.spinner("Analyzing data..."):
                json_response = analyze_data_with_langchain(df_cleaned)

            #Display recommendations as a paragraph
            st.write("### DDMind Recommendations:")
            st.write(json_response["recommendations"])

            #Extract data from JSON for dropdowns
            analysis_types = json_response.get("analysis_types", ["Basic Analysis"])
            filters = json_response.get("filters", df_cleaned.select_dtypes(include=['object']).columns.tolist())
            value_columns = json_response.get("value_columns", df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist())
            time_periods = json_response.get("time_periods", ["Monthly", "Quarterly", "Yearly"])
            date_columns = json_response.get("date_columns", [col for col in df_cleaned.columns if 'date' in col.lower()])


            #Create columns for side-by-side dropdowns
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Select Variables")
                selected_analysis = st.selectbox("Analysis Type", analysis_types)
                selected_filter = st.selectbox("Topic", filters)

                #Dynamically create subfilter dropdown
                if selected_filter:
                    #Get unique values for the selected filter
                    subfilter_options = ['All'] + list(df_cleaned[selected_filter].unique())
                    selected_subfilter = st.selectbox(f"Specific {selected_filter}", subfilter_options)

            with col2:
                st.write("###  ")  # Empty header for alignment
                selected_value = st.selectbox("Value", value_columns)
                selected_time = st.selectbox("Time Period", time_periods)
                selected_date = st.selectbox("Date", date_columns if date_columns else ["None available"])

            if st.button("Run Analysis"):
                #Prepare dataframe for analysis
                if selected_subfilter == 'All':
                    df_analysis = df_cleaned
                else:
                    df_analysis = df_cleaned[df_cleaned[selected_filter] == selected_subfilter]

                if selected_filter in df_analysis.columns and selected_value in df_analysis.columns:
                    try:
                        st.write("### Analysis Result:")

                        #Process time periods if date column is available
                        if selected_date != "None available":
                            df_analysis = process_time_period(df_analysis.copy(), selected_date, selected_time)

                            #Perform analysis with both period and filter
                            analysis_result = df_analysis.groupby(['period', selected_filter])[selected_value].agg(['sum', 'mean', 'count'])
                            result_df = analysis_result.reset_index()
                            result_df.columns = ['Time Period', selected_filter, 'Sum', 'Average', 'Count']
                            result_df = result_df.sort_values('Time Period')

                            #Create base DataFrames
                            value_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Sum')
                            avg_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Average')
                            count_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Count')

                            #Calculate additional metrics
                            total_sum = value_df.sum()
                            pct_df = value_df.div(total_sum) * 100
                            growth_df = calculate_growth(value_df)
                            concentration_df = calculate_concentration(value_df)

                        else:
                            #Fallback to regular analysis without time period
                            analysis_result = df_analysis.groupby(selected_filter)[selected_value].agg(['sum', 'mean', 'count'])
                            result_df = analysis_result.reset_index()
                            
                            #Create base DataFrames
                            value_df = result_df.set_index(selected_filter)[['Sum']]
                            avg_df = result_df.set_index(selected_filter)[['Average']]
                            count_df = result_df.set_index(selected_filter)[['Count']]
                            pct_df = value_df.div(value_df.sum()) * 100
                            growth_df = pd.DataFrame(index=value_df.index, columns=['Growth'])
                            concentration_df = calculate_concentration(value_df)

                        #Create tabs for different views
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                            "Value", "Percentage", "Average", 
                            "Percentage Growth", "Count", "Concentration"
                        ])

                        with tab1:
                            st.write(f"Value Analysis of {selected_value}")
                            st.write(value_df)

                        with tab2:
                            st.write(f"Percentage Distribution of {selected_value}")
                            st.write(pct_df.round(2))

                        with tab3:
                            st.write(f"Average Analysis of {selected_value}")
                            st.write(avg_df.round(2))

                        with tab4:
                            st.write(f"Year-over-Year Growth of {selected_value} (%)")
                            st.write(growth_df.round(2))

                        with tab5:
                            st.write(f"Count Analysis of {selected_value}")
                            st.write(count_df)

                        with tab6:
                            st.write(f"Concentration Analysis of {selected_value} (%)")
                            st.write(concentration_df.round(2))

                        #Store all analysis DataFrames in a dictionary
                        analysis_dfs = {
                            "Value": value_df,
                            "Percentage": pct_df,
                            "Average": avg_df,
                            "Growth": growth_df,
                            "Count": count_df,
                            "Concentration": concentration_df
                        }

                        #Create Excel data with all sheets
                        excel_data = to_excel_download_link(analysis_dfs)

                        #Store results in session state
                        st.session_state.analysis_dfs = analysis_dfs
                        st.session_state.analysis_complete = True
                        st.session_state.excel_data = excel_data

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                else:
                    st.warning("Selected columns are not present in the dataset. Please choose different options.")

            #Recommendations button outside Run Analysis block
            if st.session_state.analysis_complete:
                if st.button("Insights on Results"):
                    try:
                        #Combine all analysis results for recommendations
                        full_summary = "\n\n".join([
                            f"{name} Analysis:\n{df.to_string()}"
                            for name, df in st.session_state.analysis_dfs.items()
                        ])

                        with st.spinner("Generating Insights..."):
                            recommendations = generate_recommendations_from_file(full_summary)
                            st.write("### DDMind Insights:")
                            st.write(recommendations)

                    except Exception as e:
                        st.error(f"Recommendation generation error: {e}")

                #Persistent Excel download button
                st.download_button(
                    label="Download Analysis Results",
                    data=st.session_state.excel_data,
                    file_name=f"{selected_analysis}_{selected_value}_by_{selected_filter}_{selected_subfilter}_{selected_time}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="persistent_download_button"
                )
        except Exception as e:
                    st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()

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

# Set OpenAI API key
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
            "Ensure the response is valid JSON and uses the exact keys specified above."
            "Put each recommendation on a new line."
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
            # If JSON parsing fails, try to clean the response
            cleaned_response = response.content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            return json.loads(cleaned_response)

    except Exception as e:
        # Provide a fallback response if the API call fails
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
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])

        # Create period labels based on selected time period
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

def main():
    st.title("DDMind")

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

            # Display recommendations as a paragraph
            st.write("### DDMind Recommendations:")
            st.write(json_response["recommendations"])

            # Extract data from JSON for dropdowns
            analysis_types = json_response.get("analysis_types", ["Basic Analysis"])
            filters = json_response.get("filters", df_cleaned.select_dtypes(include=['object']).columns.tolist())
            value_columns = json_response.get("value_columns", df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist())
            time_periods = json_response.get("time_periods", ["Monthly", "Quarterly", "Yearly"])
            date_columns = json_response.get("date_columns", [col for col in df_cleaned.columns if 'date' in col.lower()])

            if not filters:
                st.warning("No suitable filter columns found in the dataset.")
                return
            if not value_columns:
                st.warning("No suitable value columns found in the dataset.")
                return
            if not date_columns:
                st.warning("No date columns found in the dataset. Time period analysis may not be accurate.")

            #Create columns for side-by-side dropdowns
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Select Variables")
                selected_analysis = st.selectbox("Analysis Type", analysis_types)
                selected_filter = st.selectbox("Topic", filters)

            with col2:
                st.write("###  ")  #Empty header for alignment
                selected_value = st.selectbox("Value", value_columns)
                selected_time = st.selectbox("Time Period", time_periods)
                selected_date = st.selectbox("Date", date_columns if date_columns else ["None available"])

            #Add Run Analysis button
            if st.button("Run Analysis"):
                if selected_filter in df_cleaned.columns and selected_value in df_cleaned.columns:
                    try:
                        st.write("### Analysis Result:")
                        st.write(f"Showing {selected_analysis} of {selected_value} filtered by {selected_filter} ({selected_time})")

                        #Process time periods if date column is available
                        if selected_date != "None available":
                            df_analysis = process_time_period(df_cleaned.copy(), selected_date, selected_time)

                            #Perform analysis with both period and filter
                            analysis_result = df_analysis.groupby(['period', selected_filter])[selected_value].agg(['sum', 'mean', 'count'])

                            #Reset index to make the period and filter columns regular columns
                            result_df = analysis_result.reset_index()

                            #Rename columns for clarity
                            result_df.columns = ['Time Period', selected_filter, 'Sum', 'Average', 'Count']

                            #Sort by Time Period
                            result_df = result_df.sort_values('Time Period')

                            #Create separate DataFrames for Sum, Average, and Count
                            sum_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Sum')
                            avg_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Average')
                            count_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Count')

                            #Pivot the table for 1 table
                            #pivoted_df = result_df.pivot(index= selected_filter, columns='Time Period', values=['Sum', 'Average', 'Count'])

                        else:
                            # Fallback to regular analysis without time period
                            analysis_result = df_cleaned.groupby(selected_filter)[selected_value].agg(['sum', 'mean', 'count'])
                            result_df = analysis_result.reset_index()
                            result_df.columns = [selected_filter, 'Sum', 'Average', 'Count']

                            # Create separate DataFrames for Sum, Average, and Count
                            sum_df = result_df[['Sum']]
                            avg_df = result_df[['Average']]
                            count_df = result_df[['Count']]

                            #Pivot the table for 1 table
                            #pivoted_df = result_df.pivot(index=selected_filter, columns=None, values=['Sum', 'Average', 'Count'])

                        #Display results
                        st.write("### Pivoted Results (Sum):")
                        st.write(sum_df)
                        st.write("### Pivoted Results (Average):")
                        st.write(avg_df)
                        st.write("### Pivoted Results (Count):")
                        st.write(count_df)

                        # Display results
                        #st.write(pivoted_df)

                        #Create download button for Excel
                        excel_data = to_excel_download_link(sum_df, avg_df, count_df)

                        st.download_button(
                            label="Download Analysis Results",
                            data=excel_data,
                            file_name=f"{selected_analysis}_{selected_value}_by_{selected_filter}_{selected_time}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                else:
                    st.warning("Selected columns are not present in the dataset. Please choose different options.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()

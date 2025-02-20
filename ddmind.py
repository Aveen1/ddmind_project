import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st
import pandas as pd
import os
import openai
import json
from io import BytesIO, StringIO
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tiktoken

#Import custom modules
from data_processing import (create_date_column, process_time_period, to_excel_download_link )
from data_analysis import ( calculate_growth, calculate_concentration, create_top_n_concentration, create_top_n_table)
from chart_generation import (create_line_chart, create_bar_chart, create_area_chart,create_heatmap_chart )
from ai_insights import ( analyze_data_with_langchain, generate_tab_insights, generate_recommendations_from_file )
from tabs import (create_analysis_tabs,create_sidebar,load_custom_css,create_snowball_tab, create_bridge_tab, add_total_row)

#Set OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_dataset_question(df, question):
    """Function to analyze user questions about the dataset using GPT-4"""
    try:
        #Convert dataframe info to string using StringIO
        buffer = StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        
        df_stats = f"\nDataset Statistics:\n{df.describe().to_string()}\n"
        df_head = f"First few rows:\n{df.head().to_string()}\n"
        
        prompt = f"""Given the following dataset information:

{df_info}

{df_stats}

{df_head}

Question: {question}

Please analyze this data and provide a clear, concise answer. If you need to perform any calculations, explain them briefly."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Provide clear, accurate answers based on the dataset provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing question: {str(e)}"




def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

def to_excel_download_link(analysis_dfs, filename="analysis_result.xlsx"):
    """Generates a link to download all analysis dataframes as an Excel file with separate sheets."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in analysis_dfs.items():
            df.to_excel(writer, index=True, sheet_name=sheet_name)
    excel_data = output.getvalue()
    return excel_data


def main():
    load_custom_css()
    #create_sidebar()
    #st.title("DDMind")

    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    
    if 'json_response' not in st.session_state:
        st.session_state.json_response = None

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None and (st.session_state.last_uploaded_file != uploaded_file):
        try:
            st.session_state.last_uploaded_file = uploaded_file

            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            df = create_date_column(df)
            
            st.write("### Extracted Data:")
            st.write(df)
            st.session_state.df_cleaned = df.drop_duplicates().fillna(method='ffill').fillna(method='bfill')

            with st.spinner("Analyzing data..."):
                st.session_state.json_response = analyze_data_with_langchain(st.session_state.df_cleaned)

            st.write("### DDMind Recommendations:")
            st.write(st.session_state.json_response["recommendations"])

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    if st.session_state.df_cleaned is not None and st.session_state.json_response is not None:
        df_cleaned = st.session_state.df_cleaned
        json_response = st.session_state.json_response
        analysis_types = json_response.get("analysis_types", ["Basic Analysis"])
        filters = json_response.get("filters", df_cleaned.select_dtypes(include=['object']).columns.tolist())
        value_columns = json_response.get("value_columns", df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist())
        time_periods = json_response.get("time_periods", ["Monthly", "Quarterly", "Yearly"])
        date_columns = json_response.get("date_columns", [col for col in df_cleaned.columns if 'date' in col.lower()])

        #Display the analysis form
        with st.form(key='analysis_form'):
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Select Variables")
                selected_analysis = st.selectbox("Analysis Type", ['Select...'] + analysis_types, key='analysis_type')
                selected_filter = st.selectbox("Topic", ['Select...'] + filters, key='filter')
                
                if selected_filter and selected_filter != 'Select...':
                    unique_values = sorted(df_cleaned[selected_filter].dropna().unique())
                    
                    selected_subfilters = st.multiselect(
                        f"Select {selected_filter}(s)",
                        options=unique_values,
                        default=None, 
                        key='subfilter'
                    )
                    
                    if st.checkbox("Select All", key="select_all_subfilters"):
                        selected_subfilters = list(unique_values)
                    
                else:
                    selected_subfilters = []
                    st.selectbox(
                        "Select Specific Filter",
                        ['Select Topic First'],
                        disabled=True,
                        key='subfilter_disabled'
                    )

            with col2:
                st.write("###  ")
                selected_value = st.selectbox("Value", ['Select...'] + value_columns, key='value')
                selected_time = st.selectbox("Time Period", ['Select...'] + time_periods, key='time')

                if date_columns:
                    date_options = ['Select...'] + date_columns
                else:
                    date_options = ['Select...', 'None available']
                selected_date = st.selectbox("Date", date_options, key='date')

            submit_button = st.form_submit_button("Run Analysis")

            all_selected = (
                selected_analysis != 'Select...' and
                selected_filter != 'Select...' and
                len(selected_subfilters) > 0 and  # Now checking if any subfilters are selected
                selected_value != 'Select...' and
                selected_time != 'Select...' and
                selected_date != 'Select...'
            )

            if submit_button and not all_selected:
                st.warning("Please select all options before running the analysis.")
                return

        if submit_button:
            if not selected_subfilters:  
                df_analysis = df_cleaned
            else:
                
                df_analysis = df_cleaned[df_cleaned[selected_filter].isin(selected_subfilters)]

            if selected_filter in df_analysis.columns and selected_value in df_analysis.columns:
                try:
                    st.write("### Analysis Result:")

                    if selected_date != "None available":
                        df_analysis = process_time_period(df_analysis.copy(), selected_date, selected_time)
                        
                        analysis_result = df_analysis.groupby(['period', selected_filter])[selected_value].agg(['sum', 'mean', 'count'])
                        result_df = analysis_result.reset_index()
                        result_df.columns = ['Time Period', selected_filter, 'Sum', 'Average', 'Count']
                        result_df = result_df.sort_values('Time Period')
                        
                        value_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Sum')
                        avg_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Average')
                        count_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Count')
                        total_sum = value_df.sum()
                        pct_df = value_df.div(total_sum) * 100
                        growth_df = calculate_growth(value_df)
                        concentration_df = calculate_concentration(value_df)
                        total_sum_df = pd.DataFrame(value_df.sum()).T
                        total_sum_df.index = ['Total']
                        snowball_df = growth_df.copy()
                        snowball_df = snowball_df.cumsum()
                        

                    else:
                        analysis_result = df_analysis.groupby(selected_filter)[selected_value].agg(['sum', 'mean', 'count'])
                        result_df = analysis_result.reset_index()
                    
                        value_df = result_df.set_index(selected_filter)[['sum']]
                        avg_df = result_df.set_index(selected_filter)[['mean']]
                        count_df = result_df.set_index(selected_filter)[['count']]
                        pct_df = value_df.div(value_df.sum()) * 100
                        growth_df = pd.DataFrame(index=value_df.index, columns=['Growth'])
                        concentration_df = calculate_concentration(value_df)
                        total_sum_df = pd.DataFrame(value_df.sum()).T
                        total_sum_df.index = ['Total']
                        snowball_df = growth_df.copy()
                        snowball_df = snowball_df.cumsum()

                    #Create all analysis tabs using the new function
                    create_analysis_tabs(
                        value_df, 
                        total_sum_df, 
                        pct_df, 
                        avg_df, 
                        growth_df, 
                        count_df, 
                        concentration_df,
                        selected_value,
                        selected_filter,
                        selected_analysis  
                    )

                    
                    if selected_analysis == "Retention Analysis":
                        analysis_dfs = {
                            "Snowball Analysis": add_total_row(value_df),
                            "Dollar Retention": add_total_row(value_df),  
                            "Metrics": add_total_row(value_df)  
                        }

                    elif selected_analysis == "Segmentation Analysis":
                        analysis_dfs = {
                            "Value": add_total_row(value_df),
                            "Average": add_total_row(avg_df),
                            "Percentage": add_total_row(pct_df),
                            "Growth": add_total_row(growth_df),
                            "Bridge Analysis": add_total_row(value_df),
                            "Count": add_total_row(count_df),
                            "Concentration": add_total_row(concentration_df)
                        }       

                    elif selected_analysis == "Cohort Analysis":
                        analysis_dfs = {
                            "Values": add_total_row(value_df),
                            "Count": add_total_row(value_df),
                            "Average": add_total_row(avg_df),
                            "$ Lost": add_total_row(value_df),
                            "$ Changes": add_total_row(value_df), 
                            "Lost Products": add_total_row(value_df),
                            "Lost Product Retention": add_total_row(value_df),
                            "Concentration Analysis": add_total_row(concentration_df)
                        }


                    else:
                        analysis_dfs = {
                            "Value": add_total_row(value_df),
                            "Total Sum": total_sum_df,
                            "Percentage": add_total_row(pct_df),
                            "Average": add_total_row(avg_df),
                            "Growth": add_total_row(growth_df),
                            "Count": add_total_row(count_df),
                            "Concentration": add_total_row(concentration_df),
                            "Bridge Analysis": add_total_row(value_df)
                        }

                    excel_data = to_excel_download_link(analysis_dfs)

                    st.session_state.analysis_dfs = analysis_dfs
                    st.session_state.analysis_complete = True
                    st.session_state.excel_data = excel_data

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

        if st.session_state.analysis_complete:
            if st.button("Generate Insights"):
                try:
                    full_summary = "\n\n".join([
                        f"{name} Analysis:\n{df.to_string()}"
                        for name, df in st.session_state.analysis_dfs.items()
                    ])

                    with st.spinner("Generating Insights..."):
                        recommendations = generate_recommendations_from_file(full_summary)
                        formatted_recommendations = recommendations.replace('$', '\$')

                        st.write("### DDMind Insights:")
                        st.write(formatted_recommendations)

                except Exception as e:
                    st.error(f"Recommendation generation error: {e}")

            st.download_button(
                label="Download Analysis Results",
                data=st.session_state.excel_data,
                file_name=f"{selected_analysis}_{selected_value}_by_{selected_filter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            #Add chat box for dataset questions
            st.write("### Questions on the Dataset?")
            user_question = st.text_input("Ask a question about your data:", key="dataset_question")
            
            if user_question:
                with st.spinner("Analyzing your question..."):
                    answer = analyze_dataset_question(st.session_state.df_cleaned, user_question)
                    st.write("#### Answer:")
                    st.write(answer)
                    
            





if __name__ == "__main__":
    main()
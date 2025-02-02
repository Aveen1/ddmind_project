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
import plotly.express as px
import plotly.graph_objects as go


#Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#Date column issue
def create_date_column(df):
    """
    Create a date column if one doesn't exist in the DataFrame.
    Handles various date-related columns like Year, Month, Quarter, etc.
    """
    #Check if any date column already exists
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    if not date_columns:
        #Check for year column
        year_cols = [col for col in df.columns if col.lower() in ['year', 'fiscal_year', 'fy']]
        
        if year_cols:
            year_col = year_cols[0]
            #Convert year to datetime
            try:
                #Handle string years
                df['year_temp'] = pd.to_numeric(df[year_col], errors='coerce')
                #Check if years are reasonable (between 1900 and current year + 10)
                current_year = datetime.now().year
                df.loc[df['year_temp'] < 1900, 'year_temp'] = np.nan
                df.loc[df['year_temp'] > current_year + 10, 'year_temp'] = np.nan
                
                #Create date column (defaulting to January 1st of each year)
                df['Date'] = pd.to_datetime(df['year_temp'].astype(int).astype(str) + '-01-01')
                df = df.drop('year_temp', axis=1)
                
                return df
            except Exception as e:
                st.error(f"Error creating date from year: {e}")
                return df
        else:
            #If no year column exists, create a date column based on the current date
            st.warning("No date or year column found. Using current date for analysis.")
            df['Date'] = datetime.now().date()
            
    return df

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
#time period 
def process_time_period(df, date_column, time_period):
    """Process the dataframe according to the selected time period."""
    try:
        #Create a copy to avoid modifying the original
        df = df.copy()
        
        #If the date_column is 'Date' (our created column), ensure it's handled properly
        if date_column == 'Date' and 'Year' in df.columns:
            #Use the existing Year column for period creation
            if time_period == "Yearly":
                df['period'] = df['Year'].astype(str)
            elif time_period == "Quarterly":
                df['period'] = df['Year'].astype(str) + '-Q' + '1'  #Default to Q1 if only year is available
            elif time_period == "Monthly":
                df['period'] = df['Year'].astype(str) + '-01'  #Default to January if only year is available
        else:
            #Standard date processing
            df[date_column] = pd.to_datetime(df[date_column])
            
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
        chat = ChatOpenAI(model="gpt-4", temperature=0.5)
        prompt = (
            """Analyze the provided table data to generate insights focusing on trends, patterns, and business implications. 
            Use the following examples as references for structuring your analysis, ensuring your output includes relevant metrics, growth percentages, and logical conclusions:

            Examples:
            "1. CloudPro Services grew from 15.2 million in 2020 to 28.4 million in 2022 before stabilizing at 26.7 million in 2023, demonstrating solid performance despite minor adjustments in demand."

            "2. Two new offerings, StreamFlow and TaskEase, launched in recent years, contributed 1.8 million in 2023, representing initial traction in untapped market segments."

            "3. FlexAssist Solutions showed inconsistent performance, declining from 7.5 million in 2020 to 5.1 million in 2021, but surged by 140 percent to 12.3 million in 2023, highlighting cyclical recovery potential."

            "4. Top-tier services like PrimeSuite accounted for 38 percent of revenue in 2022 but dropped to 32 percent in 2023, indicating diversification and reduced dependency on flagship products."

            "5. Customer loyalty remained a challenge, with a retention rate of 48 percent in 2023, although dollar retention surged to 115 percent, reflecting successful cross-selling strategies."

            "6. On-site Dynamics emerged as the fastest-growing segment with a 275 percent increase from 3.6 million in 2021 to 13.5 million in 2023, showcasing its ability to capitalize on market demand."

            "7. The top 15 percent of clients contributed 88 percent of revenue in 2023, a reduction from 93 percent in 2021, signaling progress in client base diversification."

            "8. The 2019 product cohort saw revenue grow from 18.1 million in 2020 to 25.2 million in 2023, reflecting the ongoing strength of established product lines."

            "9. Customer B experienced 3x growth, with revenue climbing from 12.2 million in 2021 to 37.3 million in 2023, driven by a 400 percent rise in EnterpriseLink Solutions."

            "10. The product portfolio remained strong, retaining all 11 offerings from 2020 to 2023, with the introduction of one new product annually, ensuring measured yet steady innovation."

            - Summarize key trends or patterns (e.g., growth, decline, stability) in numerical terms.
            - Highlight noteworthy anomalies or outliers (e.g., sudden surges, significant declines).
            - Explain the business implications of these changes (e.g., market diversification, customer retention challenges).
            - Make sure insights are actionable, concise, and logically derived from the data."""
        )
        messages = [
            SystemMessage(content="You are a data analysis expert."),
            HumanMessage(content=prompt + f"\n\n{file_content}")
        ]
        response = chat(messages)
        return response.content
    except Exception as e:
        return f"Error generating recommendations: {e}"
#tab insights
def generate_tab_insights(df, analysis_type, selected_value, selected_filter):
    """Generate GPT insights for specific analysis tab."""
    try:
        chat = ChatOpenAI(model="gpt-4", temperature=0.5)
        
        # Create specific prompts based on analysis type
        prompts = {
            "value": f"""Analyze the {selected_value} values across different {selected_filter} categories:
                     1. Identify the top and bottom performers
                     2. Point out any significant trends or patterns
                     3. Highlight any notable changes between time periods
                     Data: {df.to_string()}""",
                     
            "total_sum": f"""Analyze the total sum trends for {selected_value}:
                         1. Describe the overall trend (growing, declining, stable)
                         2. Calculate and mention the total growth rate
                         3. Identify peak and lowest periods
                         Data: {df.to_string()}""",
                         
            "percentage": f"""Analyze the percentage distribution of {selected_value}:
                         1. Identify dominant categories and their share
                         2. Note any significant shifts in distribution
                         3. Point out categories with growing or declining share
                         Data: {df.to_string()}""",
                         
            "average": f"""Analyze the average {selected_value} trends:
                       1. Compare averages across categories
                       2. Identify any outliers or unusual patterns
                       3. Comment on the stability of averages over time
                       Data: {df.to_string()}""",
                       
            "growth": f"""Analyze the growth patterns in {selected_value}:
                      1. Identify highest and lowest growth rates
                      2. Point out any consistent growth patterns
                      3. Flag any concerning decline trends
                      Data: {df.to_string()}""",
                      
            "count": f"""Analyze the count distribution of {selected_value}:
                     1. Identify most and least frequent categories
                     2. Comment on count distribution changes
                     3. Point out any unusual patterns
                     Data: {df.to_string()}""",
                     
            "concentration": f"""Analyze the concentration of {selected_value}:
                            1. Identify highly concentrated areas
                            2. Comment on concentration changes over time
                            3. Assess concentration risk if applicable
                            Data: {df.to_string()}"""
        }
        
        messages = [
            SystemMessage(content="""You are a data analysis expert. Provide clear, concise insights in bullet points.
                         Focus on business implications and actionable findings.
                         Keep the analysis to 3-5 key points."""),
            HumanMessage(content=prompts[analysis_type.lower()])
        ]
        
        response = chat(messages)
        return response.content
    except Exception as e:
        return f"Error generating insights: {e}"
    

#formulas for tabs
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

#charts
def create_line_chart(df, title):
    """Create a line chart using Plotly."""
    fig = px.line(df.transpose(), title=title)
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Value",
        legend_title="Categories",
        height=500
    )
    return fig

def create_bar_chart(df, title):
    """Create a bar chart using Plotly."""
    fig = px.bar(df.transpose(), title=title, barmode='group')
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Value",
        legend_title="Categories",
        height=500
    )
    return fig

def create_area_chart(df, title):
    """Create a stacked area chart using Plotly."""
    fig = px.area(df.transpose(), title=title)
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Value",
        legend_title="Categories",
        height=500
    )
    return fig




def main():
    #Custom CSS for sidebar styling
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: rgba(10, 8, 41, 255);
        }    
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    #Create sidebar
    with st.sidebar:
        st.title("DDMind.ai")
        
        # Add app description
        st.markdown("### About")
        st.info("""
        DDMind is a data analysis tool that helps you:
        - Upload and analyze Excel/CSV files
        - Get AI-powered Recommendations
        - Generate Interactive Visualizations
        - Export Detailed Analysis Reports
        """)
        
        #Add file format info
        st.markdown("### Supported File Formats")
        st.write("- Excel (.xlsx, .xls)")
        st.write("- CSV (.csv)")
        
        #Add analysis options in sidebar
        st.markdown("### Analysis Settings")
        show_raw_data = st.checkbox("Show Raw Data", value=False)
        enable_ai_insights = st.checkbox("Enable AI Insights", value=True)
        
        
        # Add footer
        st.markdown("---")
        st.markdown("Made with ❤️ by DDMind")


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

            #Create date column if needed
            df = create_date_column(df)

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

                            #Create base DataFrames,Calculate additional metrics
                            value_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Sum')
                            avg_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Average')
                            count_df = result_df.pivot(index=selected_filter, columns='Time Period', values='Count')
                            total_sum = value_df.sum()
                            pct_df = value_df.div(total_sum) * 100
                            growth_df = calculate_growth(value_df)
                            concentration_df = calculate_concentration(value_df)
                            total_sum_df = pd.DataFrame(value_df.sum()).T
                            total_sum_df.index = ['Total']

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
                        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                            "Value","Total Sum", "Percentage", "Average", 
                            "Percentage Growth", "Count", "Concentration"
                        ])

                        with tab1:
                            st.write(f"Value Analysis of {selected_value}")
                            st.write(value_df)
                            st.plotly_chart(create_line_chart(value_df, f"Value Trend of {selected_value}"))
                            st.plotly_chart(create_bar_chart(value_df, f"Value Distribution of {selected_value}"))
                            with st.expander("📊 Value Analysis Insights", expanded=True):
                                with st.spinner("Generating value insights..."):
                                    value_insights = generate_tab_insights(value_df, "value", selected_value, selected_filter)
                                    st.write(value_insights)
                        
                        with tab2:
                            st.write(f"Total Sum Analysis of {selected_value}")
                            st.write(total_sum_df)
                            st.plotly_chart(create_line_chart(total_sum_df, f"Total Sum Trend of {selected_value}"))
                            st.plotly_chart(create_bar_chart(total_sum_df, f"Total Sum Distribution of {selected_value}"))
                            with st.expander("📊 Total Sum Analysis Insights", expanded=True):
                                with st.spinner("Generating total sum insights..."):
                                    total_sum_insights = generate_tab_insights(total_sum_df, "total_sum", selected_value, selected_filter)
                                    st.write(total_sum_insights)

                        with tab3:
                            st.write(f"Percentage Distribution of {selected_value}")
                            st.write(pct_df.round(2))
                            st.plotly_chart(create_area_chart(pct_df, f"Percentage Distribution of {selected_value} Over Time"))
                            st.plotly_chart(create_bar_chart(pct_df, f"Percentage Distribution by Category"))
                            with st.expander("📊 Percentage Analysis Insights", expanded=True):
                                with st.spinner("Generating percentage insights..."):
                                    percentage_insights = generate_tab_insights(pct_df, "percentage", selected_value, selected_filter)
                                    st.write(percentage_insights)


                        with tab4:
                            st.write(f"Average Analysis of {selected_value}")
                            st.write(avg_df.round(2))
                            st.plotly_chart(create_line_chart(avg_df, f"Average Trend of {selected_value}"))
                            st.plotly_chart(create_bar_chart(avg_df, f"Average Distribution by Category"))
                            with st.expander("📊 Average Analysis Insights", expanded=True):
                                with st.spinner("Generating average insights..."):
                                    average_insights = generate_tab_insights(avg_df, "average", selected_value, selected_filter)
                                    st.write(average_insights)

                        with tab5:
                            st.write(f"Year-over-Year Growth of {selected_value} (%)")
                            st.write(growth_df.round(2))
                            st.plotly_chart(create_bar_chart(growth_df, f"Growth Rate by Category"))
                            #Add a heatmap for growth rates
                            fig_heatmap = px.imshow(growth_df,
                                                  title=f"Growth Rate Heatmap for {selected_value}",
                                                  labels=dict(x="Time Period", y="Category", color="Growth Rate (%)"))
                            st.plotly_chart(fig_heatmap)
                            with st.expander("📊 Growth Analysis Insights", expanded=True):
                                with st.spinner("Generating growth insights..."):
                                    growth_insights = generate_tab_insights(growth_df, "growth", selected_value, selected_filter)
                                    st.write(growth_insights)


                        with tab6:
                            st.write(f"Count Analysis of {selected_value}")
                            st.write(count_df)
                            st.plotly_chart(create_line_chart(count_df, f"Count Trend of {selected_value}"))
                            st.plotly_chart(create_bar_chart(count_df, f"Count Distribution by Category"))
                            with st.expander("📊 Count Analysis Insights", expanded=True):
                                with st.spinner("Generating count insights..."):
                                    count_insights = generate_tab_insights(count_df, "count", selected_value, selected_filter)
                                    st.write(count_insights)

                        with tab7:
                            st.write(f"Concentration Analysis of {selected_value} (%)")
                            st.write(concentration_df.round(2))
                            st.plotly_chart(create_area_chart(concentration_df, f"Concentration Over Time"))
                            #Add a treemap for concentration
                            if len(concentration_df.columns) > 0:
                                last_period = concentration_df.columns[-1]
                                #Create a DataFrame in the format needed for treemap
                                treemap_df = pd.DataFrame({
                                    'Category': concentration_df.index,
                                    'Value': concentration_df[last_period]
                                }).reset_index(drop=True)
                                
                                fig_treemap = px.treemap(
                                    treemap_df,
                                    path=['Category'],
                                    values='Value',
                                    title=f"Concentration Distribution for {last_period}"
                                )
                                fig_treemap.update_traces(textinfo="label+value+percent parent")
                                fig_treemap.update_layout(height=500)
                                st.plotly_chart(fig_treemap)
                            with st.expander("📊 Concentration Analysis Insights", expanded=True):
                                with st.spinner("Generating concentration insights..."):
                                    concentration_insights = generate_tab_insights(concentration_df, "concentration", selected_value, selected_filter)
                                    st.write(concentration_insights)



                        #Store all analysis DataFrames in a dictionary
                        analysis_dfs = {
                            "Value": value_df,
                            "Total Sum": total_sum_df,
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

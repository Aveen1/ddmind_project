import streamlit as st
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

#Import custom modules
from data_processing import (
    create_date_column, 
    process_time_period, 
    to_excel_download_link
    
)
from data_analysis import (
    calculate_growth, 
    calculate_concentration, 
    create_top_n_concentration, 
    create_top_n_table
)
from chart_generation import (
    create_line_chart, 
    create_bar_chart, 
    create_area_chart,
    create_heatmap_chart
)
from ai_insights import (
    analyze_data_with_langchain,
    generate_tab_insights,
    generate_recommendations_from_file
)

#Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def to_excel_download_link(analysis_dfs, filename="analysis_result.xlsx"):
    """Generates a link to download all analysis dataframes as an Excel file with separate sheets."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in analysis_dfs.items():
            df.to_excel(writer, index=True, sheet_name=sheet_name)
    excel_data = output.getvalue()
    return excel_data

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
        
        st.markdown("### About")
        st.info("""
        DDMind is a data analysis tool that helps you:
        - Upload and analyze Excel/CSV files
        - Get AI-powered Recommendations
        - Generate Interactive Visualizations
        - Export Detailed Analysis Reports
        """)
        
        st.markdown("### Supported File Formats")
        st.write("- Excel (.xlsx, .xls)")
        st.write("- CSV (.csv)")
        
        st.markdown("### Analysis Settings")
        show_raw_data = st.checkbox("Show Raw Data", value=False)
        enable_ai_insights = st.checkbox("Enable AI Insights", value=True)
        
        st.markdown("---")
        st.markdown("Made with ❤️ by DDMind")

    st.title("DDMind")

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

        with st.form(key='analysis_form'):
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Select Variables")
                selected_analysis = st.selectbox("Analysis Type", ['Select...'] + analysis_types, key='analysis_type')
                selected_filter = st.selectbox("Topic", ['Select...'] + filters, key='filter')

                if selected_filter and selected_filter != 'Select...':
                    unique_values = sorted(df_cleaned[selected_filter].dropna().unique())
                    specific_filter_options = ['Select...', 'All'] + list(unique_values)
                    selected_subfilter = st.selectbox(
                        f"Select {selected_filter}", 
                        options=specific_filter_options,
                        key='subfilter'
                    )
                else:
                    selected_subfilter = st.selectbox(
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

            #Ensure all selections are valid before running the analysis
            all_selected = (
                selected_analysis != 'Select...' and
                selected_filter != 'Select...' and
                selected_subfilter not in ['Select...', 'Select Topic First'] and
                selected_value != 'Select...' and
                selected_time != 'Select...' and
                selected_date != 'Select...'
            )



        if submit_button:
            if not all_selected:
                st.error("Please select all options before running the analysis.")
                return

            # Prepare dataframe for analysis
            if selected_subfilter == 'All':
                df_analysis = df_cleaned
            else:
                df_analysis = df_cleaned[df_cleaned[selected_filter] == selected_subfilter]

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

                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "Value", "Total Sum", "Percentage", "Average", 
                        "Percentage Growth", "Count", "Concentration Analysis"
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
                            subtab1, subtab2 = st.tabs(["General Concentration", "Top Customers"])
                            
                            with subtab1:
                                st.write(f"Concentration Analysis of {selected_value} (%)")
                                st.write(concentration_df.round(2))
                                st.plotly_chart(create_area_chart(concentration_df, f"Concentration Over Time"))
                                if len(concentration_df.columns) > 0:
                                    last_period = concentration_df.columns[-1]
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
                            
                            with subtab2:
                                st.write("### Top Customers Concentration Analysis")
                                
                                #Calculate top N concentration
                                top_n_results = create_top_n_concentration(value_df)
                                
                                #Display summary table
                                summary_df = create_top_n_table(top_n_results)
                                st.write("#### Summary of Top Customer Concentration")
                                st.write(summary_df)
                                
                                #Create visualization for top N concentration
                                fig_top_n = px.bar(
                                    summary_df,
                                    y='Concentration (%)',
                                    title="Concentration by Top N Customers",
                                    labels={'index': 'Customer Group', 'value': 'Concentration (%)'}
                                )
                                fig_top_n.update_layout(height=400)
                                st.plotly_chart(fig_top_n)
                                
                                #Display detailed customer lists
                                st.write("#### Detailed Customer Lists")
                                for group, data in top_n_results.items():
                                    with st.expander(f"{group} Details"):
                                        st.write(f"Total Value: {data['sum']:,.2f}")
                                        st.write(f"Concentration: {data['concentration']:.2f}%")
                                        st.write("Customers:")
                                        customer_df = pd.DataFrame({
                                            'Customer': data['customers'],
                                            'Value': value_df.loc[data['customers'], value_df.columns[-1]]
                                        }).sort_values('Value', ascending=False)
                                        customer_df['Cumulative %'] = (customer_df['Value'].cumsum() / customer_df['Value'].sum() * 100).round(2)
                                        st.write(customer_df)
                                
                                #Generate insights for top customer concentration
                                with st.expander("📊 Top Customer Concentration Insights", expanded=True):
                                    with st.spinner("Generating top customer insights..."):
                                        top_customer_prompt = f"""Analyze the customer concentration data:
                                            1. Comment on the distribution of value across customer segments
                                            2. Identify concentration risk levels
                                            3. Compare different customer tiers
                                            4. Suggest any risk mitigation strategies if needed
                                            Data: {summary_df.to_string()}"""
                                        
                                        messages = [
                                            SystemMessage(content="You are a data analysis expert. Provide clear, concise insights about customer concentration and associated risks."),
                                            HumanMessage(content=top_customer_prompt)
                                        ]
                                        
                                        chat = ChatOpenAI(model="gpt-4o", temperature=0)
                                        top_customer_insights = chat(messages)
                                        st.write(top_customer_insights.content)
                
                    analysis_dfs = {
                        "Value": value_df,
                        "Total Sum": total_sum_df,
                        "Percentage": pct_df,
                        "Average": avg_df,
                        "Growth": growth_df,
                        "Count": count_df,
                        "Concentration": concentration_df
                    }

                    excel_data = to_excel_download_link(analysis_dfs)

                    st.session_state.analysis_dfs = analysis_dfs
                    st.session_state.analysis_complete = True
                    st.session_state.excel_data = excel_data

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
            else:
                st.warning("Selected columns are not present in the dataset. Please choose different options.")

        if st.session_state.analysis_complete:
            if st.button("Generate Insights"):
                try:
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

            st.download_button(
                label="Download Analysis Results",
                data=st.session_state.excel_data,
                file_name=f"{selected_analysis}_{selected_value}_by_{selected_filter}_{selected_subfilter}_{selected_time}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

if __name__ == "__main__":
    main()

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import pandas as pd
import os
import openai
import json
from io import BytesIO
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

from data_analysis import (
    calculate_growth, 
    calculate_concentration, 
    create_top_n_concentration, 
    create_top_n_table
)

def create_sidebar():
    """Create and setup the sidebar"""
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
        st.checkbox("Show Raw Data", value=False)
        st.checkbox("Enable AI Insights", value=True)
        st.markdown("---")
        st.markdown("Made with ❤️ by DDMind")

def create_value_tab(value_df, selected_value, selected_filter):
    """Creates and populates the Value Analysis tab"""
    st.write(f"Value Analysis of {selected_value}")
    st.write(add_total_row(value_df))
    st.plotly_chart(create_line_chart(value_df, f"Value Trend of {selected_value}"))
    st.plotly_chart(create_bar_chart(value_df, f"Value Distribution of {selected_value}"))
    with st.expander("📊 Value Analysis Insights", expanded=True):
        with st.spinner("Generating value insights..."):
            value_insights = generate_tab_insights(value_df, "value", selected_value, selected_filter)
            st.write(value_insights)

def create_total_sum_tab(total_sum_df, selected_value, selected_filter):
    """Creates and populates the Total Sum Analysis tab"""
    st.write(f"Total Sum Analysis of {selected_value}")
    st.write(total_sum_df)
    st.plotly_chart(create_line_chart(total_sum_df, f"Total Sum Trend of {selected_value}"))
    st.plotly_chart(create_bar_chart(total_sum_df, f"Total Sum Distribution of {selected_value}"))
    with st.expander("📊 Total Sum Analysis Insights", expanded=True):
        with st.spinner("Generating total sum insights..."):
            total_sum_insights = generate_tab_insights(total_sum_df, "total_sum", selected_value, selected_filter)
            st.write(total_sum_insights)

def create_percentage_tab(pct_df, selected_value, selected_filter):
    """Creates and populates the Percentage Distribution tab"""
    st.write(f"Percentage Distribution of {selected_value}")
    pct_df = (add_total_row(pct_df.round(2)).applymap(lambda x: f"{x}%"))
    st.write(pct_df)
    st.plotly_chart(create_area_chart(pct_df, f"Percentage Distribution of {selected_value} Over Time"))
    st.plotly_chart(create_bar_chart(pct_df, f"Percentage Distribution by Category"))
    with st.expander("📊 Percentage Analysis Insights", expanded=True):
        with st.spinner("Generating percentage insights..."):
            percentage_insights = generate_tab_insights(pct_df, "percentage", selected_value, selected_filter)
            st.write(percentage_insights)

def create_average_tab(avg_df, selected_value, selected_filter):
    """Creates and populates the Average Analysis tab"""
    st.write(f"Average Analysis of {selected_value}")
    avg_df = (add_total_row(avg_df.round(2)))
    st.write(avg_df)
    st.plotly_chart(create_line_chart(avg_df, f"Average Trend of {selected_value}"))
    st.plotly_chart(create_bar_chart(avg_df, f"Average Distribution by Category"))
    with st.expander("📊 Average Analysis Insights", expanded=True):
        with st.spinner("Generating average insights..."):
            average_insights = generate_tab_insights(avg_df, "average", selected_value, selected_filter)
            st.write(average_insights)

def create_growth_tab(growth_df, selected_value, selected_filter):
    """Creates and populates the Growth Analysis tab"""
    st.write(f"Year-over-Year Growth of {selected_value} (%)")
    st.write(add_total_row(growth_df.round(2)).applymap(lambda x: f"{x}%"))
    st.plotly_chart(create_bar_chart(growth_df, f"Growth Rate by Category"))
    fig_heatmap = px.imshow(growth_df,
                           title=f"Growth Rate Heatmap for {selected_value}",
                           labels=dict(x="Time Period", y="Category", color="Growth Rate (%)"))
    st.plotly_chart(fig_heatmap)
    with st.expander("📊 Growth Analysis Insights", expanded=True):
        with st.spinner("Generating growth insights..."):
            growth_insights = generate_tab_insights(growth_df, "growth", selected_value, selected_filter)
            st.write(growth_insights)

def create_count_tab(count_df, selected_value, selected_filter):
    """Creates and populates the Count Analysis tab"""
    st.write(f"Count Analysis of {selected_value}")
    st.write(add_total_row(count_df))
    st.plotly_chart(create_line_chart(count_df, f"Count Trend of {selected_value}"))
    st.plotly_chart(create_bar_chart(count_df, f"Count Distribution by Category"))
    with st.expander("📊 Count Analysis Insights", expanded=True):
        with st.spinner("Generating count insights..."):
            count_insights = generate_tab_insights(count_df, "count", selected_value, selected_filter)
            st.write(count_insights)

def create_concentration_tab(concentration_df, value_df, selected_value, selected_filter):
    """Creates and populates the Concentration Analysis tab with its subtabs"""
    subtab1, subtab2 = st.tabs(["General Concentration", "Top Customers"])
    
    with subtab1:
        st.write(f"Concentration Analysis of {selected_value} (%)")
        st.write(add_total_row(concentration_df.round(2)).applymap(lambda x: f"{x}%"))
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
        create_top_customers_subtab(value_df)

def create_top_customers_subtab(value_df):
    """Creates and populates the Top Customers subtab within Concentration Analysis"""
    st.write("### Top Customers Concentration Analysis")
    
    top_n_results = create_top_n_concentration(value_df)
    summary_df = create_top_n_table(top_n_results)
    
    st.write("#### Summary of Top Customer Concentration")
    st.write(summary_df)
    
    fig_top_n = px.bar(
        summary_df,
        y='Concentration (%)',
        title="Concentration by Top N Customers",
        labels={'index': 'Customer Group', 'value': 'Concentration (%)'}
    )
    fig_top_n.update_layout(height=400)
    st.plotly_chart(fig_top_n)
    
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

def create_snowball_tab(snowball_df, selected_value, selected_filter):
    """Creates and populates the Snowball Analysis tab"""
    st.write(f"Snowball Analysis of {selected_value}")
    
    # Calculate cumulative growth for each category
    cumulative_df = snowball_df.cumsum()
    
    # Display the snowball metrics
    st.write("Snowball Growth Metrics")
    st.write(add_total_row(cumulative_df.round(2)))
    
    # Create waterfall chart showing contribution
    latest_period = snowball_df.columns[-1]
    fig_waterfall = go.Figure(go.Waterfall(
        name="Snowball",
        orientation="v",
        measure=["relative"] * len(snowball_df),
        x=snowball_df.index,
        y=snowball_df[latest_period],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        text=snowball_df[latest_period].round(2),
        textposition="outside"
    ))
    fig_waterfall.update_layout(title=f"Contribution to Total Growth - {latest_period}")
    st.plotly_chart(fig_waterfall)
    
    
    with st.expander("📊 Snowball Analysis Insights", expanded=True):
        with st.spinner("Generating snowball insights..."):
            snowball_insights = generate_tab_insights(snowball_df, "snowball", selected_value, selected_filter)
            st.write(snowball_insights)


def create_bridge_tab(value_df, selected_value, selected_filter):
    """Creates and populates the Bridge Analysis tab"""
    st.write(f"Bridge Analysis of {selected_value}")
    st.write("Bridge Analysis Coming Soon")

def create_metrics_tab(value_df, selected_value, selected_filter):
    """Creates and populates the Metrics Analysis tab"""
    st.write(f"Metrics Analysis of {selected_value}")
    st.write("Metrics Analysis Coming Soon")

def create_dollar_retention_tab(value_df, selected_value, selected_filter):
    """Creates and populates the Dollar Retention Analysis tab"""
    st.write("Dollar Retention Analysis Coming Soon")
    







def create_analysis_tabs(value_df, total_sum_df, pct_df, avg_df, growth_df, count_df, concentration_df, selected_value, selected_time, analysis_type):
    """Creates and manages all analysis tabs based on analysis type"""
    
    if analysis_type == "Retention Analysis":
        # Only show retention-related tabs
        tab1, tab2, tab3 = st.tabs([ "Snowball Analysis", "Dollar Retention Snowball", "Metrics"
        ])
        
        with tab1:
            create_snowball_tab(value_df, selected_value, selected_time)
        with tab2:
            create_dollar_retention_tab(value_df, selected_value, selected_time)
        with tab3:
            create_metrics_tab(value_df, selected_value, selected_time)
    
    elif analysis_type == "Segmentation Analysis":
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Values", "Percentage", "Percentage YoY Growth","Bridge Analysis", "Count", "Concentration Analysis"
        ])
        
        with tab1:
            create_value_tab(value_df, selected_value, selected_time)
        with tab2:
            create_percentage_tab(pct_df, selected_value, selected_time)
        with tab3:
            create_growth_tab(growth_df, selected_value, selected_time)
        with tab4:
            create_bridge_tab(value_df, selected_value, selected_time)
        with tab5:
            create_count_tab(count_df, selected_value, selected_time)
        with tab6:
            create_concentration_tab(concentration_df, value_df, selected_value, selected_time)


    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Values", "Total Sum", "Percentage", "Average","Percentage YoY Growth","Bridge Analysis", "Count", "Concentration Analysis"
        ])
        
        with tab1:
            create_value_tab(value_df, selected_value, selected_time)
        with tab2:
            create_total_sum_tab(total_sum_df, selected_value, selected_time)
        with tab3:
            create_percentage_tab(pct_df, selected_value, selected_time)
        with tab4:
            create_average_tab(avg_df, selected_value, selected_time)
        with tab5:
            create_growth_tab(growth_df, selected_value, selected_time)
        with tab6:
            create_bridge_tab(value_df, selected_value, selected_time)
        with tab7:
            create_count_tab(count_df, selected_value, selected_time)     
        with tab8:
            create_concentration_tab(concentration_df, value_df, selected_value, selected_time)







def add_total_row(df):
    #Add a total sum to the end of a DataFrame
    total_row = df.sum().to_frame().T
    total_row.index = ['Total']
    return pd.concat([df, total_row])
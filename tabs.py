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

def create_value_tab(value_df, selected_value, selected_filter):
    """Creates and populates the Value Analysis tab"""
    st.write(f"Value Analysis of {selected_value}")
    st.write(value_df)
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
    st.write(pct_df.round(2))
    st.plotly_chart(create_area_chart(pct_df, f"Percentage Distribution of {selected_value} Over Time"))
    st.plotly_chart(create_bar_chart(pct_df, f"Percentage Distribution by Category"))
    with st.expander("📊 Percentage Analysis Insights", expanded=True):
        with st.spinner("Generating percentage insights..."):
            percentage_insights = generate_tab_insights(pct_df, "percentage", selected_value, selected_filter)
            st.write(percentage_insights)

def create_average_tab(avg_df, selected_value, selected_filter):
    """Creates and populates the Average Analysis tab"""
    st.write(f"Average Analysis of {selected_value}")
    st.write(avg_df.round(2))
    st.plotly_chart(create_line_chart(avg_df, f"Average Trend of {selected_value}"))
    st.plotly_chart(create_bar_chart(avg_df, f"Average Distribution by Category"))
    with st.expander("📊 Average Analysis Insights", expanded=True):
        with st.spinner("Generating average insights..."):
            average_insights = generate_tab_insights(avg_df, "average", selected_value, selected_filter)
            st.write(average_insights)

def create_growth_tab(growth_df, selected_value, selected_filter):
    """Creates and populates the Growth Analysis tab"""
    st.write(f"Year-over-Year Growth of {selected_value} (%)")
    st.write(growth_df.round(2))
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
    st.write(count_df)
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

def create_analysis_tabs(value_df, total_sum_df, pct_df, avg_df, growth_df, count_df, concentration_df, selected_value, selected_filter):
    """Creates and manages all analysis tabs"""
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Value", "Total Sum", "Percentage", "Average", 
        "Percentage Growth", "Count", "Concentration Analysis"
    ])
    
    with tab1:
        create_value_tab(value_df, selected_value, selected_filter)
    
    with tab2:
        create_total_sum_tab(total_sum_df, selected_value, selected_filter)
    
    with tab3:
        create_percentage_tab(pct_df, selected_value, selected_filter)
    
    with tab4:
        create_average_tab(avg_df, selected_value, selected_filter)
    
    with tab5:
        create_growth_tab(growth_df, selected_value, selected_filter)
    
    with tab6:
        create_count_tab(count_df, selected_value, selected_filter)
    
    with tab7:
        create_concentration_tab(concentration_df, value_df, selected_value, selected_filter)
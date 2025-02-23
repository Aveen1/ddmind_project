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



def load_custom_css():
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
            /* Import Inter font from Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            /* Apply Inter font to all elements */
            * {
                font-family: 'Inter', sans-serif;
            }
            
            /* Apply specifically to Streamlit elements */
            .stMarkdown, .stButton, .stSelectbox, .stTextInput {
                font-family: 'Inter', sans-serif;
            }
            
            
            /* Table container styling */
            .dataframe {
                overflow-x: auto !important;
                width: 100% !important;
                display: block !important;
                margin: 0 !important;
                text-align: center !important;
            }
            
            /* Ensure table takes full width */
            .stTable {
                width: 100% !important;
                max-width: none !important;
                text-align: center !important;

            }
            
            /* Hide Streamlit UI elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

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
        #st.title("DDMind.ai")
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
        #st.markdown("Made with ❤️ by DDMind")

def create_value_tab(value_df, selected_value, selected_filter):
    """Creates and populates the Value Analysis tab"""
    st.write(f"#### Value Analysis of {selected_value}")
    st.dataframe(add_total_row(value_df), use_container_width=True)
    st.plotly_chart(create_line_chart(value_df, f"Value Trend of {selected_value}"))
    st.plotly_chart(create_bar_chart(value_df, f"Value Distribution of {selected_value}"))
    with st.expander("📊 Value Analysis Insights", expanded=True):
        with st.spinner("Generating value insights..."):
            value_insights = generate_tab_insights(value_df, "value", selected_value, selected_filter)
            st.write(value_insights)

def create_total_sum_tab(total_sum_df, selected_value, selected_filter):
    """Creates and populates the Total Sum Analysis tab"""
    st.write(f"#### Total Sum Analysis of {selected_value}")
    st.dataframe(total_sum_df, use_container_width=True)
    st.plotly_chart(create_line_chart(total_sum_df, f"Total Sum Trend of {selected_value}"))
    st.plotly_chart(create_bar_chart(total_sum_df, f"Total Sum Distribution of {selected_value}"))
    with st.expander("📊 Total Sum Analysis Insights", expanded=True):
        with st.spinner("Generating total sum insights..."):
            total_sum_insights = generate_tab_insights(total_sum_df, "total_sum", selected_value, selected_filter)
            st.write(total_sum_insights)

def create_percentage_tab(pct_df, selected_value, selected_filter):
    """Creates and populates the Percentage Distribution tab"""
    st.write(f"#### Percentage Distribution of {selected_value}")
    pct_df = (add_total_row(pct_df.round(2)).applymap(lambda x: f"{x}%"))
    st.dataframe(pct_df, use_container_width=True)
    st.plotly_chart(create_area_chart(pct_df, f"Percentage Distribution of {selected_value} Over Time"))
    st.plotly_chart(create_bar_chart(pct_df, f"Percentage Distribution by Category"))
    with st.expander("📊 Percentage Analysis Insights", expanded=True):
        with st.spinner("Generating percentage insights..."):
            percentage_insights = generate_tab_insights(pct_df, "percentage", selected_value, selected_filter)
            st.write(percentage_insights)

def create_average_tab(avg_df, selected_value, selected_filter):
    """Creates and populates the Average Analysis tab"""
    st.write(f"#### Average Analysis of {selected_value}")
    avg_df = (add_total_row(avg_df.round(2)))
    st.dataframe(avg_df, use_container_width=True)
    st.plotly_chart(create_line_chart(avg_df, f"Average Trend of {selected_value}"))
    st.plotly_chart(create_bar_chart(avg_df, f"Average Distribution by Category"))
    with st.expander("📊 Average Analysis Insights", expanded=True):
        with st.spinner("Generating average insights..."):
            average_insights = generate_tab_insights(avg_df, "average", selected_value, selected_filter)
            st.write(average_insights)

def create_growth_tab(growth_df, selected_value, selected_filter):
    """Creates and populates the Growth Analysis tab with correct percentage formatting"""
    rounded_df = growth_df.round(2)
    def calculate_total(col):
        valid_values = col[col.notna()]
        weights = valid_values.abs() / valid_values.abs().sum()
        return (valid_values * weights).sum() if not valid_values.empty else 0

    total_row = rounded_df.apply(calculate_total)
    result_df = pd.concat([rounded_df, pd.DataFrame(total_row).T])
    result_df.index = list(rounded_df.index) + ['Total']
    def format_percentage(x):
        if pd.isna(x):
            return ""
        return f"{x:.2f}%" if x != 0 else ""
    
    formatted_df = result_df.applymap(format_percentage)
    st.write(f"#### Year-over-Year Growth of {selected_value} (%)")
    st.dataframe(formatted_df, use_container_width=True)
    st.plotly_chart(create_bar_chart(growth_df, f"Growth Rate by Category"))
    fig_heatmap = px.imshow(
        growth_df,
        title=f"Growth Rate Heatmap for {selected_value}",
        labels=dict(x="Time Period", y="Category", color="Growth Rate (%)"),
        aspect="auto"
    )
    
    fig_heatmap.update_layout(
        height=600,
        yaxis_title="Category",
        xaxis_title="Time Period"
    )
    st.plotly_chart(fig_heatmap)
    with st.expander("📊 Growth Analysis Insights", expanded=True):
        with st.spinner("Generating growth insights..."):
            growth_insights = generate_tab_insights(growth_df, "growth", selected_value, selected_filter)
            st.write(growth_insights)

def create_count_tab(count_df, selected_value, selected_filter):
    """Creates and populates the Count Analysis tab"""
    st.write(f"#### Count Analysis of {selected_value}")
    st.dataframe(add_total_row(count_df), use_container_width=True)
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
        st.write(f"#### Concentration Analysis of {selected_value} (%)")
        st.dataframe(add_total_row(concentration_df.round(2)).applymap(lambda x: f"{x}%"), use_container_width=True)
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
    st.write("#### Top Customers Concentration Analysis")
    
    top_n_results = create_top_n_concentration(value_df)
    summary_df = create_top_n_table(top_n_results)
    
    st.write("##### Summary of Top Customer Concentration")
    st.dataframe(summary_df, use_container_width=True)
    
    fig_top_n = px.bar(
        summary_df,
        y='Concentration (%)',
        title="Concentration by Top N Customers",
        labels={'index': 'Customer Group', 'value': 'Concentration (%)'}
    )
    fig_top_n.update_layout(height=400)
    st.plotly_chart(fig_top_n)
    
    st.write("### Detailed Customer Lists")
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

def create_bridge_tab(value_df, selected_value, selected_time):
    """Creates and populates the Bridge Analysis tab with consolidated view"""
    st.write(f"#### Bridge Analysis - {selected_time}")
    
    periods = sorted(value_df.columns)    
    bridge_data = []
    summary_data = []
    
    for i in range(len(periods)-1):
        start_period = periods[i]
        end_period = periods[i+1]
        period_display = f"{start_period} to {end_period}"
        
        decrease_idx = value_df.index.get_loc('Decreases') if 'Decreases' in value_df.index else len(value_df)
        
        start_total = value_df.iloc[:decrease_idx][start_period].sum()
        end_total = value_df.iloc[:decrease_idx][end_period].sum()
        
        bridge_data.append({
            'Period': period_display,
            'Category': f"Starting Total ({start_period})",
            'Value': start_total,
            'Type': 'Total',
            'Order': 1
        })
        
        total_increases = 0
        total_decreases = 0
        
        for idx in value_df.index[:decrease_idx]:
            value = value_df.loc[idx, start_period]
            if pd.notnull(value) and value != 0:
                bridge_data.append({
                    'Period': period_display,
                    'Category': idx,
                    'Value': value,
                    'Type': 'Increase',
                    'Order': 2
                })
                total_increases += value
        
        for idx in value_df.index[decrease_idx+1:]:
            value = value_df.loc[idx, start_period]
            if pd.notnull(value) and value != 0:
                bridge_data.append({
                    'Period': period_display,
                    'Category': idx,
                    'Value': value,
                    'Type': 'Decrease',
                    'Order': 3
                })
                total_decreases += value
        
        bridge_data.append({
            'Period': period_display,
            'Category': f"Ending Total ({end_period})",
            'Value': end_total,
            'Type': 'Total',
            'Order': 4
        })
        
        net_change = end_total - start_total
        summary_data.extend([
            {
                'Period': period_display,
                'Category': '📊 Net Change',
                'Value': net_change,
                'Type': 'Summary',
                'Order': 5
            },
            {
                'Period': period_display,
                'Category': '📈 Total Increases',
                'Value': total_increases,
                'Type': 'Summary',
                'Order': 6
            },
            {
                'Period': period_display,
                'Category': '📉 Total Decreases',
                'Value': total_decreases,
                'Type': 'Summary',
                'Order': 7
            }
        ])
    
    bridge_df = pd.DataFrame(bridge_data + summary_data)
    
    pivot_df = bridge_df.pivot_table(
        index=['Category', 'Type', 'Order'],
        columns='Period',
        values='Value',
        aggfunc='sum'
    ).sort_values('Order').reset_index()
    
    display_df = pivot_df[['Category'] + [col for col in pivot_df.columns if col not in ['Category', 'Type', 'Order']]].copy()
    
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else '')
    
    st.dataframe(display_df,hide_index=True, use_container_width=True)
    
    latest_period = bridge_df['Period'].unique()[-1]
    latest_period_data = bridge_df[
        (bridge_df['Period'] == latest_period) & 
        (bridge_df['Type'] != 'Summary')
    ]
    
    fig = go.Figure()
    
    for _, row in latest_period_data.iterrows():
        measure = "absolute" if row['Type'] == 'Total' and 'Starting Total' in row['Category'] else \
                 "total" if row['Type'] == 'Total' and 'Ending Total' in row['Category'] else \
                 "relative"
        
        fig.add_trace(go.Waterfall(
            name=row['Category'],
            orientation="v",
            measure=[measure],
            x=[row['Category']],
            y=[row['Value']],
            text=[f"{row['Value']:,.2f}"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
    
    fig.update_layout(
        title={
            'text': f"{selected_value} Bridge Analysis - {latest_period}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        height=600,
        waterfallgap=0.2,
        xaxis={
            "type": "category",
            "title": "Components"
        },
        yaxis={
            "title": selected_value,
            "tickformat": ",.2f"
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Bridge Analysis Details", expanded=True):
        latest_data = latest_period_data.copy()
        start_total = latest_data[latest_data['Type'] == 'Total']['Value'].iloc[0]
        end_total = latest_data[latest_data['Type'] == 'Total']['Value'].iloc[-1]
        increases = latest_data[latest_data['Type'] == 'Increase']['Value'].sum()
        decreases = latest_data[latest_data['Type'] == 'Decrease']['Value'].sum()
        
        #Top 3 increases
        increases_df = latest_data[latest_data['Type'] == 'Increase'].nlargest(3, 'Value')
        if not increases_df.empty:
            st.write("\nLargest increases:")
            for _, row in increases_df.iterrows():
                st.write(f"- {row['Category']}: {row['Value']:,.2f} ({(row['Value']/increases * 100):.1f}% of total increases)")
        
        #Top 3 decreases
        decreases_df = latest_data[latest_data['Type'] == 'Decrease'].nsmallest(3, 'Value')
        if not decreases_df.empty:
            st.write("\nLargest decreases:")
            for _, row in decreases_df.iterrows():
                st.write(f"- {row['Category']}: {row['Value']:,.2f} ({(row['Value']/decreases * 100):.1f}% of total decreases)")
        
        with st.spinner("Generating metrics insights..."):
            bridge_insights = generate_tab_insights(latest_data, "bridge", selected_value, selected_time)
            st.write(bridge_insights)


def create_snowball_tab(value_df, selected_value, selected_time):
    """Creates and populates the Snowball Analysis tab with customer movement metrics"""
    st.write(f"#### Snowball Analysis of {selected_value}")
    
    #Calculate customer movement metrics for each period
    periods = sorted(value_df.columns)
    metrics_df = pd.DataFrame(columns=['Metric'] + periods)
    
    #Initialize metrics tracking
    metrics = {
        'Beginning Customers Balance': [],
        'New Customers': [],
        'Lost Customers': [],
        'Stop/Start': [],
        'Ending Customers Balance': []
    }
    
    for i, period in enumerate(periods):
        #Get active customers (value > 0) for current period
        current_active = set(value_df[value_df[period] > 0].index)
        
        if i == 0:
            #First period
            metrics['Beginning Customers Balance'].append(len(current_active))
            metrics['New Customers'].append(0)
            metrics['Lost Customers'].append(0)
            metrics['Stop/Start'].append(0)
            metrics['Ending Customers Balance'].append(len(current_active))
        else:
            #Get previous period's active customers
            prev_period = periods[i-1]
            prev_active = set(value_df[value_df[prev_period] > 0].index)
            
            #Calculate metrics
            beginning_balance = len(prev_active)
            new_customers = len(current_active - prev_active)
            lost_customers = len(prev_active - current_active)
            
            #Calculate stop/start (customers who were inactive in previous period but active now)
            if i > 1:
                two_periods_ago = periods[i-2]
                two_periods_ago_active = set(value_df[value_df[two_periods_ago] > 0].index)
                stop_start = len((prev_active - current_active) & two_periods_ago_active)
            else:
                stop_start = 0
            
            ending_balance = beginning_balance + new_customers - lost_customers + stop_start
            
            #Store metrics
            metrics['Beginning Customers Balance'].append(beginning_balance)
            metrics['New Customers'].append(new_customers)
            metrics['Lost Customers'].append(lost_customers)
            metrics['Stop/Start'].append(stop_start)
            metrics['Ending Customers Balance'].append(ending_balance)
    
    #Create DataFrame from metrics
    for metric_name, values in metrics.items():
        row_data = {'Metric': metric_name}
        for i, period in enumerate(periods):
            row_data[period] = values[i]
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row_data])], ignore_index=True)
    
    #Display the metrics table
    st.write("##### Customer Movement Analysis")
    st.dataframe(metrics_df.set_index('Metric'), use_container_width= True)
    
    #Create waterfall chart showing customer movement for the latest period
    latest_period = periods[-1]
    prev_period = periods[-2] if len(periods) > 1 else None
    
    if prev_period:
        waterfall_data = {
            'Measure': ['Beginning Balance', 'New Customers', 'Lost Customers', 'Stop/Start', 'Ending Balance'],
            'Value': [
                metrics['Beginning Customers Balance'][-1],
                metrics['New Customers'][-1],
                -metrics['Lost Customers'][-1],
                metrics['Stop/Start'][-1],
                metrics['Ending Customers Balance'][-1]
            ]
        }
        
        fig = go.Figure(go.Waterfall(
            name="Customer Movement",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=waterfall_data['Measure'],
            y=waterfall_data['Value'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}},
            text=waterfall_data['Value'],
            textposition="outside"
        ))
        
        fig.update_layout(
            title=f"Customer Movement Analysis - {latest_period}",
            showlegend=False
        )
        
        st.plotly_chart(fig)
    
    with st.expander("📊 Snowball Analysis Insights", expanded=True):
        with st.spinner("Generating snowball insights..."):
            snowball_insights = generate_tab_insights(metrics_df, "snowball", selected_value, selected_time)
            st.write(snowball_insights)

def create_dollar_retention_tab(value_df, selected_value, selected_time):
    """Creates and populates the Dollar Retention Analysis tab with revenue movement metrics"""
    st.write(f"#### Dollar Retention Analysis of {selected_value}")
    
    # Calculate revenue movement metrics for each period
    periods = sorted(value_df.columns)
    metrics_df = pd.DataFrame(columns=['Metric'] + periods)
    
    # Initialize metrics tracking
    metrics = {
        'Beginning Balance': [],
        'New Revenue': [],
        'Revenue Increases': [],
        'Lost Revenue': [],
        'Decreases': [],
        'Ending Balance': []
    }
    
    for i, period in enumerate(periods):
        if i == 0:
            #First period - only use values for the selected filter
            current_values = value_df[period]
            filtered_sum = current_values.sum()  #This already respects the filter since value_df is filtered
            
            metrics['Beginning Balance'].append(filtered_sum)
            metrics['New Revenue'].append(0)
            metrics['Revenue Increases'].append(0)
            metrics['Lost Revenue'].append(0)
            metrics['Decreases'].append(0)
            metrics['Ending Balance'].append(filtered_sum)
        else:
            prev_period = periods[i-1]
            current_values = value_df[period]
            prev_values = value_df[prev_period]
            
            #Calculate beginning balance - only for the filtered data
            beginning_balance = prev_values[prev_values > 0].sum()
            
            #Calculate new revenue (from customers with 0 revenue in previous period)
            new_revenue = current_values[(prev_values == 0) & (current_values > 0)].sum()
            
            #Calculate revenue increases (from existing customers)
            increases = current_values[current_values > prev_values]
            revenue_increases = (increases - prev_values[increases.index]).sum()
            
            #Calculate lost revenue (from customers who went to 0)
            lost_revenue = prev_values[(current_values == 0) & (prev_values > 0)].sum()
            
            #Calculate decreases (from existing customers who decreased but didn't go to 0)
            decreases = current_values[current_values < prev_values]
            revenue_decreases = (prev_values[decreases.index] - decreases).sum()
            
            #Calculate ending balance
            ending_balance = (beginning_balance + new_revenue + revenue_increases - 
                            lost_revenue - revenue_decreases)
            
            #Store metrics
            metrics['Beginning Balance'].append(beginning_balance)
            metrics['New Revenue'].append(new_revenue)
            metrics['Revenue Increases'].append(revenue_increases)
            metrics['Lost Revenue'].append(lost_revenue)
            metrics['Decreases'].append(-revenue_decreases)  #Store as negative value
            metrics['Ending Balance'].append(ending_balance)
    
    #Create DataFrame from metrics
    for metric_name, values in metrics.items():
        row_data = {'Metric': metric_name}
        for i, period in enumerate(periods):
            row_data[period] = values[i]
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row_data])], ignore_index=True)
    
    #Format the values
    for col in periods:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:,.2f}")
    
    #Display the metrics table
    st.write("##### Dollar Value Movement Analysis")
    st.dataframe(metrics_df.set_index('Metric') , use_container_width= True)
    
    #Create waterfall chart showing dollar movement for the latest period
    latest_period = periods[-1]
    prev_period = periods[-2] if len(periods) > 1 else None
    
    if prev_period:
        waterfall_data = {
            'Measure': ['Beginning Balance', 'New Revenue', 'Revenue Increases', 
                       'Lost Revenue', 'Decreases', 'Ending Balance'],
            'Value': [
                float(metrics_df.iloc[0][latest_period].replace(',', '')),
                float(metrics_df.iloc[1][latest_period].replace(',', '')),
                float(metrics_df.iloc[2][latest_period].replace(',', '')),
                -float(metrics_df.iloc[3][latest_period].replace(',', '')),
                float(metrics_df.iloc[4][latest_period].replace(',', '')),  
                float(metrics_df.iloc[5][latest_period].replace(',', ''))
            ]
        }
        
        fig = go.Figure(go.Waterfall(
            name="Dollar Movement",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=waterfall_data['Measure'],
            y=waterfall_data['Value'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}},
            text=[f"${x:,.2f}" for x in waterfall_data['Value']],
            textposition="outside"
        ))
        
        fig.update_layout(
            title=f"Dollar Value Movement Analysis - {latest_period}",
            showlegend=False,
            height=600,
            yaxis_title="Dollar Value"
        )
        
        st.plotly_chart(fig)
    
    with st.expander("📊 Dollar Retention Analysis Insights", expanded=True):
        with st.spinner("Generating dollar retention insights..."):
            metrics_insights = generate_tab_insights(metrics_df, "dollar_retention", selected_value, selected_time)
            st.write(metrics_insights)

def create_metrics_tab(value_df, selected_value, selected_time):
    """Creates and populates the Metrics Analysis tab with retention and revenue metrics"""
    st.write(f"#### Metrics Analysis of {selected_value}")
    
    #Initialize metrics DataFrame with all periods
    periods = sorted(value_df.columns)
    metrics_df = pd.DataFrame(index=[
        'Annual Customer Period Average',
        'Annual Average Dollar Per Customer',
        'Annual Customer Retention (%)',
        'Annual Dollar Period Average',
        'Annual Dollar Net Retention (%)',
        'Annual Dollar Retention Lost Only (%)',
        'Annual Dollar Retention (Lost + Decrease) (%)',
        'Lost Customer Average Size',
        'Size Attrition Factor',
        'New Revenue % of Beginning Revenue (%)'
    ])
    
    #Calculate metrics for each period
    for i, period in enumerate(periods):
        current_values = value_df[period]
        current_active = current_values > 0
        
        #Calculate basic period metrics
        metrics_df.loc['Annual Customer Period Average', period] = current_active.sum()
        metrics_df.loc['Annual Average Dollar Per Customer', period] = (
            current_values[current_active].mean() if current_active.any() else 0
        )
        metrics_df.loc['Annual Dollar Period Average', period] = current_values.sum()
        
        if i > 0:
            prev_period = periods[i-1]
            prev_values = value_df[prev_period]
            prev_active = prev_values > 0
            
            #Customer Retention
            retained_customers = (current_active & prev_active).sum()
            retention_rate = (retained_customers / prev_active.sum() * 100 
                            if prev_active.any() else 100)
            metrics_df.loc['Annual Customer Retention (%)', period] = retention_rate
            
            #Dollar Retention metrics
            retained_value = current_values[prev_active].sum()
            prev_total = prev_values.sum()
            
            #Net Dollar Retention
            net_retention = (retained_value / prev_total * 100 
                           if prev_total > 0 else 100)
            metrics_df.loc['Annual Dollar Net Retention (%)', period] = net_retention
            
            #Lost Only Retention
            lost_customers = prev_active & ~current_active
            lost_value = prev_values[lost_customers].sum()
            lost_only_retention = ((prev_total - lost_value) / prev_total * 100 
                                 if prev_total > 0 else 100)
            metrics_df.loc['Annual Dollar Retention Lost Only (%)', period] = lost_only_retention
            
            #Lost + Decrease Retention
            decreases = current_values < prev_values
            decrease_value = (prev_values[decreases] - current_values[decreases]).sum()
            lost_decrease_retention = ((prev_total - lost_value - decrease_value) / prev_total * 100 
                                    if prev_total > 0 else 100)
            metrics_df.loc['Annual Dollar Retention (Lost + Decrease) (%)', period] = lost_decrease_retention
            
            #Lost Customer Metrics
            if lost_customers.any():
                lost_avg_size = prev_values[lost_customers].mean()
                metrics_df.loc['Lost Customer Average Size', period] = lost_avg_size
                active_avg_size = prev_values[prev_active].mean()
                size_attrition = lost_avg_size / active_avg_size if active_avg_size > 0 else 0
                metrics_df.loc['Size Attrition Factor', period] = size_attrition
            
            #New Revenue
            new_customers = current_active & ~prev_active
            new_revenue = current_values[new_customers].sum()
            new_revenue_pct = (new_revenue / prev_total * 100 
                             if prev_total > 0 else 0)
            metrics_df.loc['New Revenue % of Beginning Revenue (%)', period] = new_revenue_pct
    
    #Format the values
    format_dict = {
        'Annual Customer Period Average': '{:.2f}',
        'Annual Average Dollar Per Customer': '{:,.2f}',
        'Annual Customer Retention (%)': '{:.2f}',
        'Annual Dollar Period Average': '{:,.2f}',
        'Annual Dollar Net Retention (%)': '{:.2f}',
        'Annual Dollar Retention Lost Only (%)': '{:.2f}',
        'Annual Dollar Retention (Lost + Decrease) (%)': '{:.2f}',
        'Lost Customer Average Size': '{:,.2f}',
        'Size Attrition Factor': '{:.2f}',
        'New Revenue % of Beginning Revenue (%)': '{:.2f}'
    }
    
    for idx, format_str in format_dict.items():
        metrics_df.loc[idx] = metrics_df.loc[idx].apply(
            lambda x: format_str.format(float(x)) if pd.notnull(x) else ''
        )
    
    #Display the metrics table
    st.write("##### Key Metrics Analysis")
    st.dataframe(metrics_df, use_container_width= True)
    
    #Generate insights
    with st.expander("📊 Metrics Analysis Insights", expanded=True):
        with st.spinner("Generating metrics insights..."):
            metrics_insights = generate_tab_insights(metrics_df, "metrics", selected_value, selected_time)
            st.write(metrics_insights)



def create_values_cohort_tab(value_df, selected_value, selected_time):
    st.write("#### Value Analysis by Cohort")

    #Get all unique years as columns
    years = sorted(value_df.columns.unique())

    #Get first time periods
    first_periods = {}
    for index in value_df.index:
        # Find first non-zero value
        first_non_zero = value_df.loc[index].replace(0, np.nan).first_valid_index()
        if first_non_zero:
            first_periods[index] = first_non_zero

    #Create cohort table
    cohort_data = []

    #Group by first period
    for start_year in years:
        #Get customers that started in this period
        starters = [customer for customer, first_period in first_periods.items() 
                    if first_period == start_year]
        
        if starters:
            row_data = {'First Time Period': start_year}
            
            #Calculate values for each period
            for year in years:
                if year >= start_year:
                    current_value = value_df.loc[starters, year].sum()
                    period_number = years.index(year) - years.index(start_year) + 1
                    row_data[year] = f"{current_value:,.2f} ({period_number})"
                else:
                    row_data[year] = ""
            
            cohort_data.append(row_data)

    #Convert to DataFrame
    cohort_df = pd.DataFrame(cohort_data).set_index('First Time Period')

    #Display the cohort table
    st.write("##### Cohort Value Analysis")
    st.dataframe(cohort_df, use_container_width=True)

    #Create heatmap visualization
    numeric_cohort_df = pd.DataFrame(index=cohort_df.index, columns=years)
    for start_year in cohort_df.index:
        for year in years:
            value = cohort_df.loc[start_year, year]
            if value:
                numeric_value = float(value.split()[0].replace(',', ''))
                numeric_cohort_df.loc[start_year, year] = numeric_value

    fig = px.imshow(
        numeric_cohort_df,
        title="Cohort Value Heatmap",
        labels=dict(x="Time Period", y="First Time Period", color="Value"),
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)

    #Generate insights using the existing function
    with st.expander("📊 Cohort Values Analysis Insights", expanded=True):
        with st.spinner("Generating cohort insights..."):
            cohort_insights = generate_tab_insights(numeric_cohort_df, "values_cohort", selected_value, selected_time)
            st.write(cohort_insights)

def create_count_cohort_tab(value_df, selected_value, selected_time):
    """Creates and populates the Count Cohort Analysis tab"""
    st.write("#### Count Analysis by Cohort")

    #Get all unique periods as columns
    periods = sorted(value_df.columns.unique())

    #Get first time periods for each customer
    first_periods = {}
    for index in value_df.index:
        # Find first non-zero value
        first_non_zero = value_df.loc[index].replace(0, np.nan).first_valid_index()
        if first_non_zero:
            first_periods[index] = first_non_zero

    #Create cohort table
    cohort_data = []

    #Group by first period
    for start_period in periods:
        #Get customers that started in this period
        starters = [customer for customer, first_period in first_periods.items() 
                   if first_period == start_period]
        
        if starters:
            row_data = {'First Time Period': start_period}
            
            #Calculate counts for each period
            for period in periods:
                if period >= start_period:
                    #Count active customers in current period
                    active_count = (value_df.loc[starters, period] > 0).sum()
                    #Calculate period number (1, 2, 3, etc.)
                    period_number = periods.index(period) - periods.index(start_period) + 1
                    row_data[period] = f"{active_count} ({period_number})"
                else:
                    row_data[period] = ""
            
            cohort_data.append(row_data)

    #Convert to DataFrame
    cohort_df = pd.DataFrame(cohort_data).set_index('First Time Period')

    #Display the cohort table
    st.write("##### Customer Count by Cohort")
    st.dataframe(cohort_df, use_container_width=True)

    # Create heatmap visualization
    numeric_cohort_df = pd.DataFrame(index=cohort_df.index, columns=periods)
    for start_period in cohort_df.index:
        for period in periods:
            value = cohort_df.loc[start_period, period]
            if value:
                numeric_value = int(value.split()[0])
                numeric_cohort_df.loc[start_period, period] = numeric_value

    fig = px.imshow(
        numeric_cohort_df,
        title="Cohort Customer Count Heatmap",
        labels=dict(x="Time Period", y="First Time Period", color="Count"),
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)

    # Generate insights
    with st.expander("📊 Count Cohort Analysis Insights", expanded=True):
        with st.spinner("Generating cohort insights..."):
            cohort_insights = generate_tab_insights(numeric_cohort_df, "count_cohort", selected_value, selected_time)
            st.write(cohort_insights)

def create_average_cohort_tab(value_df, selected_value, selected_time):
    """Creates and populates the Average Value Cohort Analysis tab"""
    st.write("#### Average Value Analysis by Cohort")

    #Get all unique periods as columns
    periods = sorted(value_df.columns.unique())

    #Get first time periods for each customer
    first_periods = {}
    for index in value_df.index:
        #Find first non-zero value
        first_non_zero = value_df.loc[index].replace(0, np.nan).first_valid_index()
        if first_non_zero:
            first_periods[index] = first_non_zero

    #Create cohort table
    cohort_data = []

    #Group by first period
    for start_period in periods:
        #Get customers that started in this period
        starters = [customer for customer, first_period in first_periods.items() 
                   if first_period == start_period]
        
        if starters:
            row_data = {'First Time Period': start_period}
            
            #Calculate average values for each period
            for period in periods:
                if period >= start_period:
                    #Get values for active customers in current period
                    period_values = value_df.loc[starters, period]
                    active_values = period_values[period_values > 0]
                    if not active_values.empty:
                        avg_value = active_values.mean()
                        #Calculate period number (1, 2, 3, etc.)
                        period_number = periods.index(period) - periods.index(start_period) + 1
                        #Format with commas and 2 decimal places
                        row_data[period] = f"{avg_value:,.2f}"
                        #Add period number in small circles before the value
                        row_data[f"{period}_number"] = period_number
                else:
                    row_data[period] = ""
                    row_data[f"{period}_number"] = None
            
            cohort_data.append(row_data)

    #Convert to DataFrame
    cohort_df = pd.DataFrame(cohort_data).set_index('First Time Period')

    #Create a styled version of the DataFrame for display
    display_df = pd.DataFrame(index=cohort_df.index, columns=periods)
    for start_period in cohort_df.index:
        for period in periods:
            value = cohort_df.loc[start_period, period]
            period_number = cohort_df.loc[start_period, f"{period}_number"]
            if value and period_number:
                display_df.loc[start_period, period] = value
            else:
                display_df.loc[start_period, period] = ""

    #Display the cohort table
    st.write("##### Average Value by Cohort")
    st.dataframe(display_df, use_container_width=True)

    #Create heatmap visualization
    numeric_cohort_df = pd.DataFrame(index=cohort_df.index, columns=periods)
    for start_period in cohort_df.index:
        for period in periods:
            value = cohort_df.loc[start_period, period]
            if value:
                numeric_value = float(value.replace(',', ''))
                numeric_cohort_df.loc[start_period, period] = numeric_value

    fig = px.imshow(
        numeric_cohort_df,
        title="Cohort Average Value Heatmap",
        labels=dict(x="Time Period", y="First Time Period", color="Average Value"),
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)

    #Generate insights
    with st.expander("📊 Average Value Cohort Analysis Insights", expanded=True):
        with st.spinner("Generating cohort insights..."):
            cohort_insights = generate_tab_insights(numeric_cohort_df, "average_cohort", selected_value, selected_time)
            st.write(cohort_insights)

def create_lost_dollars_cohort_tab(value_df, selected_value, selected_time):
    """Creates and populates the Lost Dollars Cohort Analysis tab"""
    st.write("#### Lost Dollar Value Analysis by Cohort ")
    periods = sorted(value_df.columns.unique())

    first_periods = {}
    for index in value_df.index:
        first_non_zero = value_df.loc[index].replace(0, np.nan).first_valid_index()
        if first_non_zero:
            first_periods[index] = first_non_zero

    lost_data = []

    for start_period in periods:
        starters = [customer for customer, first_period in first_periods.items() if first_period == start_period]

        if starters:
            lost_row = {'First Time Period': start_period}

            for i, period in enumerate(periods):
                if period >= start_period:
                    #Get lost customers in this period (who had revenue before but now have 0)
                    if i > 0:
                        prev_period = periods[i - 1]
                        lost_customers = value_df.loc[starters][(value_df[prev_period] > 0) & (value_df[period] == 0)].index
                        
                        if not lost_customers.empty:
                            lost_value = value_df.loc[lost_customers, prev_period].sum()
                            lost_row[period] = lost_value if lost_value > 0 else 0.0
                        else:
                            lost_row[period] = 0.0
                    else:
                        lost_row[period] = 0.0
                else:
                    lost_row[period] = ""

            lost_data.append(lost_row)

    lost_df = pd.DataFrame(lost_data).set_index('First Time Period')
    for column in lost_df.columns:
        lost_df[column] = lost_df[column].apply(
            lambda x: f"{float(x):,.2f}" if isinstance(x, (int, float)) else x
        )

    st.write("##### Lost Dollars by Cohort")
    st.dataframe(lost_df, use_container_width=True)

    numeric_lost_df = pd.DataFrame(index=lost_df.index, columns=periods)

    for start_period in lost_df.index:
        for period in periods:
            lost_value = lost_df.loc[start_period, period]
            if isinstance(lost_value, str):
                lost_value = lost_value.replace(',', '')
            numeric_lost_df.loc[start_period, period] = float(lost_value) if lost_value else 0.0

    fig = px.imshow(
        numeric_lost_df,
        title="Cohort Lost Dollars Heatmap",
        labels=dict(x="Time Period", y="First Time Period", color="Lost Value ($)"),
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig)

    with st.expander("📊 Lost Dollars Cohort Analysis Insights", expanded=True):
        with st.spinner("Generating cohort insights..."):
            cohort_insights = generate_tab_insights(numeric_lost_df, "lost_dollars_cohort", selected_value, selected_time)
            st.write(cohort_insights)

def create_dollar_decreases_cohort_tab(value_df, selected_value, selected_time):
    """Creates and populates the Dollar Decrease Cohort Analysis tab"""
    st.write("#### Dollar Value Decrease Analysis by Cohort")

    #Get all unique periods as columns
    periods = sorted(value_df.columns.unique())

    #Get first time periods for each customer
    first_periods = {}
    for index in value_df.index:
        first_non_zero = value_df.loc[index].replace(0, np.nan).first_valid_index()
        if first_non_zero:
            first_periods[index] = first_non_zero

    #Create cohort table
    cohort_data = []

    #Group by first period
    for start_period in periods:
        #Get customers that started in this period
        starters = [customer for customer, first_period in first_periods.items() 
                   if first_period == start_period]
        
        if starters:
            row_data = {'First Time Period': start_period}
            
            #Calculate dollar decrease for each period
            for period in periods:
                if period >= start_period:
                    #Get values for current and previous period
                    if periods.index(period) > 0:
                        prev_period = periods[periods.index(period) - 1]
                        curr_values = value_df.loc[starters, period]
                        prev_values = value_df.loc[starters, prev_period]
                        #Calculate lost dollars (where value decreased)
                        lost_dollars = (prev_values - curr_values).clip(lower=0).sum()
                        
                        lost_dollars = -lost_dollars  #Convert to negative
                        period_number = periods.index(period) - periods.index(start_period) + 1
                        row_data[period] = f"{lost_dollars:,.2f} ({period_number})"
                    else:
                        row_data[period] = f"0.00 (1)"
                else:
                    row_data[period] = ""
            
            cohort_data.append(row_data)

    #Convert to DataFrame
    cohort_df = pd.DataFrame(cohort_data).set_index('First Time Period')

    #Display the cohort table
    st.write("##### Dollar Value Decrease by Cohort")
    st.dataframe(cohort_df, use_container_width=True)

    #Create heatmap visualization
    numeric_cohort_df = pd.DataFrame(index=cohort_df.index, columns=periods)
    for start_period in cohort_df.index:
        for period in periods:
            value = cohort_df.loc[start_period, period]
            if value:
                numeric_value = float(value.split()[0].replace(',', ''))
                numeric_cohort_df.loc[start_period, period] = numeric_value

    fig = px.imshow(
        numeric_cohort_df,
        title="Cohort Dollar Value Decrease Heatmap",
        labels=dict(x="Time Period", y="First Time Period", color="Lost Value"),
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig)

    #Generate insights
    with st.expander("📊 Dollar Decrease Cohort Analysis Insights", expanded=True):
        with st.spinner("Generating cohort insights..."):
            cohort_insights = generate_tab_insights(numeric_cohort_df, "dollar_decreases_cohort", selected_value, selected_time)
            st.write(cohort_insights)

def create_dollar_increases_cohort_tab(value_df, selected_value, selected_time):
    """Creates and populates the Dollar Changes Cohort Analysis tab (increases only)"""
    st.write("#### Dollar Value Increases Analysis by Cohort")

    #Get all unique periods as columns
    periods = sorted(value_df.columns.unique())

    #Get first time periods for each customer
    first_periods = {}
    for index in value_df.index:
        first_non_zero = value_df.loc[index].replace(0, np.nan).first_valid_index()
        if first_non_zero:
            first_periods[index] = first_non_zero

    #Create cohort table for increases
    increase_data = []

    #Group by first period
    for start_period in periods:
        starters = [customer for customer, first_period in first_periods.items() 
                   if first_period == start_period]
        
        if starters:
            increase_row = {'First Time Period': start_period}
            
            for period in periods:
                if period >= start_period:
                    if periods.index(period) > 0:
                        prev_period = periods[periods.index(period) - 1]
                        curr_values = value_df.loc[starters, period]
                        prev_values = value_df.loc[starters, prev_period]
                        
                        #Calculate increases
                        changes = curr_values - prev_values
                        increases = changes[changes > 0].sum()
                        
                        period_number = periods.index(period) - periods.index(start_period) + 1
                        increase_row[period] = f"{increases:,.2f} ({period_number})"
                    else:
                        increase_row[period] = f"0.00 (1)"
                else:
                    increase_row[period] = ""
            
            increase_data.append(increase_row)

    #Convert to DataFrame
    increase_df = pd.DataFrame(increase_data).set_index('First Time Period')

    #Display table
    st.write("##### Value Increases by Cohort")
    st.dataframe(increase_df, use_container_width=True)

    #Create heatmap
    numeric_increase_df = pd.DataFrame(index=increase_df.index, columns=periods)
    
    for start_period in increase_df.index:
        for period in periods:
            inc_value = increase_df.loc[start_period, period]
            
            if inc_value:
                numeric_increase_df.loc[start_period, period] = float(inc_value.split()[0].replace(',', ''))

    fig = px.imshow(
        numeric_increase_df,
        title="Cohort Value Increases Heatmap",
        labels=dict(x="Time Period", y="First Time Period", color="Increase Value"),
        color_continuous_scale="Greens"
    )
    
    st.plotly_chart(fig)
    #Generate insights
    with st.expander("📊 Dollar Increases Cohort Analysis Insights", expanded=True):
        with st.spinner("Generating cohort insights..."):
            cohort_insights = generate_tab_insights(numeric_increase_df, "dollar_increases_cohort", selected_value, selected_time)
            st.write(cohort_insights)

def create_lost_cohort_tab(value_df, selected_value,selected_filter, selected_time):
    """Creates and populates the Lost Cohort Analysis tab"""
    st.write(f"#### Lost {selected_filter} Analysis by Cohort")

    #Get all unique periods as columns
    periods = sorted(value_df.columns.unique())

    #Get first time periods for each customer
    first_periods = {}
    for index in value_df.index:
        first_non_zero = value_df.loc[index].replace(0, np.nan).first_valid_index()
        if first_non_zero:
            first_periods[index] = first_non_zero

    #Create cohort table
    cohort_data = []

    #Group by first period
    for start_period in periods:
        starters = [customer for customer, first_period in first_periods.items() 
                   if first_period == start_period]
        
        if starters:
            row_data = {'First Time Period': start_period}
            
            for period in periods:
                if period >= start_period:
                    if periods.index(period) > 0:
                        prev_period = periods[periods.index(period) - 1]
                        curr_values = value_df.loc[starters, period]
                        prev_values = value_df.loc[starters, prev_period]
                        
                        #Count lost (where value became zero)
                        lost_count = ((prev_values > 0) & (curr_values == 0)).sum()
                        period_number = periods.index(period) - periods.index(start_period) + 1
                        row_data[period] = f"{lost_count} ({period_number})"
                    else:
                        row_data[period] = f"0 (1)"
                else:
                    row_data[period] = ""
            
            cohort_data.append(row_data)

    #Convert to DataFrame
    cohort_df = pd.DataFrame(cohort_data).set_index('First Time Period')

    #Display the cohort table
    st.write(f"##### Lost {selected_filter} by Cohort")
    st.dataframe(cohort_df, use_container_width=True)

    #Create heatmap visualization
    numeric_cohort_df = pd.DataFrame(index=cohort_df.index, columns=periods)
    for start_period in cohort_df.index:
        for period in periods:
            value = cohort_df.loc[start_period, period]
            if value:
                numeric_value = int(value.split()[0])
                numeric_cohort_df.loc[start_period, period] = numeric_value

    fig = px.imshow(
        numeric_cohort_df,
        title=f"Lost {selected_filter} Heatmap",
        labels=dict(x="Time Period", y="First Time Period", color="Lost"),
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig)

    #Generate insights
    with st.expander(f"📊 Lost {selected_filter} Cohort Analysis Insights", expanded=True):
        with st.spinner("Generating cohort insights..."):
            cohort_insights = generate_tab_insights(numeric_cohort_df, "lost_cohort", selected_value, selected_time)
            st.write(cohort_insights)

def create_retention_cohort_tab(value_df, selected_value, selected_filter, selected_time):
    """Creates and populates the Retention Cohort Analysis tab"""
    st.write(f"#### Lost {selected_filter} Retention Analysis by Cohort")

    #Get all unique periods as columns
    periods = sorted(value_df.columns.unique())

    #Get first time periods for each customer
    first_periods = {}
    for index in value_df.index:
        first_non_zero = value_df.loc[index].replace(0, np.nan).first_valid_index()
        if first_non_zero:
            first_periods[index] = first_non_zero

    #Create cohort table
    cohort_data = []

    #Group by first period
    for start_period in periods:
        starters = [customer for customer, first_period in first_periods.items() 
                   if first_period == start_period]
        
        if starters:
            row_data = {'First Time Period': start_period}
            initial_active = (value_df.loc[starters, start_period] > 0).sum()
            
            for period in periods:
                if period >= start_period:
                    curr_active = (value_df.loc[starters, period] > 0).sum()
                    retention_rate = (curr_active / initial_active * 100) if initial_active > 0 else 0
                    period_number = periods.index(period) - periods.index(start_period) + 1
                    row_data[period] = f"{retention_rate:.2f}% ({period_number})"
                else:
                    row_data[period] = ""
            
            cohort_data.append(row_data)

    #Convert to DataFrame
    cohort_df = pd.DataFrame(cohort_data).set_index('First Time Period')

    #Display the cohort table
    st.write(f"##### Lost {selected_filter} Retention Rates by Cohort")
    st.dataframe(cohort_df, use_container_width=True)

    #Create heatmap visualization
    numeric_cohort_df = pd.DataFrame(index=cohort_df.index, columns=periods)
    for start_period in cohort_df.index:
        for period in periods:
            value = cohort_df.loc[start_period, period]
            if value:
                numeric_value = float(value.split('%')[0])
                numeric_cohort_df.loc[start_period, period] = numeric_value

    fig = px.imshow(
        numeric_cohort_df,
        title=f" Lost {selected_filter} Retention Rate Heatmap",
        labels=dict(x="Time Period", y="First Time Period", color="Retention Rate (%)"),
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)

    #Generate insights
    with st.expander(f"📊 Lost {selected_filter} Retention Cohort Analysis Insights", expanded=True):
        with st.spinner("Generating cohort insights..."):
            cohort_insights = generate_tab_insights(numeric_cohort_df, "lost_retention_cohort", selected_value, selected_time)
            st.write(cohort_insights)





def create_analysis_tabs(value_df, total_sum_df, pct_df, avg_df, growth_df, count_df, concentration_df, selected_value, selected_time,selected_filter, analysis_type):
    """Creates and manages all analysis tabs based on analysis type"""
    
    if analysis_type == "Retention Analysis":
        #Only show retention-related tabs
        tab1, tab2, tab3 = st.tabs([ "Snowball Analysis", "Dollar Retention Snowball", "Metrics"
        ])
        
        with tab1:
            create_snowball_tab(value_df, selected_value, selected_time)
        with tab2:
            create_dollar_retention_tab(value_df, selected_value, selected_time)
        with tab3:
            create_metrics_tab(value_df, selected_value, selected_time)
    
    elif analysis_type == "Segmentation Analysis":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Values", "Average", "Percentage", "Percentage YoY Growth","Bridge Analysis", "Count", "Concentration Analysis"
        ])
        
        with tab1:
            create_value_tab(value_df, selected_value, selected_time)
        with tab2:
            create_average_tab(avg_df, selected_value, selected_time)
        with tab3:
            create_percentage_tab(pct_df, selected_value, selected_time)
        with tab4:
            create_growth_tab(growth_df, selected_value, selected_time)
        with tab5:
            create_bridge_tab(value_df, selected_value, selected_time)
        with tab6:
            create_count_tab(count_df, selected_value, selected_time)
        with tab7:
            create_concentration_tab(concentration_df, value_df, selected_value, selected_time)

    elif analysis_type == "Cohort Analysis":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9 = st.tabs([
            "Values", "Count", "Average", "$ Lost", "$ Decreases", "$ Increases", f"Lost {selected_filter}", f"Lost {selected_filter} Retention", "Concentration Analysis"
        ])
        
        with tab1:
            create_values_cohort_tab(value_df, selected_value, selected_time)
        with tab2:
            create_count_cohort_tab(value_df, selected_value, selected_time)
        with tab3:
            create_average_cohort_tab(value_df, selected_value, selected_time)
        with tab4:
            create_lost_dollars_cohort_tab(value_df, selected_value, selected_time)
        with tab5:
            create_dollar_decreases_cohort_tab(value_df, selected_value, selected_time)
        with tab6:
            create_dollar_increases_cohort_tab(value_df, selected_value, selected_time)
        with tab7:
            create_lost_cohort_tab(value_df, selected_value,selected_filter, selected_time)
        with tab8:
            create_retention_cohort_tab(value_df, selected_value,selected_filter, selected_time)
        with tab9:
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

    pass

def add_total_row(df):
    #Add a total sum to the end of a DataFrame
    total_row = df.sum().to_frame().T
    total_row.index = ['Total']
    return pd.concat([df, total_row])
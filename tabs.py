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
    
    st.write("### Summary of Top Customer Concentration")
    st.write(summary_df)
    
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

def create_snowball_tab(value_df, selected_value, selected_time):
    """Creates and populates the Snowball Analysis tab with customer movement metrics"""
    st.write(f"Snowball Analysis of {selected_value}")
    
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
            
            # Calculate metrics
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
    
    # Display the metrics table
    st.write("##### Customer Movement Analysis")
    st.write(metrics_df.set_index('Metric'))
    
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
    st.write(f"Dollar Retention Analysis of {selected_value}")
    
    #Calculate revenue movement metrics for each period
    periods = sorted(value_df.columns)
    metrics_df = pd.DataFrame(columns=['Metric'] + periods)
    
    #Initialize metrics tracking
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
            #First period
            current_values = value_df[period]
            metrics['Beginning Balance'].append(current_values.sum())
            metrics['New Revenue'].append(0)
            metrics['Revenue Increases'].append(0)
            metrics['Lost Revenue'].append(0)
            metrics['Decreases'].append(0)
            metrics['Ending Balance'].append(current_values.sum())
        else:
            prev_period = periods[i-1]
            current_values = value_df[period]
            prev_values = value_df[prev_period]
            
            #Calculate beginning balance
            beginning_balance = prev_values.sum()
            
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
    st.write(metrics_df.set_index('Metric'))
    
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
    st.write(f"Metrics Analysis of {selected_value}")
    
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
    st.dataframe(metrics_df)
    
    #Generate insights
    with st.expander("📊 Metrics Analysis Insights", expanded=True):
        with st.spinner("Generating metrics insights..."):
            metrics_insights = generate_tab_insights(metrics_df, "metrics", selected_value, selected_time)
            st.write(metrics_insights)










def create_bridge_tab(value_df, selected_value, selected_filter):
    """Creates and populates the Bridge Analysis tab"""
    st.write(f"Bridge Analysis of {selected_value}")
    st.write("Bridge Analysis Coming Soon")


def create_values_cohort_tab(value_df, selected_value, selected_time):
    """Analyzes absolute values by cohort"""
    st.write("### Value Analysis by Cohort")   
    #Display raw values
    st.write("### Raw Values by Cohort")
    st.write(value_df)  
    #Create heatmap of values
    fig = px.imshow(
        value_df,
        title="Value Heatmap by Cohort",
        labels=dict(x="Time Period", y="Customer", color="Value"),
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)
    #Calculate and display summary statistics
    summary_stats = pd.DataFrame({
        'Total Value': value_df.sum(),
        'Average Value': value_df.mean(),
        'Median Value': value_df.median(),
        'Active Customers': (value_df > 0).sum()
    })
    st.write("### Summary Statistics by Period")
    st.write(summary_stats)

def create_count_cohort_tab(value_df, selected_value, selected_time):
    """Analyzes customer counts by cohort"""
    st.write("### Customer Count Analysis by Cohort")
    
    #Calculate active customer counts
    active_counts = (value_df > 0).sum()
    total_customers = len(value_df)
    
    #Create summary DataFrame
    count_df = pd.DataFrame({
        'Active Customers': active_counts,
        'Inactive Customers': total_customers - active_counts,
        'Total Customers': total_customers,
        'Active Rate (%)': (active_counts / total_customers * 100).round(2)
    })
    
    st.write("### Customer Counts by Period")
    st.write(count_df)
    
    #Create line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=count_df.index,
        y=count_df['Active Rate (%)'],
        mode='lines+markers',
        name='Active Rate'
    ))
    fig.update_layout(
        title='Customer Activity Rate Over Time',
        xaxis_title='Time Period',
        yaxis_title='Active Rate (%)',
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig)

def create_average_cohort_tab(value_df, selected_value, selected_time):
    """Analyzes average values by cohort"""
    st.write("### Average Value Analysis by Cohort")
    
    #Calculate average values (excluding zeros)
    avg_values = value_df.replace(0, np.nan).mean()
    median_values = value_df.replace(0, np.nan).median()
    
    #Create summary DataFrame
    avg_df = pd.DataFrame({
        'Average Value': avg_values.round(2),
        'Median Value': median_values.round(2)
    })
    
    st.write("### Average Values by Period")
    st.write(avg_df)
    
    #Create line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg_df.index,
        y=avg_df['Average Value'],
        mode='lines+markers',
        name='Average Value'
    ))
    fig.add_trace(go.Scatter(
        x=avg_df.index,
        y=avg_df['Median Value'],
        mode='lines+markers',
        name='Median Value'
    ))
    fig.update_layout(
        title='Value Metrics Over Time',
        xaxis_title='Time Period',
        yaxis_title='Value'
    )
    st.plotly_chart(fig)

def create_dollar_lost_cohort_tab(value_df, selected_value, selected_time):
    """Analyzes dollar value lost by cohort"""
    st.write("### Dollar Value Lost Analysis by Cohort")
    
    #Calculate period-over-period changes
    changes = value_df.diff(axis=1)
    
    #Calculate total dollar value lost (negative changes only)
    dollar_lost = changes.clip(upper=0).abs()
    
    st.write("### Dollar Value Lost by Period")
    st.write(dollar_lost.sum().to_frame('Total Dollar Lost'))
    
    #Create heatmap
    fig = px.imshow(
        dollar_lost,
        title="Dollar Value Lost Heatmap",
        labels=dict(x="Time Period", y="Customer", color="Lost Value"),
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig)

def create_dollar_changes_cohort_tab(value_df, selected_value, selected_time):
    """Analyzes dollar value changes (increases and decreases) by cohort"""
    st.write("### Dollar Value Changes Analysis by Cohort")
    
    #Calculate period-over-period changes
    changes = value_df.diff(axis=1)
    
    #Separate increases and decreases
    increases = changes.clip(lower=0)
    decreases = changes.clip(upper=0).abs()
    
    #Create summary DataFrames
    increase_summary = pd.DataFrame({
        'Total Increases': increases.sum(),
        'Average Increase': increases[increases > 0].mean(),
        'Customers with Increases': (increases > 0).sum()
    })
    
    decrease_summary = pd.DataFrame({
        'Total Decreases': decreases.sum(),
        'Average Decrease': decreases[decreases > 0].mean(),
        'Customers with Decreases': (decreases > 0).sum()
    })
    
    st.write("### Value Increases by Period")
    st.write(increase_summary)
    st.write("### Value Decreases by Period")
    st.write(decrease_summary)
    
    #Create combined visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=increase_summary.index,
        y=increase_summary['Total Increases'],
        name='Increases',
        marker_color='green'
    ))
    fig.add_trace(go.Bar(
        x=decrease_summary.index,
        y=-decrease_summary['Total Decreases'],
        name='Decreases',
        marker_color='red'
    ))
    fig.update_layout(
        title='Dollar Value Changes Over Time',
        barmode='relative',
        xaxis_title='Time Period',
        yaxis_title='Value Change'
    )
    st.plotly_chart(fig)

def create_lost_product_cohort_tab(value_df, selected_value, selected_time):
    """Analyzes lost products/services by cohort"""
    st.write("### Lost Product Analysis by Cohort")
    
    #Calculate products/services lost (where value becomes zero)
    previous_active = value_df.shift(1, axis=1) > 0
    current_inactive = value_df == 0
    lost_products = (previous_active & current_inactive)
    
    lost_counts = lost_products.sum()
    active_counts = (value_df > 0).sum()
    
    #Create summary DataFrame
    summary_df = pd.DataFrame({
        'Lost Products': lost_counts,
        'Active Products': active_counts,
        'Loss Rate (%)': (lost_counts / active_counts.shift(1) * 100).round(2)
    })
    
    st.write("### Lost Products Summary")
    st.write(summary_df)
    
    #Create visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary_df.index,
        y=summary_df['Loss Rate (%)'],
        mode='lines+markers',
        name='Loss Rate'
    ))
    fig.update_layout(
        title='Product Loss Rate Over Time',
        xaxis_title='Time Period',
        yaxis_title='Loss Rate (%)',
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig)

def create_product_retention_cohort_tab(value_df, selected_value, selected_time):
    """Analyzes product/service retention by cohort"""
    st.write("### Product Retention Analysis by Cohort")
    
    #Calculate retention (products/services that remain active)
    initial_products = value_df.iloc[:, 0] > 0
    retention_matrix = pd.DataFrame(index=value_df.columns)
    
    for period in value_df.columns:
        current_active = value_df[period] > 0
        retention_rate = (current_active & initial_products).sum() / initial_products.sum() * 100
        retention_matrix.loc[period, 'Retention Rate (%)'] = retention_rate
    
    st.write("### Product Retention Rates")
    st.write(retention_matrix)
    
    #Create visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=retention_matrix.index,
        y=retention_matrix['Retention Rate (%)'],
        mode='lines+markers',
        name='Retention Rate'
    ))
    fig.update_layout(
        title='Product Retention Rate Over Time',
        xaxis_title='Time Period',
        yaxis_title='Retention Rate (%)',
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig)


def create_analysis_tabs(value_df, total_sum_df, pct_df, avg_df, growth_df, count_df, concentration_df, selected_value, selected_time, analysis_type):
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

    elif analysis_type == "Cohort Analysis":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Values", "Count", "Average", "$ Lost", "$ Changes", "Lost Products", "Product Retention", "Concentration Analysis"
        ])
        
        with tab1:
            create_values_cohort_tab(value_df, selected_value, selected_time)
        with tab2:
            create_count_cohort_tab(value_df, selected_value, selected_time)
        with tab3:
            create_average_cohort_tab(value_df, selected_value, selected_time)
        with tab4:
            create_dollar_lost_cohort_tab(value_df, selected_value, selected_time)
        with tab5:
            create_dollar_changes_cohort_tab(value_df, selected_value, selected_time)
        with tab6:
            create_lost_product_cohort_tab(value_df, selected_value, selected_time)
        with tab7:
            create_product_retention_cohort_tab(value_df, selected_value, selected_time)
        with tab8:
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
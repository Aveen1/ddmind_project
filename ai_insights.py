from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
import streamlit as st
import pandas as pd
import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_dataframe(df, max_tokens=10000):
    """Truncate dataframe preview to fit within token limit."""
    preview = df.head().to_string()
    tokens = num_tokens_from_string(preview)
    
    while tokens > max_tokens and len(df) > 1:
        df = df.iloc[:len(df)//2]
        preview = df.head().to_string()
        tokens = num_tokens_from_string(preview)
    
    return preview

def analyze_data_with_langchain(df):
    """Uses LangChain to structure the ChatGPT prompt and retrieve recommendations."""
    
    MAX_TOTAL_TOKENS = 100000000 
    MAX_DATASET_TOKENS = 10000
    
    dataset_preview = truncate_dataframe(df, MAX_DATASET_TOKENS)
    
    prompt_template = (
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
        "Make sure to add Cohort, Retention, Segmentation Analysis in the analysis list, focusing on these aspects if they provide meaningful insights given the specific dataset and business questions at hand."
        "Put each recommendation on a new line and elaborate the recommendations."
    )

    prompt = prompt_template.format(
        dataset_preview=dataset_preview,
        columns=df.columns.tolist()
    )


    system_message = "You are an expert data analyst. Always provide responses in valid JSON format."
    total_tokens = num_tokens_from_string(system_message) + num_tokens_from_string(prompt)
    
    if total_tokens > MAX_TOTAL_TOKENS:
        st.warning(f"Input size exceeds token limit ({total_tokens}/{MAX_TOTAL_TOKENS})")
        return {
            "analysis_types": ["Basic Analysis"],
            "filters": [],
            "value_columns": [],
            "time_periods": ["Monthly", "Quarterly", "Yearly"],
            "date_columns": [],
            "recommendations": "Dataset too large for detailed analysis. Please reduce the input size."
        }

    try:
        chat = ChatOpenAI(model="gpt-4", temperature=0)
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        response = chat(messages)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            cleaned_response = response.content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            return json.loads(cleaned_response)

    except Exception as e:
        st.error(f"Analysis generation error: {e}")
        return {
            "analysis_types": ["Basic Analysis"],
            "filters": [],
            "value_columns": [],
            "time_periods": ["Monthly", "Quarterly", "Yearly"],
            "date_columns": [],
            "recommendations": "Unable to generate detailed recommendations."
        }

def generate_tab_insights(df, analysis_type, selected_value, selected_filter):
    """Generate GPT insights for specific analysis tab."""
    try:
        MAX_TOKENS = 100000000
        df_preview = truncate_dataframe(df, MAX_TOKENS // 2) 
        
        chat = ChatOpenAI(model="gpt-4o", temperature=0)
        
        prompts = {
                    #Segmentation insights
                    "value": f"""
                Perform a detailed analysis of {selected_value} across {selected_filter} categories:
                1. Identify top 3-5 performing categories and their contribution to total
                2. Detect seasonal patterns and year-over-year trends
                3. Compare performance across different segments
                4. Calculate growth rates for each category
                5. Highlight significant deviations from historical patterns

                Example: For Monthly Revenue across Customer Segments:
                - Enterprise segment leads with 45 percent of total revenue
                - SMB segment shows 20 percent YoY growth
                - Seasonal peaks in Q4 across all segments
                    """,

                    "total_sum": f"""
                Analyze total sum trends for {selected_value} focusing on:
                1. Calculate overall growth rate (CAGR) and period-over-period changes
                2. Identify peak periods and their drivers
                3. Break down contributions by segment/category
                4. Compare against industry benchmarks
                5. Project future trends based on historical patterns

                Example: Total Revenue Sum Analysis:
                - 25 percent CAGR over past 3 years
                - Q4 represents 40 percent of annual total
                - Top 3 segments contribute 80 percent of total
                    """,

                    "percentage": f"""
                Analyze percentage distribution of {selected_value} examining:
                1. Calculate relative share for each category
                2. Track distribution changes over time
                3. Identify categories gaining/losing share
                4. Measure concentration and diversity metrics
                5. Compare against target distributions

                Example: Revenue Distribution Analysis:
                - Product A share increased from 20 percent to 35 percent
                - Top 5 categories represent 85 percent of total
                - New categories growing at 3x market rate
                    """,

                    "average": f"""
                Analyze average {selected_value} trends with focus on:
                1. Calculate mean values across categories
                2. Identify standard deviations and outliers
                3. Compare averages across time periods
                4. Break down by segment/category
                5. Highlight significant variations

                Example: Average Order Value Analysis:
                - Mean order value increased 15 percent YoY
                - Enterprise segment 2.5x above average
                - Significant variation across regions
                    """,

                    "growth": f"""
                Analyze growth patterns in {selected_value} considering:
                1. Calculate growth rates by category/segment
                2. Identify highest and lowest growth areas
                3. Compare against historical growth rates
                4. Detect acceleration/deceleration patterns
                5. Project future growth trajectories

                Example: Growth Rate Analysis:
                - Overall growth rate of 35 percent YoY
                - New products growing 2x faster than mature
                - Growth acceleration in emerging markets
                    """,

                    "count": f"""
                Analyze count distribution of {selected_value} examining:
                1. Calculate frequency distributions
                2. Identify most/least common categories
                3. Track count changes over time
                4. Compare against expected distributions
                5. Highlight unusual patterns

                Example: Transaction Count Analysis:
                - 50 percent increase in daily transaction volume
                - Peak activity during morning hours
                - Weekend counts 30 percent lower than weekdays
                    """,

                    "concentration": f"""
                Analyze concentration of {selected_value} focusing on:
                1. Calculate concentration ratios
                2. Identify high-density areas/segments
                3. Track concentration changes over time
                4. Assess risk of concentration
                5. Compare against desired distribution

                Example: Customer Concentration Analysis:
                - Top 10 customers represent 40 percent of revenue
                - Concentration decreased 5 percent YoY
                - Geographic concentration in three regions
                    """,

                    #Retention insights
                    "snowball": f"""
                Analyze snowball effect of {selected_value} examining:
                1. Identify compound growth patterns
                2. Calculate network effects metrics
                3. Track viral coefficient
                4. Measure expansion rates
                5. Project future growth potential

                Example: User Growth Analysis:
                - Viral coefficient of 1.3
                - Each user brings 2.5 new users annually
                - 40percent faster growth in referral channels
                    """,

                    "dollar_retention": f"""
                Analyze dollar retention rates for {selected_value} focusing on:
                1. Calculate gross and net dollar retention
                2. Break down by customer segment
                3. Track changes in retention rates
                4. Identify expansion opportunities
                5. Compare against industry benchmarks

                Example: Dollar Retention Analysis:
                - 120 percent net dollar retention
                - Enterprise segment at 135 percent NDR
                - 25 percent expansion in existing accounts
                    """,

                    "metrics": f"""
                Analyze key metrics for {selected_value} examining:
                1. Track core KPI performance
                2. Compare against targets
                3. Identify leading indicators
                4. Calculate correlation between metrics
                5. Highlight areas needing attention

                Example: Key Metrics Analysis:
                - CAC decreased 15 percent while LTV increased 20 percent
                - Engagement metrics up 30 percent YoY
                - Strong correlation between usage and retention
                    """,

                    #Cohort insights
                    "values_cohort": f"""
                Analyze cohort behavior for {selected_value} focusing on:
                1. Compare cohort performance over time
                2. Calculate cohort-specific metrics
                3. Identify best/worst performing cohorts
                4. Track cohort maturation patterns
                5. Project lifetime value by cohort

                Example: Value Cohort Analysis:
                - 2023 Q4 cohort 40 percent stronger than average
                - Early adoption predicts 2x lifetime value
                - Cohort quality improving over time
                    """,

                    "count_cohort": f"""
                Analyze cohort count distribution for {selected_value} examining:
                1. Track cohort size changes
                2. Calculate retention by cohort size
                3. Compare acquisition patterns
                4. Identify optimal cohort size
                5. Measure cohort stability

                Example: Count Cohort Analysis:
                - Larger cohorts retain 15 percent better
                - Optimal cohort size of 500-1000 users
                - Acquisition efficiency improving by 20 percent
                    """,

                    "average_cohort": f"""
                Analyze cohort averages for {selected_value} focusing on:
                1. Calculate mean values by cohort
                2. Track average value changes over time
                3. Compare cohort performance
                4. Identify high-value cohorts
                5. Project future cohort value

                Example: Average Cohort Analysis:
                - Recent cohorts show 25 percent higher average value
                - Value maturation occurs in months 3-6
                - Consistent growth across cohorts
                    """,

                    "lost_dollars_cohort": f"""
                Analyze lost revenue patterns in {selected_value} examining:
                1. Calculate churn impact by cohort
                2. Identify high-risk periods
                3. Track recovery effectiveness
                4. Compare loss patterns across segments
                5. Recommend prevention strategies

                Example: Lost Dollar Analysis:
                - 70 percent of losses occur in months 1-3
                - Recovery rate of 25 percent for at-risk accounts
                - Proactive engagement reduces losses by 40 percent
                    """,

                    "dollar_decreases_cohort": f"""
                Analyze revenue decrease patterns for {selected_value} focusing on:
                1. Track magnitude of decreases
                2. Identify common decrease triggers
                3. Calculate impact by segment
                4. Compare against normal variation
                5. Develop early warning indicators

                Example: Revenue Decrease Analysis:
                - Average decrease of 15 percent in affected accounts
                - Usage drops precede 80 percent of decreases
                - Early intervention success rate of 60 percent
                    """,

                    "dollar_increases_cohort": f"""
                Analyze revenue increase patterns for {selected_value} examining:
                1. Track expansion patterns
                2. Identify growth triggers
                3. Calculate upgrade rates
                4. Compare against expansion targets
                5. Develop growth strategies

                Example: Revenue Increase Analysis:
                - 35 percent of accounts expand within 6 months
                - Feature adoption drives 70 percent of increases
                - Proactive outreach yields 2x expansion rate
                    """,

                    "lost_cohort": f"""
                Analyze lost customer patterns for {selected_value} focusing on:
                1. Calculate churn rates by cohort
                2. Identify churn predictors
                3. Track win-back success
                4. Compare against retention targets
                5. Develop retention strategies

                Example: Lost Customer Analysis:
                - Churn rate 20 percent lower in engaged segments
                - 40 percent of churned customers return within 1 year
                - Early warning system prevents 30 percent of churn
                    """,

                    "lost_retention_cohort": f"""
                Analyze product retention for {selected_value} examining:
                1. Calculate product-specific retention
                2. Identify critical usage periods
                3. Track feature adoption impact
                4. Compare across product tiers
                5. Develop engagement strategies

                Example: Product Retention Analysis:
                - 85 percent retention in premium tier
                - Feature adoption drives 2x retention
                - First 30 days critical for long-term retention
                    """
                }
        
        prompt = f"{prompts[analysis_type.lower()]} Data: {df_preview}"
        
        system_message = "You are a data analysis expert. Provide clear, concise insights in bullet points."
        total_tokens = num_tokens_from_string(system_message) + num_tokens_from_string(prompt)
        
        if total_tokens > MAX_TOKENS:
            return "Dataset too large for detailed analysis. Please reduce the input size or analyze a smaller subset of the data."
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        
        response = chat(messages)
        return response.content
    except Exception as e:
        return f"Error generating insights: {e}"

def generate_recommendations_from_file(file_content):
    """Send analysis result file content to GPT for generating recommendations."""
    try:
        MAX_TOKENS = 100000000        
        content_tokens = num_tokens_from_string(file_content)
        if content_tokens > MAX_TOKENS // 2:
            encoding = tiktoken.get_encoding('cl100k_base')
            decoded_tokens = encoding.decode(encoding.encode(file_content)[:MAX_TOKENS // 2])
            file_content = decoded_tokens
        
        chat = ChatOpenAI(model="gpt-4o", temperature=0)
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
        
        system_message = "You are an expert data analysis expert."
        total_tokens = num_tokens_from_string(system_message) + num_tokens_from_string(prompt) + num_tokens_from_string(file_content)
        
        if total_tokens > MAX_TOKENS:
            return "Input size exceeds token limit. Please reduce the file content size."
            
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt + f"\n\n{file_content}")
        ]
        response = chat(messages)
        return response.content
    except Exception as e:
        return f"Error generating recommendations: {e}"
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
import streamlit as st
import pandas as pd

def analyze_data_with_langchain(df):
    """Uses LangChain to structure the ChatGPT prompt and retrieve recommendations."""
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
        dataset_preview=df.head().to_string(),
        columns=df.columns.tolist()
    )

    try:
        chat = ChatOpenAI(model="gpt-4", temperature=0)
        messages = [
            SystemMessage(content="You are an expert data analyst. Always provide responses in valid JSON format."),
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
        chat = ChatOpenAI(model="gpt-4o", temperature=0)
        
        prompts = {
            "value": f"Analyze {selected_value} values across {selected_filter} categories, identifying top performers, trends, and patterns.",
            "total_sum": f"Analyze total sum trends for {selected_value}, describing overall trend and growth rate.",
            "percentage": f"Analyze percentage distribution of {selected_value}, identifying dominant categories and distribution shifts.",
            "average": f"Analyze average {selected_value} trends, comparing across categories and identifying outliers.",
            "growth": f"Analyze growth patterns in {selected_value}, identifying highest growth rates and patterns.",
            "count": f"Analyze count distribution of {selected_value}, identifying most frequent categories and patterns.",
            "concentration": f"Analyze concentration of {selected_value}, identifying concentrated areas and changes."
        }
        
        messages = [
            SystemMessage(content="You are a data analysis expert. Provide clear, concise insights in bullet points."),
            HumanMessage(content=f"{prompts[analysis_type.lower()]} Data: {df.to_string()}")
        ]
        
        response = chat(messages)
        return response.content
    except Exception as e:
        return f"Error generating insights: {e}"

def generate_recommendations_from_file(file_content):
    """Send analysis result file content to GPT for generating recommendations."""
    try:
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
        messages = [
            SystemMessage(content="You are a data analysis expert."),
            HumanMessage(content=prompt + f"\n\n{file_content}")
        ]
        response = chat(messages)
        return response.content
    except Exception as e:
        return f"Error generating recommendations: {e}"
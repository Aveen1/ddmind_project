import streamlit as st
import pandas as pd
import openai
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json

#Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_top_columns(file, num_columns=5):
    """Extract columns from an uploaded Excel file."""
    try:
        df = pd.read_excel(file)
        return df.iloc[:, :num_columns]  #Select the first 'num_columns' columns
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return None

def get_chatgpt_analysis_recommendations(df):
    """Send the entire dataset to GPT-4 and get analysis recommendations."""
    prompt = (
        "List all potential analyses and Make a table with the following columns:\n"
        "1) Analysis Number\n"
        "2) Analyses Title\n"
        "3) Short Description\n"
        "4) Recommended independent variables.\n"
        "5) Recommended dependent variables\n"
        "6) Time Period to select (monthly, quarterly, yearly).\n\n"
        f"The dataset structure is:\n{df.head(5).to_string()}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        gpt_response = response["choices"][0]["message"]["content"].strip()

        #Attempt to parse JSON from the GPT response
        try:
            analyses = pd.read_json(gpt_response)
            st.dataframe(analyses)
            return analyses
            
        except ValueError:
            st.warning("GPT response is not valid JSON. Attempting to parse as structured text.")

            #Parse structured table-like response
            rows = [row.split("\t") for row in gpt_response.split("\n") if row.strip()]
            df = pd.DataFrame(rows[1:], columns=rows[0])  #Use first row as header
            st.dataframe(df)
            return df
        
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return None

def analyze_columns(df):
    """Analyze the columns in the DataFrame."""
    return list(df.columns)

#Function for Regression Analysis
def perform_regression_analysis(df, independent_var, dependent_var):
    """Perform regression analysis."""
    try:
        X = df[independent_var]
        y = df[dependent_var]

        if X.ndim == 1:
            X = sm.add_constant(X)  #Add a constant term for the intercept

        model = sm.OLS(y, X).fit()
        st.write("### Regression Summary:")
        st.text(model.summary())
    except Exception as e:
        st.error(f"Error performing regression analysis: {e}")

#Function for Time Series Analysis
def perform_time_series_analysis(df, dependent_var, period):
    """Perform time series analysis."""
    try:
        if 'Year' not in df.columns:
            st.error("The dataset must contain a 'Year' column for time series analysis.")
            return

        # Convert 'Year' to datetime format
        df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
        df = df.dropna(subset=['Year'])
        df = df.set_index('Year')

        if period == "Yearly":
            ts_data = df[dependent_var].resample('A').mean()
        elif period == "Quarterly":
            ts_data = df[dependent_var].resample('Q').mean()
        elif period == "Monthly":
            ts_data = df[dependent_var].resample('M').mean()
        else:
            st.error("Invalid period selected.")
            return

        st.write(f"### {period} Time Series Data:")
        st.line_chart(ts_data)
    except Exception as e:
        st.error(f"Error performing time series analysis: {e}")

def main():
    st.title("Excel Analysis Assistant with ChatGPT")
    st.write("Upload an Excel file, and we'll help you figure out the analyses that can be done based on the sample data.")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("### Extracted Data:")
        st.dataframe(df)

        columns = analyze_columns(df)

        #ChatGPT Button
        if st.button("Get ChatGPT Analysis Recommendations"):
            st.write("Getting ChatGPT recommendations based on the uploaded dataset...")
            recommendations = get_chatgpt_analysis_recommendations(df)
            if recommendations is not None:
                st.write("### ChatGPT Recommendations:")
                st.dataframe(recommendations)
                

                
        
        #Step 2: Ask for Independent and Dependent Variables
        st.write("### Select Values:")
        independent_var = st.selectbox("Select Filter", columns)
        dependent_var = st.selectbox("Select Value", columns)

        #Step 3: Ask for Time Periods
        period = st.selectbox("Select Time Period:", ["Yearly", "Quarterly", "Monthly"])

        #Step 4: Generate Analysis Options
        analysis_options = ["Regression Analysis", "Time Series Analysis", "Correlation Analysis"]
        analysis_choice = st.selectbox("Select Analysis Type:", analysis_options)

        #Analysis Button
        if st.button("Run Analysis"):
            if analysis_choice == "Regression Analysis":
                st.write(f"Running Regression Analysis with {independent_var} as independent and {dependent_var} as dependent variables...")
                perform_regression_analysis(df, independent_var, dependent_var)

            elif analysis_choice == "Time Series Analysis":
                st.write(f"Running Time Series Analysis with {dependent_var} over {period} periods...")
                perform_time_series_analysis(df, dependent_var, period)

            elif analysis_choice == "Correlation Analysis":
                st.write(f"Calculating Correlation between {independent_var} and {dependent_var}...")
                correlation = df[independent_var].corr(df[dependent_var])
                st.write(f"Correlation: {correlation}")

if __name__ == "__main__":
    main()

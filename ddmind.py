import streamlit as st
import pandas as pd
import openai
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_top_columns(file, num_columns=5):
    """Extract columns from an uploaded Excel file."""
    try:
        df = pd.read_excel(file)
        return df.iloc[:, :num_columns]  #Select the first 'num_columns' columns
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return None

def get_chatgpt_analysis_recommendations(independent_var, dependent_var):
    """Send selected independent and dependent variables to GPT-4o and get analysis recommendations."""
    prompt = (
        f"I have selected the following variables for analysis:\n"
        f"Independent Variable: {independent_var}\n"
        f"Dependent Variable: {dependent_var}\n\n"
        "What analyses would you recommend or suggest based on these variables?"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  #Use your specific model
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,  #Optional, adjust creativity level
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        return None

def analyze_columns(df):
    """Analyze the columns in the DataFrame"""
    columns = list(df.columns)
    return columns

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

        #Convert 'Year' to datetime format
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

        #Step 2: Ask for Independent Variables
        st.write("### Select Variables:")
        independent_var = st.selectbox("Select Independent Variable:", columns)
        dependent_var = st.selectbox("Select Dependent Variable:", columns)

        #Step 3: Ask for Time Periods
        period = st.selectbox("Select Time Period:", ["Yearly", "Quarterly", "Monthly"])

        #Step 4: Generate Analysis Options
        analysis_options = ["Regression Analysis", "Time Series Analysis", "Correlation Analysis"]
        analysis_choice = st.selectbox("Select Analysis Type:", analysis_options)

        if st.button("Run Analysis"):
            if analysis_choice == "Regression Analysis":
                st.write(f"Running Regression Analysis with {independent_var} as independent and {dependent_var} as dependent variables...")
                #regression logic
                perform_regression_analysis(df, independent_var, dependent_var)

            elif analysis_choice == "Time Series Analysis":
                st.write(f"Running Time Series Analysis with {dependent_var} over {period} periods...")
                #time series logic
                perform_time_series_analysis(df, dependent_var, period)


            elif analysis_choice == "Correlation Analysis":
                st.write(f"Calculating Correlation between {independent_var} and {dependent_var}...")
                correlation = df[independent_var].corr(df[dependent_var])
                st.write(f"Correlation: {correlation}")

        if st.button("Get ChatGPT Analysis Recommendations"):
            st.write(f"Getting ChatGPT recommendations for Independent Variable: {independent_var} and Dependent Variable: {dependent_var}...")
            recommendations = get_chatgpt_analysis_recommendations(independent_var, dependent_var)
            if recommendations:
                st.write("### ChatGPT Recommendations:")
                st.write(recommendations)

if __name__ == "__main__":
    main()

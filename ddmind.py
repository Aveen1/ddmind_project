import streamlit as st
import pandas as pd
import openai
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_top_columns(file, num_columns):
    """Extract columns from an uploaded Excel file."""
    try:
        df = pd.read_excel(file)
        return df.iloc[:, :num_columns]  # Select the first 'num_columns' columns
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return None

def preprocess_data(df):
    """Preprocess and clean the extracted data."""
    try:
        steps = {
            "Remove duplicate rows": lambda df: df.drop_duplicates(),
            "Remove null values": lambda df: df.dropna(),
            "Handle date columns": lambda df: handle_date_columns(df),
            #"Standardize column names": lambda df: standardize_column_names(df),
            #"Ensure 'Year' column is valid": lambda df: validate_year_column(df),
            #"Validate and correct data types": lambda df: validate_data_types(df),
            "Remove low-variance columns": lambda df: remove_low_variance_columns(df),
            #"Reset index": lambda df: df.reset_index(drop=True)
        }

        st.write("### Data Cleaning Steps:")
        for step, func in steps.items():
            if st.checkbox(step, value=True):
                df = func(df)
                #st.success(f"{step} completed.")

        st.success("All selected data cleaning steps completed successfully!")

        #Display the cleaned dataset
        st.write("### Cleaned Data:")
        st.dataframe(df)

        return df

    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return None

def validate_year_column(df):
    """Ensure 'Year' column is valid and properly formatted."""
    if 'Year' in df.columns:
        #Remove non-numeric characters and standardize format
        df['Year'] = df['Year'].astype(str).str.replace(r"[^\d]", "", regex=True)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
    else:
        st.error("The dataset must contain a 'Year' column.")
    return df



def handle_date_columns(df):
    """Convert and fix date formats."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df.dropna()

def standardize_column_names(df):
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def validate_data_types(df):
    """Validate and correct data types of columns."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                pass
    return df.dropna()

def remove_low_variance_columns(df):
    """Remove columns with low variance."""
    low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    return df.drop(columns=low_variance_cols)

#Function for ChatGPT Recommendations
def get_chatgpt_analysis_recommendations(df):
    """Send the entire dataset to GPT-4 and get analysis recommendations."""
    prompt = (
        "List all potential analyses (don't use sterics or dash, just highlight the title) with all this information including: \n"
        "Analysis Title\n"
        "Short Description\n"
        "Recommended Filter\n"
        "Recommended Value\n"
        "Time Period to select (monthly, quarterly, yearly).\n\n"
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

        #Print GPT response as plain text
        st.write("### DDMind Analysis Recommendations:")
        st.text(gpt_response)

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")

def analyze_columns(df):
    """Analyze the columns in the DataFrame."""
    return list(df.columns)

#Function for Segment Analysis
def perform_segment_analysis(df, filter_column, value_column):
    try:
        segment_summary = df.groupby(filter_column)[value_column].mean().reset_index()
        st.write(f"### Segment Analysis ({filter_column}):")
        st.dataframe(segment_summary)
        st.bar_chart(segment_summary.set_index(filter_column))
    except Exception as e:
        st.error(f"Error performing segment analysis: {e}")

#Function for Retention Analysis
def perform_retention_analysis(df, filter_column, value_column):
    
    try:
        #Convert the filter column to datetime if it's Year and Month
        if 'Year' in df.columns and 'Month' in df.columns:
            df[filter_column] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
        else:
            df[filter_column] = pd.to_datetime(df[filter_column])

        #Calculate cohort and retention periods
        df['cohort'] = df.groupby(value_column)[filter_column].transform('min')
        df['period'] = ((df[filter_column] - df['cohort']) / pd.Timedelta(days=30)).apply(lambda x: int(x))

        #Generate the retention table
        retention = df.pivot_table(index='cohort', columns='period', values=value_column, aggfunc='nunique').fillna(0)
        retention_percentage = retention.div(retention.iloc[:, 0], axis=0)

        #Display the results in Streamlit
        st.write("### Retention Analysis:")
        st.dataframe(retention_percentage)

        #Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Retention Analysis")
        sns.heatmap(retention_percentage, annot=True, fmt=".0%", cmap="Blues", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error performing retention analysis: {e}")


#Function for Regression Analysis
def perform_regression_analysis(df, independent_var, dependent_var):
    """Perform regression analysis."""
    try:
        X = df[independent_var]
        y = df[dependent_var]

        if X.ndim == 1:
            X = sm.add_constant(X)  # Add a constant term for the intercept

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
    st.title("Data Analysis with DDMind")
    st.write("Upload an Excel file, and we'll help you figure out the analyses that can be done based on the sample data.")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("### Extracted Data:")
        st.dataframe(df)

        # Preprocess the data
        df = preprocess_data(df)

        if df is not None:
            columns = analyze_columns(df)

            # ChatGPT Button
            if st.button("Get DDMind Analysis Recommendations"):
                st.write("Getting DDMind recommendations based on the extracted dataset...")
                get_chatgpt_analysis_recommendations(df)

            # Step 2: Ask for Independent and Dependent Variables
            st.write("### Select Variables:")
            filter_column = st.selectbox("Select Filter", columns)
            value_column = st.selectbox("Select Value", columns)

            # Step 3: Ask for Time Periods
            period = st.selectbox("Select Time Period:", ["Yearly", "Quarterly", "Monthly"])

            #Step 4: Generate Analysis Options
            analysis_options = ["Segment Analysis","Retention Analysis","Regression Analysis", "Correlation Analysis"]
            analysis_choice = st.selectbox("Select Analysis Type:", analysis_options)

            # Analysis Button
            if st.button("Run Analysis"):
                if analysis_choice == "Segment Analysis":
                    perform_segment_analysis(df, filter_column, value_column)

                elif analysis_choice == "Retention Analysis":
                    perform_retention_analysis(df, filter_column, value_column)

                elif analysis_choice == "Regression Analysis":
                    perform_regression_analysis(df, filter_column, value_column)

                elif analysis_choice == "Correlation Analysis":
                    correlation = df[filter_column].corr(df[value_column])
                    st.write(f"Correlation: {correlation}")

if __name__ == "__main__":
    main()

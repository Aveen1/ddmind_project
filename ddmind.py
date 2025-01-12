import streamlit as st
import pandas as pd
import openai

#Set up OpenAI API key
openai.api_key = "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"

def extract_top_columns(file, num_columns=5):
    "Extract top columns from an uploaded Excel file"
    try:
        df = pd.read_excel(file)
        return df.iloc[:, :num_columns]  #Select the first 'num_columns' columns
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return None

def get_chatgpt_analysis_recommendations(sample_data):
    "Send sample data to ChatGPT and get analysis recommendations"
    prompt = (
        "Here is a sample of data from an Excel sheet:\n"
        f"{sample_data.to_string(index=False)}\n\n"
        "Based on this sample, what analyses would you recommend or suggest that can be done?"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data analyst"},
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Error interacting with ChatGPT: {e}")
        return None

def main():
    st.title("Excel Analysis Assistant with ChatGPT")
    st.write("Upload an Excel file, and we'll help you figure out the analyses that can be done based on the sample data.")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file:
        num_columns = st.slider("Number of Top Columns to Extract", min_value=1, max_value=10, value=5)
        sample_data = extract_top_columns(uploaded_file, num_columns=num_columns)

        if sample_data is not None:
            st.write("### Extracted Sample Data:")
            st.dataframe(sample_data)

            if st.button("Ask ChatGPT for Analysis Suggestions"):
                suggestions = get_chatgpt_analysis_recommendations(sample_data)

                if suggestions:
                    st.write("### Recommended Analyses:")
                    st.text(suggestions)

if __name__ == "__main__":
    main()

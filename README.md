# DDMind Project

## 📊 Overview

**DDMind** is a data analysis and visualization tool designed to process complex datasets and generate insightful visualizations. It includes AI-driven insights, data cleaning processes, and flexible chart generation.

## 🚀 Features

- **AI Insights**: Leverages machine learning to detect patterns and trends.
- **Data Cleaning**: Handles missing values, outliers, and data inconsistencies.
- **Data Processing**: Transforms raw data into structured formats.
- **Chart Generation**: Produces dynamic visualizations for better data interpretation.
- **Backup System**: Ensures data integrity with automatic backups.

## 🏗️ Project Structure

```
ddmind_project/
├── .devcontainer/          # Development container setup
├── __pycache__/            # Python cache files
├── graphs/                 # Generated charts and graphs
├── ai_insights.py          # AI-driven data analysis
├── analysis.txt            # Analysis notes and logs
├── backup.py               # Backup scripts
├── chart_generation.py     # Chart and graph creation
├── check.py                # Validation checks
├── data_analysis.py        # Core data analysis logic
├── data_cleaning.py        # Data preprocessing and cleaning
├── data_processing.py      # Data transformation scripts
├── ddmind.py               # Main entry point of the app
├── requirements.txt        # Python dependencies
├── tabs.py                 # Tab management for UI
└── .gitignore              # Ignored files and directories
```

## 🏃‍♀️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aveen1/ddmind_project.git
cd ddmind_project
```

2. **Set up a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scriptsctivate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🎯 Usage

Run the main script:
```bash
python ddmind.py
```
Ensure your data files are placed in the appropriate directory before running the application.

## 📈 Versioning

We use **Git tags** to track stable versions.

- **List all tags**:
```bash
git tag
```
- **Checkout a specific version**:
```bash
git checkout tags/<tag_name>
```

Refer to [CHANGELOG.md](./CHANGELOG.md) for detailed version history.

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit pull requests.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

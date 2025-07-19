# Gemini Pro Financial Decoder

An AI-powered financial analysis tool that leverages Google's Gemini Pro model to analyze and summarize financial statements. Upload your financial documents and get instant AI-generated insights along with interactive visualizations.

---

## Features

- **Multi-Document Analysis**: Upload and analyze three key financial statements:
  - Balance Sheet
  - Profit & Loss Statement
  - Cash Flow Statement
- **Flexible File Support**: Compatible with both CSV and Excel (.xlsx) file formats
- **AI-Powered Insights**: Uses Gemini Pro 1.5 via LangChain for generating detailed financial summaries
- **Interactive Visualizations**: 
  - Automatic line charts for numerical data trends
  - Detailed data tables for comprehensive review
- **Secure Configuration**: Implements environment variable-based API key management
- **Project Documentation**: Complete documentation including design thinking, requirements analysis, and implementation details available in `project_documentation/`

---

## Tech Stack

- **Core Language**: Python
- **Web Framework**: Streamlit
- **AI & ML**:
  - Google Generative AI (Gemini Pro 1.5)
  - LangChain integration
- **Data Processing**: 
  - Pandas for data handling
  - Streamlit's built-in visualization tools
- **Configuration**: python-dotenv for environment management

---

## Project Structure

```
├── main.py                 # Main application file
├── requirements.txt        # Project dependencies
└── project_documentation/  # Comprehensive documentation
    ├── Introduction/      # Project overview and planning
    ├── Design Thinking/   # Brainstorming and ideation
    ├── Literature Survey/ # Research background
    ├── Project Design/    # Technical architecture
    └── Requirement Analysis/ # Functional & non-functional requirements
```

---

## Setup

1. Create a `.env` file in the root directory
2. Add your Gemini API key:
   ```
   GEMINI_KEY=your_api_key_here
   ```
3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   streamlit run main.py
   ```

---

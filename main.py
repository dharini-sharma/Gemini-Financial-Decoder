import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

# Load environment variables
load_dotenv()

# Configure API
api_key = os.getenv("GEMINI_KEY")
genai.configure(api_key=api_key)

# Initialize LLM
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key, temperature=0.7)

# PYDANTIC MODELS FOR STRUCTURED OUTPUT

class FinancialRatio(BaseModel):
    name: str = Field(description="Name of the financial ratio")
    value: str = Field(description="Calculated value of the ratio (can be numeric or descriptive)")
    interpretation: str = Field(description="What this ratio means")
    benchmark: str = Field(description="Industry benchmark or ideal range")

class FinancialAnalysis(BaseModel):
    summary: str = Field(description="Executive summary of financial health")
    key_ratios: List[FinancialRatio] = Field(description="List of calculated financial ratios")
    strengths: List[str] = Field(description="Key financial strengths")
    concerns: List[str] = Field(description="Areas of concern")
    recommendations: List[str] = Field(description="Specific recommendations")
    risk_level: str = Field(description="Overall risk assessment: Low/Medium/High")

# Create parsers
balance_sheet_parser = PydanticOutputParser(pydantic_object=FinancialAnalysis)
profit_loss_parser = PydanticOutputParser(pydantic_object=FinancialAnalysis)
cash_flow_parser = PydanticOutputParser(pydantic_object=FinancialAnalysis)

# ENHANCED PROMPT TEMPLATES
prompt_templates = {
    "balance_sheet": PromptTemplate(    
        input_variables=["balance_sheet_data"],
        template="""Analyze this balance sheet data: {balance_sheet_data}
        
        Provide a comprehensive financial analysis including:
        1. Key liquidity ratios (current ratio, quick ratio) with calculations
        2. Debt-to-equity ratio with industry context
        3. Working capital analysis
        4. Asset composition insights
        5. Financial health assessment with specific recommendations
        
        {format_instructions}
        
        Ensure all ratio calculations are precise and provide specific numerical values where possible.
        If exact calculations cannot be made due to data limitations, provide qualitative assessments.
        """,
        partial_variables={"format_instructions": balance_sheet_parser.get_format_instructions()}
    ),

    "profit_loss": PromptTemplate(
        input_variables=["profit_loss_data"],
        template="""Analyze this profit and loss statement: {profit_loss_data}
        
        Provide detailed profitability analysis including:
        1. Profitability ratios (gross margin %, operating margin %, net margin %)
        2. Revenue growth trends with percentage calculations
        3. Cost structure breakdown and efficiency metrics
        4. EBITDA analysis and operational insights
        5. Performance comparison and optimization recommendations
        
        {format_instructions}
        
        Focus on actionable insights and provide specific recommendations for improvement.
        """,
        partial_variables={"format_instructions": profit_loss_parser.get_format_instructions()}
    ),

    "cash_flow": PromptTemplate(
        input_variables=["cash_flow_data"],
        template="""Analyze this cash flow statement: {cash_flow_data}
        
        Provide comprehensive cash flow analysis including:
        1. Operating cash flow efficiency and conversion ratios
        2. Free cash flow calculation and sustainability
        3. Working capital impact and optimization opportunities
        4. Investment and financing activities analysis
        5. Liquidity position and cash management recommendations
        
        {format_instructions}
        
        Emphasize cash generation capabilities and financial sustainability.
        """,
        partial_variables={"format_instructions": cash_flow_parser.get_format_instructions()}
    )
}

# FILE HANDLING FUNCTIONS

def upload_file():
    """Function to upload files using Streamlit"""
    st.markdown("#Upload Financial Statements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        balance_sheet = st.file_uploader("Balance Sheet", type=["csv", "xlsx"])
    with col2:
        profit_loss = st.file_uploader("Profit & Loss Statement", type=["csv", "xlsx"])
    with col3:
        cash_flow = st.file_uploader("Cash Flow Statement", type=["csv", "xlsx"])

    return balance_sheet, profit_loss, cash_flow

def load_file(file):
    """Load uploaded file into pandas DataFrame"""
    if file is not None:
        try:
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                return pd.read_excel(file)
        except Exception as e:
            st.error(f"Error loading file {file.name}: {str(e)}")
            return None
    return None

# ANALYSIS FUNCTIONS

def generate_structured_summary(prompt_type, data):  
    """Generate structured financial analysis using LangChain parsers"""
    if data is None or data.empty:
        return None
    
    try:
        # Convert data to a more readable format
        data_summary = data.describe().to_dict() if not data.empty else data.to_dict()
        
        # Get the appropriate parser
        parser_map = {
            "balance_sheet": balance_sheet_parser,
            "profit_loss": profit_loss_parser,
            "cash_flow": cash_flow_parser
        }
        parser = parser_map[prompt_type]
        
        # Generate prompt
        prompt = prompt_templates[prompt_type].format(**{f"{prompt_type}_data": data_summary})
        
        # Get LLM response
        response = llm(prompt)
        
        # Parse structured response
        parsed_response = parser.parse(response)
        return parsed_response
        
    except Exception as e:
        st.error(f"Error processing {prompt_type}: {str(e)}")
        # Fallback to basic analysis
        return generate_basic_summary(prompt_type, data)

def generate_basic_summary(prompt_type, data):
    """Fallback function for basic summary without structured parsing"""
    if data is not None:
        data_dict = data.to_dict()
        
        # Use basic template without structured output
        basic_template = f"""Analyze this {prompt_type.replace('_', ' ')} data: {data_dict}
        
        Provide a comprehensive analysis with key insights, ratios, and recommendations."""
        
        response = llm(basic_template)
        return response
    return "Error: No data provided."

# DISPLAY FUNCTIONS
def display_financial_analysis(analysis, title):
    """Display structured financial analysis with professional formatting"""
    if analysis is None:
        st.warning(f"No data available for {title}")
        return
    
    # Check if analysis is structured (Pydantic model) or basic string
    if hasattr(analysis, 'summary'):
        display_structured_analysis(analysis, title)
    else:
        display_basic_analysis(analysis, title)

def display_structured_analysis(analysis, title):
    """Display structured Pydantic analysis"""
    st.subheader(f"📊 {title}")
    
    # Executive Summary
    st.markdown("### Executive Summary")
    st.info(analysis.summary)
    
    # Risk Assessment Badge
    risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    st.markdown(f"**Risk Level:** {risk_colors.get(analysis.risk_level, '⚪')} {analysis.risk_level}")
    
    # Key Ratios Table
    if analysis.key_ratios:
        st.markdown("### Key Financial Ratios")
        ratio_data = []
        for ratio in analysis.key_ratios:
            ratio_data.append({
                "Ratio": ratio.name,
                "Value": ratio.value,
                "Interpretation": ratio.interpretation,
                "Benchmark": ratio.benchmark
            })
        
        ratio_df = pd.DataFrame(ratio_data)
        st.dataframe(ratio_df, use_container_width=True)
    
    # Strengths and Concerns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#Strengths")
        for strength in analysis.strengths:
            st.success(f"• {strength}")
    
    with col2:
        st.markdown("#Areas of Concern")
        for concern in analysis.concerns:
            st.warning(f"• {concern}")
    
    # Recommendations
    st.markdown("#Recommendations")
    for i, rec in enumerate(analysis.recommendations, 1):
        st.markdown(f"**{i}.** {rec}")

def display_basic_analysis(analysis, title):
    """Display basic string analysis"""
    st.subheader(f"{title}")
    st.write(analysis)

def create_financial_charts(data, title, chart_type):
    """Create financial visualizations"""
    if data is None or data.empty:
        return
    
    st.markdown(f"#{title} Visualizations")
    
    numeric_data = data.select_dtypes(include=['number'])
    if numeric_data.empty:
        st.info("No numeric data available for visualization")
        st.markdown("**Raw Data Preview:**")
        st.dataframe(data.head(), use_container_width=True)
        return
    
    # Create different chart types based on financial statement
    col1, col2 = st.columns(2)
    
    with col1:
        if chart_type == "balance_sheet":
            st.markdown("**Asset vs Liability Trends**")
            st.bar_chart(numeric_data)
        elif chart_type == "profit_loss":
            st.markdown("**Revenue & Expense Analysis**")
            st.line_chart(numeric_data)
        elif chart_type == "cash_flow":
            st.markdown("**Cash Flow Patterns**")
            st.area_chart(numeric_data)
    
    with col2:
        st.markdown("**Data Summary Statistics**")
        st.dataframe(numeric_data.describe())


# MAIN APPLICATION
def main():
    # App configuration
    st.set_page_config(
        page_title="Gemini Pro Financial Decoder",
        page_icon="💰",
        layout="wide"
    )
    
    # App title and description
    st.title("Gemini Pro Financial Decoder")
    st.markdown("### Advanced AI-Powered Financial Statement Analysis")
    st.markdown("Upload your financial statements and get comprehensive analysis with AI-generated insights, ratios, and recommendations.")
    
    # File uploads
    balance_sheet_file, profit_loss_file, cash_flow_file = upload_file()
    
    # Analysis button
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button("Generate Professional Financial Analysis", type="primary")
    
    if analyze_button:
        if not any([balance_sheet_file, profit_loss_file, cash_flow_file]):
            st.error("Please upload at least one financial statement to analyze.")
            return
        
        with st.spinner("Generating comprehensive financial analysis..."):
            # Load data files
            balance_sheet_data = load_file(balance_sheet_file)
            profit_loss_data = load_file(profit_loss_file)
            cash_flow_data = load_file(cash_flow_file)
            
            # Generate analyses
            analyses = {}
            
            if balance_sheet_data is not None:
                analyses['balance_sheet'] = generate_structured_summary("balance_sheet", balance_sheet_data)
            
            if profit_loss_data is not None:
                analyses['profit_loss'] = generate_structured_summary("profit_loss", profit_loss_data)
            
            if cash_flow_data is not None:
                analyses['cash_flow'] = generate_structured_summary("cash_flow", cash_flow_data)
        
        # Display results
        st.success("Analysis Complete!")
        
        # Balance Sheet Analysis
        if 'balance_sheet' in analyses:
            display_financial_analysis(analyses['balance_sheet'], "Balance Sheet Analysis")
            create_financial_charts(balance_sheet_data, "Balance Sheet", "balance_sheet")
            st.divider()
        
        # Profit & Loss Analysis
        if 'profit_loss' in analyses:
            display_financial_analysis(analyses['profit_loss'], "Profit & Loss Analysis")
            create_financial_charts(profit_loss_data, "Profit & Loss", "profit_loss")
            st.divider()
        
        # Cash Flow Analysis
        if 'cash_flow' in analyses:
            display_financial_analysis(analyses['cash_flow'], "Cash Flow Analysis")
            create_financial_charts(cash_flow_data, "Cash Flow", "cash_flow")
        
        # Export functionality
        if analyses:
            st.markdown("#Export Analysis")
            export_data = {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analyses": {k: str(v) for k, v in analyses.items()}
            }
            
            st.download_button(
                label="Download Financial Analysis Report",
                data=str(export_data),
                file_name=f"financial_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    # Sidebar with information
    with st.sidebar:
        st.markdown("#About This Tool")
        st.markdown("""
        This financial analyzer uses **Google Gemini AI** with **LangChain** to provide:
        
        - **Structured Analysis**: Consistent, professional format
        - **Key Ratios**: Automated calculation of financial metrics
        - **Risk Assessment**: AI-powered risk evaluation
        - **Actionable Insights**: Specific recommendations
        - **Visual Charts**: Data visualization for trends
        
        **Supported Formats:** CSV, Excel (XLSX)
        """)
        
        st.markdown("#Features")
        st.markdown("""
        Balance Sheet Analysis  
        Profit & Loss Analysis  
        Cash Flow Analysis  
        Automated Ratio Calculations  
        Risk Assessment  
        Professional Visualizations  
        Export Reports  
        """)

if __name__ == "__main__":
    main()

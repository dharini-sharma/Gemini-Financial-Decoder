import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_KEY")

genai.configure(api_key=api_key)

model = genai.list_models()

for model in model:
    print(model)

llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key, temperature=1.0)

prompt_templates = {
    "balance_sheet": PromptTemplate(    
        input_variables=["balance_sheet_data"],
        template="""Analyze this balance sheet data: {balance_sheet_data}
        
        Calculate and provide:
        1. Key liquidity ratios (current ratio, quick ratio)
        2. Debt-to-equity ratio
        3. Working capital analysis
        4. Asset composition insights
        5. Financial health assessment with specific recommendations"""
    ),

    "profit_loss": PromptTemplate(
        input_variables=["profit_loss_data"],
        template="""Analyze this profit and loss statement data: {profit_loss_data}
        
        Calculate and provide:
        1. Profitability ratios (gross margin, operating margin, net margin)
        2. Revenue growth trends and analysis
        3. Cost structure breakdown (COGS, operating expenses, overhead)
        4. EBITDA and operating efficiency metrics
        5. Year-over-year performance comparison
        6. Expense management insights and optimization recommendations
        7. Revenue diversification and sustainability assessment"""
    ),

    "cash_flow": PromptTemplate(
        input_variables=["cash_flow_data"],
        template="""Analyze this cash flow statement data: {cash_flow_data}
        
        Calculate and provide:
        1. Operating cash flow analysis and cash conversion efficiency
        2. Free cash flow calculation and implications
        3. Cash flow ratios (operating cash flow to sales, cash coverage ratio)
        4. Working capital impact on cash generation
        5. Investment activities analysis (CapEx trends, asset acquisitions)
        6. Financing activities breakdown (debt payments, dividend policy)
        7. Cash burn rate and runway analysis (if applicable)
        8. Liquidity position and cash management recommendations
        9. Seasonal cash flow patterns and business cycle insights"""
    )
}
# Function to upload files using Streamlit

def upload_file():
    balance_sheet = st.file_uploader("Upload Balance Sheet", type=["csv", "xlsx"])
    profit_loss = st.file_uploader("Upload Profit and Loss Statement", type=["csv", "xlsx"])
    cash_flow = st.file_uploader("Upload Cash Flow Statement", type=["csv", "xlsx"])

    return balance_sheet, profit_loss, cash_flow

def load_file(file):
    if file is not None:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)  # Fix: was pd.read_csv
    return None


#Generate Summary from Templates 

def generate_summary(prompt_type, data):  
  if data is not None:  
    data_dict = data.to_dict()  
    if prompt_type == "balance_sheet":  
      prompt = prompt_templates [prompt_type].format(balance_sheet_data=data_dict)  
    elif prompt_type == "profit_loss":  
      prompt = prompt_templates [prompt_type].format(profit_loss_data=data_dict)  
    elif prompt_type == "cash_flow":  
      prompt = prompt_templates [prompt_type].format(cash_flow_data=data_dict)  
    response = llm(prompt)
    return response  
  return "Error: No data provided."  

#visulization function
def create_visuals(data, title):  
    if data is not None:  
        st.subheader(title)  
        st.write(data)  
        st.line_chart(data.select_dtypes (include=['number']))

#App title and file uploads
st.title("Gemini Pro Financial Decoder")
balance_sheet_file, profit_loss_file, cash_flow_file = upload_file()

#Button to generate reports
if st.button("Generate Reports"):
  with st.spinner("Generating summaries and visualizations..."):
    balance_sheet_data = load_file(balance_sheet_file)
    profit_loss_data = load_file(profit_loss_file)
    cash_flow_data = load_file(cash_flow_file)
  #Generate summaries and create visualizations
  balance_sheet_summary = generate_summary("balance_sheet", balance_sheet_data)
  profit_loss_summary = generate_summary("profit_loss", profit_loss_data)
  cash_flow_summary = generate_summary("cash_flow", cash_flow_data)

  st.subheader("Balance Sheet Summary")
  st.write(balance_sheet_summary)
  create_visuals(balance_sheet_data,"Balance Sheet Data")

  st.subheader("Profit Loss Summary")
  st.write(profit_loss_summary)
  create_visuals(profit_loss_data,"Profit & Loss Data")

  st.subheader("Cash Flow Summary")
  st.write(cash_flow_summary)
  create_visuals(cash_flow_data,"Cash Flow Data")
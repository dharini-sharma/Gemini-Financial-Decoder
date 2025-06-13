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

llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=api_key, temperature=1.0)

prompt_templates = {
    "balance_sheet": PromptTemplate(    
        input_variables=["balance_sheet_data"],
        template="""Given the balance sheet data: {balance_sheet_data},
provide a clear and concise summary highlighting key financial metrics and insights.""" 
    ),

    "profit_loss": PromptTemplate(
        input_variables=["profit_loss_data"],
        template="""Given the profit and loss statement data: {profit_loss_data},
provide a clear and concise summary highlighting key financial metrics and insights.""" 
    ),

    "cash_flow": PromptTemplate(
        input_variables=["cash_flow_data"],
        template="""Given the cash flow statement data: {cash_flow_data},
provide a clear and concise summary highlighting key financial metrics and insights."""
    )
}

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
            return pd.read_csv(file)
        
    return None
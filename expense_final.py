# cd "C:\Users\charl\Documents\K-ID\Corporate Credit AI"
# .venv\Scripts\activate.bat
# streamlit run expense_final.py

import streamlit as st
import pandas as pd
import yaml
import time
import plotly.express as px
import plotly.graph_objects as go
import re
import smtplib
from email.message import EmailMessage
import threading
from datetime import datetime, date, timedelta
from itertools import combinations
import io
import os
import signal
import sys
import pickle

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import webbrowser

# Page config
st.set_page_config(
    page_title="Expense Validation Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Purple Theme
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #1a0d2e 0%, #2d1b3d 25%, #3e2659 50%, #4a2c6b 75%, #5c3a7d 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1d391kg .stMarkdown {
        background: linear-gradient(180deg, #2d1b3d 0%, #3e2659 100%);
    }
    
    /* Main content area */
    .block-container {
        background: rgba(45, 27, 61, 0.3);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(156, 102, 234, 0.2);
    }
    
    /* Headers */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #9c66ea, #c084fc, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 0 0 30px rgba(156, 102, 234, 0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(45, 27, 61, 0.6);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #c084fc;
        background: transparent;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #7c3aed, #9c66ea);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(139, 92, 246, 0.1));
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(156, 102, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #7c3aed, #9c66ea);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #6d28d9, #8b5cf6);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
    }
    
    /* Sidebar elements */
    .css-1d391kg h2, .css-1d391kg h3 {
        color: #c084fc;
    }
    
    /* Text elements */
    h1, h2, h3 {
        color: #e2e8f0;
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(45, 27, 61, 0.6);
        border: 1px solid rgba(156, 102, 234, 0.3);
        border-radius: 10px;
    }
    
    /* Success boxes */
    .stSuccess {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 10px;
    }
    
    /* Warning boxes */
    .stWarning {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 10px;
    }
    
    /* Error boxes */
    .stError {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(45, 27, 61, 0.6);
        border-radius: 8px;
        border: 1px solid rgba(156, 102, 234, 0.2);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(45, 27, 61, 0.4);
        border: 2px dashed rgba(156, 102, 234, 0.4);
        border-radius: 10px;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(45, 27, 61, 0.3);
        border-radius: 10px;
        border: 1px solid rgba(156, 102, 234, 0.2);
    }
    
    /* Enhanced chart container styling */
    .chart-container {
        background: rgba(45, 27, 61, 0.4);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(156, 102, 234, 0.3);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }

    /* Gradient text for section headers */
    .section-header {
        background: linear-gradient(45deg, #c084fc, #e879f9, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-shadow: 0 0 20px rgba(156, 102, 234, 0.3);
    }

    /* Enhanced metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(156, 102, 234, 0.25);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'policy' not in st.session_state:
    st.session_state.policy = None
if 'validation_complete' not in st.session_state:
    st.session_state.validation_complete = False
if 'SCHEDULED_EMAILS' not in st.session_state:
    st.session_state.SCHEDULED_EMAILS = {}
if 'POLICY_SCHEDULED_EMAILS' not in st.session_state:
    st.session_state.POLICY_SCHEDULED_EMAILS = {}
if 'FIXED_ASSET_SCHEDULED_EMAILS' not in st.session_state:
    st.session_state.FIXED_ASSET_SCHEDULED_EMAILS = {}
    
EMAIL_LOCK = threading.Lock()
    
# Default policy YAML
DEFAULT_POLICY = """
general_policy:
  require_receipts: true
  require_budget_approval: true
  domain_leader_approval_limit_usd: 500
  finance_approval_required_over_usd: 500
  finance_email: "finance@k-id.com"

reimbursable_categories:
  - business_travel
  - business_meals
  - business_entertainment
  - miscellaneous_business_expenses
  - software_tools
  - equipment
  - professional_development
  - mobile_phone_usage

travel:
  hotels:
    default_max_usd: 250
    high_cost_city_max_usd: 350
    high_cost_cities: [London, Tokyo, New York]
  rental_car:
    max_daily_usd: 100

meals_and_entertainment:
  work_trip_meals:
    daily_limit_usd: 75
    gratuity_limit_percent: 20
  business_meals:
    max_per_person_usd: 100
  entertainment:
    max_per_person_usd: 100

communications:
  mobile_reimbursement_max_usd: 50

equipment:
  general_employee_limit_usd: 2000
  pde_limit_usd: 2800
  cycle_years: 3
"""

# Employee email dictionary
EMPLOYEE_EMAIL_MAPPING = {
    "Aakash Mandhar": "aakash@k-id.com",
    "Tushar Ajmera": "tushar@k-id.com",
    "Susan": "susanchen@k-id.com",
    "Fil Baumanis": "fil@k-id.com",
    "Kay Vasey": "kay@k-id.com",
    "Luc Delany": "luc@k-id.com",
    "Wesley": "wesleysitu@k-id.com",
    "Liz": "liz@k-id.com",
    "Kevin Loh": "kevin@k-id.com",
    "Tiffany Friedel": "tiffany@k-id.com",
    "Marshall Nu": "marshall@k-id.com",
    "Arunan Rabindran": "arun@k-id.com",
    "Aaron Lam": "aaron.lam@k-id.com",
    "Joseph Newman": "joe@k-id.com",
    "Braxton Sheum": "braxton@k-id.com",
    "Natalie Shou": "natalie@k-id.com",
    "Alyssa Aw": "alyssa@k-id.com",
    "Arunan": "arunan@k-id.com",
    "Miguel Kyle Khonrad Lejano Martinez": "miguel@k-id.com",
    "Olav Bus": "olav@k-id.com",
    "Crystal Wong": "crystal@k-id.com",
    "Benjamin Fox": "ben@k-id.com",
    "Markus Juuti": "marklee@k-id.com",
    "Tristen": "tristen@k-id.com",
    "Julian Corbett": "julian@k-id.com",
    "Beatrice Cavicchioli": "beatrice@k-id.com",
    "Lennart Ng": "lennart@k-id.com",
    "Carolyn Yan": "carolyn@k-id.com",
    "Lulu Xia": "lulu@k-id.com",
    "Sebastian Chew": "sebastian@k-id.com",
    "Keemin Ngiam": "keemin@k-id.com",
    "Nina Cheuck": "nina@k-id.com",
    "Timothy Ma": "timothy@k-id.com",
    "Adam Snyder": "adam@k-id.com",
    "Denise Villanueva": "denise@k-id.com",
    "Benjamin Chen": "benc@k-id.com",  # Note: Second Benjamin with different email
    "Ibrahim Midian": "ibrahim@k-id.com",
    "Erich Bao": "erich@k-id.com",
    "Ruosi Wang": "ruosi@k-id.com",
    "Shireen Ho": "shireen@k-id.com",
    "Hilson Wong": "hilson@k-id.com",
    "Bernie": "bernie@k-id.com",
    "Kieran Donovan": "kieran@k-id.com",
    "Michel Paupulaire": "mpaupulaire@k-id.com",
    "Greg Leib": "greg@k-id.com",
    "Rupali Sharma": "rupali@k-id.com",
    "Charleston Yap": "charlestonyap@k-id.com",
    "Andrew Huth": "ahuth@k-id.com",
    "Joanna Shields": "joanna@k-id.com",
    "Jeff Wu": "jwu@k-id.com",
    "Andre Malan": "andre@k-id.com"
}

# MAIN BACKBONE OF PROGRAM

def load_policy_from_yaml(yaml_content):
    """Load policy from YAML content"""
    try:
        return yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML: {e}")
        return None

def standardize_column_names(df):
    """
    Standardize column names to match expected format
    Maps the new format columns to the standard ones used in processing
    """
    # Define the mapping from new format to standard format
    column_mapping = {
        # Employee information
        'Employee - ID': 'Employee_ID',
        'Employee': 'Employee_Name',
        
        # Transaction details
        'Vendor name': 'Vendor',
        'Amount (by category) - Currency': 'Currency',
        'Amount (by category)': 'Amount',
        'Purchase date': 'Date',
        'Approval State': 'Approval_State',
        'Object Type': 'Object_Type',
        'Memo': 'Memo',
        'Category Name': 'Category',
        'Transaction Id': 'Transaction_ID',
        'Has Receipt': 'HasReceipt',
        'Check Date': 'Check_Date',
        'Attendees': 'Attendees',  # <-- ADD THIS LINE
        
        # Merchant information
        'Merchant currency': 'Merchant_Currency',
        'Merchant Amount (by category) - Currency': 'Merchant_Amount_Currency',
        'Merchant Amount (by category)': 'Merchant_Amount'
    }
    
    # Create a copy of the dataframe
    df_standardized = df.copy()
    
    # Apply the mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df_standardized = df_standardized.rename(columns={old_name: new_name})
    
    return df_standardized

def parse_attendees(attendees_str, employee_name):
    """
    Parse attendees string and return list of attendee names
    Special handling for Shireen Ho - she's excluded when attendees are specified
    For others, employee is always included in attendees
    """
    is_shireen = employee_name.lower().strip() == 'shireen ho'
    
    if pd.isna(attendees_str) or not attendees_str or str(attendees_str).strip() == '':
        # No attendees specified - employee is the only attendee (including Shireen)
        return [employee_name]
    
    # Parse the attendees string
    attendees = [name.strip() for name in str(attendees_str).split(',')]
    # Remove empty strings
    attendees = [name for name in attendees if name]
    
    if not attendees:  # If after cleaning we have no attendees
        return [employee_name]
    
    if is_shireen:
        # For Shireen Ho: if attendees are specified, she's excluded
        return attendees
    else:
        # For everyone else: ensure employee is in the attendees list
        attendees_lower = [name.lower().strip() for name in attendees]
        if employee_name.lower().strip() not in attendees_lower:
            attendees.append(employee_name)
        return attendees

def calculate_per_pax_amount(amount, attendees_list):
    """
    Calculate per-pax amount based on attendees list
    """
    num_attendees = len(attendees_list) if attendees_list else 1
    return amount / num_attendees

def find_minimal_excess_transactions(transactions, daily_limit):
    """
    Find the minimal set of transactions to flag that reduces spending to within daily limits.
    
    Args:
        transactions: List of (index, amount) tuples
        daily_limit: The daily spending limit
    
    Returns:
        dict: {
            'transactions_to_flag': list of indices to flag,
            'total_flagged_amount': total amount being flagged,
            'excess_amount': amount over the limit,
            'remaining_company_funded': amount still funded by company,
            'employee_out_of_pocket': amount employee pays out of pocket
        }
    """
    if not transactions:
        return {
            'transactions_to_flag': [],
            'total_flagged_amount': 0,
            'excess_amount': 0,
            'remaining_company_funded': 0,
            'employee_out_of_pocket': 0
        }
    
    total_amount = sum(amount for _, amount in transactions)
    
    if total_amount <= daily_limit:
        # No violations
        return {
            'transactions_to_flag': [],
            'total_flagged_amount': 0,
            'excess_amount': 0,
            'remaining_company_funded': total_amount,
            'employee_out_of_pocket': 0
        }
    
    excess = total_amount - daily_limit
    
    # Strategy 1: Find single transaction that covers excess with minimal overage
    single_candidates = [(idx, amount) for idx, amount in transactions if amount >= excess]
    if single_candidates:
        # Choose the smallest transaction that covers the excess
        best_single = min(single_candidates, key=lambda x: x[1])
        single_solution = {
            'transactions_to_flag': [best_single[0]],
            'total_flagged_amount': best_single[1],
            'excess_amount': excess,
            'remaining_company_funded': total_amount - best_single[1],
            'employee_out_of_pocket': best_single[1]
        }
    else:
        single_solution = None
    
    # Strategy 2: Greedy combination starting from largest transactions
    # This minimizes the number of transactions flagged
    sorted_desc = sorted(transactions, key=lambda x: x[1], reverse=True)
    greedy_large = []
    flagged_sum = 0
    
    for idx, amount in sorted_desc:
        if flagged_sum < excess:
            greedy_large.append((idx, amount))
            flagged_sum += amount
    
    large_first_solution = {
        'transactions_to_flag': [idx for idx, _ in greedy_large],
        'total_flagged_amount': flagged_sum,
        'excess_amount': excess,
        'remaining_company_funded': total_amount - flagged_sum,
        'employee_out_of_pocket': flagged_sum
    }
    
    # Strategy 3: Greedy combination starting from smallest transactions
    # This might minimize the total flagged amount in some cases
    sorted_asc = sorted(transactions, key=lambda x: x[1])
    greedy_small = []
    flagged_sum = 0
    
    for idx, amount in sorted_asc:
        if flagged_sum < excess:
            greedy_small.append((idx, amount))
            flagged_sum += amount
    
    small_first_solution = {
        'transactions_to_flag': [idx for idx, _ in greedy_small],
        'total_flagged_amount': flagged_sum,
        'excess_amount': excess,
        'remaining_company_funded': total_amount - flagged_sum,
        'employee_out_of_pocket': flagged_sum
    }
    
    # Choose the solution that minimizes employee out-of-pocket cost
    candidates = [large_first_solution, small_first_solution]
    if single_solution:
        candidates.append(single_solution)
    
    # Select solution with minimum employee out-of-pocket amount
    best_solution = min(candidates, key=lambda x: x['employee_out_of_pocket'])
    
    return best_solution

def get_violation_assignee(employee_name, attendees_list):
    """
    Determine who should be assigned violations for a transaction
    For Shireen Ho with attendees: violations go to attendees
    For everyone else: violations stay with the employee
    """
    is_shireen = employee_name.lower().strip() == 'shireen ho'
    
    if is_shireen and len(attendees_list) > 0:
        # For Shireen with attendees, she should not be in the attendees list
        # Violations should be distributed among the actual attendees
        return attendees_list
    else:
        # For everyone else, violation stays with the employee
        return [employee_name]


def process_daily_limits_optimized(df, limits):
    """
    Process daily limits with optimized flagging and proper violation assignment
    Fixed version that prevents duplicate processing and inflated costs
    """
    
    for key, cap in limits.items():
        if cap['type'] == 'daily':
            mask = df['Category'].isin(cap['match'])
            
            if mask.any():
                df[f'{key}_daily_exceed'] = False
                
                if cap.get('per_pax', False):
                    # Per-pax daily limit processing with violation assignment
                    df_expanded = []

                    # Expand each transaction by attendees
                    for idx, row in df[mask].iterrows():
                        attendees = row['Attendees_List']
                        violation_assignees = row['Violation_Assignees']
                        per_pax_amount = row['Per_Pax_Amount']

                        for attendee in attendees:
                            # Determine if this attendee should receive violations
                            receives_violation = attendee in violation_assignees
                            
                            df_expanded.append({
                                'Original_Index': idx,
                                'Employee': row['Employee'],
                                'Date': row['Date'],
                                'Category': row['Category'],
                                'Attendee': attendee,
                                'Receives_Violation': receives_violation,
                                'Per_Pax_Amount': per_pax_amount,
                                'Amount': row['Amount']
                            })

                    if df_expanded:
                        expanded_df = pd.DataFrame(df_expanded)

                        # Group by attendee who receives violations, date, category
                        violation_groups = expanded_df[expanded_df['Receives_Violation'] == True].groupby(['Attendee', 'Date', 'Category'])
                        
                        for (attendee, date, category), group in violation_groups:
                            # Get unique transactions for this attendee/date/category
                            unique_transactions = group.drop_duplicates('Original_Index')
                            transactions = [(idx, row['Per_Pax_Amount']) for idx, row in unique_transactions.iterrows() 
                                          if row['Original_Index'] in group['Original_Index'].values]

                            total_per_pax = sum(amount for _, amount in transactions)

                            if total_per_pax > cap['value']:
                                excess = total_per_pax - cap['value']

                                # Create transaction list with original indices for optimization
                                optimization_transactions = [(row['Original_Index'], row['Per_Pax_Amount']) 
                                                           for _, row in unique_transactions.iterrows()]

                                # Find optimal flagging solution
                                solution = find_minimal_excess_transactions(optimization_transactions, cap['value'])

                                if solution['transactions_to_flag']:
                                    total_flagged_per_pax = solution['total_flagged_amount']

                                    for original_idx in solution['transactions_to_flag']:
                                        df.loc[original_idx, f'{key}_daily_exceed'] = True

                                        current_per_pax = df.loc[original_idx, 'Per_Pax_Amount']
                                        proportion = current_per_pax / total_flagged_per_pax if total_flagged_per_pax > 0 else 0
                                        transaction_excess = excess * proportion

                                        df.loc[original_idx, 'Employee_Out_of_Pocket'] += transaction_excess
                                        current_company_funded = df.loc[original_idx, 'Company_Funded_Amount']
                                        df.loc[original_idx, 'Company_Funded_Amount'] = current_company_funded - transaction_excess
                
                else:
                    # FIXED: Total amount daily limit processing with violation assignment
                    # Group by (violation assignee, date, category) and process each group only once
                    processed_groups = set()
                    
                    for idx, row in df[mask].iterrows():
                        violation_assignees = row['Violation_Assignees']
                        
                        # For each violation assignee, check their daily total
                        for assignee in violation_assignees:
                            group_key = (assignee, row['Date'], row['Category'])
                            
                            # Skip if we've already processed this group
                            if group_key in processed_groups:
                                continue
                                
                            processed_groups.add(group_key)
                            
                            assignee_mask = (df['Category'].isin(cap['match']) & 
                                           (df['Date'] == row['Date']))
                            
                            # Find all transactions for this assignee on this date/category
                            assignee_transactions = []
                            for check_idx, check_row in df[assignee_mask].iterrows():
                                if assignee in check_row['Violation_Assignees']:
                                    assignee_transactions.append((check_idx, check_row['Amount']))
                            
                            if assignee_transactions:
                                total_amount = sum(amount for _, amount in assignee_transactions)
                                
                                if total_amount > cap['value']:
                                    excess = total_amount - cap['value']
                                    solution = find_minimal_excess_transactions(assignee_transactions, cap['value'])
                                    
                                    if solution['transactions_to_flag']:
                                        total_flagged_amount = solution['total_flagged_amount']
                                        
                                        # Calculate the proportion of excess each flagged transaction should bear
                                        for flagged_idx in solution['transactions_to_flag']:
                                            # Only process if not already flagged for this limit type
                                            if not df.loc[flagged_idx, f'{key}_daily_exceed']:
                                                df.loc[flagged_idx, f'{key}_daily_exceed'] = True
                                                
                                                transaction_amount = df.loc[flagged_idx, 'Amount']
                                                proportion = transaction_amount / total_flagged_amount if total_flagged_amount > 0 else 0
                                                transaction_excess = excess * proportion
                                                
                                                df.loc[flagged_idx, 'Employee_Out_of_Pocket'] += transaction_excess
                                                current_company_funded = df.loc[flagged_idx, 'Company_Funded_Amount']
                                                df.loc[flagged_idx, 'Company_Funded_Amount'] = current_company_funded - transaction_excess
                
                # Ensure the column exists
                if f'{key}_daily_exceed' not in df.columns:
                    df[f'{key}_daily_exceed'] = False
    
    return df


def generate_cost_summary(df):
    """
    Generate a summary of cost breakdowns for management reporting.
    
    Returns:
        dict: Summary statistics about cost allocation
    """
    total_transactions = len(df)
    flagged_transactions = len(df[df['Employee_Out_of_Pocket'] > 0])
    
    total_amount = df['Amount'].sum()
    total_company_funded = df['Company_Funded_Amount'].sum()
    total_employee_out_of_pocket = df['Employee_Out_of_Pocket'].sum()
    
    # Per-employee breakdown
    employee_summary = df.groupby('Employee').agg({
        'Amount': 'sum',
        'Company_Funded_Amount': 'sum',
        'Employee_Out_of_Pocket': 'sum'
    }).reset_index()
    
    employee_summary['Compliance_Rate'] = (
        employee_summary['Company_Funded_Amount'] / employee_summary['Amount'] * 100
    ).round(2)
    
    return {
        'overview': {
            'total_transactions': total_transactions,
            'flagged_transactions': flagged_transactions,
            'compliance_rate': f"{(total_company_funded / total_amount * 100):.2f}%" if total_amount > 0 else "N/A",
            'total_amount': total_amount,
            'company_funded': total_company_funded,
            'employee_out_of_pocket': total_employee_out_of_pocket
        },
        'by_employee': employee_summary.to_dict('records')
    }


def process_expense_data(df, policy, mode="Full Validation"):
    """Process expense data and apply validation rules based on the selected mode"""
    if df is None or policy is None:
        return None
    
    # Standardize column names
    df = standardize_column_names(df)
    
    # Map required columns
    required_mappings = {
        'Category': ['Category', 'Category Name'],
        'Amount': ['Amount', 'Amount (by category)'],
        'Date': ['Date', 'Purchase date'],
        'Employee': ['Employee_Name', 'Employee', 'Employee - ID']
    }
    
    for standard_col, possible_cols in required_mappings.items():
        found = False
        for possible_col in possible_cols:
            if possible_col in df.columns:
                if standard_col != possible_col:
                    df = df.rename(columns={possible_col: standard_col})
                found = True
                break
        if not found:
            st.error(f"Required column '{standard_col}' not found. Available columns: {list(df.columns)}")
            return None
    
    # Handle Employee column
    if 'Employee_Name' in df.columns and 'Employee' not in df.columns:
        df = df.rename(columns={'Employee_Name': 'Employee'})
    elif 'Employee_ID' in df.columns and 'Employee' not in df.columns:
        df = df.rename(columns={'Employee_ID': 'Employee'})
    
    # Convert date and amount columns
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce')
    except Exception as e:
        st.error(f"Error converting data: {e}")
        return None
    
    if 'Approval_State' in df.columns:
        df['Approval_State'] = df['Approval_State'].fillna('').astype(str)
    
    # Filter for 2025 data
    df = df[df['Date'] >= pd.Timestamp('2025-01-01')].copy()
    if len(df) == 0:
        st.warning("No transactions found for 2025.")
        return df
    
    # Handle receipt info
    if 'HasReceipt' not in df.columns:
        df['HasReceipt'] = df.get('Has Receipt', True)
    df['HasReceipt'] = df['HasReceipt'].map({
        'Yes': True, 'No': False, 'Y': True, 'N': False,
        'TRUE': True, 'FALSE': False, 'True': True, 'False': False,
        '1': True, '0': False, 1: True, 0: False
    }).fillna(True)
    
    # Initialize default columns for all modes
    df['Company_Funded_Amount'] = df['Amount']
    df['Employee_Out_of_Pocket'] = 0.0
    
    if mode == "Receipt-Only Mode":
        # Receipt-only mode: Only check for missing receipts on transactions >= $20
        df = df[df['Amount'] >= 20].copy()
        
        # Special filter: Exclude Jeff Wu's Grab/Uber transactions
        if 'Vendor' in df.columns:
            df = df[~((df['Employee'].str.lower() == 'jeff wu') & 
                      (df['Vendor'].str.lower().str.contains('grab|uber', na=False)))]
        
        # Receipt-only validation function
        def validate_receipt_only(row):
            if policy.get('general_policy', {}).get('require_receipts', True) and not row.get('HasReceipt', True):
                return 'Fail', 'Missing receipt'
            return 'Pass', 'OK'
        
        validation_results = df.apply(validate_receipt_only, axis=1)
        df['Status'] = [result[0] for result in validation_results]
        df['Message'] = [result[1] for result in validation_results]
        
    else:
        # Full validation mode: Apply all policy checks
        # Process attendees with special logic for Shireen Ho
        df['Attendees'] = df.get('Attendees', '')
        df['Attendees_List'] = df.apply(lambda row: parse_attendees(row['Attendees'], row['Employee']), axis=1)
        df['Number_of_Attendees'] = df['Attendees_List'].apply(len)
        df['Per_Pax_Amount'] = df.apply(lambda row: calculate_per_pax_amount(row['Amount'], row['Attendees_List']), axis=1)
        
        # Define policy limits
        limits = {
            'work_trip_meals': {'match': ['Biz Travel: Meals'], 'type': 'daily', 'value': policy['meals_and_entertainment']['work_trip_meals']['daily_limit_usd'], 'per_pax': True},
            'business_meals': {'match': ['Biz Meals, Gifts & Entertainment'], 'type': 'per_txn', 'value': policy['meals_and_entertainment']['business_meals']['max_per_person_usd'], 'per_pax': True},
            'hotels': {'match': ['Biz Travel: Hotels'], 'type': 'per_txn', 'value': policy['travel']['hotels']['default_max_usd'], 'per_pax': False},
            'transport': {'match': ['Biz Travel & Client Meeting: Transport'], 'type': 'daily', 'value': policy['travel']['rental_car']['max_daily_usd'], 'per_pax': False},
            'mobile': {'match': ['Telecommunications'], 'type': 'per_txn', 'value': policy['communications']['mobile_reimbursement_max_usd'], 'per_pax': False},
            'equipment': {'match': ['Computer Equipment'], 'type': 'per_txn', 'value': policy['equipment']['general_employee_limit_usd'], 'per_pax': False}
        }
        
        # Apply per-transaction limits
        for key, cap in limits.items():
            if cap['type'] == 'per_txn':
                df[f'{key}_exceed'] = False
                mask = df['Category'].isin(cap['match'])
                if cap.get('per_pax', False):
                    exceeded = df.loc[mask, 'Per_Pax_Amount'] > cap['value']
                    df.loc[mask & exceeded, f'{key}_exceed'] = True
                    for idx in df[mask & exceeded].index:
                        excess = df.loc[idx, 'Per_Pax_Amount'] - cap['value']
                        df.loc[idx, 'Employee_Out_of_Pocket'] += excess
                        df.loc[idx, 'Company_Funded_Amount'] -= excess
                else:
                    exceeded = df.loc[mask, 'Amount'] > cap['value']
                    df.loc[mask & exceeded, f'{key}_exceed'] = True
                    if key != 'hotels':
                        for idx in df[mask & exceeded].index:
                            excess = df.loc[idx, 'Amount'] - cap['value']
                            df.loc[idx, 'Employee_Out_of_Pocket'] += excess
                            df.loc[idx, 'Company_Funded_Amount'] -= excess
        
        df['Violation_Assignees'] = df.apply(lambda row: get_violation_assignee(row['Employee'], row['Attendees_List']), axis=1)
        # Apply daily limits
        df = process_daily_limits_optimized(df, limits)
        
        # Full validation function
        def validate_full(row):
            msgs = []
            has_violations = False

            # Check spending limits
            for key, cap in limits.items():
                if row.get(f'{key}_exceed', False):
                    has_violations = True
                    msgs.append(f"{key.replace('_', ' ').title()} {'per-pax ' if cap.get('per_pax', False) else ''}{'daily ' if cap['type'] == 'daily' else ''}limit exceeded")

                # Also check daily limits
                if row.get(f'{key}_daily_exceed', False):
                    has_violations = True
                    msgs.append(f"{key.replace('_', ' ').title()} {'per-pax ' if cap.get('per_pax', False) else ''}daily limit exceeded")

            # Check receipts
            if policy.get('general_policy', {}).get('require_receipts', True) and not row.get('HasReceipt', True):
                has_violations = True
                msgs.append("Missing receipt")

            # Check approval status
            if row.get('Approval_State', '').lower() in ['pending', 'rejected', 'unapproved']:
                has_violations = True
                msgs.append(f"Approval issue: {row['Approval_State']}")

            # Check high amount approval
            if row['Amount'] > policy['general_policy']['finance_approval_required_over_usd'] and row.get('Approval_State', '').lower() not in ['approved']:
                has_violations = True
                msgs.append("High amount requires finance approval")

            # NEW: Handle violation assignment for Shireen Ho
            violation_assignees = row.get('Violation_Assignees', [row['Employee']])
            employee_name = row['Employee']
            is_shireen = employee_name.lower().strip() == 'shireen ho'

            # If Shireen has attendees and there are violations, redirect violations to attendees
            if is_shireen and len(violation_assignees) > 0 and violation_assignees != [employee_name] and has_violations:
                # For Shireen with attendees, violations should not be recorded under her name
                # Instead, we'll mark this transaction as "Pass" for Shireen and handle violations separately
                status = 'Pass'
                message = f"Transaction booked for: {', '.join(violation_assignees)}"

                # The actual violations will be tracked through the violation_assignees system
                # This ensures Shireen doesn't get flagged for violations on transactions she booked for others
            else:
                # Normal processing for everyone else (including Shireen's personal transactions)
                status = 'Fail' if has_violations else 'Pass'
                message = "; ".join(msgs) if msgs else "OK"

                # Add violation assignee info for transparency
                if len(violation_assignees) > 1 or (len(violation_assignees) == 1 and violation_assignees[0] != employee_name):
                    assignee_str = ", ".join(violation_assignees)
                    if has_violations:
                        message += f" (Assigned to: {assignee_str})"

            return status, message
        
        validation_results = df.apply(validate_full, axis=1)
        df['Status'] = [result[0] for result in validation_results]
        df['Message'] = [result[1] for result in validation_results]
    
    return df









# WEEKLY BUDGETTING SYSTEM

# Enhanced department mapping with flexible abbreviations
DEPARTMENT_MAPPING = {
    # Direct mappings
    'Sales': 'Sales, Growth & Customer Success',
    'MarComms': 'Marketing & Comms', 
    'Marketing & Comms': 'Marketing & Comms',
    'LPR': 'Legal Policy Rockstars',
    'Legal Policy Rockstars': 'Legal Policy Rockstars',
    'PDE': 'Product, Design & Engineering',
    'Product, Design & Engineering': 'Product, Design & Engineering',
    'Strategy': 'Strategic Growth',
    'Strategic Growth': 'Strategic Growth',
    'Finance': 'Finance',
    'CEO': 'CEO',
    
    # Flexible abbreviations and variations
    'Marketing': 'Marketing & Comms',
    'Legal': 'Legal Policy Rockstars',
    'Product': 'Product, Design & Engineering',
    'Engineering': 'Product, Design & Engineering',
    'Design': 'Product, Design & Engineering',
    'Growth': 'Sales, Growth & Customer Success',
    'Customer Success': 'Sales, Growth & Customer Success',
    'Policy': 'Legal Policy Rockstars',
    'Rockstars': 'Legal Policy Rockstars',
    'Strategic': 'Strategic Growth',
    'Comms': 'Marketing & Comms',
    'Mkt': 'Marketing & Comms',
    'Eng': 'Product, Design & Engineering',
    'Dev': 'Product, Design & Engineering',
    'Development': 'Product, Design & Engineering',
    'Legal Policy': 'Legal Policy Rockstars',
    'Customer': 'Sales, Growth & Customer Success',
    'Business Development': 'Sales, Growth & Customer Success',
    'BD': 'Sales, Growth & Customer Success',
    'Biz Dev': 'Sales, Growth & Customer Success',
    'Sales Team': 'Sales, Growth & Customer Success',
    'Marketing Team': 'Marketing & Comms',
    'Legal Team': 'Legal Policy Rockstars',
    'Product Team': 'Product, Design & Engineering',
    'Engineering Team': 'Product, Design & Engineering',
    'Design Team': 'Product, Design & Engineering',
    'Finance Team': 'Finance',
    'Strategy Team': 'Strategic Growth'
}

TEAM_LEADERS = {
    "Marketing & Comms": {"leader": "Luc Delany", "members": ["Sebastian Chew", "Natalie Shou", "Miguel Kyle Khonrad Martinez", "Braxton Sheum", "Olav Bus"]},
    "Family Trust & Safety": {"leader": "Jeff Wu", "members": ["Ibrahim Midian", "Kay Vasey"]},
    "Finance": {"leader": "Lulu Xia", "members": ["Liz Lu", "Shireen Ho", "Marshall Nu", "Tushar Ajmera"]},
    "Legal Policy Rockstars": {"leader": "Timothy Ma", "members": ["Tristen Goetz", "Joanna Shields", "Joseph Newman", "Lennart Ng", "Carolyn Yan", "Hilson Wong", "Beatrice Cavicchioli"]},
    "People": {"leader": "Tiffany Friedel", "members": ["Susan Chen", "Alyssa Aw"]},
    "Product, Design & Engineering": {"leader": "Aakash Mandhar", "members": ["Andre Malan", "Kevin Loh","Greg Leib","Denise Villanueva","Tulcy Patel","Nina Cheuck","Rupali Sharma","Arun Rao","Adam Snyder","Aaron Lam","Markus Juuti","Crystal Wong","Michel Paupulaire","Fil Baumanis","Andrew Huth","Wesley Situ"]},
    "Sales, Growth & Customer Success": {"leader": "Erich Bao", "members": ["Benjamin Chen", "Ruosi Wang", "Benjamin Fox"]},
    "Strategic Growth": {"leader": "Julian Corbett", "members": ["Keemin Ngiam"]},
    "CEO": {"leader": "Kieran Donovan", "members": []}
}

def flexible_department_mapping(dept_name):
    """
    Flexible department mapping that handles various formats and abbreviations
    """
    if pd.isna(dept_name) or dept_name == '':
        return None
    
    # Convert to string and clean
    dept_clean = str(dept_name).strip()
    
    # Direct lookup first
    if dept_clean in DEPARTMENT_MAPPING:
        return DEPARTMENT_MAPPING[dept_clean]
    
    # Case-insensitive lookup
    for key, value in DEPARTMENT_MAPPING.items():
        if dept_clean.lower() == key.lower():
            return value
    
    # Partial matching for complex cases
    dept_lower = dept_clean.lower()
    
    # Check for partial matches
    if any(word in dept_lower for word in ['legal', 'policy', 'lpr']):
        return 'Legal Policy Rockstars'
    elif any(word in dept_lower for word in ['marketing', 'comms', 'marcomms']):
        return 'Marketing & Comms'
    elif any(word in dept_lower for word in ['sales', 'growth', 'customer']):
        return 'Sales, Growth & Customer Success'
    elif any(word in dept_lower for word in ['product', 'design', 'engineering', 'pde']):
        return 'Product, Design & Engineering'
    elif any(word in dept_lower for word in ['strategy', 'strategic']):
        return 'Strategic Growth'
    elif 'finance' in dept_lower:
        return 'Finance'
    elif 'ceo' in dept_lower:
        return 'CEO'
    
    # If no match found, return the original (cleaned) name
    return dept_clean

def map_employee_to_department(employee_name):
    """
    Map an employee name to their department using the custom team leaders dictionary.
    Now uses session state to allow for dynamic updates.
    
    Args:
        employee_name: Name of the employee from expense data
    
    Returns:
        str: Department name if found, None if not found
    """
    if not employee_name or pd.isna(employee_name):
        return None
    
    clean_name = str(employee_name).strip().lower()
    
    # Use custom mappings if available, otherwise use default
    team_leaders = st.session_state.get('custom_team_leaders', TEAM_LEADERS)
    
    for department, info in team_leaders.items():
        # Check leader
        if info["leader"].strip().lower() == clean_name:
            return department
        
        # Check members
        for member in info["members"]:
            if member.strip().lower() == clean_name:
                return department

    # If no match found
    return None

def display_department_mapping_editor():
    """
    Display and allow editing of department mappings
    """
    st.subheader("🏢 Department Mapping Management")
    st.caption("View and modify employee-department assignments")
    
    # Initialize session state for team leaders if not exists
    if 'custom_team_leaders' not in st.session_state:
        st.session_state.custom_team_leaders = TEAM_LEADERS.copy()
    
    # Tab layout for better organization
    tab1, tab2 = st.tabs(["📋 View Current Mappings", "✏️ Edit Mappings"])
    
    with tab1:
        st.markdown("**Current Department Structure:**")
        
        # Display current mappings in an organized way
        for department, info in st.session_state.custom_team_leaders.items():
            with st.expander(f"🏢 {department} ({len(info['members']) + 1} people)", expanded=False):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown("**Team Leader:**")
                    st.write(f"👑 {info['leader']}")
                
                with col2:
                    st.markdown("**Team Members:**")
                    if info['members']:
                        for member in info['members']:
                            st.write(f"• {member}")
                    else:
                        st.write("_No team members_")
    
    with tab2:
        st.markdown("**Modify Department Assignments:**")
        
        # Department selector for modifications
        selected_dept = st.selectbox(
            "Select Department to Edit:",
            options=list(st.session_state.custom_team_leaders.keys()),
            key="edit_dept_selector"
        )
        
        if selected_dept:
            dept_info = st.session_state.custom_team_leaders[selected_dept]
            
            st.markdown(f"#### Editing: {selected_dept}")
            
            # Edit team leader
            col1, col2 = st.columns([1, 1])
            
            with col1:
                new_leader = st.text_input(
                    "Team Leader:",
                    value=dept_info['leader'],
                    key=f"leader_{selected_dept}"
                )
            
            with col2:
                if st.button("Update Leader", key=f"update_leader_{selected_dept}"):
                    st.session_state.custom_team_leaders[selected_dept]['leader'] = new_leader
                    st.success(f"✅ Leader updated for {selected_dept}")
                    st.rerun()
            
            # Edit team members
            st.markdown("**Team Members:**")
            
            # Display current members with remove option
            if dept_info['members']:
                for i, member in enumerate(dept_info['members']):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• {member}")
                    with col2:
                        if st.button("🗑️", key=f"remove_{selected_dept}_{i}", help=f"Remove {member}"):
                            st.session_state.custom_team_leaders[selected_dept]['members'].remove(member)
                            st.success(f"✅ Removed {member} from {selected_dept}")
                            st.rerun()
            else:
                st.write("_No team members_")
            
            # Add new member
            st.markdown("**Add New Member:**")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_member = st.text_input(
                    "Member Name:",
                    placeholder="Enter full name",
                    key=f"new_member_{selected_dept}"
                )
            
            with col2:
                if st.button("➕ Add Member", key=f"add_member_{selected_dept}"):
                    if new_member and new_member.strip():
                        # Check if member already exists in any department
                        existing_dept = None
                        for dept, info in st.session_state.custom_team_leaders.items():
                            if new_member.strip() in info['members'] or new_member.strip() == info['leader']:
                                existing_dept = dept
                                break
                        
                        if existing_dept and existing_dept != selected_dept:
                            st.warning(f"⚠️ {new_member} is already in {existing_dept}. Move them first!")
                        elif new_member.strip() not in dept_info['members']:
                            st.session_state.custom_team_leaders[selected_dept]['members'].append(new_member.strip())
                            st.success(f"✅ Added {new_member} to {selected_dept}")
                            st.rerun()
                        else:
                            st.warning(f"⚠️ {new_member} is already in this department")
                    else:
                        st.error("❌ Please enter a valid name")
            
            # Move member between departments
            st.markdown("**Move Member Between Departments:**")
            
            # Get all employees for dropdown
            all_employees = []
            for dept, info in st.session_state.custom_team_leaders.items():
                all_employees.append(f"{info['leader']} (Leader - {dept})")
                for member in info['members']:
                    all_employees.append(f"{member} ({dept})")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                selected_employee = st.selectbox(
                    "Select Employee:",
                    options=all_employees,
                    key=f"move_employee_{selected_dept}"
                )
            
            with col2:
                target_dept = st.selectbox(
                    "Move to Department:",
                    options=[dept for dept in st.session_state.custom_team_leaders.keys() if dept != selected_dept],
                    key=f"target_dept_{selected_dept}"
                )
            
            with col3:
                if st.button("🔄 Move", key=f"move_btn_{selected_dept}"):
                    if selected_employee and target_dept:
                        # Parse employee info
                        employee_name = selected_employee.split(' (')[0]
                        current_dept = selected_employee.split(' (')[1].split(')')[0]
                        is_leader = 'Leader' in selected_employee
                        
                        if not is_leader:
                            # Remove from current department
                            current_dept_clean = current_dept
                            if employee_name in st.session_state.custom_team_leaders[current_dept_clean]['members']:
                                st.session_state.custom_team_leaders[current_dept_clean]['members'].remove(employee_name)
                                
                                # Add to target department
                                if employee_name not in st.session_state.custom_team_leaders[target_dept]['members']:
                                    st.session_state.custom_team_leaders[target_dept]['members'].append(employee_name)
                                    st.success(f"✅ Moved {employee_name} from {current_dept_clean} to {target_dept}")
                                    st.rerun()
                                else:
                                    st.warning(f"⚠️ {employee_name} already exists in {target_dept}")
                        else:
                            st.warning("⚠️ Cannot move team leaders. Please change leadership first.")
        
        # Reset to defaults option
        st.markdown("---")
        st.markdown("**Reset Options:**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Reset to Default Mappings", type="secondary"):
                st.session_state.custom_team_leaders = TEAM_LEADERS.copy()
                st.success("✅ Department mappings reset to defaults")
                st.rerun()
        
        with col2:
            # Download current mappings as JSON
            if st.button("💾 Download Current Mappings"):
                import json
                mapping_json = json.dumps(st.session_state.custom_team_leaders, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=mapping_json,
                    file_name="department_mappings.json",
                    mime="application/json"
                )

def load_budget_file(uploaded_file):
    """Load budget data from uploaded file (CSV or Excel)"""
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            budget_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            budget_df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file.")
            return None
        
        # Display the uploaded data for user verification
        st.subheader("📊 Uploaded Budget Data Preview")
        st.dataframe(budget_df.head(10))  # Show first 10 rows
        
        return budget_df
    
    except Exception as e:
        st.error(f"Error reading budget file: {str(e)}")
        return None

def process_budget_data(budget_df, focus_on_travel=True):
    """Process the uploaded budget DataFrame into usable format"""
    try:
        # Make a copy to avoid modifying original
        df = budget_df.copy()
        
        # Clean up column names - remove extra spaces and standardize
        df.columns = [str(col).strip() for col in df.columns]
        
        # The first column should be department/team names
        dept_column = df.columns[0]
        df = df.rename(columns={dept_column: 'Department'})
        
        # Convert Department column to string and clean up
        df['Department'] = df['Department'].astype(str)
        df['Department'] = [str(dept).strip() for dept in df['Department']]
        
        # Remove any rows where Department is NaN, empty, or contains 'Total'
        df = df[df['Department'] != 'nan']
        df = df[df['Department'] != '']
        df = df[~df['Department'].str.contains('Total', case=False, na=False)]
        
        # Get date columns (everything except the first column)
        date_columns = [col for col in df.columns if col != 'Department']
        
        # Convert budget data to numeric, handling currency symbols
        for col in date_columns:
            if col in df.columns:
                # Convert to string first, then remove currency symbols and convert to float
                df[col] = df[col].astype(str)
                df[col] = [str(val).replace('$', '').replace(',', '').replace(' ', '') for val in df[col]]
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create a dictionary structure for easy lookup
        budget_dict = {}
        
        for _, row in df.iterrows():
            dept = row['Department']
            
            # Use flexible department mapping
            mapped_dept = flexible_department_mapping(dept)
            
            if mapped_dept is None:
                continue  # Skip if department couldn't be mapped
            
            dept_budgets = {}
            for col in date_columns:
                if pd.notna(row[col]) and row[col] != 0:
                    # Convert date format from 'Mon-YY' to 'Mon YYYY'
                    try:
                        standardized_date = standardize_date_format(col)
                        
                        if standardized_date:
                            dept_budgets[standardized_date] = float(row[col])
                    except (ValueError, TypeError) as e:
                        continue
            
            if dept_budgets:  # Only add if we have valid budget data
                # If department already exists, merge the budgets
                if mapped_dept in budget_dict:
                    for period, amount in dept_budgets.items():
                        budget_dict[mapped_dept][period] = budget_dict[mapped_dept].get(period, 0) + amount
                else:
                    budget_dict[mapped_dept] = dept_budgets
        
        return budget_dict
    
    except Exception as e:
        st.error(f"Error processing budget data: {str(e)}")
        st.error("Please check the file format and ensure it contains department names in the first column and budget amounts in subsequent columns.")
        return None

def standardize_date_format(date_str):
    """
    Convert various date formats to 'Mon YYYY' format
    
    Examples:
    'Jun-25' -> 'Jun 2025'
    'Jul-25' -> 'Jul 2025'
    'Jan 2025' -> 'Jan 2025' (already correct)
    'January 2025' -> 'Jan 2025'
    """
    try:
        date_str = str(date_str).strip()
        
        # Handle 'Mon-YY' format (e.g., 'Jun-25')
        if '-' in date_str and len(date_str.split('-')) == 2:
            month_part, year_part = date_str.split('-')
            
            # Convert 2-digit year to 4-digit year
            if len(year_part) == 2:
                year_int = int(year_part)
                # Assume 00-30 is 2000-2030, 31-99 is 1931-1999
                if year_int <= 30:
                    full_year = 2000 + year_int
                else:
                    full_year = 1900 + year_int
            else:
                full_year = int(year_part)
            
            # Standardize month to 3-letter abbreviation
            month_abbr = standardize_month_name(month_part)
            
            return f"{month_abbr} {full_year}"
        
        # Handle 'Mon YYYY' format (already correct)
        elif ' ' in date_str and len(date_str.split(' ')) == 2:
            month_part, year_part = date_str.split(' ')
            month_abbr = standardize_month_name(month_part)
            return f"{month_abbr} {year_part}"
        
        # Handle other formats
        else:
            # Try to parse as date
            try:
                parsed_date = pd.to_datetime(date_str)
                return parsed_date.strftime('%b %Y')
            except:
                return None
    
    except Exception as e:
        return None

def standardize_month_name(month_str):
    """
    Convert month name to 3-letter abbreviation
    """
    month_mapping = {
        'january': 'Jan', 'jan': 'Jan',
        'february': 'Feb', 'feb': 'Feb',
        'march': 'Mar', 'mar': 'Mar',
        'april': 'Apr', 'apr': 'Apr',
        'may': 'May',
        'june': 'Jun', 'jun': 'Jun',
        'july': 'Jul', 'jul': 'Jul',
        'august': 'Aug', 'aug': 'Aug',
        'september': 'Sep', 'sep': 'Sep',
        'october': 'Oct', 'oct': 'Oct',
        'november': 'Nov', 'nov': 'Nov',
        'december': 'Dec', 'dec': 'Dec'
    }
    
    month_clean = month_str.lower().strip()
    return month_mapping.get(month_clean, month_str.capitalize())

def filter_expenses_for_travel(df):
    """
    Filter expense data to only include travel-related expenses
    """
    if 'Category' not in df.columns:
        st.warning("No 'Category' column found in expense data. Cannot filter for travel expenses.")
        return df
    
    # Filter for travel-related categories
    travel_categories = ['Biz Travel', 'Biz Travel: Meals', 'Biz Travel: Hotels', 'Biz Travel & Client Meeting: Transport']
    travel_mask = df['Category'].str.contains('|'.join(travel_categories), case=False, na=False)
    travel_df = df[travel_mask].copy()
    
    if len(travel_df) == 0:
        st.warning("No travel expenses found in the data.")
        return df
    
    st.info(f"📍 Filtered to {len(travel_df)} travel expenses out of {len(df)} total expenses")
    return travel_df

def calculate_budget_status_for_period(raw_df, budget_dict, start_date, end_date, focus_on_travel=True, test_mode=False):
    """Calculate budget status for a specific date range with transaction details"""
    
    # Use custom mappings if available, otherwise use default
    team_leaders = st.session_state.get('custom_team_leaders', TEAM_LEADERS)
    
    # Make a copy and filter by date range
    df = raw_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
    
    if len(df) == 0:
        return {}
    
    # Filter for travel expenses if needed
    if focus_on_travel:
        if 'Category Name' in df.columns:
            df = filter_expenses_for_travel_raw(df, 'Category Name')
        elif 'Category' in df.columns:
            df = filter_expenses_for_travel_raw(df, 'Category')
    
    # Handle amount column
    amount_column = 'Amount (by category)' if 'Amount (by category)' in df.columns else 'Amount'
    df['Amount'] = pd.to_numeric(df[amount_column], errors='coerce')
    df = df[df['Amount'].notna()]
    
    # Map employees to departments
    df['Department'] = df['Employee'].apply(map_employee_to_department)
    df = df[df['Department'].notna()]
    
    # Calculate period length and pro-rate budget
    period_days = (end_date - start_date).days + 1
    
    # Determine which months are covered by the period
    period_months = []
    current_date = pd.Timestamp(start_date)
    while current_date <= pd.Timestamp(end_date):
        month_str = current_date.strftime('%b %Y')
        if month_str not in period_months:
            period_months.append(month_str)
        current_date += pd.DateOffset(months=1)
    
    budget_status = {}
    
    for dept in budget_dict.keys():
        # Get team info from custom mappings
        if dept in team_leaders:
            team_info = {
                'leader': team_leaders[dept]['leader'],
                'members': team_leaders[dept]['members'],
                'leader_email': f"{team_leaders[dept]['leader'].lower().replace(' ', '.')}@company.com"  # Generate email
            }
        else:
            # Fallback to default if not found
            team_info = {'leader': 'Unknown', 'members': [], 'leader_email': 'unknown@company.com'}
        
        team_members = team_info.get('members', [])
        
        if not team_members:
            continue
        
        # Calculate allocated budget for the period
        allocated_budget = 0
        for month in period_months:
            if month in budget_dict[dept]:
                month_budget = budget_dict[dept][month]
                # Pro-rate based on how many days of the month are in our period
                month_start = pd.Timestamp(month + ' 01')
                month_end = month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                
                # Find overlap between period and month
                overlap_start = max(pd.Timestamp(start_date), month_start)
                overlap_end = min(pd.Timestamp(end_date), month_end)
                
                if overlap_start <= overlap_end:
                    days_in_month = month_end.day
                    overlap_days = (overlap_end - overlap_start).days + 1
                    prorated_budget = month_budget * (overlap_days / days_in_month)
                    allocated_budget += prorated_budget
        
        # Filter expenses for this department
        dept_expenses = df[df['Department'] == dept]
        
        # Calculate member-wise expenses and collect transaction details
        member_expenses = {}
        member_transaction_details = {}
        total_member_expenses = 0
        
        for member in team_members:
            member_dept_expenses = dept_expenses[dept_expenses['Employee'] == member]
            member_amount = member_dept_expenses['Amount'].sum()
            member_expenses[member] = float(member_amount)
            total_member_expenses += member_amount
            
            # Store transaction details for this member
            member_transactions = []
            for _, transaction in member_dept_expenses.iterrows():
                member_transactions.append({
                    'date': transaction['Date'],
                    'amount': transaction['Amount'],
                    'category': transaction.get('Category Name', transaction.get('Category', 'Other')),
                    'description': transaction.get('Description', ''),
                    'vendor': transaction.get('Vendor', '')
                })
            
            member_transaction_details[member] = member_transactions
        
        # Use the sum of member expenses as total spent
        total_spent = total_member_expenses
        
        # Calculate metrics
        remaining_budget = allocated_budget - total_spent
        utilization_rate = (total_spent / allocated_budget * 100) if allocated_budget > 0 else 0
        
        # Determine status
        if total_spent > allocated_budget:
            status = "Over Budget"
        elif utilization_rate > 80:
            status = "Near Limit"
        else:
            status = "On Track"
        
        budget_status[dept] = {
            'leader': team_info.get('leader', 'Unknown'),
            'leader_email': team_info.get('leader_email', 'unknown@company.com'),
            'department': dept,
            'team_members': team_members,
            'allocated_budget': allocated_budget,
            'total_spent': total_spent,
            'remaining_budget': remaining_budget,
            'utilization_rate': utilization_rate,
            'status': status,
            'member_expenses': member_expenses,
            'member_transaction_details': member_transaction_details,
            'transaction_count': len(dept_expenses),
            'test_mode': test_mode
        }
    
    return budget_status

# Additional helper function to debug the expense calculation
def debug_expense_calculation(df, department, team_members):
    """Debug helper to understand expense calculation discrepancies"""
    
    dept_expenses = df[df['Department'] == department]
    
    print(f"\n=== DEBUGGING {department} ===")
    print(f"Total department expense rows: {len(dept_expenses)}")
    print(f"Team members: {team_members}")
    print(f"Unique employees in expenses: {dept_expenses['Employee'].unique().tolist()}")
    
    # Check for employees in expenses but not in team
    expense_employees = set(dept_expenses['Employee'].unique())
    team_employees = set(team_members)
    
    unmatched_in_expenses = expense_employees - team_employees
    unmatched_in_team = team_employees - expense_employees
    
    if unmatched_in_expenses:
        print(f"⚠️ Employees in expenses but not in team: {list(unmatched_in_expenses)}")
        for emp in unmatched_in_expenses:
            emp_amount = dept_expenses[dept_expenses['Employee'] == emp]['Amount'].sum()
            print(f"  {emp}: ${emp_amount:.2f}")
    
    if unmatched_in_team:
        print(f"ℹ️ Team members with no expenses: {list(unmatched_in_team)}")
    
    # Show all expenses for this department
    print("\nAll department expenses:")
    for _, row in dept_expenses.iterrows():
        print(f"  {row['Employee']}: ${row['Amount']:.2f} on {row['Date']}")
    
    print(f"Total: ${dept_expenses['Amount'].sum():.2f}")
    print("=" * 50)

def filter_expenses_for_travel_raw(df, category_column):
    """
    Filter expense data to only include travel-related expenses for raw data
    """
    if category_column not in df.columns:
        st.warning(f"No '{category_column}' column found in expense data. Cannot filter for travel expenses.")
        return df
    
    # Filter for travel-related categories
    travel_categories = ['Biz Travel', 'Travel', 'Hotel', 'Airfare', 'Meals', 'Transportation']
    
    # Create mask for travel expenses
    travel_mask = df[category_column].str.contains('|'.join(travel_categories), case=False, na=False)
    travel_df = df[travel_mask].copy()
    
    if len(travel_df) == 0:
        st.warning("No travel expenses found in the data.")
        return df
    
    st.info(f"📍 Filtered to {len(travel_df)} travel expenses out of {len(df)} total expenses")
    return travel_df

# K-ID Brand Colors
BRAND_COLORS = {
    'purple': '#715DEC',
    'purple_dark': '#4E39D4',
    'purple_light': '#EBE8FF',
    'orange': '#FC6C0F',
    'blackberry': '#0F0740',
    'blackberry_light': '#2C216F',
    'black': '#505050',
    'gradient_start': '#AF7EF0',
    'gradient_end': '#715DEC'
}

# Status configurations
STATUS_CONFIG = {
    "On Track": {
        "emoji": "✅",
        "color": BRAND_COLORS['purple'],
        "bg_color": BRAND_COLORS['purple_light'],
        "border_color": BRAND_COLORS['purple']
    },
    "Near Limit": {
        "emoji": "⚠️",
        "color": BRAND_COLORS['orange'],
        "bg_color": "#FFF3E0",
        "border_color": BRAND_COLORS['orange']
    },
    "Over Budget": {
        "emoji": "🚨",
        "color": "#dc3545",
        "bg_color": "#f8d7da",
        "border_color": "#f5c6cb"
    }
}

def create_enhanced_member_transaction_display(member, team_info, period):
    """Create enhanced transaction display for email using existing data"""
    
    # Get member's transaction data
    member_amount = float(team_info['member_expenses'].get(member, 0))
    
    if member_amount == 0:
        return '<div style="font-size: 12px; color: #6c757d; font-style: italic;">No transactions recorded</div>'
    
    # Get member's transaction details
    member_transactions = team_info.get('member_transaction_details', {}).get(member, [])
    
    if not member_transactions:
        return f'<div style="font-size: 12px; color: {BRAND_COLORS["black"]};">No transaction details available</div>'
    
    # Sort transactions by date (most recent first) and amount (largest first)
    sorted_transactions = sorted(member_transactions, 
                                key=lambda x: (x.get('date', ''), -x.get('amount', 0)), 
                                reverse=True)
    
    # Generate enhanced display
    display_html = f'<div style="font-size: 12px; color: {BRAND_COLORS["black"]};">'
    
    # Transaction count and total
    transaction_count = len(member_transactions)
    display_html += f'<div style="margin-bottom: 6px; font-weight: 500; color: {BRAND_COLORS["purple"]};">{transaction_count} transaction{"s" if transaction_count != 1 else ""} totaling ${member_amount:,.2f}</div>'
    
    # Show top 2-3 transactions with key details
    max_transactions_to_show = min(3, len(sorted_transactions))
    
    for i, transaction in enumerate(sorted_transactions[:max_transactions_to_show]):
        date_str = transaction.get('date', 'N/A')
        amount = transaction.get('amount', 0)
        category = transaction.get('category', 'Other')
        vendor = transaction.get('vendor', 'Unknown vendor')
        
        # Format date
        try:
            if len(date_str) == 10 and date_str.count('-') == 2:
                date_parts = date_str.split('-')
                date_display = f"{date_parts[1]}/{date_parts[2]}"
            else:
                date_display = date_str
        except:
            date_display = date_str
        
        # Truncate vendor name if too long
        vendor_display = vendor if len(vendor) <= 25 else vendor[:22] + "..."
        
        # Create compact transaction line
        display_html += f'''
        <div style="margin-bottom: 4px; padding: 4px 6px; background-color: #f8f9fa; border-radius: 3px; border-left: 2px solid {BRAND_COLORS['purple_light']};">
            <span style="font-weight: 600; color: {BRAND_COLORS['purple']};">${amount:,.2f}</span>
            <span style="color: {BRAND_COLORS['blackberry']}; margin: 0 4px;">•</span>
            <span style="color: {BRAND_COLORS['blackberry']};">{category}</span>
            <span style="color: {BRAND_COLORS['blackberry']}; margin: 0 4px;">•</span>
            <span style="color: #6c757d; font-size: 11px;">{vendor_display}</span>
            <span style="color: #6c757d; font-size: 11px; float: right;">{date_display}</span>
        </div>
        '''
    
    # If more transactions, show summary
    if len(sorted_transactions) > 3:
        remaining_count = len(sorted_transactions) - 3
        remaining_total = sum(t.get('amount', 0) for t in sorted_transactions[3:])
        
        display_html += f'''
        <div style="margin-top: 6px; padding: 3px 6px; background-color: #e9ecef; border-radius: 3px; text-align: center;">
            <span style="color: {BRAND_COLORS['blackberry']}; font-size: 11px;">
                +{remaining_count} more (${remaining_total:,.2f})
            </span>
        </div>
        '''
    
    display_html += '</div>'
    
    return display_html

def create_budget_alert_html_email(team_info, period, focus_on_travel=True, mode="Full Validation"):
    """Create complete HTML email with K-ID branding and employee transaction summary"""
    
    budget_type = "Travel Budget" if focus_on_travel else "Budget"
    status_config = STATUS_CONFIG.get(team_info['status'], {
        "emoji": "📊",
        "color": BRAND_COLORS['purple'],
        "bg_color": BRAND_COLORS['purple_light'],
        "border_color": BRAND_COLORS['purple']
    })
    
    # Build member expenses table with enhanced transaction display
    member_expenses_html = ""
    for member_data in team_info['member_expenses'].items():
        if len(member_data) == 2:
            member, amount = member_data
        else:
            continue
        expense_display = f"${float(amount):,.2f}" if float(amount) > 0 else '<span style="color: #6c757d;">$0.00 (No expenses)</span>'
        
        # Get enhanced transaction display for this member
        transaction_display = create_enhanced_member_transaction_display(member, team_info, period)
        
        member_expenses_html += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #e9ecef;">
                <div style="font-weight: 600; color: {BRAND_COLORS['blackberry']}; margin-bottom: 6px;">
                    {member}
                </div>
                {transaction_display}
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e9ecef; text-align: right; vertical-align: top;">
                <div style="font-size: 16px; font-weight: 600; color: {BRAND_COLORS['blackberry']};">
                    {expense_display}
                </div>
            </td>
        </tr>
        """
    
    # Status-specific recommendations
    recommendations_html = _get_status_recommendations(team_info, status_config)
    
    # Test mode warning
    test_warning = _get_test_mode_warning(team_info) if team_info.get('test_mode', False) else ""
    
    # Email recipient
    recipient_email = "lulu@k-id.com" if team_info.get('test_mode', False) else team_info['leader_email']
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="use_container_width=device-use_container_width, initial-scale=1.0">
        <title>{budget_type} Alert - {team_info['department']} Team</title>
    </head>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: {BRAND_COLORS['blackberry']}; margin: 0; padding: 0; background-color: #f8f9fa;">
        <div style="max-use_container_width: 800px; margin: 0 auto; background-color: #ffffff;">
            <!-- Header with K-ID branding -->
            <div style="background: linear-gradient(135deg, {BRAND_COLORS['gradient_start']} 0%, {BRAND_COLORS['gradient_end']} 100%); padding: 30px; text-align: center; border-radius: 0;">
                <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 28px; font-weight: 600;">
                    {budget_type} Alert {status_config['emoji']}
                </h1>
                <p style="color: #ffffff; margin: 0; font-size: 18px; opacity: 0.9;">
                    {team_info['department']} Team
                </p>
            </div>
            
            <!-- Main content -->
            <div style="padding: 40px;">
                <div style="margin-bottom: 30px;">
                    <h2 style="color: {BRAND_COLORS['blackberry']}; margin: 0 0 15px 0; font-size: 20px;">
                        Dear {team_info['leader']},
                    </h2>
                    <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['black']};">
                        This is an automated alert regarding your team's {budget_type.lower()} status for the period <strong>{period['start']} to {period['end']}</strong>.
                    </p>
                </div>
                
                {test_warning}
                
                <!-- Budget Overview -->
                <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 2px 10px rgba(113, 93, 236, 0.1);">
                    <h2 style="color: {BRAND_COLORS['purple']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                        <span style="margin-right: 10px;">💰</span> {budget_type} Overview
                    </h2>
                    <table style="use_container_width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Allocated Budget:</td>
                            <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; font-size: 16px;">${team_info['allocated_budget']:,.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Total Spent:</td>
                            <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; font-size: 16px;">${team_info['total_spent']:,.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Remaining Budget:</td>
                            <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; color: {status_config['color']}; font-weight: 600; font-size: 16px;">${team_info['remaining_budget']:,.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Utilization Rate:</td>
                            <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; font-size: 16px;">{team_info['utilization_rate']:.1f}%</td>
                        </tr>
                        <tr>
                            <td style="padding: 15px 0; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Status:</td>
                            <td style="padding: 15px 0; text-align: right; color: {status_config['color']}; font-weight: bold; font-size: 16px;">{team_info['status']} {status_config['emoji']}</td>
                        </tr>
                    </table>
                </div>
                
                <!-- Team Member Spending with Transaction Summary -->
                <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 2px 10px rgba(113, 93, 236, 0.1);">
                    <h2 style="color: {BRAND_COLORS['purple']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                        <span style="margin-right: 10px;">📊</span> Team Member Spending & Activity
                    </h2>
                    <table style="use_container_width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background-color: {BRAND_COLORS['purple_light']};">
                                <th style="padding: 15px; text-align: left; color: {BRAND_COLORS['blackberry']}; font-weight: 600;">Team Member & Activity</th>
                                <th style="padding: 15px; text-align: right; color: {BRAND_COLORS['blackberry']}; font-weight: 600;">Amount Spent</th>
                            </tr>
                        </thead>
                        <tbody>
                            {member_expenses_html}
                        </tbody>
                    </table>
                </div>
                
                {recommendations_html}
                
                <!-- Next Steps -->
                <div style="background: linear-gradient(135deg, {BRAND_COLORS['blackberry_light']} 0%, {BRAND_COLORS['blackberry']} 100%); padding: 30px; border-radius: 12px; margin: 30px 0; color: #ffffff;">
                    <h2 style="color: #ffffff; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                        <span style="margin-right: 10px;">📋</span> Next Steps
                    </h2>
                    <ul style="margin: 0; padding-left: 20px; font-size: 16px;">
                        <li style="margin-bottom: 10px;"><strong>View Detailed Dashboard</strong> - Check detailed transactions</li>
                        <li style="margin-bottom: 10px;"><strong>Request Budget Adjustment</strong> - If required</li>
                        <li style="margin-bottom: 10px;"><strong>Team Policy Reminder</strong> - Share guidelines with team</li>
                    </ul>
                    <p style="margin: 20px 0 0 0; font-size: 16px;">Please review your team's spending and take appropriate action if necessary.</p>
                </div>
                
                <!-- Footer -->
                <div style="margin-top: 40px; padding-top: 30px; border-top: 2px solid {BRAND_COLORS['purple_light']}; text-align: center;">
                    <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                        Best regards,<br>
                        <strong style="color: {BRAND_COLORS['purple']};">Finance Team</strong>
                    </p>
                    <p style="margin: 0; font-size: 14px; color: {BRAND_COLORS['black']};">
                        This is an automated alert. Questions? Contact 
                        <a href="mailto:finance@k-id.com" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">finance@k-id.com</a>
                    </p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_body, recipient_email

def _get_status_recommendations(team_info, status_config):
    """Generate status-specific recommendations HTML"""
    if team_info['status'] == "Over Budget":
        return f"""
        <div style="background-color: {status_config['bg_color']}; border: 2px solid {status_config['border_color']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
            <h3 style="color: {status_config['color']}; margin: 0 0 15px 0; font-size: 20px;">
                🚨 URGENT ACTION REQUIRED
            </h3>
            <p style="margin: 0 0 15px 0; font-size: 16px;">
                Your team has exceeded the allocated budget by <strong>${abs(team_info['remaining_budget']):,.2f}</strong>. Please:
            </p>
            <ul style="margin: 0; padding-left: 20px; font-size: 16px;">
                <li style="margin-bottom: 8px;">Review all pending expenses immediately</li>
                <li style="margin-bottom: 8px;">Defer any non-essential purchases</li>
                <li style="margin-bottom: 8px;">Contact Finance to discuss budget adjustment if necessary</li>
            </ul>
        </div>
        """
    elif team_info['status'] == "Near Limit":
        return f"""
        <div style="background-color: {status_config['bg_color']}; border: 2px solid {status_config['border_color']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
            <h3 style="color: {status_config['color']}; margin: 0 0 15px 0; font-size: 20px;">
                ⚠️ CAUTION - APPROACHING LIMIT
            </h3>
            <p style="margin: 0 0 15px 0; font-size: 16px;">
                Your team is nearing the budget limit with only <strong>${team_info['remaining_budget']:,.2f}</strong> remaining. Please:
            </p>
            <ul style="margin: 0; padding-left: 20px; font-size: 16px;">
                <li style="margin-bottom: 8px;">Monitor remaining budget closely</li>
                <li style="margin-bottom: 8px;">Prioritize only essential expenditures</li>
                <li style="margin-bottom: 8px;">Seek pre-approval for any large purchases</li>
            </ul>
        </div>
        """
    return ""

def _get_member_transaction_summary(member, team_info, period):
    """Generate mini transaction summary for a team member"""
    
    # Get member's transaction data from the raw data
    # This would need to be passed or accessed from the team_info
    member_amount = float(team_info['member_expenses'].get(member, 0))
    
    if member_amount == 0:
        return '<div style="font-size: 12px; color: #6c757d; font-style: italic;">No transactions recorded</div>'
    
    # You'll need to modify your calculate_budget_status_for_period function
    # to also store transaction details for each member
    member_transactions = team_info.get('member_transaction_details', {}).get(member, [])
    
    if not member_transactions:
        return f'<div style="font-size: 12px; color: {BRAND_COLORS["black"]};">{len(member_transactions)} transactions</div>'
    
    # Generate summary based on available transaction data
    summary_parts = []
    
    # Transaction count
    transaction_count = len(member_transactions)
    summary_parts.append(f"{transaction_count} transaction{'s' if transaction_count != 1 else ''}")
    
    # Most frequent category or largest expense
    if member_transactions:
        # Group by category and find most common
        categories = {}
        largest_expense = 0
        largest_category = ""
        
        for transaction in member_transactions:
            category = transaction.get('category', 'Other')
            amount = transaction.get('amount', 0)
            
            if category not in categories:
                categories[category] = {'count': 0, 'total': 0}
            categories[category]['count'] += 1
            categories[category]['total'] += amount
            
            if amount > largest_expense:
                largest_expense = amount
                largest_category = category
        
        # Find most frequent category
        most_frequent_category = max(categories.keys(), key=lambda x: categories[x]['count']) if categories else "Other"
        
        # Add category info
        if most_frequent_category:
            category_count = categories[most_frequent_category]['count']
            if category_count > 1:
                summary_parts.append(f"mainly {most_frequent_category.lower()}")
            else:
                summary_parts.append(f"{most_frequent_category.lower()}")
        
        # Add largest expense if significant
        if largest_expense > member_amount * 0.5:  # If single transaction is >50% of total
            summary_parts.append(f"largest: ${largest_expense:,.0f}")
    
    # Join summary parts
    summary_text = " • ".join(summary_parts)
    
    return f'<div style="font-size: 12px; color: {BRAND_COLORS["black"]};">{summary_text}</div>'

def _get_test_mode_warning(team_info):
    """Generate test mode warning HTML"""
    return f"""
    <div style="background-color: #d1ecf1; border: 2px solid #bee5eb; padding: 25px; border-radius: 12px; margin: 30px 0;">
        <h3 style="color: #0c5460; margin: 0 0 15px 0; font-size: 20px;">
            ⚠️ THIS IS A TEST EMAIL
        </h3>
        <p style="margin: 0; font-size: 16px;">
            Original recipient: <strong>{team_info.get('leader_email', 'Unknown')}</strong>
        </p>
    </div>
    """

def _normalize_team_data(team_budget_data):
    """Normalize team data to consistent list format"""
    if isinstance(team_budget_data, dict):
        team_list = []
        for team_name, team_info in team_budget_data.items():
            if isinstance(team_info, dict):
                if 'department' not in team_info:
                    team_info['department'] = team_name
                team_list.append(team_info)
        return team_list
    elif isinstance(team_budget_data, list):
        return team_budget_data
    else:
        raise ValueError("team_budget_data must be a list of dictionaries or a dictionary of team data")

def _validate_team_info(team_info, index):
    """Validate team information and return validation result"""
    if not isinstance(team_info, dict):
        return False, f"Item {index} is not a dictionary. Got: {type(team_info)}"
    
    required_fields = ['department', 'leader', 'allocated_budget', 'total_spent', 
                      'remaining_budget', 'utilization_rate', 'status', 'member_expenses']
    missing_fields = [field for field in required_fields if field not in team_info]
    
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    return True, ""

def _create_email_message(team_info, period, smtp_config, test_mode, focus_on_travel):
    """Create email message with HTML content"""
    html_body, recipient_email = create_budget_alert_html_email(team_info, period, focus_on_travel, mode=mode)
    
    msg = EmailMessage()
    msg['From'] = smtp_config['user']
    msg['To'] = recipient_email
    
    budget_type = "Travel Budget" if focus_on_travel else "General Budget"
    status_config = STATUS_CONFIG.get(team_info['status'], {"emoji": "📊"})
    test_prefix = "[TEST MODE] " if test_mode else ""
    
    msg['Subject'] = f"{test_prefix}{budget_type} Alert - {team_info['department']} Team {status_config['emoji']}"
    msg.add_alternative(html_body, subtype='html')
    
    return msg, recipient_email, html_body

def add_scheduled_email(email_id, email_data):
    """Add email to the session state scheduled emails dictionary"""
    global EMAIL_LOCK
    with EMAIL_LOCK:
        st.session_state.SCHEDULED_EMAILS[email_id] = email_data
    return email_id

def add_scheduled_policy_email(email_id, email_data):
    """Add policy email to the session state scheduled emails dictionary"""
    global EMAIL_LOCK
    with EMAIL_LOCK:
        st.session_state.POLICY_SCHEDULED_EMAILS[email_id] = email_data
    return email_id

def send_budget_alert_emails_with_ui(team_budget_data, smtp_config, test_mode=True, delay_minutes=5, focus_on_travel=True):
    """Send budget alert emails and launch static UI for management"""
    emails_scheduled = 0
    emails_skipped = 0
    
    try:
        # Normalize team data format
        team_list = _normalize_team_data(team_budget_data)
        
        # Process each team
        for i, team_info in enumerate(team_list):
            # Validate team information
            is_valid, error_msg = _validate_team_info(team_info, i)
            if not is_valid:
                print(f"❌ Error: Team {team_info.get('leader', 'Unknown')} - {error_msg}")
                emails_skipped += 1
                continue
            
            leader_name = team_info.get('leader', 'Unknown')
            
            # Check for email address in non-test mode
            if not test_mode and not team_info.get('leader_email'):
                print(f"No email found for team leader: {leader_name}. Skipping notification.")
                emails_skipped += 1
                continue
            
            # Set test mode flag for email generation
            team_info['test_mode'] = test_mode
            
            # Generate email content
            period = {
                'start': team_info.get('period_start', 'N/A'),
                'end': team_info.get('period_end', 'N/A')
            }
            
            try:
                msg, recipient_email, html_body = _create_email_message(
                    team_info, period, smtp_config, test_mode, focus_on_travel
                )
            except Exception as e:
                print(f"❌ Error generating email content for {leader_name}: {e}")
                emails_skipped += 1
                continue
            
            # Create email data for delayed sending
            department = team_info.get('department', 'Unknown Department')
            email_id = f"BUDGET_{department.replace(' ', '_')}_{leader_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            email_data = {
                'id': email_id,
                'employee': f"{leader_name} ({department})",
                'to_addr': recipient_email,
                'cc_addr': 'finance@k-id.com',
                'subject': msg['Subject'],  # Extract subject from the message
                'from_addr': smtp_config['user'],
                'html_body': html_body,  # This is already returned from create_email_message
                'message': msg,
                'smtp_config': smtp_config,
                'excel_file': None,
                'send_immediately': False,
                'scheduled_time': datetime.now() + timedelta(minutes=delay_minutes),
                'has_attachment': False,
                'violations_count': 0
            }

            add_scheduled_email(email_id, email_data)
            
            # Start the delayed sending in a separate thread
            thread = threading.Thread(target=send_delayed_email, args=(email_data, delay_minutes, None, st.session_state.SCHEDULED_EMAILS), daemon=True)
            thread.start()
            
            status_config = STATUS_CONFIG.get(team_info.get('status', 'Unknown'), {"emoji": "📊"})
            print(f"📧 Budget alert scheduled for {leader_name} - {department} ({recipient_email})")
            print(f"   Email ID: {email_id}")
            print(f"   Status: {team_info.get('status', 'Unknown')} {status_config['emoji']}")
            emails_scheduled += 1
        
        print(f"\n📈 BUDGET ALERT SCHEDULING COMPLETE:")
        print(f"   Scheduled: {emails_scheduled}")
        print(f"   Skipped: {emails_skipped}")
        print(f"   Delay: {delay_minutes} minutes")
        
        # Generate static HTML and serve it
        generate_and_serve_static_ui(emails_scheduled, delay_minutes, mode, email_type="budget")
        
        # Open browser automatically
        webbrowser.open('http://localhost:5000')
        
        print(f"\n🎯 BUDGET ALERT UI LAUNCHED!")
        print(f"   📧 {emails_scheduled} budget alerts scheduled")
        print(f"   🌐 UI available at: http://localhost:5000")
        print(f"   ⏰ Emails will send in {delay_minutes} minutes")
        print(f"   💡 Static snapshot of current email status")
        
        return {
            'success': True,
            'scheduled_count': emails_scheduled,
            'skipped_count': emails_skipped,
            'delay_minutes': delay_minutes,
            'message': f'Successfully scheduled {emails_scheduled} budget alert emails with {delay_minutes} minute delay'
        }
        
    except Exception as e:
        print(f"❌ Failed to schedule budget alert emails: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to schedule budget alert emails: {e}'
        }





# EMAIL UI SYSTEM
email_manager_running = False

def get_employee_email(employee_name):
    """
    Get employee email from the mapping, with fallback logic for name variations
    """
    if not employee_name:
        return None
    
    # Clean the input name
    employee_name = str(employee_name).strip()
    
    # Direct lookup first
    if employee_name in EMPLOYEE_EMAIL_MAPPING:
        return EMPLOYEE_EMAIL_MAPPING[employee_name]
    
    # Try variations for common name formats
    name_variations = [
        employee_name.strip(),
        employee_name.title(),
        employee_name.lower().title(),
    ]
    
    # Check for partial matches (first name only)
    first_name = employee_name.split()[0] if ' ' in employee_name else employee_name
    if first_name in EMPLOYEE_EMAIL_MAPPING:
        return EMPLOYEE_EMAIL_MAPPING[first_name]
    
    # Check all variations
    for variation in name_variations:
        if variation in EMPLOYEE_EMAIL_MAPPING:
            return EMPLOYEE_EMAIL_MAPPING[variation]
    
    # Try case-insensitive lookup
    employee_name_lower = employee_name.lower()
    for key, email in EMPLOYEE_EMAIL_MAPPING.items():
        if key.lower() == employee_name_lower:
            return email
    
    # Try partial matching for common variations
    for key, email in EMPLOYEE_EMAIL_MAPPING.items():
        # Check if the input name is contained in the key or vice versa
        if employee_name.lower() in key.lower() or key.lower() in employee_name.lower():
            return email
    
    return None

def signal_handler(sig, frame):
    print('\n\n🛑 Email scheduler interrupted by user')
    print('📧 Scheduled emails will continue running in background')
    print('🌐 Static UI server will remain active')
    print('💡 You can now use the terminal for commands')
    sys.exit(0)

def cancel_all_scheduled_emails():
    """Cancel all scheduled BUDGET emails using passed dictionary reference"""
    global EMAIL_LOCK
    cancelled_count = 0
    
    # Get the actual dictionary reference from session state
    budget_dict = st.session_state.SCHEDULED_EMAILS
    
    print(f"🔍 CANCEL BUDGET FUNCTION: Starting...")
    print(f"🔍 CANCEL BUDGET FUNCTION: Dict has {len(budget_dict)} emails")
    print(f"🔍 CANCEL BUDGET FUNCTION: Dict ID: {id(budget_dict)}")
    
    with EMAIL_LOCK:
        # Work directly with the dictionary reference
        email_ids_to_cancel = []
        
        for email_id, email_data in budget_dict.items():
            if not email_data.get('cancelled', False) and not email_data.get('sent', False) and not email_data.get('failed', False):
                email_ids_to_cancel.append(email_id)
        
        # Cancel them using the dictionary reference
        for email_id in email_ids_to_cancel:
            if email_id in budget_dict:
                budget_dict[email_id]['cancelled'] = True
                budget_dict[email_id]['cancelled_time'] = datetime.now()
                cancelled_count += 1
                print(f"✅ MARKED FOR CANCELLATION: BUDGET email {email_id}")
    
    return cancelled_count

def cancel_all_scheduled_policy_emails():
    """Cancel all scheduled POLICY emails using passed dictionary reference"""
    global EMAIL_LOCK
    cancelled_count = 0
    
    # Get the actual dictionary reference from session state
    policy_dict = st.session_state.POLICY_SCHEDULED_EMAILS
    
    print(f"🔍 CANCEL POLICY FUNCTION: Starting...")
    print(f"🔍 CANCEL POLICY FUNCTION: Dict has {len(policy_dict)} emails")
    print(f"🔍 CANCEL POLICY FUNCTION: Dict ID: {id(policy_dict)}")
    
    with EMAIL_LOCK:
        # Check if dictionary exists and has content
        if not policy_dict:
            print(f"🔍 CANCEL POLICY FUNCTION: Dictionary is empty or None")
            return 0
        
        email_ids_to_cancel = []
        
        for email_id, email_data in policy_dict.items():
            if not email_data.get('cancelled', False) and not email_data.get('sent', False) and not email_data.get('failed', False):
                email_ids_to_cancel.append(email_id)
                print(f"🔍 CANCEL POLICY FUNCTION: Found cancellable email: {email_id}")
        
        print(f"🔍 CANCEL POLICY FUNCTION: Found {len(email_ids_to_cancel)} emails to cancel")
        
        # Cancel them using the dictionary reference
        for email_id in email_ids_to_cancel:
            if email_id in policy_dict:
                policy_dict[email_id]['cancelled'] = True
                policy_dict[email_id]['cancelled_time'] = datetime.now()
                cancelled_count += 1
                print(f"✅ MARKED FOR CANCELLATION: POLICY email {email_id}")
    
    print(f"🔍 CANCEL POLICY FUNCTION: Completed. Cancelled {cancelled_count} emails")
    return cancelled_count

def cancel_all_scheduled_fixed_asset_emails():
    
    # Initialize if not exists
    if 'FIXED_ASSET_SCHEDULED_EMAILS' not in st.session_state:
        st.session_state.FIXED_ASSET_SCHEDULED_EMAILS = {}
        return 0

    cancelled_count = 0
    EMAIL_LOCK = threading.Lock()

    with EMAIL_LOCK:
        for email_id, email_data in st.session_state.FIXED_ASSET_SCHEDULED_EMAILS.items():
            if not email_data.get('cancelled', False) and not email_data.get('sent', False) and not email_data.get('failed', False):
                email_data['cancelled'] = True
                email_data['cancelled_time'] = datetime.now()
                cancelled_count += 1
                print(f"❌ Cancelled fixed asset email: {email_id}")

    print(f"❌ Cancelled {cancelled_count} scheduled fixed asset emails")
    return cancelled_count

def create_excel_report(employee_name, violations, output_dir="temp_reports"):
    """Create an Excel report for employees with many violations"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for Excel
    excel_data = []
    for i, violation in enumerate(violations, 1):
        memo_text = violation.get('Memo', '')
        if not memo_text or str(memo_text).strip() == '' or str(memo_text).lower() == 'nan':
            memo_text = "No memo provided"
        
        receipt_status = "Yes" if violation.get('HasReceipt', True) else "No"
        company_funded = violation.get('Company_Funded_Amount', violation['Amount'])
        employee_out_of_pocket = violation.get('Employee_Out_of_Pocket', 0.0)
        
        excel_data.append({
            'Transaction #': i,
            'Date': violation['Date'],
            'Total Amount': violation['Amount'],
            'Company Funded': company_funded,
            'Employee Out-of-Pocket': employee_out_of_pocket,
            'Category': violation['Category'],
            'Vendor': violation.get('Vendor', 'N/A'),
            'Transaction ID': violation.get('Transaction_ID', 'N/A'),
            'Has Receipt': receipt_status,
            'Memo': memo_text,
            'Policy Issues': violation['Message']
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{employee_name.replace(' ', '_')}_violations_{timestamp}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    # Create Excel with formatting
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Policy Violations', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Policy Violations']
        
        # Auto-adjust column use_container_widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_use_container_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].use_container_width = adjusted_use_container_width
    
    return filepath


def send_delayed_email(email_data, delay_minutes=5, policy_dict=None, budget_dict=None):
    """Send an email after a specified delay, checking passed dictionaries for cancellation"""
    global EMAIL_LOCK
    
    email_id = email_data['id']
    mode = email_data.get('mode', 'Full Validation')
    is_policy_email = email_id.startswith('POLICY_')
    
    # Use the passed dictionaries instead of session state
    email_dict = policy_dict if is_policy_email else budget_dict
    
    if email_dict is None:
        print(f"❌ No dictionary passed for email {email_id}")
        return False
    
    # Check if this email exists in the dictionary
    email_exists = email_id in email_dict
    
    email_type_label = "Policy" if is_policy_email else "Budget"
    violation_type = "missing receipt" if mode == "Receipt-Only Mode" else ("policy violation" if is_policy_email else "budget alert")

    print(f"📧 {email_type_label} {violation_type} email {email_id} scheduled for {delay_minutes} minutes from now...")
    
    # Wait for the delay period, checking every 10 seconds
    total_seconds = delay_minutes * 60
    check_interval = 10
    elapsed = 0
    
    while elapsed < total_seconds:
        # Use thread lock when checking status
        with EMAIL_LOCK:
            current_email_status = email_dict.get(email_id, {})
            is_cancelled = current_email_status.get('cancelled', False)
            send_immediately = current_email_status.get('send_immediately', False)

        if is_cancelled:
            print(f"❌ {email_type_label} {violation_type} email {email_id} cancelled before sending (loop check)")
            # Update the dictionary with lock
            with EMAIL_LOCK:
                if email_id in email_dict:
                    email_dict[email_id]['cancelled'] = True
                    email_dict[email_id]['cancelled_time'] = datetime.now()
            return False

        # Check if marked for immediate sending
        if send_immediately:
            print(f"🚀 {email_type_label} {violation_type} email {email_id} marked for immediate sending")
            break

        time.sleep(min(check_interval, total_seconds - elapsed))
        elapsed += check_interval

        # Show countdown every 2 minutes
        if elapsed % 120 == 0 and elapsed < total_seconds:
            remaining = (total_seconds - elapsed) // 60
            print(f"⏰ {email_type_label} {violation_type} email {email_id} will be sent in {remaining} minutes")
    
    # Final check before sending with lock
    with EMAIL_LOCK:
        final_email_status = email_dict.get(email_id, {})
        is_cancelled = final_email_status.get('cancelled', False)
        print(f"🔍 DEBUG FINAL CHECK {email_id}: cancelled={is_cancelled}")
        print(f"🔍 DEBUG FINAL CHECK {email_id}: Dict ID={id(email_dict)}, email exists in dict={email_id in email_dict}")
        if email_id in email_dict:
            print(f"🔍 DEBUG FINAL CHECK {email_id}: Full email data cancelled status={email_dict[email_id].get('cancelled', 'NOT_SET')}")

    if is_cancelled:
        print(f"❌ {email_type_label} {violation_type} email {email_id} cancelled before sending (final check)")
        with EMAIL_LOCK:
            if email_id in email_dict:
                email_dict[email_id]['cancelled'] = True
                email_dict[email_id]['cancelled_time'] = datetime.now()
        return False
    
    # Send the email
    try:
        import smtplib
        import os
        
        print(f"🔍 DEBUG SENDING: About to send {email_id}")
        
        with smtplib.SMTP(email_data['smtp_config']['server'], email_data['smtp_config']['port']) as smtp:
            smtp.starttls()
            smtp.login(email_data['smtp_config']['user'], email_data['smtp_config']['password'])
            
            # Get all recipients (TO + CC)
            msg = email_data['message']
            to_addrs = [email_data['to_addr']]
            
            # Add CC recipients if present
            if 'Cc' in msg:
                cc_addrs = [addr.strip() for addr in msg['Cc'].split(',')]
                to_addrs.extend(cc_addrs)
            
            # Send to all recipients (TO and CC)
            smtp.send_message(msg, to_addrs=to_addrs)
        
        print(f"✅ {email_type_label} {violation_type} email {email_id} sent successfully to {email_data['employee']} ({email_data['to_addr']})")
        print(f"   📊 Mode: {mode}")
        if is_policy_email:
            print(f"   📧 Violations: {email_data.get('violations_count', 'Unknown')}")
        
        # Mark as sent in dictionary with lock
        with EMAIL_LOCK:
            if email_id in email_dict:
                email_dict[email_id]['sent'] = True
                email_dict[email_id]['sent_time'] = datetime.now()
        
        # Clean up Excel file if it was created (mainly for policy emails)
        if 'excel_file' in email_data and email_data['excel_file'] and os.path.exists(email_data['excel_file']):
            os.remove(email_data['excel_file'])
            print(f"🗑️ Temporary Excel file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to send {email_type_label.lower()} {violation_type} email {email_id}: {e}")
        # Mark as failed in dictionary with lock
        with EMAIL_LOCK:
            if email_id in email_dict:
                email_dict[email_id]['failed'] = True
                email_dict[email_id]['failed_time'] = datetime.now()
                email_dict[email_id]['error'] = str(e)
        return False
    
def create_email_body_html(emp, violations, mode="Full Validation", has_excel_attachment=False):
    """Create HTML email body - simplified for Excel attachments"""
    
    if mode == "Receipt-Only Mode":
        greeting = f"Hi {emp},"
        intro = """
        <p>We've identified some transactions that are missing receipts. Please upload the required receipts 
        to ensure compliance with our expense policy.</p>
        """
        violation_header = "Missing Receipts"
        violation_icon = "🧾"
    else:
        greeting = f"Hi {emp},"
        intro = """
        <p>We've identified some expenses that require your attention to ensure compliance with our corporate 
        expense policy. Please review and take the necessary actions.</p>
        """
        violation_header = "Policy Violations"
        violation_icon = "⚠️"
    
    # Build violations HTML
    violations_html = ""
    for i, v in enumerate(violations[:10], 1):  # Show first 10 violations
        violations_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 8px; font-weight: bold;">{i}</td>
            <td style="padding: 8px;">{v['Date']}</td>
            <td style="padding: 8px;">${v['Amount']:.2f}</td>
            <td style="padding: 8px;">{v['Category']}</td>
            <td style="padding: 8px;">{v['Vendor']}</td>
            <td style="padding: 8px; color: #e74c3c;">{v['Message']}</td>
        </tr>
        """
    
    if has_excel_attachment:
        # For >10 violations, just reference the Excel file
        total_amount = sum([v['Amount'] for v in violations])
        total_employee_cost = sum([v['Employee_Out_of_Pocket'] for v in violations])
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="use_container_width=device-use_container_width, initial-scale=1.0">
            <title>k-ID Corporate Card Policy Alert</title>
        </head>
        <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: {BRAND_COLORS['blackberry']}; margin: 0; padding: 0; background-color: #f8f9fa;">
            <div style="max-use_container_width: 800px; margin: 0 auto; background-color: #ffffff;">
                <!-- Header with K-ID branding -->
                <div style="background: linear-gradient(135deg, {BRAND_COLORS['gradient_start']} 0%, {BRAND_COLORS['gradient_end']} 100%); padding: 30px; text-align: center; border-radius: 0;">
                    <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 28px; font-weight: 600;">
                        Corporate Card Policy Alert 📊
                    </h1>
                    <p style="color: #ffffff; margin: 0; font-size: 18px; opacity: 0.9;">
                        k-ID Expense Policy Checker System
                    </p>
                </div>
                
                <!-- Main content -->
                <div style="padding: 40px;">
                    <p style="margin: 0 0 20px 0; font-size: 14px; color: {BRAND_COLORS['black']}; font-style: italic;">
                        This is an automated notification from k-ID's Expense Policy Checker system.
                    </p>
                    
                    <div style="margin-bottom: 30px;">
                        <h2 style="color: {BRAND_COLORS['blackberry']}; margin: 0 0 15px 0; font-size: 20px;">
                            Hi {emp},
                        </h2>
                        <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['black']};">
                            Hope you're doing great! We've identified some items with your recent corporate card expenses that need your attention to ensure full compliance with our k-ID Corporate Card Policy.
                        </p>
                    </div>
                    
                    <!-- Summary Section -->
                    <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 2px 10px rgba(113, 93, 236, 0.1);">
                        <h3 style="color: {BRAND_COLORS['purple']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">📊</span> Summary
                        </h3>
                        <table style="use_container_width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Total Transactions Requiring Action:</td>
                                <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; font-size: 16px; color: {BRAND_COLORS['purple']}; font-weight: 600;">{len(violations)}</td>
                            </tr>
                            <tr>
                                <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Total Transaction Amount:</td>
                                <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; font-size: 16px; color: {BRAND_COLORS['purple']}; font-weight: 600;">${total_amount:.2f}</td>
                            </tr>
                            {f'''<tr>
                                <td style="padding: 15px 0; font-weight: 600; color: {BRAND_COLORS['blackberry']};">⚠️ Amount You May Need to Cover:</td>
                                <td style="padding: 15px 0; text-align: right; font-size: 16px; color: {BRAND_COLORS['orange']}; font-weight: bold;">${total_employee_cost:.2f}</td>
                            </tr>''' if total_employee_cost > 0 else ''}
                        </table>
                    </div>
                    
                    </div>
                    
                    <!-- Important Note Section -->
                    <div style="background-color: #f8f9fa; border-left: 4px solid {BRAND_COLORS['purple']}; padding: 15px; margin: 20px 0; border-radius: 4px;">
                        <p style="margin: 0; font-size: 14px; color: {BRAND_COLORS['blackberry']}; font-style: italic;">
                            <strong>Note:</strong> Expenses categorized under 'Biz Travel: Airfare' and 'Biz Travel: Hotels' will be flagged whenever amounts exceed company budgets (<strong>$2,500</strong> for airfares and <strong>$250</strong> per-night rates for hotels). This limitation arises because the Rippling platform does not provide an API to directly extract detailed receipt information. Please manually review these expenses to ensure policy compliance.
                        </p>
                    </div>
                    
                    <!-- Excel Report Section -->
                    <div style="background: linear-gradient(135deg, {BRAND_COLORS['purple_light']} 0%, #ffffff 100%); border: 2px solid {BRAND_COLORS['purple']}; padding: 30px; border-radius: 12px; margin: 30px 0;">
                        <h3 style="color: {BRAND_COLORS['purple_dark']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">📎</span> Detailed Excel Report Attached
                        </h3>
                        <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                            Due to the number of violations ({len(violations)} transactions), we've attached a comprehensive Excel report containing:
                        </p>
                        <ul style="margin: 0 0 15px 0; padding-left: 20px; font-size: 16px; color: {BRAND_COLORS['black']};">
                            <li style="margin-bottom: 8px;">Complete transaction details (date, amount, vendor)</li>
                            <li style="margin-bottom: 8px;">Specific policy violation information</li>
                            <li style="margin-bottom: 8px;">Receipt status for each transaction</li>
                            <li style="margin-bottom: 8px;">Company vs. employee cost allocation</li>
                            <li style="margin-bottom: 8px;">Transaction IDs and categories</li>
                        </ul>
                        <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['purple_dark']}; font-weight: 600;">
                            Please review the Excel file carefully - it contains all the information you need to resolve these policy violations.
                        </p>
                    </div>
                    
                    <!-- Next Steps Section -->
                    <div style="background: linear-gradient(135deg, {BRAND_COLORS['blackberry_light']} 0%, {BRAND_COLORS['blackberry']} 100%); padding: 30px; border-radius: 12px; margin: 30px 0; color: #ffffff;">
                        <h3 style="color: #ffffff; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">✅</span> Next Steps
                        </h3>
                        <ol style="margin: 0; padding-left: 20px; font-size: 16px;">
                            <li style="margin-bottom: 15px;"><strong>Open and review the attached Excel report</strong> - it contains detailed information about each violation</li>
                            <li style="margin-bottom: 15px;"><strong>Upload missing receipts</strong> to Rippling within 15 days:
                                <ul style="margin: 10px 0; padding-left: 20px;">
                                    <li style="margin-bottom: 5px;">Forward receipt emails to receipts@rippling.com</li>
                                    <li style="margin-bottom: 5px;">Take photos and upload via the <strong>Rippling App</strong></li>
                                </ul>
                            </li>
                            <li style="margin-bottom: 15px;"><strong>For spending limit violations:</strong> Amounts exceeding policy limits may be deducted from payroll unless pre-approved</li>
                            <li style="margin-bottom: 15px;"><strong>Ensure proper approvals</strong> are obtained in Rippling for transactions requiring approval</li>
                            <li style="margin-bottom: 15px;"><strong>Contact finance team</strong> for high-value transaction approvals at <a href="mailto:finance@k-id.com" style="color: #ffffff; text-decoration: underline;">finance@k-id.com</a></li>
                        </ol>
                    </div>
                    
                    <!-- Lost Receipt Section -->
                    <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['orange']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
                        <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                            <strong style="color: {BRAND_COLORS['orange']};">Lost Receipt?</strong> Complete the <a href="https://app.pandadoc.com/a/#/templates/ePp27TxfNsABXwWdxQHBkT" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">k-ID Lost Receipt Disclaimer Form</a> and obtain approval from your domain leader. <a href="https://www.loom.com/share/8f4bbd55b4de4ea8a67cfa787db0558f?sid=1fa4559e-1979-4166-ab4c-f29c4d9cf746" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">Watch tutorial here</a>.
                        </p>
                    </div>
                    
                    <!-- Important Deadlines -->
                    <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
                        <h4 style="color: {BRAND_COLORS['purple']}; margin: 0 0 15px 0; font-size: 20px; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">⏰</span> Important Deadlines
                        </h4>
                        <ul style="margin: 0; padding-left: 20px; font-size: 16px; color: {BRAND_COLORS['black']};">
                            <li style="margin-bottom: 10px;"><strong>Receipts:</strong> Must be uploaded within 45 days of purchase (60 days = payroll deduction risk)</li>
                            <li style="margin-bottom: 10px;"><strong>Policy Violations:</strong> Must be resolved within 15 business days</li>
                            <li style="margin-bottom: 10px;"><strong>High-Value Items:</strong> Require immediate finance approval</li>
                        </ul>
                    </div>
                    
                    <!-- Footer -->
                    <div style="margin-top: 40px; padding-top: 30px; border-top: 2px solid {BRAND_COLORS['purple_light']}; text-align: center;">
                        <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                            🔎 You can refer to our company's <a href="https://www.notion.so/kidentify/Travel-Expense-Policy-1d25e61a5a2480328acffcfba458b5ba" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">expense policy</a> for full guidelines.
                        </p>
                        <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                            If you believe this notice was sent in error or if you have questions, please contact the Finance Team at <a href="mailto:finance@k-id.com" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">finance@k-id.com</a>.
                        </p>
                        <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                            Thank you for your prompt attention to these items. Your compliance helps keep k-ID financially healthy and audit-ready!
                        </p>
                        <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                            Best regards,<br><br>
                            <strong style="color: {BRAND_COLORS['purple']};">k-ID Finance Team</strong>
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    else:
        # For ≤10 violations, include detailed transaction list
        total_amount = sum([v['Amount'] for v in violations])
        total_employee_cost = sum([v['Employee_Out_of_Pocket'] for v in violations])
        
        # Categorize violations
        receipt_violations = []
        spending_violations = []
        approval_violations = []
        high_amount_violations = []
        other_violations = []
        
        for violation in violations:
            message = violation['Message'].lower()
            if 'missing receipt' in message:
                receipt_violations.append(violation)
            elif 'limit exceeded' in message or 'daily limit' in message:
                spending_violations.append(violation)
            elif 'approval' in message:
                approval_violations.append(violation)
            elif 'high amount' in message or 'finance approval' in message:
                high_amount_violations.append(violation)
            else:
                other_violations.append(violation)
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="use_container_width=device-use_container_width, initial-scale=1.0">
            <title>k-ID Corporate Card Policy Alert</title>
        </head>
        <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: {BRAND_COLORS['blackberry']}; margin: 0; padding: 0; background-color: #f8f9fa;">
            <div style="max-use_container_width: 800px; margin: 0 auto; background-color: #ffffff;">
                <!-- Header with K-ID branding -->
                <div style="background: linear-gradient(135deg, {BRAND_COLORS['gradient_start']} 0%, {BRAND_COLORS['gradient_end']} 100%); padding: 30px; text-align: center; border-radius: 0;">
                    <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 28px; font-weight: 600;">
                        Corporate Card Policy Alert 📊
                    </h1>
                    <p style="color: #ffffff; margin: 0; font-size: 18px; opacity: 0.9;">
                        k-ID Expense Policy Checker System
                    </p>
                </div>
                
                <!-- Main content -->
                <div style="padding: 40px;">
                    <p style="margin: 0 0 20px 0; font-size: 14px; color: {BRAND_COLORS['black']}; font-style: italic;">
                        This is an automated notification from k-ID's Expense Policy Checker system.
                    </p>
                    
                    <div style="margin-bottom: 30px;">
                        <h2 style="color: {BRAND_COLORS['blackberry']}; margin: 0 0 15px 0; font-size: 20px;">
                            Hi {emp},
                        </h2>
                        <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['black']};">
                            Hope you're doing great! We've identified some items with your recent corporate card expenses that need your attention to ensure full compliance with our k-ID Corporate Card Policy.
                        </p>
                    </div>
                    
                    <!-- Summary Section -->
                    <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 2px 10px rgba(113, 93, 236, 0.1);">
                        <h3 style="color: {BRAND_COLORS['purple']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">📊</span> Summary
                        </h3>
                        <table style="use_container_width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Total Transactions Requiring Action:</td>
                                <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; font-size: 16px; color: {BRAND_COLORS['purple']}; font-weight: 600;">{len(violations)}</td>
                            </tr>
                            <tr>
                                <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Total Transaction Amount:</td>
                                <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; font-size: 16px; color: {BRAND_COLORS['purple']}; font-weight: 600;">${total_amount:.2f}</td>
                            </tr>
                            {f'''<tr>
                                <td style="padding: 15px 0; font-weight: 600; color: {BRAND_COLORS['blackberry']};">⚠️ Amount You May Need to Cover:</td>
                                <td style="padding: 15px 0; text-align: right; font-size: 16px; color: {BRAND_COLORS['orange']}; font-weight: bold;">${total_employee_cost:.2f}</td>
                            </tr>''' if total_employee_cost > 0 else ''}
                        </table>
                    </div>
                    </div>
                    
                    <!-- Important Note Section -->
                    <div style="background-color: #f8f9fa; border-left: 4px solid {BRAND_COLORS['purple']}; padding: 15px; margin: 20px 0; border-radius: 4px;">
                        <p style="margin: 0; font-size: 14px; color: {BRAND_COLORS['blackberry']}; font-style: italic;">
                            <strong>Note:</strong> Expenses categorized under 'Biz Travel: Airfare' and 'Biz Travel: Hotels' will be flagged whenever amounts exceed company budgets (<strong>$2,500</strong> for airfares and <strong>$250</strong> per-night rates for hotels). This limitation arises because the Rippling platform does not provide an API to directly extract detailed receipt information. Please manually review these expenses to ensure policy compliance.
                        </p>
                    </div>
        """
        
        # Add section for receipt violations
        if receipt_violations:
            html_body += f"""
            <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['orange']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
                <h3 style="color: {BRAND_COLORS['orange']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                    <span style="margin-right: 10px;">🧾</span> Missing Receipts ({len(receipt_violations)} transaction(s))
                </h3>
                <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                    The following transactions are missing receipts. Please upload them as soon as possible:
                </p>
                <ul style="margin: 0 0 15px 0; padding-left: 20px; font-size: 16px; color: {BRAND_COLORS['black']};">
                    <li style="margin-bottom: 8px;">Forward the receipt email to receipts@rippling.com</li>
                    <li style="margin-bottom: 8px;">Take a photo and upload via the <strong>Rippling App</strong></li>
                </ul>
                <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                    <strong>Lost Receipt?</strong> Complete the <a href="https://app.pandadoc.com/a/#/templates/ePp27TxfNsABXwWdxQHBkT" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">k-ID Lost Receipt Disclaimer Form</a> and obtain approval from your domain leader. <a href="https://www.loom.com/share/8f4bbd55b4de4ea8a67cfa787db0558f?sid=1fa4559e-1979-4166-ab4c-f29c4d9cf746" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">Watch tutorial here</a>.
                </p>
            </div>
            """
        
        # Add detailed transaction table
        html_body += f"""
        <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 2px 10px rgba(113, 93, 236, 0.1);">
            <h3 style="color: {BRAND_COLORS['purple']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                <span style="margin-right: 10px;">📋</span> Transaction Details
            </h3>
            <div style="overflow-x: auto;">
                <table style="use_container_width: 100%; border-collapse: collapse; min-use_container_width: 1000px;">
                    <thead>
                        <tr style="background: linear-gradient(135deg, {BRAND_COLORS['purple_light']} 0%, {BRAND_COLORS['purple']} 100%);">
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 40px;">#</th>
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 100px;">📅 Date</th>
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 80px;">💰 Amount</th>
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 80px;">🏢 Company</th>
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 80px;">👤 Employee</th>
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 120px;">🏪 Vendor</th>
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 100px;">🏷️ Category</th>
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 60px;">🧾 Receipt</th>
                            <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 200px;">⚠️ Policy Issues</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for i, violation in enumerate(violations, 1):
            # Handle empty memo
            memo_text = violation.get('Memo', '')
            if not memo_text or str(memo_text).strip() == '' or str(memo_text).lower() == 'nan':
                memo_text = "No memo provided"

            receipt_status = "Yes" if violation.get('HasReceipt', True) else "No"
            
            # Get cost breakdown
            company_funded = violation.get('Company_Funded_Amount', violation['Amount'])
            employee_out_of_pocket = violation.get('Employee_Out_of_Pocket', 0.0)
            
            # Color code row based on violation type
            row_bg_color = "#ffffff" if i % 2 == 0 else "#f8f9fa"
            if employee_out_of_pocket > 0:
                row_bg_color = "#fff5f5"
            elif "Missing receipt" in violation['Message']:
                row_bg_color = "#fffbf0"
            
            html_body += f"""
                        <tr style="background-color: {row_bg_color};">
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['purple']}; text-align: center;">{i}</td>
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['black']}; font-size: 14px;">{violation['Date']}</td>
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['purple']}; font-weight: 600; font-size: 14px;">${violation['Amount']:.2f}</td>
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['purple']}; font-weight: 600; font-size: 14px;">${company_funded:.2f}</td>
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {'#d32f2f' if employee_out_of_pocket > 0 else BRAND_COLORS['black']}; font-weight: {'bold' if employee_out_of_pocket > 0 else '400'}; font-size: 14px;">${employee_out_of_pocket:.2f}</td>
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['black']}; font-size: 14px;">{violation.get('Vendor', 'N/A')}</td>
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['black']}; font-size: 14px;">{violation['Category']}</td>
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {'#d32f2f' if receipt_status == 'No' else '#2e7d32'}; font-weight: 600; font-size: 14px; text-align: center;">{receipt_status}</td>
                            <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['orange']}; font-size: 14px; font-weight: 600;">{violation['Message']}</td>
                        </tr>
            """
        
        html_body += """
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 15px; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid {BRAND_COLORS['purple']};">
                <p style="margin: 0; font-size: 14px; color: {BRAND_COLORS['blackberry']};">
                    <strong>Legend:</strong> Company = Amount covered by company | Employee = Amount you may need to cover | 
                    <span style="color: #d32f2f;">Red amounts</span> indicate potential employee responsibility
                </p>
            </div>
        </div>
        """
        
        # Add action items section
        html_body += f"""
        <div style="background: linear-gradient(135deg, {BRAND_COLORS['blackberry_light']} 0%, {BRAND_COLORS['blackberry']} 100%); padding: 30px; border-radius: 12px; margin: 30px 0; color: #ffffff;">
            <h3 style="color: #ffffff; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                <span style="margin-right: 10px;">✅</span> Next Steps
            </h3>
            <ol style="margin: 0; padding-left: 20px; font-size: 16px;">
                <li style="margin-bottom: 15px;"><strong>Review the transaction table above</strong> and understand the policy requirements</li>
        """
        
        if receipt_violations:
            html_body += "<li style='margin-bottom: 15px;'><strong>Upload missing receipts</strong> to Rippling within 15 days</li>"
        
        if spending_violations:
            html_body += "<li style='margin-bottom: 15px;'><strong>For spending limit violations:</strong> Amounts exceeding policy limits may be deducted from payroll unless pre-approved</li>"
        
        if approval_violations:
            html_body += "<li style='margin-bottom: 15px;'><strong>Ensure proper approvals</strong> are obtained in Rippling for transactions requiring approval</li>"
        
        if high_amount_violations:
            html_body += "<li style='margin-bottom: 15px;'><strong>Contact finance team</strong> for high-value transaction approvals at <a href='mailto:finance@k-id.com' style='color: #ffffff; text-decoration: underline;'>finance@k-id.com</a></li>"
        
        html_body += """
            </ol>
        </div>
        
        <!-- Lost Receipt Section -->
        <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['orange']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
            <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                <strong style="color: {BRAND_COLORS['orange']};">Lost Receipt?</strong> Complete the <a href="https://app.pandadoc.com/a/#/templates/ePp27TxfNsABXwWdxQHBkT" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">k-ID Lost Receipt Disclaimer Form</a> and obtain approval from your domain leader. <a href="https://www.loom.com/share/8f4bbd55b4de4ea8a67cfa787db0558f?sid=1fa4559e-1979-4166-ab4c-f29c4d9cf746" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">Watch tutorial here</a>.
            </p>
        </div>
        
        <!-- Important Deadlines -->
        <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
            <h4 style="color: {BRAND_COLORS['purple']}; margin: 0 0 15px 0; font-size: 20px; display: flex; align-items: center;">
                <span style="margin-right: 10px;">⏰</span> Important Deadlines
            </h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 16px; color: {BRAND_COLORS['black']};">
                <li style="margin-bottom: 10px;"><strong>Receipts:</strong> Must be uploaded within 45 days of purchase (60 days = payroll deduction risk)</li>
                <li style="margin-bottom: 10px;"><strong>Policy Violations:</strong> Must be resolved within 15 business days</li>
                <li style="margin-bottom: 10px;"><strong>High-Value Items:</strong> Require immediate finance approval</li>
            </ul>
        </div>
        
        <!-- Footer -->
        <div style="margin-top: 40px; padding-top: 30px; border-top: 2px solid {BRAND_COLORS['purple_light']}; text-align: center;">
            <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                🔎 You can refer to our company's <a href="https://www.notion.so/kidentify/Travel-Expense-Policy-1d25e61a5a2480328acffcfba458b5ba" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">expense policy</a> for full guidelines.
            </p>
            <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                If you believe this notice was sent in error or if you have questions, please contact the Finance Team at <a href="mailto:finance@k-id.com" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">finance@k-id.com</a>.
            </p>
            <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                Thank you for your prompt attention to these items. Your compliance helps keep k-ID financially healthy and audit-ready!
            </p>
            <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                Best regards,<br><br>
                <strong style="color: {BRAND_COLORS['purple']};">k-ID Finance Team</strong>
            </p>
        </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    return html_body


def create_missing_receipt_email_body_html(emp, violations, test_mode=False, has_excel_attachment=False):
    """Create HTML email body specifically for missing receipt reminders"""
    
    # Calculate total amount
    total_amount = sum(violation['Amount'] for violation in violations)
    
    html_body = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="use_container_width=device-use_container_width, initial-scale=1.0">
                    <title>k-ID Corporate Card Receipt Reminder</title>
                </head>
                <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: {BRAND_COLORS['blackberry']}; margin: 0; padding: 0; background-color: #f8f9fa;">
                    <div style="max-use_container_width: 800px; margin: 0 auto; background-color: #ffffff;">
                        <!-- Header with K-ID branding -->
                        <div style="background: linear-gradient(135deg, {BRAND_COLORS['gradient_start']} 0%, {BRAND_COLORS['gradient_end']} 100%); padding: 30px; text-align: center; border-radius: 0;">
                            <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 28px; font-weight: 600;">
                                Corporate Card Receipt Reminder 🧾
                            </h1>
                            <p style="color: #ffffff; margin: 0; font-size: 18px; opacity: 0.9;">
                                k-ID Expense Policy Checker System
                            </p>
                        </div>
                        
                        <!-- Main content -->
                        <div style="padding: 40px;">
                            <p style="margin: 0 0 20px 0; font-size: 14px; color: {BRAND_COLORS['black']}; font-style: italic;">
                                This is an automated notification from k-ID's Expense Policy Checker system.
                            </p>
                            
                            <div style="margin-bottom: 30px;">
                                <h2 style="color: {BRAND_COLORS['blackberry']}; margin: 0 0 15px 0; font-size: 20px;">
                                    Hi {emp},
                                </h2>
                                {"<div style='background-color: #fff3cd; border: 2px solid #f0ad4e; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center;'><p style='margin: 0; font-size: 18px; color: #856404; font-weight: bold;'>🔴 THIS IS A TEST MODE EMAIL 🔴</p></div>" if test_mode and emp.lower() == "lulu" else ""}
                                <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['black']};">
                                    Hope you're doing great! Just a quick nudge about your recent corporate card expenses. We noticed some receipts haven't made their way onto Rippling yet. As a quick refresher, our k-ID Corporate Card Policy asks that all expenses be paired with their original receipts.
                                </p>
                            </div>
                """
                
    # Add Excel attachment notification if applicable
    if has_excel_attachment:
        html_body += f"""
                            <!-- Excel Attachment Notification -->
                            <div style="background: linear-gradient(135deg, #ffedd5 0%, #ffead0 100%); border: 2px solid #f97316; padding: 25px; border-radius: 12px; margin: 30px 0;">
                                <h3 style="color: #f97316; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                                    <span style="margin-right: 10px;">📎</span> Detailed Report Attached
                                </h3>
                                <p style="margin: 0; font-size: 16px; color: #2c3e50;">
                                    Since you have {len(violations)} transactions with missing receipts, we've attached a detailed Excel report for your convenience. This spreadsheet contains all transaction details and can help you track your progress as you upload receipts.
                                </p>
                            </div>
        """
    
    html_body += f"""
                            <!-- Summary Section -->
                            <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 2px 10px rgba(113, 93, 236, 0.1);">
                                <h3 style="color: {BRAND_COLORS['purple']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                                    <span style="margin-right: 10px;">📊</span> Summary
                                </h3>
                                <table style="use_container_width: 100%; border-collapse: collapse;">
                                    <tr>
                                        <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Total Transactions Missing Receipts:</td>
                                        <td style="padding: 15px 0; border-bottom: 1px solid {BRAND_COLORS['purple_light']}; text-align: right; font-size: 16px; color: {BRAND_COLORS['purple']}; font-weight: 600;">{len(violations)}</td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 15px 0; font-weight: 600; color: {BRAND_COLORS['blackberry']};">Total Transaction Amount:</td>
                                        <td style="padding: 15px 0; text-align: right; font-size: 16px; color: {BRAND_COLORS['purple']}; font-weight: 600;">${total_amount:.2f}</td>
                                    </tr>
                                </table>
                            </div>
    """
    
    # Only show transaction details in email if ≤10 violations (otherwise only in Excel)
    if not has_excel_attachment:
        html_body += f"""
                            <!-- Transaction Details Section (Table Format) -->
                            <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 2px 10px rgba(113, 93, 236, 0.1);">
                                <h3 style="color: {BRAND_COLORS['purple']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                                    <span style="margin-right: 10px;">📋</span> Transaction Details
                                </h3>
                                <div style="overflow-x: auto;">
                                    <table style="use_container_width: 100%; border-collapse: collapse; min-use_container_width: 1000px;">
                                        <thead>
                                            <tr style="background: linear-gradient(135deg, {BRAND_COLORS['purple_light']} 0%, {BRAND_COLORS['purple']} 100%);">
                                                <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 40px;">#</th>
                                                <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 100px;">📅 Date</th>
                                                <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 80px;">💰 Amount</th>
                                                <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 120px;">🏪 Vendor</th>
                                                <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 100px;">🏷️ Category</th>
                                                <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 100px;">🆔 Transaction ID</th>
                                                <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 200px;">📝 Memo</th>
                                                <th style="padding: 15px 10px; text-align: left; color: #ffffff; font-weight: 600; border: 1px solid {BRAND_COLORS['purple']}; min-use_container_width: 100px;">⚠️ Issue</th>
                                            </tr>
                                        </thead>
                                        <tbody>
        """
        
        # Add each transaction as a table row
        for i, violation in enumerate(violations, 1):
            memo_text = violation.get('Memo', '')
            if not memo_text or str(memo_text).strip() == '' or str(memo_text).lower() == 'nan':
                memo_text = "No memo provided"
            
            # Alternate row colors for better readability
            row_color = "#f8f9fa" if i % 2 == 0 else "#ffffff"
            
            html_body += f"""
                                            <tr style="background-color: {row_color};">
                                                <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; font-weight: 600; color: {BRAND_COLORS['purple']};">{i}</td>
                                                <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['black']};">{violation['Date']}</td>
                                                <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['purple']}; font-weight: 600;">${violation['Amount']:.2f}</td>
                                                <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['black']};">{violation.get('Vendor', 'N/A')}</td>
                                                <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['black']};">{violation['Category']}</td>
                                                <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['black']}; font-family: monospace; font-size: 14px;">{violation.get('Transaction_ID', 'N/A')}</td>
                                                <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['black']}; max-use_container_width: 200px; word-wrap: break-word;">{memo_text}</td>
                                                <td style="padding: 12px 10px; border: 1px solid {BRAND_COLORS['purple_light']}; color: {BRAND_COLORS['orange']}; font-weight: 600;">Missing receipt</td>
                                            </tr>
            """
        
        html_body += """
                                        </tbody>
                                    </table>
                                </div>
                            </div>
        """
    
    html_body += f"""
                            <!-- How to Upload Section -->
                            <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['orange']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
                                <h3 style="color: {BRAND_COLORS['orange']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                                    <span style="margin-right: 10px;">📤</span> How to Upload Receipts
                                </h3>
                                <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                                    We'd love it if you could upload your receipts soon to help keep our financial processes running smoothly. You can do so easily:
                                </p>
                                <ul style="margin: 0 0 15px 0; padding-left: 20px; font-size: 16px; color: {BRAND_COLORS['black']};">
                                    <li style="margin-bottom: 8px;">Forwarding the email containing the receipt to receipts@rippling.com</li>
                                    <li style="margin-bottom: 8px;">Taking a photo and uploading it via the <strong>Rippling App</strong></li>
                                </ul>
                            </div>
                            
                            <!-- Lost Receipt Section -->
                            <div style="background: linear-gradient(135deg, {BRAND_COLORS['purple_light']} 0%, #ffffff 100%); border: 2px solid {BRAND_COLORS['purple']}; padding: 30px; border-radius: 12px; margin: 30px 0;">
                                <h3 style="color: {BRAND_COLORS['purple_dark']}; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                                    <span style="margin-right: 10px;">📋</span> Lost Receipt?
                                </h3>
                                <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                                    If you no longer have the receipt, please complete the k-ID Lost Receipt Disclaimer Form and obtain approval from your domain leader.
                                </p>
                                <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                                    We have created a simple process for you. Here's how <strong>(OR Watch a video tutorial <a href="https://www.loom.com/share/8f4bbd55b4de4ea8a67cfa787db0558f?sid=1fa4559e-1979-4166-ab4c-f29c4d9cf746" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">here</a>):</strong>
                                </p>
                                <ol style="margin: 0 0 15px 0; padding-left: 20px; font-size: 16px; color: {BRAND_COLORS['black']};">
                                    <li style="margin-bottom: 8px;">Click on the link <a href="https://app.pandadoc.com/a/#/templates/ePp27TxfNsABXwWdxQHBkT" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">k-ID Lost Receipt Disclaimer Form</a></li>
                                    <li style="margin-bottom: 8px;">Click on <strong>Use this Template</strong> on the web page</li>
                                    <li style="margin-bottom: 8px;">Enter the Email ID of your <strong>Domain leader</strong> for approval</li>
                                    <li style="margin-bottom: 8px;">Fill out the required details in form</li>
                                    <li style="margin-bottom: 8px;">Click on <strong>continue</strong>, then tap <strong>send</strong></li>
                                    <li style="margin-bottom: 8px;">An approval request will be sent via PanDoc to the domain leader</li>
                                    <li style="margin-bottom: 8px;">Once approved, you will receive a signed copy via email and on your PanDoc page</li>
                                    <li style="margin-bottom: 8px;"><strong>Download and upload</strong> the signed form in Rippling under the respective expenses</li>
                                </ol>
                                <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['purple_dark']}; font-weight: 600;">
                                    Please note: This form should be used only in <strong>exceptional cases</strong>. To keep our organization fully compliant with <strong>taxation and audit requirements</strong>, we kindly request that you always upload the original receipts in Rippling.
                                </p>
                            </div>
                            
                            <!-- Important Requirements -->
                            <div style="background-color: #ffffff; border: 2px solid {BRAND_COLORS['purple_light']}; padding: 25px; border-radius: 12px; margin: 30px 0;">
                                <h4 style="color: {BRAND_COLORS['purple']}; margin: 0 0 15px 0; font-size: 20px; display: flex; align-items: center;">
                                    <span style="margin-right: 10px;">⚠️</span> Important Requirements
                                </h4>
                                <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                                    Additionally, please <strong>upload original expense receipts</strong>—<strong>credit card statements are not valid proof</strong>. Receipts must also include the <strong>names of attendees</strong> in accordance with our internal expense policy.
                                </p>
                            </div>
                            
                            <!-- Timeline Section -->
                            <div style="background: linear-gradient(135deg, {BRAND_COLORS['blackberry_light']} 0%, {BRAND_COLORS['blackberry']} 100%); padding: 30px; border-radius: 12px; margin: 30px 0; color: #ffffff;">
                                <h3 style="color: #ffffff; margin: 0 0 20px 0; font-size: 22px; display: flex; align-items: center;">
                                    <span style="margin-right: 10px;">⏰</span> Important Timeline
                                </h3>
                                <ul style="margin: 0; padding-left: 20px; font-size: 16px;">
                                    <li style="margin-bottom: 15px;">If the receipt isn't uploaded within 45 days of the purchase date, we'll send another reminder and loop in your manager</li>
                                    <li style="margin-bottom: 15px;">After 60 days, any undocumented amounts may need to be deducted from payroll (and we'd really prefer to avoid that!)</li>
                                </ul>
                            </div>
                            
                            <!-- Footer -->
                            <div style="margin-top: 40px; padding-top: 30px; border-top: 2px solid {BRAND_COLORS['purple_light']}; text-align: center;">
                                <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                                    Thank you so much for handling this—it's a big help from your side to keep our organisation 100% compliant in all aspects! Let us know if you run into any issues or need assistance.
                                </p>
                                <p style="margin: 0 0 15px 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                                    If you believe this notice was sent in error or if you have questions, please contact the Finance Team at <a href="mailto:finance@k-id.com" style="color: {BRAND_COLORS['purple']}; text-decoration: none;">finance@k-id.com</a>.
                                </p>
                                <p style="margin: 0; font-size: 16px; color: {BRAND_COLORS['blackberry']};">
                                    Best regards,<br><br>
                                    <strong style="color: {BRAND_COLORS['purple']};">k-ID Finance Team</strong>
                                </p>
                            </div>
                        </div>
                    </div>
                </body>
                </html>
    """
    
    return html_body


def send_policy_notification_email_with_ui(employee_violations, smtp_config, test_mode=True, mode="Full Validation", delay_minutes=5):
    """Send email notifications and launch static UI for management"""
    global EMAIL_LOCK
    emails_scheduled = 0
    emails_skipped = 0
    
    try:   
        # Process each employee
        for emp, violations in employee_violations.items():
            if test_mode:
                if emp.lower() not in ["lulu xia", "lulu"]:
                    print(f"Test mode: Skipping {emp} (not Lulu)")
                    emails_skipped += 1
                    continue
                to_addr = "lulu@k-id.com"
                
                # Different subject lines based on mode
                if mode == "Receipt-Only Mode":
                    subject = "[TEST MODE] Missing Receipt Reminder"
                else:
                    subject = "[TEST MODE] Corporate Card Expense Policy Reminder"
            else:
                to_addr = get_employee_email(emp)
                if not to_addr:
                    print(f"No email found for employee: {emp}. Skipping notification.")
                    emails_skipped += 1
                    continue
                
                # Different subject lines based on mode
                if mode == "Receipt-Only Mode":
                    subject = "[ALERT] Missing Receipt Reminder"
                else:
                    subject = "[ALERT] Corporate Card Expense Policy Reminder"
            
            # Check if employee has more than 10 violations
            has_many_violations = len(violations) > 10
            excel_file = None
            
            if has_many_violations:
                print(f"📊 Employee {emp} has {len(violations)} violations. Creating Excel report...")
                excel_file = create_excel_report(emp, violations, mode)
            
            msg = EmailMessage()
            msg['From'] = smtp_config['user']
            msg['To'] = to_addr
            msg['Cc'] = 'finance@k-id.com'
            msg['Subject'] = subject
            
            # Create HTML email body based on mode
            if mode == "Receipt-Only Mode":
                html_body = create_missing_receipt_email_body_html(
                    emp, violations, test_mode=test_mode, has_excel_attachment=has_many_violations
                )
            else:
                html_body = create_email_body_html(
                    emp, violations, mode=mode, has_excel_attachment=has_many_violations
                )
            
            msg.add_alternative(html_body, subtype='html')
            
            # Add Excel attachment if needed
            if excel_file and os.path.exists(excel_file):
                with open(excel_file, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(excel_file)
                msg.add_attachment(file_data, maintype='application', 
                                 subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                                 filename=file_name)
            
            # Create email data for delayed sending
            email_id = f"POLICY_{emp.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            email_data = {
                'id': email_id,
                'employee': emp,
                'to_addr': to_addr,
                'subject': subject,
                'from_addr': smtp_config['user'],
                'html_body': html_body,
                'message': msg,
                'smtp_config': smtp_config,
                'excel_file': excel_file,
                'send_immediately': False,
                'scheduled_time': datetime.now() + timedelta(minutes=delay_minutes),
                'has_attachment': has_many_violations,
                'violations_count': len(violations),
                'mode': mode,
                'cancelled': False,  # Initialize cancellation status
                'sent': False,       # Initialize sent status
                'failed': False      # Initialize failed status
            }
            
            # Use thread lock when adding to POLICY session state
            with EMAIL_LOCK:
                st.session_state.POLICY_SCHEDULED_EMAILS[email_id] = email_data
            
            # Start the delayed sending in a separate thread
            thread = threading.Thread(target=send_delayed_email, args=(email_data, delay_minutes, st.session_state.POLICY_SCHEDULED_EMAILS, None), daemon=True)
            thread.start()
            
            violation_type = "missing receipt" if mode == "Receipt-Only Mode" else "policy violation"
            print(f"📧 {violation_type.title()} reminder scheduled for {emp} ({to_addr})")
            print(f"   Email ID: {email_id}")
            emails_scheduled += 1
        
        print(f"\n📈 SCHEDULING COMPLETE:")
        print(f"   Scheduled: {emails_scheduled}")
        print(f"   Skipped: {emails_skipped}")
        print(f"   Mode: {mode}")
        print(f"   Delay: {delay_minutes} minutes")
        print(f"   Policy Session dict ID: {id(st.session_state.POLICY_SCHEDULED_EMAILS)}")
        
        # Generate static HTML and serve it
        generate_and_serve_static_ui(emails_scheduled, delay_minutes, mode, email_type="policy")
        
        # Open browser automatically (optional)
        webbrowser.open('http://localhost:5000')
        
        violation_type = "missing receipt" if mode == "Receipt-Only Mode" else "policy violation"
        
        return {
            'success': True,
            'scheduled_count': emails_scheduled,
            'skipped_count': emails_skipped,
            'delay_minutes': delay_minutes,
            'mode': mode,
            'message': f'Successfully scheduled {emails_scheduled} {violation_type} emails with {delay_minutes} minute delay'
        }
        
    except Exception as e:
        print(f"❌ Failed to schedule emails: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to schedule emails: {e}'
        }

class StaticEmailData:
    @staticmethod
    def generate_snapshot():
        """CORRECTED: Generate a static snapshot of current email data with better error handling"""
        import streamlit as st
        from datetime import datetime
        
        # CORRECTED: Initialize all email dictionaries if they don't exist
        if 'SCHEDULED_EMAILS' not in st.session_state:
            st.session_state.SCHEDULED_EMAILS = {}
        if 'POLICY_SCHEDULED_EMAILS' not in st.session_state:
            st.session_state.POLICY_SCHEDULED_EMAILS = {}
        if 'FIXED_ASSET_SCHEDULED_EMAILS' not in st.session_state:
            st.session_state.FIXED_ASSET_SCHEDULED_EMAILS = {}
            
        # Combine all email dictionaries safely
        try:
            budget_emails = dict(st.session_state.SCHEDULED_EMAILS)
            policy_emails = dict(st.session_state.POLICY_SCHEDULED_EMAILS)
            fixed_asset_emails = dict(st.session_state.FIXED_ASSET_SCHEDULED_EMAILS)
        except Exception as e:
            print(f"⚠️ Error accessing session state email data: {e}")
            budget_emails = {}
            policy_emails = {}
            fixed_asset_emails = {}
        
        emails_snapshot = {**budget_emails, **policy_emails, **fixed_asset_emails}
        
        # CORRECTED: Safe counting with error handling
        try:
            pending_count = len([e for e in emails_snapshot.values() 
                               if not e.get('cancelled', False) 
                               and not e.get('sent', False) 
                               and not e.get('failed', False)])
            cancelled_count = len([e for e in emails_snapshot.values() if e.get('cancelled', False)])
        except Exception as e:
            print(f"⚠️ Error counting emails: {e}")
            pending_count = 0
            cancelled_count = 0
        
        emails_data = []
        for email_id, data in emails_snapshot.items():
            try:
                # CORRECTED: Safe status determination
                if data.get('cancelled', False):
                    status = "cancelled"
                elif data.get('sent', False):
                    status = "sent"
                elif data.get('failed', False):
                    status = "failed"
                else:
                    status = "pending"
                
                # CORRECTED: Safe datetime handling
                scheduled_time = data.get('scheduled_time', datetime.now())
                if isinstance(scheduled_time, str):
                    scheduled_time_str = scheduled_time
                else:
                    scheduled_time_str = scheduled_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # CORRECTED: Extract email details with safe defaults
                email_info = {
                    'id': email_id,
                    'employee': data.get('employee', 'Unknown'),
                    'status': status,
                    'scheduled_time': scheduled_time_str,
                    'to_addr': data.get('to_addr', 'N/A'),
                    'subject': data.get('subject', 'N/A'),
                    'from_addr': data.get('from_addr', 'N/A'),
                    'html_body': data.get('html_body', ''),
                    'has_attachment': data.get('has_attachment', False),
                    'violations_count': data.get('violations_count', 0),
                    'test_mode': data.get('test_mode', False),
                    # CORRECTED: Add email type identification
                    'email_type': 'fixed_asset' if 'FIXED_ASSET_' in email_id else 
                                 'policy' if 'POLICY_' in email_id else 'budget'
                }
                emails_data.append(email_info)
                
            except Exception as e:
                print(f"⚠️ Error processing email {email_id}: {e}")
                # Add a basic error entry
                emails_data.append({
                    'id': email_id,
                    'employee': 'Error',
                    'status': 'error',
                    'scheduled_time': 'N/A',
                    'to_addr': 'N/A',
                    'subject': 'Processing Error',
                    'from_addr': 'N/A',
                    'html_body': '',
                    'has_attachment': False,
                    'violations_count': 0,
                    'test_mode': False,
                    'email_type': 'unknown'
                })
        
        return {
            'status': {
                'pending_emails': pending_count,
                'cancelled_emails': cancelled_count,
                'total_scheduled': len(emails_snapshot),
                'budget_emails': len(budget_emails),
                'policy_emails': len(policy_emails),
                'fixed_asset_emails': len(fixed_asset_emails)
            },
            'emails': emails_data
        }
    
# ENHANCED STATIC UI WITH BUTTONS AND k-ID COLOR THEME

def generate_and_serve_static_ui(emails_scheduled, delay_minutes, mode="Full Validation", email_type="policy"):
    """Generate static HTML content with interactive buttons and k-ID color theme"""
    
    # Get snapshot of current data
    snapshot = StaticEmailData.generate_snapshot()
    
    violation_type = "missing receipt" if mode == "Receipt-Only Mode" else "policy violation"
    
    # Generate static HTML with enhanced styling and buttons
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="use_container_width=device-use_container_width, initial-scale=1.0">
        <title>k-ID Email Manager - Static View</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #AF7EF7 0%, #715DEC 100%);
                color: #0F0740;
                line-height: 1.6;
                min-height: 100vh;
            }}
            
            .container {{
                max-use_container_width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(15, 7, 64, 0.1);
                border: 1px solid #EBE8FF;
            }}
            
            .header h1 {{
                color: #0F0740;
                margin-bottom: 10px;
                font-size: 2.5em;
                font-weight: 700;
            }}
            
            .header p {{
                color: #505050;
                margin-bottom: 5px;
            }}
            
            .status-cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .status-card {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(15, 7, 64, 0.1);
                border: 1px solid #EBE8FF;
                transition: transform 0.2s ease;
            }}
            
            .status-card:hover {{
                transform: translateY(-2px);
            }}
            
            .status-card h3 {{
                color: #2C216F;
                margin-bottom: 15px;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .status-card .number {{
                font-size: 36px;
                font-weight: 700;
                margin-bottom: 10px;
            }}
            
            .pending {{ color: #FC6C0F; }}
            .cancelled {{ color: #4E39D4; }}
            .total {{ color: #715DEC; }}
            
            .emails-section {{
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(15, 7, 64, 0.1);
                border: 1px solid #EBE8FF;
            }}
            
            .emails-section h2 {{
                margin-bottom: 25px;
                color: #0F0740;
                font-size: 1.8em;
                font-weight: 600;
            }}
            
            .email-card {{
                border: 2px solid #EBE8FF;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                background: #FAFAFA;
                transition: all 0.3s ease;
            }}
            
            .email-card:hover {{
                box-shadow: 0 4px 20px rgba(113, 93, 236, 0.15);
            }}
            
            .email-card.pending {{
                border-left: 6px solid #FC6C0F;
            }}
            
            .email-card.cancelled {{
                border-left: 6px solid #4E39D4;
                opacity: 0.7;
            }}
            
            .email-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                flex-wrap: wrap;
                gap: 10px;
            }}
            
            .email-id {{
                font-family: 'Courier New', monospace;
                font-size: 12px;
                color: #505050;
                background: #EBE8FF;
                padding: 4px 8px;
                border-radius: 6px;
            }}
            
            .status-badge {{
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .status-badge.pending {{
                background: #FC6C0F;
                color: white;
            }}
            
            .status-badge.cancelled {{
                background: #4E39D4;
                color: white;
            }}
            
            .email-details {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            
            .email-details strong {{
                color: #0F0740;
            }}
            
            .email-actions {{
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }}
            
            .btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                text-decoration: none;
                display: inline-block;
            }}
            
            .btn-primary {{
                background: #715DEC;
                color: white;
            }}
            
            .btn-primary:hover {{
                background: #4E39D4;
                transform: translateY(-1px);
            }}
            
            .btn-danger {{
                background: #FC6C0F;
                color: white;
            }}
            
            .btn-danger:hover {{
                background: #E55A0A;
                transform: translateY(-1px);
            }}
            
            .btn-secondary {{
                background: #EBE8FF;
                color: #2C216F;
            }}
            
            .btn-secondary:hover {{
                background: #D1C7FF;
            }}
            
            .btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }}
            
            .email-preview {{
                margin-top: 15px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                border: 1px solid #EBE8FF;
            }}
            
            .email-preview h4 {{
                margin-bottom: 10px;
                color: #0F0740;
                font-size: 1.1em;
            }}
            
            .email-body {{
                max-height: 200px;
                overflow-y: auto;
                padding: 15px;
                background: #FAFAFA;
                border-radius: 8px;
                font-size: 13px;
                border: 1px solid #EBE8FF;
            }}
            
            .note {{
                background: linear-gradient(135deg, #EBE8FF 0%, #F5F3FF 100%);
                border: 2px solid #715DEC;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
            }}
            
            .note h3 {{
                color: #0F0740;
                margin-bottom: 10px;
                font-size: 1.2em;
            }}
            
            .note p {{
                color: #2C216F;
            }}
            
            .commands-section {{
                background: #0F0740;
                color: white;
                padding: 25px;
                border-radius: 12px;
                margin-top: 30px;
            }}
            
            .commands-section h3 {{
                color: #EBE8FF;
                margin-bottom: 20px;
                font-size: 1.5em;
            }}
            
            .command-item {{
                background: #2C216F;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-left: 4px solid #715DEC;
            }}
            
            .command-item h4 {{
                color: #FC6C0F;
                margin-bottom: 8px;
            }}
            
            .command-code {{
                font-family: 'Courier New', monospace;
                background: #0F0740;
                padding: 8px 12px;
                border-radius: 6px;
                margin: 8px 0;
                color: #EBE8FF;
                border: 1px solid #4E39D4;
            }}
            
            .command-item p {{
                color: #EBE8FF;
                font-size: 14px;
            }}
            
            .global-actions {{
                display: flex;
                gap: 15px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            
            .attachment-badge {{
                background: #715DEC;
                color: white;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                margin-left: 10px;
            }}
        </style>
        <script>
            // Functions to handle button clicks
            function showCommandForAction(action, emailId = null) {{
                let command = '';
                switch(action) {{
                    case 'cancel':
                        command = emailId ? `cancel_scheduled_email('${{emailId}}')` : 'cancel_all_scheduled_emails()';
                        break;
                    case 'send':
                        command = `# Mark email for immediate sending:\\nscheduled_emails['${{emailId}}']['send_immediately'] = True`;
                        break;
                    case 'status':
                        command = `# Check scheduled emails:\\nfor email_id, data in scheduled_emails.items():\\n    print(f"{{email_id}}: {{data.get('cancelled', False)}}")`;
                        break;
                }}
                
                alert(`Copy this command and paste it in your Python terminal:\\n\\n${{command}}`);
            }}
            
            function copyToClipboard(text) {{
                navigator.clipboard.writeText(text).then(function() {{
                    alert('Command copied to clipboard!');
                }}, function() {{
                    alert('Failed to copy. Please copy manually:\\n\\n' + text);
                }});
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📧 k-ID Email Manager</h1>
                <p><strong>Static snapshot taken at:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Emails scheduled to send in:</strong> {delay_minutes} minutes</p>
            </div>
            
            <div class="status-cards">
                <div class="status-card">
                    <h3>Pending Emails</h3>
                    <div class="number pending">{snapshot['status']['pending_emails']}</div>
                </div>
                <div class="status-card">
                    <h3>Cancelled Emails</h3>
                    <div class="number cancelled">{snapshot['status']['cancelled_emails']}</div>
                </div>
                <div class="status-card">
                    <h3>Total Scheduled</h3>
                    <div class="number total">{snapshot['status']['total_scheduled']}</div>
                </div>
            </div>
            
            <div class="note">
                <h3>📝 How to Use This Interface</h3>
                <p>This is a static view of your email status. Click the action buttons below to get the terminal commands you need to manage your emails. The actual email sending process continues in the background.</p>
            </div>
            
            <div class="emails-section">
                <h2>📋 Scheduled Emails</h2>
                
                <div class="global-actions">
                    <button class="btn btn-danger" onclick="showCommandForAction('cancel')">
                        🚫 Cancel All Emails
                    </button>
                    <button class="btn btn-secondary" onclick="showCommandForAction('status')">
                        📊 Check Status
                    </button>
                </div>
    """
    
    # Add email cards with buttons
    for email in snapshot['emails']:
        status_class = email['status']
        email_id = email['id']
        is_cancelled = status_class == 'cancelled'

        html_content += f"""
                <div class="email-card {status_class}">
                    <div class="email-header">
                        <div class="email-id">{email_id}</div>
                        <div class="status-badge {status_class}">{email['status']}</div>
                    </div>

                    <div class="email-details">
                        <div><strong>👤 Employee:</strong> {email['employee']}</div>
                        <div><strong>📅 Scheduled:</strong> {email['scheduled_time']}</div>
                        <div><strong>📧 To:</strong> {email['to_addr']}</div>
                        <div><strong>📋 Subject:</strong> {email['subject']}</div>
                    </div>

                    <div class="email-actions">
        """

        if not is_cancelled:
            html_content += f"""
                        <button class="btn btn-primary" onclick="showCommandForAction('send', '{email_id}')">
                            🚀 Send Immediately
                        </button>
                        <button class="btn btn-danger" onclick="showCommandForAction('cancel', '{email_id}')">
                            🚫 Cancel Email
                        </button>
            """
        else:
            html_content += f"""
                        <button class="btn btn-secondary" disabled>
                            ✅ Already Cancelled
                        </button>
            """

        html_content += """
                    </div>
        """

        # Add attachment info
        if email['has_attachment']:
            html_content += f"""
                    <div style="margin-top: 15px;">
                        <strong>📎 Attachment:</strong> 
                        <span class="attachment-badge">Excel Report ({email['violations_count']} violations)</span>
                    </div>
            """

        # Add email preview
        html_content += f"""
                    <div class="email-preview">
                        <h4>📧 Email Preview:</h4>
                        <div class="email-body">
                            <iframe srcdoc="{email['html_body'].replace('"', '&quot;')}"></iframe>
                        </div>
                    </div>
                </div>
        """
    
    # Add terminal commands section
    html_content += f"""
            </div>
            
            <div class="commands-section">
                <h3>🖥️ Terminal Commands Reference</h3>
                <p style="margin-bottom: 20px; color: #EBE8FF;">Copy and paste these commands into your Python terminal to manage emails:</p>
                
                <div class="command-item">
                    <h4>🚫 Cancel a specific email</h4>
                    <div class="command-code">cancel_scheduled_email('EMAIL_ID_HERE')</div>
                    <p>Replace EMAIL_ID_HERE with the actual email ID from above</p>
                </div>
                
                <div class="command-item">
                    <h4>🚫 Cancel all scheduled emails</h4>
                    <div class="command-code">cancel_all_scheduled_emails()</div>
                    <p>This will cancel all emails that haven't been sent yet</p>
                </div>
                
                <div class="command-item">
                    <h4>🚀 Send email immediately</h4>
                    <div class="command-code">scheduled_emails['EMAIL_ID_HERE']['send_immediately'] = True</div>
                    <p>Replace EMAIL_ID_HERE with the actual email ID. This marks the email for immediate sending.</p>
                </div>
                
                <div class="command-item">
                    <h4>📊 Check email status</h4>
                    <div class="command-code">for email_id, data in scheduled_emails.items():
    status = "cancelled" if data.get('cancelled', False) else "pending"
    print(f"{{email_id}}: {{status}}")</div>
                    <p>This will show you the current status of all scheduled emails</p>
                </div>
                
                <div class="command-item">
                    <h4>📋 View specific email details</h4>
                    <div class="command-code">print(scheduled_emails['EMAIL_ID_HERE'])</div>
                    <p>Replace EMAIL_ID_HERE to see detailed information about a specific email</p>
                </div>
                
                <div class="command-item">
                    <h4>🔄 Refresh this page</h4>
                    <div class="command-code"># Re-run the email scheduling function to get updated UI
send_policy_notification_email_with_ui(employee_violations, smtp_config, test_mode, delay_minutes)</div>
                    <p>This will regenerate the static UI with current email status</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open('static_email_manager.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Start simple HTTP server
    start_static_server()
    
def start_static_server(port=5000):
    """CORRECTED: Start a simple HTTP server for the static HTML with better error handling"""
    import http.server
    import socketserver
    import socket
    import os
    import threading
    
    def find_available_port(start_port):
        """Find an available port starting from start_port"""
        for test_port in range(start_port, start_port + 20):  # Try more ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', test_port))
                    return test_port
            except OSError:
                continue
        return None
    
    # Find available port
    available_port = find_available_port(port)
    if available_port is None:
        print("❌ Could not find available port in range")
        return None
    
    class StaticHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '':
                self.path = '/static_email_manager.html'
            elif self.path == '/api/emails':
                # CORRECTED: Add API endpoint for email data
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                try:
                    import json
                    email_data = StaticEmailData.generate_snapshot()
                    self.wfile.write(json.dumps(email_data).encode())
                except Exception as e:
                    error_response = {'error': str(e), 'status': {'pending_emails': 0, 'cancelled_emails': 0, 'total_scheduled': 0}, 'emails': []}
                    self.wfile.write(json.dumps(error_response).encode())
                return
                
            return super().do_GET()
    
    def run_server():
        with socketserver.TCPServer(("", available_port), StaticHandler) as httpd:
            print(f"🌐 Static Email UI server started at http://localhost:{available_port}")
            httpd.serve_forever()
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    return server_thread

# Main UI
st.markdown('<h1 class="main-header">💰 Expense Validation Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Policy configuration
    st.subheader("Policy Settings")
    policy_option = st.radio("Policy Source:", ["Use Default", "Upload YAML", "Edit Inline"])
    
    if policy_option == "Use Default":
        st.session_state.policy = load_policy_from_yaml(DEFAULT_POLICY)
    elif policy_option == "Upload YAML":
        uploaded_yaml = st.file_uploader("Upload Policy YAML", type=['yaml', 'yml'])
        if uploaded_yaml:
            yaml_content = uploaded_yaml.read().decode('utf-8')
            st.session_state.policy = load_policy_from_yaml(yaml_content)
    else:  # Edit Inline
        yaml_content = st.text_area("Edit Policy YAML:", value=DEFAULT_POLICY, height=300)
        if st.button("Apply Policy"):
            st.session_state.policy = load_policy_from_yaml(yaml_content)
    # Add Validation Mode Selection
    st.subheader("🔍 Validation Mode")
    validation_mode = st.radio(
        "Select Validation Mode:",
        ["Full Validation", "Receipt-Only Mode"],
        help="Choose 'Full Validation' for all policy checks or 'Receipt-Only Mode' for missing receipt reminders only."
    )
    st.session_state.validation_mode = validation_mode
    
    # Show mode description
    if validation_mode == "Receipt-Only Mode":
        st.info("📋 **Receipt-Only Mode**: Only checks for missing receipts on transactions ≥$20. Excludes Jeff Wu's Grab/Uber transactions.")
    else:
        st.info("📋 **Full Validation**: Comprehensive policy checking including spending limits, approvals, and receipts.")
    
    # Email configuration
    st.subheader("📧 Email Settings")
    
    # Real-time email configuration (saves immediately when you type)
    smtp_server = st.text_input(
        "SMTP Server", 
        value=st.session_state.get('smtp_server', 'smtp.gmail.com'),
        key="sidebar_smtp_server"
    )
    
    smtp_port = st.number_input(
        "SMTP Port", 
        value=st.session_state.get('smtp_port', 587),
        min_value=1, 
        max_value=65535,
        key="sidebar_smtp_port"
    )
    
    smtp_user = st.text_input(
        "Email Username", 
        value=st.session_state.get('smtp_user', ''),
        placeholder="your.email@company.com",
        key="sidebar_smtp_user"
    )
    
    smtp_pass = st.text_input(
        "Email Password", 
        value=st.session_state.get('smtp_pass', ''),
        type="password",
        placeholder="Your email password",
        key="sidebar_smtp_pass"
    )
    
    # Save values to session state immediately
    st.session_state.smtp_server = smtp_server
    st.session_state.smtp_port = smtp_port
    st.session_state.smtp_user = smtp_user
    st.session_state.smtp_pass = smtp_pass
    
    # Show current configuration status
    email_config_complete = all([smtp_server, smtp_user, smtp_pass])
    
    if email_config_complete:
        st.success("✅ Email configured")
        st.info(f"Server: {smtp_server}")
        st.info(f"User: {smtp_user}")
    else:
        st.warning("⚠️ Email not configured")
        remaining_fields = []
        if not smtp_server: remaining_fields.append("SMTP Server")
        if not smtp_user: remaining_fields.append("Email Username") 
        if not smtp_pass: remaining_fields.append("Email Password")
        st.info(f"Missing: {', '.join(remaining_fields)}")

# Main content
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Dashboard", "📋 Data Upload", "🔍 Validation Results", "📧 Email Notifications", "🔄Leaders Weekly Budget Update", "🤖 Personal Expense AI Detection", "🏢 Fixed Asset Tracker"])

with tab2:
    st.header("📋 Data Upload")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Sample Data")
        if st.button("Load Sample Data", help="Load test data for validation"):
            sample_data = {
                "Employee - ID": [
                    "EMP003", "EMP003", "EMP003", "EMP003", "EMP003", 
                    "EMP001", "EMP002", "EMP001", "EMP002"
                ],
                "Employee": [
                    "Charleston Yap", "Charleston Yap", "Charleston Yap", "Charleston Yap", "Charleston Yap",
                    "John Doe", "Jane Smith", "John Doe", "Jane Smith"
                ],
                "Vendor name": [
                    # Charleston's 5 transactions (mix of violations)
                    "Expensive Restaurant A", "Tech Store Premium", "Luxury Hotel Chain", 
                    "Amazon Business", "Airline Delta",
                    # Other employees
                    "Restaurant C", "Tech Store", "Hotel Chain", "Mobile Provider"
                ],
                "Amount (by category) - Currency": ["USD"] * 9,
                "Amount (by category)": [
                    # Charleston's amounts (designed to trigger various violations)
                    850.00,    # High amount - needs finance approval + missing attendees
                    3200.00,   # Over equipment limit + missing receipt
                    320.00,    # Over daily hotel limit
                    450.00,    # Missing receipt + high amount
                    1200.00,   # Flight - high amount + missing receipt
                    # Other employees (normal amounts)
                    125.50, 180.00, 95.00, 75.00
                ],
                "Purchase date": [
                    # Charleston's dates
                    "2025-02-15", "2025-02-16", "2025-02-18", "2025-02-25", "2025-02-23",
                    # Other employees
                    "2025-02-15", "2025-02-16", "2025-02-20", "2025-02-18"
                ],
                "Approval State": [
                    # Charleston's approvals
                    "Approved", "Approved", "Approved", "Approved", "Approved",
                    # Other employees
                    "Approved", "Pending", "Approved", "Approved"
                ],
                "Object Type": [
                    # Charleston's object types
                    "Expense", "Equipment", "Travel", "Equipment", "Travel",
                    # Other employees
                    "Expense", "Equipment", "Travel", "Communication"
                ],
                "Memo": [
                    # Charleston's memos (some empty to trigger violations)
                    "Executive dinner with clients", "New development laptop", "Hotel for conference", 
                    "Development tools", "Flight to client meeting",
                    # Other employees
                    "Business dinner", "New laptop", "Hotel stay", "Monthly phone"
                ],
                "Category Name": [
                    # Charleston's categories
                    "Biz Travel: Meals", "Equipment", "Hotel", "Equipment", "Airfare",
                    # Other employees
                    "Biz Travel: Meals", "Equipment", "Hotel", "Mobile"
                ],
                "Transaction Id": [
                    # Charleston's transaction IDs
                    "TXN101", "TXN102", "TXN104", "TXN111", "TXN109",
                    # Other employees
                    "TXN001", "TXN002", "TXN004", "TXN005"
                ],
                "Has Receipt": [
                    # Charleston's receipts (some missing to trigger violations)
                    "Yes", "No", "Yes", "No", "No",
                    # Other employees
                    "Yes", "Yes", "Yes", "Yes"
                ],
                "Check Date": [
                    # Charleston's check dates
                    "2025-02-16", "2025-02-17", "2025-02-19", "2025-02-26", "2025-02-24",
                    # Other employees
                    "2025-02-16", "2025-02-17", "2025-02-21", "2025-02-19"
                ],
                "Merchant currency": ["USD"] * 9,
                "Merchant Amount (by category) - Currency": ["USD"] * 9,
                "Merchant Amount (by category)": [
                    # Charleston's merchant amounts (same as regular amounts)
                    850.00, 3200.00, 320.00, 450.00, 1200.00,
                    # Other employees
                    125.50, 180.00, 95.00, 75.00
                ],
                "Attendees": [
                    # Charleston's attendees (some missing for meal violations)
                    "", "Charleston Yap", "Charleston Yap", "Charleston Yap", "Charleston Yap",
                    # Other employees
                    "John Doe, Jane Smith", "", "John Doe", ""
                ]
            }

            # Policy limits for testing (you can adjust these in your main code)
            POLICY_LIMITS = {
                "Biz Travel: Meals": {"daily": 75.00, "per_transaction": 150.00},
                "Equipment": {"daily": 2000.00, "per_transaction": 2500.00},
                "Hotel": {"daily": 300.00, "per_transaction": 400.00},
                "Travel: Transportation": {"daily": 100.00, "per_transaction": 200.00}, 
                "Travel: Gas": {"daily": 80.00, "per_transaction": 100.00},
                "Airfare": {"daily": 2000.00, "per_transaction": 3000.00},
                "Mobile": {"daily": 100.00, "per_transaction": 150.00}
            }

            # Expected violations for Charleston Yap (5 transactions):
            charleston_expected_violations = {
                "TXN101": ["High amount requiring finance approval ($850)", "Missing attendees for business meal"],
                "TXN102": ["Missing receipt", "Amount exceeds equipment transaction limit ($3200 > $2500)"],
                "TXN104": ["Amount exceeds hotel transaction limit ($320 > $300)"],
                "TXN111": ["Missing receipt", "High amount requiring finance approval ($450)"],
                "TXN109": ["Missing receipt", "High amount requiring finance approval ($1200)"]
            }

            st.session_state.df = pd.DataFrame(sample_data)
            st.session_state.original_df = st.session_state.df.copy()
            st.session_state.validation_complete = False  # Reset validation status
            st.success("Sample data loaded! 🎉")
            st.info("Note: This replaces any previously uploaded data.")
    
    with col2:
        # File Upload - Modified to preserve session state
        st.subheader("📁 File Upload")

        # Add disclaimer text
        st.markdown("""
        <div style="background: rgba(156, 102, 234, 0.1); border: 1px solid rgba(156, 102, 234, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <p style="margin: 0; font-size: 0.9em; color: #c084fc;">
                <strong>📋 File Requirements:</strong><br>
                Upload the <strong>Spend Management Report</strong> from Rippling:<br>
                <em>Rippling Home → Tools → Recipes → Finance → Spend Management Report → Start Building Report → Spend Transaction Report Entry → Transaction Details (Select required fields mentioned below)</em>
            </p>
            <p style="margin: 8px 0 0 0; font-size: 0.85em; color: #a78bfa;">
                <strong>Required fields to select:</strong> Attendees, Memo, Transaction ID, Has Receipt (in addition to default selections)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if we already have an uploaded file in session state
        if 'uploaded_file_processed' not in st.session_state:
            st.session_state.uploaded_file_processed = False

        uploaded_file = st.file_uploader("Upload Expense Report", 
                                       type=['xlsx', 'csv'],
                                       help="Upload your Excel or CSV expense report",
                                       key="expense_file_uploader")

        # Only process the file if it's new (not already processed)
        if uploaded_file and not st.session_state.uploaded_file_processed:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)

                st.session_state.df = df
                st.session_state.original_df = df.copy()
                st.session_state.validation_complete = False  # Reset validation status
                st.session_state.uploaded_file_processed = True  # Mark as processed
                st.success(f"✅ Uploaded {len(df)} transactions")
                st.info("📝 **Note:** File loaded successfully with your column format.")

                # Show column detection
                st.subheader("🔗 Column Detection")
                st.markdown("*Detected columns in your file:*")

                expected_columns = [
                    "Employee - ID", "Employee", "Vendor name", 
                    "Amount (by category) - Currency", "Amount (by category)",
                    "Purchase date", "Approval State", "Object Type", "Memo",
                    "Category Name", "Transaction Id", "Has Receipt", "Check Date",
                    "Attendees",  # <-- ADD THIS LINE
                    "Merchant currency", "Merchant Amount (by category) - Currency",
                    "Merchant Amount (by category)"
                ]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Found Columns:**")
                    found_cols = [col for col in expected_columns if col in df.columns]
                    for col in found_cols:
                        st.markdown(f"• {col}")

                with col2:
                    st.markdown("**❓ Missing Expected Columns:**")
                    missing_cols = [col for col in expected_columns if col not in df.columns]
                    if missing_cols:
                        for col in missing_cols:
                            st.markdown(f"• {col}")
                    else:
                        st.markdown("• None - All expected columns found!")

                if len(found_cols) >= 4:  # Minimum required columns
                    st.success("✅ File format looks good! Ready for validation.")
                else:
                    st.warning("⚠️ Some key columns may be missing. Validation may have issues.")

            except Exception as e:
                st.error(f"❌ Error uploading file: {e}")

        # Add a reset button to allow new file uploads
        if st.session_state.uploaded_file_processed:
            if st.button("🔄 Upload New File", help="Clear current data and upload a new file"):
                st.session_state.uploaded_file_processed = False
                st.session_state.df = None
                st.session_state.original_df = None
                st.session_state.validation_complete = False
                st.rerun()
    
    # Show current data
    if st.session_state.df is not None:
        st.subheader("👀 Current Data Preview")
        st.markdown(f"*Showing first 10 rows of {len(st.session_state.df)} total transactions*")
        
        # Show key columns first, then all columns
        key_columns = ['Employee', 'Category Name', 'Amount (by category)', 'Attendees', 'Purchase date', 'Transaction Id', 'Has Receipt', 'Memo', 'Approval State']
        available_key_columns = [col for col in key_columns if col in st.session_state.df.columns]

        if available_key_columns:
            st.dataframe(st.session_state.df[available_key_columns].head(10), use_container_width=True)
        else:
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Show full column list in expander
        with st.expander("📋 View All Columns"):
            st.write("**All available columns:**")
            st.write(list(st.session_state.df.columns))

with tab1:
    st.header("📊 Validation Dashboard")
    
    if st.session_state.df is not None and st.session_state.policy is not None:
        if st.button("🚀 Run Validation", type="primary"):
            with st.spinner("Validating expenses..."):
                mode = st.session_state.get('validation_mode', 'Full Validation')
                validated_df = process_expense_data(st.session_state.df, st.session_state.policy, mode=mode)
                if validated_df is not None:
                    st.session_state.df = validated_df
                    st.session_state.processed_df = validated_df
                    st.session_state.validation_complete = True
                    st.success(f"✅ Validation completed! Processed {len(validated_df)} transactions.")
                    st.info("📧 You can now send policy alert notifications in the 'Email Notifications' tab.")
                else:
                    st.error("❌ Failed to process data - please check your file format and try again.")
        
        if st.session_state.validation_complete and 'Status' in st.session_state.df.columns:
            # Summary metrics
            total_transactions = len(st.session_state.df)
            passed_transactions = len(st.session_state.df[st.session_state.df['Status'] == 'Pass'])
            failed_transactions = total_transactions - passed_transactions
            pass_rate = (passed_transactions / total_transactions) * 100 if total_transactions > 0 else 0
            
            # Create a more visually appealing metrics layout
            st.markdown('<div class="section-header">📊 Validation Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("📊 Total Transactions", f"{total_transactions:,}")
            with col2:
                st.metric("✅ Passed", f"{passed_transactions:,}", 
                         delta=f"{pass_rate:.1f}%", delta_color="normal")
            with col3:
                st.metric("❌ Failed", f"{failed_transactions:,}",
                         delta=f"{100-pass_rate:.1f}%", delta_color="inverse")
            with col4:
                total_amount = st.session_state.df['Amount'].sum()
                st.metric("💰 Total Amount", f"${total_amount:,.0f}")
            with col5:
                if 'Employee_Out_of_Pocket' in st.session_state.df.columns:
                    total_out_of_pocket = st.session_state.df['Employee_Out_of_Pocket'].sum()
                    cost_impact = (total_out_of_pocket / total_amount * 100) if total_amount > 0 else 0
                    st.metric("💸 Employee Out-of-Pocket Cost", f"${total_out_of_pocket:,.0f}", 
                             delta=f"{cost_impact:.1f}% of total", delta_color="inverse")
                    
            # Advanced Visualizations with purple theme
            st.markdown('<div class="section-header">📈 Performance Analytics</div>', unsafe_allow_html=True)

            # Define consistent color schemes
            purple_colors = ['#7c3aed', '#9c66ea', '#c084fc', '#e879f9', '#fbbf24', '#34d399', '#60a5fa']
            purple_gradient = ['#1a0d2e', '#2d1b3d', '#3e2659', '#4a2c6b', '#5c3a7d', '#7c3aed', '#9c66ea']

            col1, col2 = st.columns(2)
            
            with col1:
                # Status distribution with enhanced styling and insights
                status_counts = st.session_state.df['Status'].value_counts()
                total_records = len(st.session_state.df)
                pass_rate = (status_counts.get('Pass', 0) / total_records * 100) if total_records > 0 else 0

                fig_status = px.pie(
                    values=status_counts.values, 
                    names=status_counts.index,
                    title=f"Validation Status Distribution<br><sub>Pass Rate: {pass_rate:.1f}%</sub>",
                    color_discrete_map={'Pass': '#ffffff', 'Fail': '#7c3aed'},
                    hole=0.4  # Creates a donut chart for modern look
                )

                # Enhanced styling
                fig_status.update_traces(
                    textposition='outside',
                    textinfo='percent+label+value',
                    textfont_size=12
                )

                fig_status.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    annotations=[dict(text=f'{total_records}<br>Total', x=0.5, y=0.5, font_size=16, showarrow=False)]
                )

                st.plotly_chart(fig_status, use_container_width=True)

            with col2:
                # Amount by status with enhanced insights
                status_amounts = st.session_state.df.groupby('Status')['Amount'].sum().reset_index()
                total_amount = status_amounts['Amount'].sum()

                # Create pie chart instead of bar chart for consistency
                fig_amounts = px.pie(
                    status_amounts, 
                    values='Amount', 
                    names='Status',
                    title=f"Amount Distribution by Status<br><sub>Total: ${total_amount:,.0f}</sub>",
                    color='Status',
                    color_discrete_map={'Pass': '#ffffff', 'Fail': '#7c3aed'},
                    hole=0.4
                )

                fig_amounts.update_traces(
                    textposition='outside',
                    textinfo='percent+label',
                    texttemplate='%{label}<br>$%{value:,.0f}<br>(%{percent})',
                    textfont_size=12,
                    marker=dict(line=dict(color='#ffffff', use_container_width=2))
                )

                fig_amounts.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    annotations=[dict(text=f'${total_amount:,.0f}<br>Total', x=0.5, y=0.5, font_size=16, showarrow=False)]
                )

                st.plotly_chart(fig_amounts, use_container_width=True)
            
            # Failure reasons analysis
            if failed_transactions > 0:
                st.markdown('<div class="section-header">🔍 Failure Deep Dive</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    # Horizontal funnel chart for failure reasons
                    failed_df = st.session_state.df[st.session_state.df['Status'] == 'Fail']
                    failure_reasons = []
                    for msg in failed_df['Message']:
                        if pd.isna(msg):
                            continue
                        reasons = msg.split(';')
                        failure_reasons.extend([reason.strip() for reason in reasons])

                    if failure_reasons:
                        reason_counts = pd.Series(failure_reasons).value_counts().head(8)

                        # Create a more sophisticated horizontal bar with gradient
                        fig_reasons = go.Figure()

                        colors = purple_gradient[:len(reason_counts)]

                        fig_reasons.add_trace(go.Bar(
                            y=reason_counts.index,
                            x=reason_counts.values,
                            orientation='h',
                            marker=dict(
                                color=colors,
                                line=dict(color='#9c66ea', use_container_width=1)
                            ),
                            text=[f'{val}' for val in reason_counts.values],
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
                        ))

                        fig_reasons.update_layout(
                            title={'text': 'Top Failure Reasons', 'font': {'color': '#e2e8f0', 'size': 18}},
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#e2e8f0', 'family': 'Arial'},
                            xaxis={'gridcolor': 'rgba(156, 102, 234, 0.2)'},
                            yaxis={'gridcolor': 'rgba(156, 102, 234, 0.2)', 'categoryorder': 'total ascending'},
                            height=400
                        )
                        st.plotly_chart(fig_reasons, use_container_width=True)

                with col2:
                    # Simple horizontal bar chart for employee violations
                    violation_counts = failed_df['Employee'].value_counts().head(10)  # Top 10 employees

                    fig_violations = px.bar(
                        x=violation_counts.values,
                        y=violation_counts.index,
                        orientation='h',
                        title='Top 10 Employees by Violation Count',
                        color=violation_counts.values,
                        color_continuous_scale='Purples',
                        text=violation_counts.values
                    )

                    fig_violations.update_traces(
                        texttemplate='%{text}',
                        textposition='outside',
                        textfont_size=12
                    )

                    fig_violations.update_layout(
                        xaxis_title="Number of Violations",
                        yaxis_title="Employee",
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e2e8f0', 'family': 'Arial'},
                        title={'font': {'color': '#e2e8f0', 'size': 18}},
                        height=400,
                        yaxis={'categoryorder': 'total ascending'}
                    )

                    st.plotly_chart(fig_violations, use_container_width=True)
                
                # Employee violation summary
                st.subheader("👥 Employee Violation Summary")
                employee_violations = failed_df.groupby('Employee').agg({
                    'Amount': ['count', 'sum']
                }).round(2)
                employee_violations.columns = ['Violation Count', 'Total Amount']
                employee_violations = employee_violations.sort_values('Violation Count', ascending=False)
                st.dataframe(employee_violations, use_container_width=True)
            # Employee Out-of-Pocket Cost Analysis
            if 'Employee_Out_of_Pocket' in st.session_state.df.columns:
                total_out_of_pocket = st.session_state.df['Employee_Out_of_Pocket'].sum()
                employees_with_costs = len(st.session_state.df[st.session_state.df['Employee_Out_of_Pocket'] > 0]['Employee'].unique())

                if total_out_of_pocket > 0:
                    st.markdown('<div class="section-header">💸 Cost Impact Analysis</div>', unsafe_allow_html=True)

                    # Cost breakdown by employee
                    employee_costs = st.session_state.df[st.session_state.df['Employee_Out_of_Pocket'] > 0].groupby('Employee').agg({
                        'Employee_Out_of_Pocket': 'sum',
                        'Amount': 'count'
                    }).round(2)
                    employee_costs.columns = ['Out_of_Pocket_Amount', 'Transaction_Count']
                    employee_costs = employee_costs.sort_values('Out_of_Pocket_Amount', ascending=False)

                    col1, col2 = st.columns(2)

                    with col1:
                        # 3D-style bar chart with employee costs
                        fig_employee_3d = go.Figure()

                        fig_employee_3d.add_trace(go.Bar(
                            x=employee_costs.index,
                            y=employee_costs['Out_of_Pocket_Amount'],
                            marker=dict(
                                color=employee_costs['Out_of_Pocket_Amount'],
                                colorscale='Purples',
                                showscale=True,
                                colorbar=dict(title='Amount ($)', tickfont=dict(color='#e2e8f0')),
                                line=dict(color='#9c66ea', use_container_width=1)
                            ),
                            text=[f'${val:,.0f}' for val in employee_costs['Out_of_Pocket_Amount']],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Out-of-Pocket: $%{y:,.0f}<br>Transactions: %{customdata}<extra></extra>',
                            customdata=employee_costs['Transaction_Count']
                        ))

                        fig_employee_3d.update_layout(
                            title={'text': 'Employee Cost Impact', 'font': {'color': '#e2e8f0', 'size': 18}},
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#e2e8f0', 'family': 'Arial'},
                            xaxis={'tickangle': -45, 'gridcolor': 'rgba(156, 102, 234, 0.2)'},
                            yaxis={'gridcolor': 'rgba(156, 102, 234, 0.2)', 'title': 'Amount ($)'},
                            height=400
                        )
                        st.plotly_chart(fig_employee_3d, use_container_width=True)

                    with col2:
                        # Simple bar chart for cost distribution by category
                        category_costs = st.session_state.df[st.session_state.df['Employee_Out_of_Pocket'] > 0].groupby('Category')['Employee_Out_of_Pocket'].sum().sort_values(ascending=True)

                        fig_costs = px.bar(
                            x=category_costs.values,
                            y=category_costs.index,
                            orientation='h',
                            title='Employee Out-of-Pocket Costs by Category',
                            color=category_costs.values,
                            color_continuous_scale='Purples',
                            text=category_costs.values
                        )

                        fig_costs.update_traces(
                            texttemplate='$%{text:,.0f}',
                            textposition='outside',
                            textfont_size=12
                        )

                        fig_costs.update_layout(
                            xaxis_title="Cost ($)",
                            yaxis_title="Category",
                            showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#e2e8f0', 'family': 'Arial'},
                            title={'font': {'color': '#e2e8f0', 'size': 18}},
                            height=400,
                            xaxis={'tickformat': '$,.0f'}
                        )

                        st.plotly_chart(fig_costs, use_container_width=True)

                    # Enhanced detailed breakdown with sparklines
                    st.markdown("### 📋 Executive Cost Summary")

                    # Create summary metrics
                    avg_cost_per_employee = total_out_of_pocket / employees_with_costs if employees_with_costs > 0 else 0

                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:
                        st.metric("💰 Total Cost Impact", f"${total_out_of_pocket:,.0f}")
                    with summary_col2:
                        st.metric("👥 Affected Employees", f"{employees_with_costs}")
                    with summary_col3:
                        cost_percentage = (total_out_of_pocket / total_amount * 100) if total_amount > 0 else 0
                        st.metric("📈 Cost Impact %", f"{cost_percentage:.1f}%")

                    # Enhanced table with formatting
                    employee_costs_display = employee_costs.copy()
                    employee_costs_display['Out_of_Pocket_Amount'] = employee_costs_display['Out_of_Pocket_Amount'].apply(lambda x: f"${x:,.2f}")
                    employee_costs_display.columns = ['💸 Out-of-Pocket Amount', '📊 Transaction Count']

                    st.dataframe(
                        employee_costs_display, 
                        use_container_width=True,
                        height=300
                    )
    else:
        st.info("📋 Please upload expense data and configure policy settings in the sidebar to begin validation.")
        
        # Show policy preview if available
        if st.session_state.policy:
            with st.expander("👀 Current Policy Preview"):
                st.json(st.session_state.policy)

with tab3:
    st.header("Validation Results")
    
    if st.session_state.validation_complete and st.session_state.df is not None:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect("Filter by Status:", 
                                         st.session_state.df['Status'].unique(),
                                         default=st.session_state.df['Status'].unique())
        
        with col2:
            employee_filter = st.multiselect("Filter by Employee:",
                                           st.session_state.df['Employee'].unique(),
                                           default=st.session_state.df['Employee'].unique())
        
        with col3:
            min_amount = st.number_input("Minimum Amount:", value=0.0)
        
        # Apply filters
        filtered_df = st.session_state.df[
            (st.session_state.df['Status'].isin(status_filter)) &
            (st.session_state.df['Employee'].isin(employee_filter)) &
            (st.session_state.df['Amount'] >= min_amount)
        ]
        
        # Display results
        st.subheader(f"Filtered Results ({len(filtered_df)} transactions)")
        
        # Configure display columns
        display_columns = ['Employee', 'Date', 'Category', 'Amount', 'Attendees', 'Attendees_Count', 'Per_Pax_Amount', 'Vendor', 'Transaction_ID', 'HasReceipt', 'Memo', 'Status', 'Message']
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_columns],
            use_container_width=True,
            column_config={
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    format="$%.2f"
                ),
                "Date": st.column_config.DateColumn(
                    "Date",
                    format="YYYY-MM-DD"
                ),
                "Transaction_ID": st.column_config.TextColumn(
                    "Transaction ID",
                    help="Unique transaction identifier"
                ),
                "HasReceipt": st.column_config.CheckboxColumn(
                    "Has Receipt",
                    help="Whether receipt is available"
                ),
                "Memo": st.column_config.TextColumn(
                    "Memo",
                    help="Transaction notes (may be empty)"
                )
            }
        )
        
        # Download results
        if st.button("Download Results as CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"expense_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("Run validation first to see results.")


with tab4:
    st.header("📧 Notifications")
    
    mode = st.session_state.get('validation_mode', 'Full Validation')
    st.info("ℹ️ After changing the validation mode in the sidebar, click 'Run Validation' to update the results.")
    
    if 'processed_df' not in st.session_state or st.session_state.processed_df is None or len(st.session_state.processed_df) == 0:
        st.info("📄 Run expense policy check first to generate notifications.")
    else:
        violations_df = st.session_state.processed_df[st.session_state.processed_df['Status'] == 'Fail'].copy()
        if len(violations_df) == 0:
            st.success(f"🎉 No {'receipts missing' if mode == 'Receipt-Only Mode' else 'policy violations'} found!")
            st.success("🎉 Excellent! No policy violations found.")
            st.info("All employees are compliant with the expense policy.")
        else:
            employee_violations = {}
            for _, row in violations_df.iterrows():
                emp = row['Employee']
                employee_violations.setdefault(emp, []).append({
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Amount': row['Amount'],
                    'Company_Funded_Amount': row.get('Company_Funded_Amount', row['Amount']),
                    'Employee_Out_of_Pocket': row.get('Employee_Out_of_Pocket', 0.0),
                    'Category': row['Category'],
                    'Vendor': row.get('Vendor', 'N/A'),
                    'Transaction_ID': row.get('Transaction_ID', 'N/A'),
                    'Memo': row.get('Memo', ''),
                    'HasReceipt': row.get('HasReceipt', True),
                    'Message': row['Message']
                })
            
            test_mode = st.toggle("🧪 Test Mode", value=True, help="Send notifications to Lulu only if enabled.")
            if test_mode and "Lulu" in employee_violations:
                st.info(f"🧪 Test Mode: Only Lulu's {len(employee_violations['Lulu'])} violation(s) will be notified.")
            elif test_mode:
                st.info("🧪 Test Mode: No violations for Lulu.")
            
            st.subheader("📋 Summary by Employee" if mode == "Full Validation" else "📋 Missing Receipts by Employee")
            for emp, violations in employee_violations.items():
                with st.expander(f"{emp} ({len(violations)} {'violation(s)' if mode == 'Full Validation' else 'missing receipt(s)'})"):
                    total_amount = sum(v['Amount'] for v in violations)
                    total_company_funded = sum(v['Company_Funded_Amount'] for v in violations)
                    total_employee_cost = sum(v['Employee_Out_of_Pocket'] for v in violations)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Amount", f"${total_amount:.2f}")
                    with col2:
                        if mode == "Full Validation":
                            st.metric("Company Funded", f"${total_company_funded:.2f}")
                    with col3:
                        if mode == "Full Validation":
                            st.metric("Employee Out-of-Pocket", f"${total_employee_cost:.2f}" if total_employee_cost > 0 else "$0.00")
                    for i, v in enumerate(violations, 1):
                        color = "🔴" if v['Employee_Out_of_Pocket'] > 0 and mode == "Full Validation" else "🟡" if "Missing receipt" in v['Message'] else "🟠"
                        st.markdown(f"""
                        {color} **Transaction #{i}**
                        - **📅 Date:** {v['Date']}
                        - **💰 Amount:** ${v['Amount']:.2f}
                        {f"- **🏢 Company Funded:** ${v['Company_Funded_Amount']:.2f}" if mode == "Full Validation" else ""}
                        {f"- **👤 Employee Cost:** ${v['Employee_Out_of_Pocket']:.2f}" if mode == "Full Validation" and v['Employee_Out_of_Pocket'] > 0 else ""}
                        - **🏷️ Category:** {v['Category']}
                        - **🏪 Vendor:** {v['Vendor']}
                        - **🆔 Transaction ID:** {v['Transaction_ID']}
                        - **📝 Memo:** {v['Memo'] or 'No memo'}
                        - **🧾 Receipt:** {'✅ Yes' if v['HasReceipt'] else '❌ No'}
                        - **⚠️ Issue:** {v['Message']}
                        ---
                        """)
            if employee_violations:
                # Validate email addresses for all employees with violations - FIX FOR ERROR 2
                st.subheader("📧 Email Validation")
                email_validation_results = {}

                for emp in employee_violations.keys():
                    if test_mode and emp.lower() not in ["lulu xia", "lulu"]:
                        # In test mode, only validate lulu's email
                        continue

                    email = get_employee_email(emp) if not test_mode else "lulu@k-id.com"
                    email_validation_results[emp] = email

                    if email:
                        st.success(f"✅ {emp}: {email}")
                    else:
                        st.error(f"❌ {emp}: No email found")

                # Show summary
                valid_emails = len([email for email in email_validation_results.values() if email])
                total_employees = len(employee_violations)
                st.info(f"📊 Email Validation Summary: {valid_emails}/{total_employees} employees have valid emails")
            
            # Updated recipient selection section
            if employee_violations and any(email_validation_results.values()):
                st.subheader("👥 Select Recipients")
                if test_mode:
                    lulu_violations = employee_violations.get("Lulu", [])
                    lulu_has_email = email_validation_results.get("Lulu") is not None

                    if lulu_violations and lulu_has_email:
                        selected_recipients = st.multiselect(
                            f"Select employees to receive {mode.lower()} reminders:",
                            options=["Lulu"],
                            default=["Lulu"],
                            help="Test mode: Only Lulu available"
                        )
                    elif lulu_violations and not lulu_has_email:
                        st.error("🧪 **Test Mode**: Lulu has violations but no email found. Cannot send test email.")
                        selected_recipients = []
                    else:
                        st.info(f"🧪 **Test Mode**: Lulu has no violations in {mode}, so no test email available.")
                        selected_recipients = []
                else:
                    # Only show employees with valid emails
                    available_recipients = [emp for emp in employee_violations.keys() if email_validation_results.get(emp)]

                    violation_type = "missing receipt" if mode == "Receipt-Only Mode" else "policy violation"
                    selected_recipients = st.multiselect(
                        f"Select employees to receive {violation_type} reminders:",
                        options=available_recipients,
                        default=available_recipients,
                        help="Only employees with valid email addresses are shown"
                    )
            else:
                selected_recipients = []
                if employee_violations:
                    st.warning("⚠️ No employees have valid email addresses. Cannot send notifications.")
                else:
                    violation_type = "missing receipts" if mode == "Receipt-Only Mode" else "policy violations"
                    st.info(f"ℹ️ No {violation_type} found. No notifications needed.")

            # Filter employee_violations based on selection
            filtered_employee_violations = {emp: violations for emp, violations in employee_violations.items() 
                                          if emp in selected_recipients}
            
            # Send policy violation reminders
            st.subheader("📧 Send Policy Violation Reminder Emails")

            if all([smtp_server, smtp_user, smtp_pass]):
                # Count how many emails will be sent
                emails_to_send = len(filtered_employee_violations)

                if test_mode:
                    send_button_text = f"🧪 Send Test Policy Reminder ({emails_to_send} recipient)" if emails_to_send == 1 else f"🧪 Send Test Policy Reminders ({emails_to_send} recipients)"
                else:
                    send_button_text = f"📧 Send Policy Reminder ({emails_to_send} recipient)" if emails_to_send == 1 else f"📧 Send Policy Reminders ({emails_to_send} recipients)"

                if emails_to_send > 0:
                    if st.button(send_button_text, type="primary"):
                        smtp_config = {
                            'server': smtp_server,
                            'port': smtp_port,
                            'user': smtp_user,
                            'password': smtp_pass
                        }

                        # Show sending progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        with st.spinner("Sending policy alert reminder emails..."):
                            # Capture the output to show more detailed results
                            import io
                            import contextlib

                            output_buffer = io.StringIO()
                            with contextlib.redirect_stdout(output_buffer):
                                success = send_policy_notification_email_with_ui(filtered_employee_violations, smtp_config, test_mode, mode=mode)

                            # Get the output
                            output_text = output_buffer.getvalue()

                            # Show results
                            if success:
                                # After successful email sending:
                                st.session_state.policy_email_result = {
                                    'success': True,
                                    'scheduled_count': len(filtered_employee_violations)
                                }

                                if test_mode:
                                    st.success(f"✅ Test policy violation reminder sent successfully!")
                                else:
                                    st.success(f"✅ Policy violation reminders sent successfully!")

                                # Show detailed output
                                if output_text:
                                    with st.expander("📋 Detailed Results"):
                                        st.text(output_text)
                            else:
                                # After failed email sending:
                                st.session_state.policy_email_result = {
                                    'success': False,
                                    'scheduled_count': 0
                                }

                                st.error("❌ Failed to send policy violation reminders")

                                # Show error details
                                if output_text:
                                    with st.expander("📋 Error Details"):
                                        st.text(output_text)

                        progress_bar.progress(100)
                        status_text.text("Complete!")

                    # MOVED THE CANCEL LOGIC OUTSIDE THE SEND BUTTON BLOCK
                    # Cancel button with enhanced feedback - NOW ALWAYS AVAILABLE WHEN EMAILS ARE SCHEDULED
                    if 'policy_email_result' in st.session_state and st.session_state.policy_email_result.get('success') and st.session_state.policy_email_result.get('scheduled_count', 0) > 0:
                        st.markdown("---")

                        # Show current scheduled emails info using POLICY session state
                        with EMAIL_LOCK:
                            total_emails = len(st.session_state.POLICY_SCHEDULED_EMAILS)
                            active_emails = sum(1 for email_data in st.session_state.POLICY_SCHEDULED_EMAILS.values() 
                                              if not email_data.get('cancelled', False) 
                                              and not email_data.get('sent', False)
                                              and not email_data.get('failed', False))

                        st.write(f"📧 Current scheduled policy emails: {total_emails}")
                        st.write(f"⏳ Active policy emails pending: {active_emails}")
                        print(f"🔍 UI DEBUG: POLICY dict ID from UI: {id(st.session_state.POLICY_SCHEDULED_EMAILS)}")
                        st.write(f"🔍 DEBUG: Policy Dict ID: {id(st.session_state.POLICY_SCHEDULED_EMAILS)}")

                        if active_emails > 0:
                            if st.button("❌ Cancel Policy Emails", type="secondary"):
                                with st.spinner("Cancelling policy emails..."):
                                    try:
                                        print("🔍 Policy cancel button clicked - starting cancellation process")

                                        # Show what we have before cancelling
                                        st.write(f"🔍 Debug: Found {total_emails} total policy emails in system")
                                        st.write(f"🔍 Debug: Found {active_emails} active policy emails to cancel")

                                        # Cancel all POLICY emails using session state
                                        cancelled_count = cancel_all_scheduled_policy_emails()
                                        time.sleep(2)  # Give threads time to process

                                        with EMAIL_LOCK:
                                            # Verify cancellation worked
                                            still_active = sum(1 for email_data in st.session_state.POLICY_SCHEDULED_EMAILS.values() 
                                                              if not email_data.get('cancelled', False) 
                                                              and not email_data.get('sent', False)
                                                              and not email_data.get('failed', False))

                                        st.write(f"📊 Verification: {still_active} emails still active after cancellation")

                                        # Provide feedback based on results
                                        if cancelled_count > 0:
                                            st.success(f"✅ Successfully cancelled {cancelled_count} scheduled policy emails")
                                            st.info("📧 Cancelled policy emails will not be sent when their scheduled time arrives")

                                            # Show updated status
                                            with EMAIL_LOCK:
                                                remaining_active = sum(1 for email_data in st.session_state.POLICY_SCHEDULED_EMAILS.values() 
                                                                     if not email_data.get('cancelled', False) 
                                                                     and not email_data.get('sent', False)
                                                                     and not email_data.get('failed', False))
                                            st.write(f"📊 Remaining active scheduled policy emails: {remaining_active}")

                                            # Clear the result so the cancel section disappears
                                            time.sleep(2)
                                            if 'policy_email_result' in st.session_state:
                                                del st.session_state.policy_email_result
                                            st.rerun()

                                        elif total_emails == 0:
                                            st.info("ℹ️ No policy emails found in the system to cancel")
                                        else:
                                            st.warning("⚠️ No policy emails were cancelled (they may have already been sent or cancelled)")

                                    except Exception as e:
                                        st.error(f"❌ Error cancelling policy emails: {str(e)}")
                                        st.write(f"Exception details: {type(e).__name__}: {str(e)}")
                                        print(f"🔍 EXCEPTION in cancel: {str(e)}")

                                        # Clear the session state even if there's an error
                                        if 'policy_email_result' in st.session_state:
                                            del st.session_state.policy_email_result
                        else:
                            st.info("ℹ️ No active policy emails to cancel")
                            # Also add a dismiss button when no active emails
                            if st.button("✅ Dismiss", key="dismiss_policy_no_active"):
                                if 'policy_email_result' in st.session_state:
                                    del st.session_state.policy_email_result
                                st.rerun()
                else:
                    st.info("No recipients selected or no policy violations for selected employees.")
            else:
                st.warning("⚠️ Please configure email settings in the sidebar to send policy violation reminders.")
                
        # Add summary statistics if violations exist - FIXED: Use session state
        if 'processed_df' in st.session_state and st.session_state.processed_df is not None:
            # Filter for violations from session state
            violations_df = st.session_state.processed_df[st.session_state.processed_df['Status'] == 'Fail']
            
            if len(violations_df) > 0:
                st.subheader("📊 Violation Statistics")
                
                # Create summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_violations = len(violations_df)
                    st.metric("Total Violations", total_violations)
                
                with col2:
                    total_employees = violations_df['Employee'].nunique()
                    st.metric("Employees Affected", total_employees)
                
                with col3:
                    total_amount = violations_df['Amount'].sum()
                    st.metric("Total Amount at Risk", f"${total_amount:.2f}")
                
                with col4:
                    total_employee_cost = violations_df.get('Employee_Out_of_Pocket', pd.Series([0])).sum()
                    st.metric("Total Employee Cost", f"${total_employee_cost:.2f}")
                
                # Violation breakdown by type
                st.subheader("📈 Violation Breakdown")
                
                # Parse violation messages to categorize them
                violation_categories = {
                    'Missing Receipt': 0,
                    'Spending Limits': 0,
                    'Approval Issues': 0,
                    'High Amount': 0,
                    'Other': 0
                }
                
                for _, row in violations_df.iterrows():
                    message = row['Message'].lower()
                    if 'missing receipt' in message:
                        violation_categories['Missing Receipt'] += 1
                    elif 'limit exceeded' in message or 'daily limit' in message:
                        violation_categories['Spending Limits'] += 1
                    elif 'approval' in message:
                        violation_categories['Approval Issues'] += 1
                    elif 'high amount' in message or 'finance approval' in message:
                        violation_categories['High Amount'] += 1
                    else:
                        violation_categories['Other'] += 1
                
                # Display breakdown
                for category, count in violation_categories.items():
                    if count > 0:
                        st.write(f"• **{category}**: {count} violation(s)")
        
with tab5:
    st.title("💰 Budget Dashboard")
    
    # Check if raw expense data is available from tab 2
    if 'df' not in st.session_state or st.session_state.df is None:
        st.empty()
        st.error("📊 **Missing Expense Data** - Please upload your expense data in the 'Data Upload' tab first.")
        st.stop()
    
    # Main container with better spacing
    st.markdown("---")
    
    with st.expander("🏢 Manage Department Mappings", expanded=False):
        display_department_mapping_editor()

    st.markdown("---")
    
    # Step 1: Budget File Upload (Streamlined)
    with st.container():
        st.subheader("📁 Budget Configuration")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_budget_file = st.file_uploader(
                "Upload Budget Limits File (CSV/Excel)",
                type=['csv', 'xlsx', 'xls'],
                help="First column: department names, other columns: monthly budgets (Jan 2025, Feb 2025, etc.)"
            )
        
        with col2:
            focus_on_travel = st.toggle("🧳 Travel Budget Only", 
                                       value=True, 
                                       help="Focus analysis on travel expenses only")
        
        if uploaded_budget_file is not None:
            # Load and process budget file
            budget_df = load_budget_file(uploaded_budget_file)
            
            if budget_df is not None:
                # Process budget data
                budget_dict = process_budget_data(budget_df, focus_on_travel)
                
                if budget_dict:
                    st.session_state.budget_dict = budget_dict
                    st.session_state.focus_on_travel = focus_on_travel
                    
                    # Success message with summary
                    total_depts = len(budget_dict)
                    total_budget = sum(sum(budgets.values()) for budgets in budget_dict.values())
                    
                    st.success(f"✅ **Budget loaded**: {total_depts} departments, ${total_budget:,.0f} total budget")
                    
                    # Collapsible detailed view
                    with st.expander("📊 View Budget Details", expanded=False):
                        summary_data = []
                        for dept, budgets in budget_dict.items():
                            total_budget = sum(budgets.values())
                            months_count = len(budgets)
                            
                            summary_data.append({
                                'Department': dept,
                                'Total Budget': f"${total_budget:,.0f}",
                                'Months': months_count,
                                'Avg Monthly': f"${total_budget/months_count:,.0f}"
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Step 2: Budget Analysis (Only show if budget is loaded)
    if 'budget_dict' in st.session_state and 'df' in st.session_state:
        budget_dict = st.session_state.budget_dict
        raw_df = st.session_state.df
        focus_on_travel = st.session_state.get('focus_on_travel', True)
        
        with st.container():
            st.subheader("📊 Budget Analysis")
            
            # Get date range from expense data
            if 'Date' in raw_df.columns:
                raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
                min_date = raw_df['Date'].min().date()
                max_date = raw_df['Date'].max().date()
                
                # Compact date range selector
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    start_date = st.date_input("📅 Start Date", value=min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("📅 End Date", value=max_date, min_value=min_date, max_value=max_date)
                with col3:
                    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
                
                if analyze_button and start_date <= end_date:
                    with st.spinner("Analyzing budget..."):
                        budget_status = calculate_budget_status_for_period(
                            raw_df, budget_dict, start_date, end_date, focus_on_travel
                        )
                        
                        st.session_state.budget_status = budget_status
                        st.session_state.analysis_period = {'start': start_date, 'end': end_date}
                        
                        if budget_status:
                            st.success(f"✅ Analysis complete: {len(budget_status)} departments analyzed")
                        else:
                            st.warning("⚠️ No budget data found for the selected period")
                elif analyze_button:
                    st.error("❌ Start date must be before end date")
            else:
                st.error("❌ Date column not found in expense data")
        
        st.markdown("---")
        
        # Step 3: Results Dashboard (Only show if analysis is complete)
        if 'budget_status' in st.session_state and 'analysis_period' in st.session_state:
            budget_status = st.session_state.budget_status
            period = st.session_state.analysis_period
            
            if budget_status:
                budget_type = "Travel Budget" if focus_on_travel else "Budget"
                period_days = (period['end'] - period['start']).days + 1
                
                # Key Metrics Dashboard
                st.subheader(f"📈 {budget_type} Overview")
                st.caption(f"Period: {period['start']} to {period['end']} ({period_days} days)")
                
                # Calculate summary metrics
                total_allocated = sum(info['allocated_budget'] for info in budget_status.values())
                total_spent = sum(info['total_spent'] for info in budget_status.values())
                utilization_rate = (total_spent / total_allocated * 100) if total_allocated > 0 else 0
                
                # Status counts
                status_counts = {}
                for info in budget_status.values():
                    status = info['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Total Budget", f"${total_allocated:,.0f}")
                with col2:
                    st.metric("💸 Total Spent", f"${total_spent:,.0f}")
                with col3:
                    delta_color = "normal" if utilization_rate <= 80 else "inverse"
                    st.metric("📊 Utilization", f"{utilization_rate:.0f}%")
                with col4:
                    over_budget = status_counts.get('Over Budget', 0)
                    st.metric("🚨 Over Budget", f"{over_budget} teams", delta_color="inverse" if over_budget > 0 else "normal")
                
                # Status breakdown
                if len(status_counts) > 1:
                    st.subheader("🎯 Status Breakdown")
                    status_cols = st.columns(len(status_counts))
                    for i, (status, count) in enumerate(status_counts.items()):
                        with status_cols[i]:
                            emoji = {"On Track": "✅", "Near Limit": "⚠️", "Over Budget": "🚨"}.get(status, "📊")
                            st.metric(f"{emoji} {status}", f"{count} teams")
                
                st.markdown("---")
                
                # Department Results
                st.subheader("🏢 Department Results")
                
                # Create clean results table
                results_data = []
                for dept, info in budget_status.items():
                    # Status emoji
                    status_emoji = {"On Track": "✅", "Near Limit": "⚠️", "Over Budget": "🚨"}.get(info['status'], "📊")
                    
                    results_data.append({
                        'Department': dept,
                        'Team Leader': info['leader'],
                        'Budget': f"${info['allocated_budget']:,.0f}",
                        'Spent': f"${info['total_spent']:,.0f}",
                        'Remaining': f"${info['remaining_budget']:,.0f}",
                        'Usage': f"{info['utilization_rate']:.0f}%",
                        'Status': f"{status_emoji} {info['status']}",
                        'Team Size': len(info['team_members'])
                    })
                
                results_df = pd.DataFrame(results_data)
                
                # Sort by utilization rate (highest first)
                results_df['_sort_utilization'] = results_df['Usage'].str.rstrip('%').astype(float)
                results_df = results_df.sort_values('_sort_utilization', ascending=False).drop('_sort_utilization', axis=1)
                
                # Display with conditional formatting
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Team Detail View
                st.markdown("---")
                st.subheader("👥 Team Breakdown")
                
                # Department selector
                selected_dept = st.selectbox(
                    "Select Department:",
                    options=list(budget_status.keys()),
                    key="dept_selector"
                )
                
                if selected_dept:
                    dept_info = budget_status[selected_dept]
                    
                    # Department header
                    status_emoji = {"On Track": "✅", "Near Limit": "⚠️", "Over Budget": "🚨"}.get(dept_info['status'], "📊")
                    st.markdown(f"### {status_emoji} {selected_dept}")
                    st.caption(f"Team Leader: {dept_info['leader']} • Team Size: {len(dept_info['team_members'])}")
                    
                    # Department metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Budget", f"${dept_info['allocated_budget']:,.0f}")
                    with col2:
                        st.metric("Spent", f"${dept_info['total_spent']:,.0f}")
                    with col3:
                        remaining = dept_info['remaining_budget']
                        delta_color = "normal" if remaining >= 0 else "inverse"
                        st.metric("Remaining", f"${remaining:,.0f}")
                    
                    # Team member expenses
                    if dept_info['member_expenses']:
                        st.markdown("**Team Member Expenses:**")
                        
                        # Create member breakdown
                        member_data = []
                        for member in dept_info['team_members']:
                            amount = dept_info['member_expenses'].get(member, 0)
                            percentage = (amount / dept_info['total_spent'] * 100) if dept_info['total_spent'] > 0 else 0
                            
                            member_data.append({
                                'Member': member,
                                'Amount': f"${amount:,.0f}",
                                'Share': f"{percentage:.0f}%",
                                'Status': '💸 Has Expenses' if amount > 0 else '💤 No Expenses'
                            })
                        
                        # Sort by amount (descending)
                        member_data.sort(key=lambda x: float(x['Amount'].replace('$', '').replace(',', '')), reverse=True)
                        
                        member_df = pd.DataFrame(member_data)
                        st.dataframe(member_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Alert Section (Simplified)
                st.subheader("🚨 Budget Alerts")
                
                # Find teams needing attention
                attention_needed = [dept for dept, info in budget_status.items() if info['status'] in ['Over Budget', 'Near Limit']]
                
                if attention_needed:
                    st.warning(f"⚠️ **{len(attention_needed)} team(s) need attention**")
                    
                    # Show teams in alert boxes
                    for dept in attention_needed:
                        info = budget_status[dept]
                        status_emoji = {"Near Limit": "⚠️", "Over Budget": "🚨"}.get(info['status'], "📊")
                        
                        if info['status'] == 'Over Budget':
                            st.error(f"{status_emoji} **{dept}** - {info['leader']} | Over by ${abs(info['remaining_budget']):,.0f}")
                        else:
                            st.warning(f"{status_emoji} **{dept}** - {info['leader']} | ${info['remaining_budget']:,.0f} remaining")
                    
                    # Email configuration check
                    smtp_server = st.session_state.get('smtp_server', '')
                    smtp_user = st.session_state.get('smtp_user', '')
                    smtp_pass = st.session_state.get('smtp_pass', '')
                    email_config_complete = all([smtp_server, smtp_user, smtp_pass])
                    
                    st.markdown("**📧 Email Notifications:**")
                    
                    if email_config_complete:
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            test_mode = st.toggle("🧪 Test Mode", 
                                                value=True, 
                                                help="Send test email to Lulu only")

                        with col2:
                            if st.button("📧 Send Alerts", type="primary", use_container_width=True):
                                smtp_config = {
                                    'server': smtp_server,
                                    'port': st.session_state.get('smtp_port', 587),
                                    'user': smtp_user,
                                    'password': smtp_pass
                                }

                                # Filter for teams needing attention
                                alert_teams = {dept: budget_status[dept] for dept in attention_needed}

                                with st.spinner("Sending alerts..."):
                                    # Add period and test mode info to team data
                                    for dept, info in alert_teams.items():
                                        info['period_start'] = period['start'].strftime('%Y-%m-%d')
                                        info['period_end'] = period['end'].strftime('%Y-%m-%d')
                                        info['test_mode'] = test_mode

                                    try:
                                        result = send_budget_alert_emails_with_ui(
                                            alert_teams, 
                                            smtp_config, 
                                            test_mode=test_mode,
                                            delay_minutes=5,
                                            focus_on_travel=focus_on_travel
                                        )

                                        if result['success']:
                                            st.success(f"✅ **Alerts sent successfully!**")
                                            st.info(f"📧 {result['scheduled_count']} emails scheduled with {result['delay_minutes']} minute delay")
                                            # Store the result in session state so the cancel button persists
                                            st.session_state.email_result = result
                                        else:
                                            st.error(f"❌ Failed to send alerts: {result['message']}")
                                    except Exception as e:
                                        st.error(f"❌ Error sending alerts: {str(e)}")

                        # Cancel button - show if emails were successfully scheduled
                        if 'email_result' in st.session_state and st.session_state.email_result.get('success') and st.session_state.email_result.get('scheduled_count', 0) > 0:
                            st.markdown("---")

                            # Show current scheduled emails info using session state
                            with EMAIL_LOCK:
                                total_emails = len(st.session_state.SCHEDULED_EMAILS)
                                active_emails = sum(1 for email_data in st.session_state.SCHEDULED_EMAILS.values() 
                                                  if not email_data.get('cancelled', False) 
                                                  and not email_data.get('sent', False))

                            st.write(f"Current scheduled emails: {total_emails}")
                            st.write(f"Active emails pending: {active_emails}")
                            st.write(f"Dictionary ID: {id(st.session_state.SCHEDULED_EMAILS)}")  # Debug info

                            if active_emails > 0:
                                if st.button("❌ Cancel Scheduled Emails", type="secondary", 
                                            help="Cancel all scheduled emails before they are sent"):
                                    with st.spinner("Cancelling emails..."):
                                        try:
                                            # Show what we have before cancelling
                                            st.write(f"Debug: Found {total_emails} total emails in system")
                                            st.write(f"Debug: Found {active_emails} active emails to cancel")

                                            # Cancel all emails using session state
                                            cancelled_count = cancel_all_scheduled_emails()
                                            time.sleep(2)
                                            with EMAIL_LOCK:
                                                # Verify cancellation worked
                                                still_active = sum(1 for email_data in st.session_state.SCHEDULED_EMAILS.values() 
                                                                  if not email_data.get('cancelled', False) 
                                                                  and not email_data.get('sent', False)
                                                                  and not email_data.get('failed', False))

                                            st.write(f"📊 Verification: {still_active} emails still active after cancellation")

                                            # Provide feedback based on results
                                            if cancelled_count > 0:
                                                st.success(f"✅ Successfully cancelled {cancelled_count} scheduled emails")
                                                st.info("📧 Cancelled emails will not be sent when their scheduled time arrives")

                                                # Wait a moment for threads to process the cancellation
                                                time.sleep(1)

                                                # Show updated status
                                                with EMAIL_LOCK:
                                                    remaining_active = sum(1 for email_data in st.session_state.SCHEDULED_EMAILS.values() 
                                                                         if not email_data.get('cancelled', False) 
                                                                         and not email_data.get('sent', False))
                                                st.write(f"Remaining active scheduled emails: {remaining_active}")

                                            elif total_emails == 0:
                                                st.info("ℹ️ No emails found in the system to cancel")
                                            else:
                                                st.warning("⚠️ No emails were cancelled (they may have already been sent or cancelled)")

                                            time.sleep(2)    
                                            
                                            # Clear the session state so button disappears
                                            if 'email_result' in st.session_state:
                                                del st.session_state.email_result
                                            st.rerun()

                                            # Refresh the UI
                                            

                                        except Exception as e:
                                            st.error(f"❌ Error cancelling emails: {str(e)}")
                                            st.write(f"Exception details: {type(e).__name__}: {str(e)}")

                                            # Still clear the session state even if there's an error
                                            if 'email_result' in st.session_state:
                                                del st.session_state.email_result
                            else:
                                st.info("ℹ️ No active emails to cancel")

                            # Show email details for debugging
                            if total_emails > 0:
                                st.write("**Current Email Status:**")
                                with EMAIL_LOCK:
                                    for email_id, email_data in st.session_state.SCHEDULED_EMAILS.items():
                                        cancelled = email_data.get('cancelled', False)
                                        sent = email_data.get('sent', False)
                                        failed = email_data.get('failed', False)
                                        st.write(f"- {email_id}: cancelled={cancelled}, sent={sent}, failed={failed}")
                        
                    else:
                        st.info("💡 Configure email settings in the sidebar to send alerts")
                else:
                    st.success("✅ **All teams are within budget!** No alerts needed.")
            else:
                st.info("📊 No budget data available for the selected period")
    else:
        st.info("📋 Upload budget limits file and run analysis to view results")
        

with tab6:
    st.title("🤖 Personal Expense AI Detection")
    
    if st.session_state.df is None or st.session_state.policy is None:
        st.info("📋 Please upload expense data and configure policy settings in the **Data Upload** tab to begin validation.")
        st.stop()
    else:
        # Import should be at top of file, but showing here for completeness
        from Personal_Expense_AI import PersonalExpenseDetector, FixedAssetDetector

        # Initialize or load detector
        if 'expense_detector' not in st.session_state:
            model_dir = "trained_models"
            required_files = ["best_model.pkl", "scaler.pkl", "category_encoder.pkl", "nlp_model.pkl"]
            
            try:
                detector = PersonalExpenseDetector(auto_train=False)
                
                # Check if all required files exist
                file_status = {}
                for file in required_files:
                    file_path = os.path.join(model_dir, file)
                    file_status[file] = os.path.exists(file_path)
                
                if all(file_status.values()):
                    # Load all model components
                    with open(f"{model_dir}/best_model.pkl", 'rb') as f:
                        detector.best_model = pickle.load(f)
                    with open(f"{model_dir}/scaler.pkl", 'rb') as f:
                        detector.scaler = pickle.load(f)
                    with open(f"{model_dir}/category_encoder.pkl", 'rb') as f:
                        detector.category_encoder = pickle.load(f)
                    with open(f"{model_dir}/nlp_model.pkl", 'rb') as f:
                        detector.nlp_model = pickle.load(f)
                    
                    # Set flags
                    detector.is_ml_trained = True
                    detector.category_encoder_fitted = True
                    detector.best_model_name = "Loaded Model"
                    
                st.session_state.expense_detector = detector
                
            except Exception as e:
                # Fallback to basic detector
                st.session_state.expense_detector = PersonalExpenseDetector(auto_train=False)
        
        # Get detector from session state (this runs every time, not just on first load)
        detector = st.session_state.expense_detector
        
        # Model Status Display
        st.subheader("Detection System Status")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if detector.is_ml_trained:
                st.success("✅ ML Model: Trained and Active")
                if hasattr(detector, 'best_model_name') and detector.best_model_name:
                    st.info(f"**Model Type:** {detector.best_model_name}")
            else:
                st.warning("⚠️ ML Model: Not Available")
                st.caption("Using rule-based detection only")
        
        with col2:
            if hasattr(detector, 'scaler') and detector.scaler is not None:
                st.success("✅ Feature Scaler: Loaded")
            else:
                st.error("❌ Feature Scaler: Missing")
        
        with col3:
            if hasattr(detector, 'category_encoder') and detector.category_encoder is not None:
                st.success("✅ Category Encoder: Loaded")
            else:
                st.error("❌ Category Encoder: Missing")
        
        with col4:
            if hasattr(detector, 'nlp_model') and detector.nlp_model is not None:
                st.success("✅ NLP Model: Loaded")
            else:
                st.error("❌ NLP Model: Missing")
        
        # Training setup section for non-ML users
        if not detector.is_ml_trained:
            with st.expander("🤖 ML Detection"):
                st.markdown("""
                **Enhanced ML Features Include:**
                - 🎯 **Cost-sensitive learning** for optimal unauthorised transactions detection
                - 🔗 **Ensemble modeling** combining multiple ML algorithms for one optimal hybrid model
                - 📊 **Automatic hyperparameter tuning** with cross-validation
                - 🎚️ **Threshold optimization** to minimize business costs
                - 📈 **SHAP explanations** for model interpretability (SHAP helps explain why a machine learning model made a certain decision — like showing which parts of a receipt made the AI think an expense was personal)
                
                **If pre-trained model not loaded:**
                
                1. **Prepare Dataset:** Ensure `credit_card_transactions.csv` is in your project directory (https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
                2. **Run Training:** Use the training options below (NOTE: Training the models would take ~30 hours)
                3. **Restart Application:** The trained model will load automatically
                
                **Dataset Requirements:**
                - Columns: `amt`, `category`, `trans_date_trans_time`, `job`, `is_fraud`
                - Format: CSV with unauthorised transactions labels for supervised learning
                """)
                
                # Training options
                col1 = st.columns(1)[0]
                
                with col1:
                    st.markdown("**🚀 Training (ML + NLP)**")
                    if st.button("🔧 Train Models", type="primary", key="full_train"):
                        if not os.path.exists("credit_card_transactions.csv"):
                            st.error("❌ Dataset file 'credit_card_transactions.csv' not found!")
                            st.info("Please add the dataset file to your project directory first.")
                        else:
                            with st.spinner("Training complete system (ML + NLP)... This may take several minutes..."):
                                try:
                                    success = detector.pretrain_all_models()
                                    if success:
                                        st.success("✅ Complete system trained and saved successfully!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Complete training failed")
                                except Exception as e:
                                    st.error(f"Complete training error: {e}")

        # Display training results if available
        if detector.is_ml_trained and hasattr(detector, 'training_stats') and detector.training_stats:
            st.divider()
            with st.expander("📊 View Training Results", expanded=False):
                detector.display_training_results()
                
                # Cost analysis if available
                if hasattr(detector, 'cv_results') and detector.best_model_name in detector.cv_results:
                    best_results = detector.cv_results[detector.best_model_name]
                    if 'cost_ratio_used' in best_results:
                        st.subheader("💰 Cost-Sensitive Analysis")
                        st.caption("💡 Cost-sensitive = System considers business impact of missing unauthorised transactions vs false alarms")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cost Ratio Used", f"1:{best_results['cost_ratio_used']}")
                            st.caption("💡 Higher ratio = More sensitive to unauthorised transactions detection")
                        with col2:
                            if 'optimal_threshold' in best_results:
                                st.metric("Optimal Threshold", f"{best_results['optimal_threshold']:.3f}")
                                st.caption("💡 Threshold = Decision boundary (0.0-1.0)")
                        with col3:
                            if 'total_business_cost' in best_results:
                                st.metric("Test Set Business Cost", f"{best_results['total_business_cost']:.0f}")
                                st.caption("💡 Lower cost = Better business outcome")

        # Main Analysis Section
        st.divider()
        st.subheader("📊 Transaction Analysis")

        # File upload for analysis
        st.markdown("""
        **📋 Data Requirements:**
        Your spend management report should include these columns:
        - **Memo**: Transaction description/details
        - **Amount (by category)**: Transaction amount in dollars
        - **Title**: Employee job title (from Rippling: Employment → Employment Details)
        - **Category name**: Expense category classification
        - **Purchase date**: When the transaction occurred
        
        """)
        
        transaction_file = st.file_uploader(
            "Upload Corporate Transactions", 
            type=['csv', 'xlsx'], 
            key="corporate_transactions",
            help="Upload CSV or Excel file with corporate transaction data"
        )

        # Initialize category mapping in session state
        if 'category_mapping' not in st.session_state:
            # Default mapping suggestions
            st.session_state.category_mapping = {
                'Software': 'misc_net',
                'Miscellaneous': 'misc_pos',
                'Biz Travel: Meals': 'food_dining',
                'Team building': 'entertainment',
                'Advertising & Marketing': 'misc_pos',
                'Computer Equipment': 'shopping_net',
                'Biz Meals, Gifts & Entertainment': 'entertainment',
                'Biz Travel & Client Meeting: Transport': 'gas_transport',
                'Biz Travel: Hotels': 'travel',
                'Telecommunications': 'misc_net',
                'Network Services': 'misc_net',
                'Professional Services': 'misc_pos',
                'Biz Travel: Airfare': 'travel',
                'Global Summit': 'travel',
                'Insurance': 'misc_pos',
                'Office Supplies': 'shopping_pos',
                'Legal': 'misc_pos',
                'Utilities': 'misc_pos',
                'Medical': 'health_fitness',
                'Other benefits': 'misc_pos',
                'Fees, Licenses, & Taxes': 'misc_pos',
                'Charity': 'misc_pos',
                'Training Events': 'misc_pos',
                'Travel - client facing': 'travel',
                'Contractor': 'misc_pos'
            }

        # Category Mapping Section
        if transaction_file:
            try:
                # Read uploaded file to get categories
                file_extension = transaction_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    temp_df = pd.read_csv(transaction_file)
                elif file_extension in ['xlsx', 'xls']:
                    temp_df = pd.read_excel(transaction_file)
                else:
                    st.error("❌ Unsupported file format. Please upload CSV or XLSX files.")
                    st.stop()

                # Check if Category name column exists
                if 'Category name' in temp_df.columns:
                    unique_categories = temp_df['Category name'].unique()
                    unique_categories = [cat for cat in unique_categories if pd.notna(cat)]
                    
                    st.divider()
                    st.subheader("🏷️ Category Mapping Configuration")
                    
                    st.markdown("""
                    **Map your expense categories to standardized categories for better detection accuracy.**
                    
                    **Available Target Categories:**
                    - `misc_net` - Miscellaneous online services
                    - `grocery_pos` - Grocery stores (in-person)
                    - `entertainment` - Entertainment & recreation
                    - `gas_transport` - Transportation & fuel
                    - `misc_pos` - Miscellaneous purchases (in-person)
                    - `grocery_net` - Grocery delivery/online
                    - `shopping_net` - Online shopping
                    - `shopping_pos` - Retail shopping (in-person)
                    - `food_dining` - Restaurants & dining
                    - `personal_care` - Beauty & personal care
                    - `health_fitness` - Health & fitness
                    - `travel` - Travel expenses
                    - `kids_pets` - Children & pets
                    - `home` - Home & household
                    """)
                    
                    with st.expander("📝 Configure Category Mappings", expanded=True):
                        # Create two columns for better layout
                        col1, col2 = st.columns(2)
                        
                        # Available target categories
                        target_categories = [
                            'misc_net', 'grocery_pos', 'entertainment', 'gas_transport', 
                            'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos',
                            'food_dining', 'personal_care', 'health_fitness', 'travel',
                            'kids_pets', 'home'
                        ]
                        
                        # Display mapping controls
                        updated_mapping = {}
                        
                        for i, category in enumerate(sorted(unique_categories)):
                            # Determine which column to use
                            if i % 2 == 0:
                                current_col = col1
                            else:
                                current_col = col2
                                
                            with current_col:
                                # Get current mapping or default
                                current_mapping = st.session_state.category_mapping.get(category, 'misc_pos')
                                
                                # Create selectbox for mapping
                                mapped_category = st.selectbox(
                                    f"**{category}**",
                                    options=target_categories,
                                    index=target_categories.index(current_mapping) if current_mapping in target_categories else target_categories.index('misc_pos'),
                                    key=f"mapping_{category}",
                                    help=f"Map '{category}' to a standardized category"
                                )
                                updated_mapping[category] = mapped_category
                        
                        # Update session state
                        st.session_state.category_mapping = updated_mapping
                        
                        # Display current mappings summary
                        st.markdown("**📋 Current Mappings Summary:**")
                        mapping_df = pd.DataFrame([
                            {'Source Category': k, 'Target Category': v} 
                            for k, v in st.session_state.category_mapping.items()
                        ])
                        st.dataframe(mapping_df, use_container_width=True, height=200)
                        
                        # Save/Load mapping presets
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("💾 Save Mapping Preset", key="save_mapping"):
                                # Convert mapping to JSON for download
                                import json
                                mapping_json = json.dumps(st.session_state.category_mapping, indent=2)
                                st.download_button(
                                    label="📥 Download Mapping File",
                                    data=mapping_json,
                                    file_name=f"category_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        
                        with col2:
                            uploaded_mapping = st.file_uploader(
                                "📤 Load Mapping Preset",
                                type=['json'],
                                key="load_mapping",
                                help="Upload a previously saved mapping configuration"
                            )
                            if uploaded_mapping:
                                try:
                                    import json
                                    loaded_mapping = json.load(uploaded_mapping)
                                    st.session_state.category_mapping.update(loaded_mapping)
                                    st.success("✅ Mapping preset loaded successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error loading mapping: {e}")
                        
                        with col3:
                            if st.button("🔄 Reset to Defaults", key="reset_mapping"):
                                # Reset to default mappings
                                st.session_state.category_mapping = {
                                    'Software': 'misc_net',
                                    'Miscellaneous': 'misc_pos',
                                    'Biz Travel: Meals': 'food_dining',
                                    'Team building': 'entertainment',
                                    'Advertising & Marketing': 'misc_pos',
                                    'Computer Equipment': 'shopping_net',
                                    'Biz Meals, Gifts & Entertainment': 'entertainment',
                                    'Biz Travel & Client Meeting: Transport': 'gas_transport',
                                    'Biz Travel: Hotels': 'travel',
                                    'Telecommunications': 'misc_net',
                                    'Network Services': 'misc_net',
                                    'Professional Services': 'misc_pos',
                                    'Biz Travel: Airfare': 'travel',
                                    'Global Summit': 'travel',
                                    'Insurance': 'misc_pos',
                                    'Office Supplies': 'shopping_pos',
                                    'Legal': 'misc_pos',
                                    'Utilities': 'misc_pos',
                                    'Medical': 'health_fitness',
                                    'Other benefits': 'misc_pos',
                                    'Fees, Licenses, & Taxes': 'misc_pos',
                                    'Charity': 'misc_pos',
                                    'Training Events': 'misc_pos',
                                    'Travel - client facing': 'travel',
                                    'Contractor': 'misc_pos'
                                }
                                st.success("✅ Mapping reset to defaults!")
                                st.rerun()
                                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

        # Analysis configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            threshold = st.slider(
                "Personal Expense Threshold", 
                min_value=0, max_value=100, value=30,
                help="Transactions scoring above this threshold will be flagged as personal"
            )
            st.caption("💡 **Threshold Guide:**")
            st.caption("• **0** = Flag everything (too strict) ")  
            st.caption("• **50** = Balanced approach")
            st.caption("• **100** = Never flag anything (too lenient)")
            st.caption("🎯 **For unauthorised transactions detection: ~30 is typically optimal**")
        with col2:
            if detector.is_ml_trained:
                analysis_mode = st.selectbox(
                    "Detection Mode", 
                    ["Combined (Rules + ML)", "Rules Only", "ML Only"],
                    help="Choose detection method"
                )
            else:
                analysis_mode = st.selectbox(
                    "Detection Mode", 
                    ["Rules Only"],
                    help="ML mode requires trained model"
                )
                st.info("💡 ML modes available after model training")
        with col3:
            show_details = st.checkbox(
                "Show Detection Details", 
                value=True,
                help="Include confidence factors and reasoning"
            )

        # Cost sensitivity settings for ML predictions
        if detector.is_ml_trained:
            with st.expander("⚙️ Advanced ML Settings"):
                st.caption("💡 These settings control how the AI makes decisions")
                cost_ratio = st.slider(
                    "Unauthorised transactions Detection Cost Ratio (1:X)", 
                    min_value=1, max_value=20, value=10,
                    help="Cost of missing fraud vs false alarm (higher = more sensitive)"
                )
                st.info(f"Current setting: Missing unauthorised transactions costs {cost_ratio}x more than false alarms")
                st.caption("💡 **Cost Ratio Guide:** Higher values make the system more likely to flag suspicious transactions")

        # Analysis execution
        if transaction_file and st.button("🔍 Analyze Transactions", type="primary"):
            st.info("⏱️ **Processing Time:** Analysis typically takes 5-15 minutes depending on dataset size and complexity.")
            try:
                # Read uploaded file
                file_extension = transaction_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    transactions_df = pd.read_csv(transaction_file)
                elif file_extension in ['xlsx', 'xls']:
                    transactions_df = pd.read_excel(transaction_file)
                else:
                    st.error("❌ Unsupported file format. Please upload CSV or XLSX files.")
                    st.stop()

                # Validate required columns
                required_cols = ['Vendor name', 'Amount (by category)', 'Purchase date']
                missing_cols = [col for col in required_cols if col not in transactions_df.columns]

                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info("**Required columns:**")
                    st.write("- **Vendor name**: Merchant/vendor name")
                    st.write("- **Amount (by category)**: Transaction amount")
                    st.write("- **Purchase date**: Transaction date")
                    st.write("- **Description** (optional): Additional details")
                else:
                    # Apply category mapping if Category name column exists
                    if 'Category name' in transactions_df.columns:
                        transactions_df['mapped_category'] = transactions_df['Category name'].map(
                            st.session_state.category_mapping
                        ).fillna('misc_pos')
                        st.info(f"✅ Applied category mapping to {len(transactions_df)} transactions")
                    
                    # Map columns to expected format
                    column_mapping = {
                        'Category name': 'category',
                        'Amount (by category)': 'amt', 
                        'Title': 'job',
                        'Purchase date': 'trans_date_trans_time'
                    }
                    
                    # Apply column mapping
                    for old_col, new_col in column_mapping.items():
                        if old_col in transactions_df.columns:
                            transactions_df[new_col] = transactions_df[old_col]
                    
                    # Use mapped category if available
                    if 'mapped_category' in transactions_df.columns:
                        transactions_df['category'] = transactions_df['mapped_category']
                    
                    # Run analysis
                    with st.spinner(f"Analyzing {len(transactions_df)} transactions..."):
                        results_df = detector.analyze_transactions(
                            transactions_df=transactions_df,
                            threshold=threshold
                        )

                    # Store results for export
                    st.session_state.analysis_results = results_df

                    # Display comprehensive results
                    detector.display_analysis_results(results_df, threshold)

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Please check your file format and data structure")

        # Individual Transaction Tester
        st.divider()
        st.subheader("🧪 Individual Transaction Tester")

        col1, col2 = st.columns(2)
        with col1:
            test_merchant = st.text_input(
                "Merchant Name", 
                placeholder="e.g., Zara, Netflix, Office Depot",
                help="Enter the merchant or vendor name"
            )
            test_amount = st.number_input(
                "Amount ($)", 
                value=100.0, 
                step=10.0,
                min_value=0.01,
                help="Transaction amount in dollars"
            )

        with col2:
            test_description = st.text_input(
                "Description (optional)", 
                placeholder="e.g., clothing purchase, team lunch",
                help="Additional transaction details"
            )
            test_date = st.date_input(
                "Transaction Date", 
                value=datetime.now().date(),
                help="Date of the transaction"
            )

        if st.button("🎯 Test Transaction", type="secondary"):
            if test_merchant.strip():
                # Create test transaction
                test_transaction = {
                    'merchant': test_merchant,
                    'Vendor name': test_merchant,  # Alternative naming
                    'amount': test_amount,
                    'Amount (by category)': test_amount,  # Alternative naming
                    'description': test_description,
                    'Description': test_description,  # Alternative naming
                    'date': test_date.isoformat(),
                    'Purchase date': test_date.isoformat(),  # Alternative naming
                    'category': 'general'  # Default category
                }

                # Get predictions
                prediction = detector.predict_personal_expense(test_transaction)
                
                # Display results
                st.subheader("🎯 Detection Results")

                # Score metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rule-based Score", f"{prediction['rule_score']:.1f}/100")
                    st.caption("💡 Based on predefined patterns")
                with col2:
                    if detector.is_ml_trained:
                        st.metric("ML Score", f"{prediction['ml_score']:.1f}/100")
                        st.caption("💡 AI learned from data patterns")
                    else:
                        st.metric("ML Score", "Not Available")
                        st.caption("💡 AI learned from data patterns")
                with col3:
                    st.metric("Final Score", f"{prediction['final_score']:.1f}/100")
                    st.caption("💡 Combined intelligent decision")
                with col4:
                    # Add cost-sensitive prediction if ML is available
                    if detector.is_ml_trained:
                        cost_pred = detector.predict_with_cost_optimization(test_transaction, cost_ratio if 'cost_ratio' in locals() else 10)
                        threshold_used = cost_pred.get('threshold_used', 0.5)
                        st.metric("Unauthorised transactions Threshold", f"{threshold_used:.3f}")
                        st.caption("💡 Decision boundary (0.0-1.0)")

                # Risk assessment
                final_score = prediction['final_score']
                if final_score >= 70:
                    st.error("🚨 **HIGH RISK** - Likely personal expense")
                elif final_score >= 40:
                    st.warning("⚠️ **MEDIUM RISK** - Requires manual review")
                else:
                    st.success("✅ **LOW RISK** - Likely legitimate business expense")

                # Enhanced detection reasoning
                if show_details:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction['confidence_factors']:
                            st.info("**Rule-based Detection Factors:**")
                            for factor in prediction['confidence_factors'][:5]:  # Show top 5 factors
                                st.write(f"• {factor}")
                    
                    with col2:
                        if detector.is_ml_trained and 'cost_pred' in locals():
                            st.info("**ML Analysis:**")
                            st.write(f"• Unauthorised transactions Probability: {cost_pred.get('fraud_probability', 0):.1f}%")
                            st.write(f"• Confidence Level: {cost_pred.get('confidence', 'Unknown')}")
                            st.write(f"• Threshold Used: {cost_pred.get('threshold_used', 0.5):.3f}")

                # Detailed breakdown if available
                if 'detailed_breakdown' in prediction and prediction['detailed_breakdown']:
                    with st.expander("🔍 Detailed Analysis Breakdown"):
                        for category, details in prediction['detailed_breakdown'].items():
                            st.write(f"**{category.title()}:**")
                            if isinstance(details, dict):
                                for key, value in details.items():
                                    st.write(f"  - {key}: {value}")
                            else:
                                st.write(f"  - {details}")

            else:
                st.warning("⚠️ Please enter a merchant name to test")

        # Model interpretability section
        if detector.is_ml_trained and hasattr(detector, 'explain_with_shap'):
            st.divider()
            with st.expander("🔍 Model Interpretability (SHAP Analysis)"):
                st.markdown("""
                **SHAP (SHapley Additive exPlanations)** provides insights into how the ML model makes decisions.
                This analysis shows which features are most important for unauthorised transactions detection.
                """)
                
                if st.button("Generate SHAP Analysis", key="shap_analysis"):
                    with st.spinner("Generating SHAP explanations..."):
                        try:
                            # This would require training data to be available
                            st.info("SHAP analysis requires training data. Feature importance is available in training results above.")
                        except Exception as e:
                            st.error(f"SHAP analysis error: {e}")

        # Information and Help Section
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Detection Methods")
            st.markdown("""
            **Active Detection Techniques:**
            - 🧠 **Ensemble Model** Multiple ML algorithm trained on 1M+ data points, best models combined on a weighted basis to form one optimal hybrid model
            - 🏪 **Fuzzy merchant matching** with confidence scoring
            - 🔤 **Multi-level keyword analysis** (high/medium confidence)
            - 📊 **Semantic NLP analysis** with embeddings
            - ⏰ **Time-based indicators** (weekends, off-hours) (if dataset permits)
            - 💰 **Statistical amount analysis** with z-scores
            - 🎯 **Pattern recognition** with regex matching
            """)
            if detector.is_ml_trained:
                st.markdown("""
                - 🤖 **Cost-sensitive ML predictions**
                - 🔗 **Ensemble modeling** (Random Forest, XGBoost, etc.)
                - 🎚️ **Optimized thresholds** for business cost minimization
                """)

        with col2:
            st.subheader("🏷️ Monitoring Categories")
            st.markdown("""
            **Personal Expense Categories:**
            - 👕 **Clothing & Fashion** (Zara, H&M, etc.)
            - 🎬 **Entertainment & Streaming** (Netflix, Spotify, Cinema etc.)
            - 💄 **Personal Care & Beauty** (Sephora, salons)
            - 🏋️ **Fitness & Health** (gyms, supplements)
            - 🛒 **Personal Shopping** (Amazon personal items)
            - 💎 **Luxury Goods** (jewelry, high-end brands)
            - 🍔 **Personal Dining** (individual meals, fast food)
            - 🚗 **Personal Transportation** (rideshares, personal fuel)
            """)

        # Enhanced usage guide
        with st.expander("❓ Advanced Usage Guide & Tips"):
            st.markdown("""
            ### How to Use This Tool
            
            **1. Upload Transaction Data**
            - Format: CSV or Excel file
            - Required columns: `Vendor name`, `Amount (by category)`, `Purchase date`
            
            **2. Configure Detection**
            - Set threshold based on your risk tolerance
            - Choose detection mode (Rules, ML, or Combined)
            - Adjust cost sensitivity for ML predictions
            - Enable details to see reasoning behind flags
            
            **3. Review Results**
            - Check flagged transactions manually
            - Use score distributions to understand patterns
            - Review SHAP explanations for ML decisions
            - Export results for further analysis
            
            ### Scoring System
            - **0-39:** Low risk (likely business expense)
            - **40-69:** Medium risk (requires review)
            - **70-100:** High risk (likely personal expense)
            
            ### Cost-Sensitive ML Features
            - **Threshold Optimization:** Automatically finds the best threshold to minimize business costs
            - **Cost Ratio:** Configure how much more expensive missing unauthorised transactions is vs false alarms
            - **Ensemble Voting:** Combines multiple ML models for better accuracy
            
            ### Best Practices for System
            ✅ **Do:**
            - Train with your organization's historical data for best results
            - Use cost-sensitive settings appropriate to your business
            - Combine rule-based and ML approaches for optimal performance
            - Review SHAP explanations to understand ML decisions
            - Adjust thresholds based on audit findings
            
            ❌ **Don't:**
            - Rely solely on automated detection for final decisions
            - Use overly strict cost ratios without business justification
            - Ignore the semantic analysis results from NLP
            - Skip manual review of medium-risk transactions
            """)
        
        # Export functionality
        if 'analysis_results' in st.session_state:
            st.divider()
            st.subheader("📥 Export Results")
            
            results_df = st.session_state.analysis_results
            
            col1, col2 = st.columns(2)
            with col1:
                # Full results export
                full_csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Full Analysis",
                    data=full_csv,
                    file_name=f"expense_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Flagged only export
                if 'is_flagged' in results_df.columns:
                    flagged_df = results_df[results_df['is_flagged']]
                    if not flagged_df.empty:
                        flagged_csv = flagged_df.to_csv(index=False)
                        st.download_button(
                            label="🚨 Download Flagged Only",
                            data=flagged_csv,
                            file_name=f"flagged_expenses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

        # Performance monitoring section
        if detector.is_ml_trained and hasattr(detector, 'training_stats'):
            st.divider()
            with st.expander("📈 Performance Monitoring"):
                st.markdown("**Model Performance Tracking:**")
                
                # Display key metrics
                if detector.training_stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Accuracy", 
                                f"{detector.training_stats.get('best_test_accuracy', 0):.3f}")
                    with col2:
                        st.metric("Precision", 
                                f"{detector.training_stats.get('best_test_precision', 0):.3f}")
                    with col3:
                        st.metric("Recall", 
                                f"{detector.training_stats.get('best_test_recall', 0):.3f}")
                
                st.info("💡 **Tip:** Monitor these metrics over time. Declining performance may indicate need for retraining with fresh data.")
                
with tab7:
    st.header("🏢 Fixed Asset Detector")
    st.markdown("**Identify potential fixed asset transactions using NLP and rule-based analysis**")
    
    # Initialize session state for fixed asset detector
    if 'fixed_asset_detector' not in st.session_state:
        st.session_state.fixed_asset_detector = None
    if 'fa_analysis_results' not in st.session_state:
        st.session_state.fa_analysis_results = None
    if 'fa_uploaded_data' not in st.session_state:
        st.session_state.fa_uploaded_data = None
    
    # Settings section with improved layout
    st.subheader("🔧 Fixed Asset Settings")

    # Create columns for better layout
    settings_col1, settings_col2, settings_col3, settings_col4 = st.columns([2, 2, 2, 2])

    with settings_col1:
        # Detection threshold with inline guide
        fa_threshold = st.slider(
            "Detection Threshold",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Minimum score required to classify as fixed asset (higher = more strict)"
        )
        
        # Threshold guide right below the slider
        st.markdown("""
        <div class="threshold-guide">
        <strong>🎚️ Threshold Guide:</strong><br>
        <small>
        • 80-100: Very strict (high confidence only)<br>
        • 60-79: Strict (high-medium confidence)<br>
        • 40-59: Balanced (recommended)<br>
        • 20-39: Lenient (includes low confidence)<br>
        • 1-19: Very lenient (very low confidence)
        </small>
        </div>
        """, unsafe_allow_html=True)

    with settings_col2:
        # Category filter
        fa_category_filter = st.selectbox(
            "Category Filter",
            options=['all', 'electronics_it', 'furniture_office', 'machinery_equipment', 
                    'vehicles_transport', 'building_improvements', 'software_licenses'],
            format_func=lambda x: {
                'all': '📊 All Categories',
                'electronics_it': '💻 Electronics & IT',
                'furniture_office': '🪑 Office Furniture',
                'machinery_equipment': '⚙️ Machinery & Equipment',
                'vehicles_transport': '🚛 Vehicles & Transport',
                'building_improvements': '🏗️ Building Improvements',
                'software_licenses': '💿 Software Licenses'
            }.get(x, x)
        )
        
        # Detection categories right beside the filter
        st.markdown("""
        <div class="category-box">
        <strong>🎯 Detection Categories:</strong><br>
        <small>
        💻 Electronics & IT Equipment<br>
        🪑 Office Furniture & Storage<br>
        ⚙️ Machinery & Equipment<br>
        🚛 Vehicles & Transport<br>
        🏗️ Building Improvements<br>
        💿 Software Licenses
        </small>
        </div>
        """, unsafe_allow_html=True)

    with settings_col3:
        # Confidence level filter
        fa_confidence_filter = st.selectbox(
            "Confidence Level",
            options=['all', 'Very High', 'High', 'Medium', 'Low', 'Very Low'],
            format_func=lambda x: f"🎯 {x}" if x != 'all' else "📈 All Confidence Levels"
        )

    with settings_col4:
        # Amount range filter
        st.markdown("**💰 Amount Range**")
        fa_min_amount = st.number_input(
            "Min Amount ($)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="fa_min_amount"
        )
        fa_max_amount = st.number_input(
            "Max Amount ($)",
            min_value=0.0,
            value=10000.0,
            step=50.0,
            key="fa_max_amount"
        )

    st.markdown("---")
    
    # File upload section
    st.subheader("📤 Upload Transaction Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file with transaction data",
        type=['csv', 'xlsx', 'xls'],
        key="fa_file_uploader",
        help="Upload CSV or Excel file with columns: vendor/merchant, description, amount, date"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel files
                df = pd.read_excel(uploaded_file)

            st.session_state.fa_uploaded_data = df
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} transactions")
            
            column_aliases = {
                'Amount (by category)': 'amount',
                'Vendor name': 'vendor',
                'Memo': 'description',
                'Purchase Date': 'date'
            }

            # Rename columns based on alias mapping
            df = df.rename(columns={col: column_aliases[col] for col in df.columns if col in column_aliases})
            
            # Display data preview in expander
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show column mapping suggestions
                st.markdown("**Column Mapping:**")
                cols = df.columns.tolist()
                
                col_map_1, col_map_2, col_map_3, col_map_4 = st.columns(4)
                with col_map_1:
                    vendor_col = st.selectbox("Vendor/Merchant Column", options=cols, 
                                            index=next((i for i, col in enumerate(cols) if 'vendor' in col.lower() or 'merchant' in col.lower()), 0))
                with col_map_2:
                    desc_col = st.selectbox("Description Column", options=cols,
                                          index=next((i for i, col in enumerate(cols) if 'description' in col.lower() or 'memo' in col.lower()), 0))
                with col_map_3:
                    amount_col = st.selectbox("Amount Column", options=cols,
                                            index=next((i for i, col in enumerate(cols) if 'amount' in col.lower()), 0))
                with col_map_4:
                    date_col = st.selectbox("Date Column", options=cols,
                                          index=next((i for i, col in enumerate(cols) if 'date' in col.lower()), 0))
            
            # Analysis button
            if st.button("🔍 Analyze Fixed Assets", type="primary", use_container_width=True):
                if st.session_state.fixed_asset_detector is None:
                    st.session_state.fixed_asset_detector = FixedAssetDetector()
                
                with st.spinner("🔄 Analyzing transactions for fixed assets..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress updates
                    for i in range(101):
                        progress_bar.progress(i)
                        if i < 20:
                            status_text.text("📝 Pre-processing transaction data...")
                        elif i < 60:
                            status_text.text("🔍 Analyzing vendor patterns...")
                        elif i < 80:
                            status_text.text("💰 Evaluating amounts and thresholds...")
                        elif i < 95:
                            status_text.text("📊 Calculating confidence scores...")
                        else:
                            status_text.text("✅ Finalizing results...")
                        time.sleep(0.02)  # Simulate processing time
                    
                    # Perform actual analysis
                    results_df = st.session_state.fixed_asset_detector.analyze_transactions(df, threshold=fa_threshold)
                    st.session_state.fa_analysis_results = results_df
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"🎉 Analysis complete! Processed {len(results_df)} transactions")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Results section
    if st.session_state.fa_analysis_results is not None:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        results_df = st.session_state.fa_analysis_results
        
        # Apply filters
        filtered_df = results_df.copy()
        
        # Category filter
        if fa_category_filter != 'all':
            filtered_df = filtered_df[filtered_df['category'] == fa_category_filter]
        
        # Confidence filter
        if fa_confidence_filter != 'all':
            filtered_df = filtered_df[filtered_df['confidence_level'] == fa_confidence_filter]
        
        # Amount range filter
        if 'amount' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['amount'] >= fa_min_amount) & 
                (filtered_df['amount'] <= fa_max_amount)
            ]
        
        # Summary metrics with improved styling
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_transactions = len(filtered_df)
        fixed_assets = len(filtered_df[filtered_df['is_fixed_asset'] == True])
        avg_score = filtered_df['score'].mean() if len(filtered_df) > 0 else 0
        total_value = filtered_df[filtered_df['is_fixed_asset'] == True]['amount'].sum() if 'amount' in filtered_df.columns else 0
        high_confidence = len(filtered_df[filtered_df['confidence_level'].isin(['Very High', 'High'])])
        
        with col1:
            st.metric("📊 Total Transactions", f"{total_transactions:,}")
        with col2:
            st.metric("🏢 Fixed Assets Detected", f"{fixed_assets:,}", 
                     delta=f"{(fixed_assets/total_transactions*100):.1f}%" if total_transactions > 0 else "0%")
        with col3:
            st.metric("🎯 Average Score", f"{avg_score:.1f}")
        with col4:
            st.metric("💰 Total Asset Value", f"${total_value:,.0f}" if total_value > 0 else "$0")
        with col5:
            st.metric("⭐ High Confidence", f"{high_confidence:,}")
        
        # Tabs for different views (consolidated to 3 tabs)
        tab_analytics, tab_results, tab_export, tab_email = st.tabs([
            "📊 Analytics Dashboard", "📋 All Results", "💾 Export", "📧 Email"
        ])
        
        with tab_analytics:
            if len(filtered_df) > 0:
                # Enhanced visualizations with dark theme
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**🎯 Confidence Distribution**")
                    confidence_counts = filtered_df['confidence_level'].value_counts()

                    # Enhanced confidence colors with better contrast
                    confidence_colors = {
                        'Very High': '#00D4AA',    # Bright teal
                        'High': '#715DEC',        # Purple
                        'Medium': '#FF6B35',      # Orange-red
                        'Low': '#FFD23F',         # Yellow
                        'Very Low': '#FF4757'     # Red
                    }

                    # Create donut chart instead of pie for better readability
                    fig_confidence = px.pie(
                        values=confidence_counts.values,
                        names=confidence_counts.index,
                        color=confidence_counts.index,
                        color_discrete_map=confidence_colors,
                        height=350,
                        hole=0.4  # Creates donut chart
                    )

                    # Add percentage annotations inside the donut
                    total = sum(confidence_counts.values)
                    fig_confidence.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont_size=11,
                        textfont_color='white',
                        textfont_family="Arial Black",
                        pull=[0.1 if x == confidence_counts.max() else 0.05 for x in confidence_counts.values]  # Pull out largest slice
                    )

                    fig_confidence.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            x=1.02,
                            y=0.5,
                            font=dict(color='white', size=10)
                        ),
                        margin=dict(t=30, b=30, l=20, r=100),
                        font=dict(color='white'),
                        paper_bgcolor='#1E1E1E',  # Dark background
                        plot_bgcolor='#1E1E1E',
                        title=dict(
                            text=f"<b>Total Items: {total}</b>",
                            x=0.5,
                            y=0.02,
                            font=dict(size=12, color='white')
                        )
                    )
                    st.plotly_chart(fig_confidence, use_container_width=True)

                with col2:
                    st.markdown("**📊 Score Distribution**")

                    # Create bins for better visualization
                    score_bins = pd.cut(filtered_df['score'], bins=10, precision=1)
                    score_counts = score_bins.value_counts().sort_index()

                    # Convert interval index to string for better display
                    bin_labels = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in score_counts.index]

                    fig_scores = px.bar(
                        x=bin_labels,
                        y=score_counts.values,
                        color=score_counts.values,
                        color_continuous_scale=['#FF4757', '#FFD23F', '#00D4AA'],
                        height=350,
                        labels={'x': 'Score Range', 'y': 'Count', 'color': 'Count'}
                    )

                    # Add value labels on bars
                    fig_scores.update_traces(
                        texttemplate='%{y}',
                        textposition='outside',
                        textfont=dict(color='white', size=10)
                    )

                    fig_scores.update_layout(
                        xaxis_title="Score Range",
                        yaxis_title="Count",
                        showlegend=False,
                        margin=dict(t=20, b=60, l=40, r=20),
                        font=dict(color='white'),
                        paper_bgcolor='#1E1E1E',
                        plot_bgcolor='#1E1E1E',
                        xaxis=dict(
                            tickangle=-45,
                            gridcolor='#444',
                            tickfont=dict(size=9)
                        ),
                        yaxis=dict(gridcolor='#444'),
                        title=dict(
                            text=f"<b>Avg Score: {filtered_df['score'].mean():.1f}</b>",
                            x=0.5,
                            y=0.95,
                            font=dict(size=12, color='white')
                        )
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)

                with col3:
                    st.markdown("**💰 Value Distribution**")
                    if 'amount' in filtered_df.columns and len(filtered_df[filtered_df['is_fixed_asset'] == True]) > 0:
                        fixed_assets_df = filtered_df[filtered_df['is_fixed_asset'] == True]

                        # Create more granular value ranges
                        bins = [0, 100, 500, 1000, 2500, 5000, 10000, float('inf')]
                        labels = ['<$100', '$100-500', '$500-1K', '$1K-2.5K', '$2.5K-5K', '$5K-10K', '>$10K']

                        fixed_assets_df = fixed_assets_df.copy()
                        fixed_assets_df['value_range'] = pd.cut(
                            fixed_assets_df['amount'], 
                            bins=bins,
                            labels=labels
                        )

                        value_counts = fixed_assets_df['value_range'].value_counts()

                        # Calculate total value for each range
                        value_sums = fixed_assets_df.groupby('value_range')['amount'].sum()

                        fig_values = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            color=value_sums[value_counts.index],
                            color_continuous_scale=['#FF4757', '#FFD23F', '#715DEC', '#00D4AA'],
                            height=350,
                            labels={'x': 'Value Range', 'y': 'Count', 'color': 'Total Value ($)'}
                        )

                        # Add count labels on bars
                        fig_values.update_traces(
                            texttemplate='%{y}',
                            textposition='outside',
                            textfont=dict(color='white', size=10)
                        )

                        fig_values.update_layout(
                            xaxis_title="Value Range",
                            yaxis_title="Asset Count",
                            margin=dict(t=20, b=60, l=40, r=20),
                            font=dict(color='white'),
                            paper_bgcolor='#1E1E1E',
                            plot_bgcolor='#1E1E1E',
                            xaxis=dict(
                                tickangle=-45,
                                gridcolor='#444',
                                tickfont=dict(size=9)
                            ),
                            yaxis=dict(gridcolor='#444'),
                            coloraxis_colorbar=dict(
                                title=dict(
                                    text="Total Value ($)",
                                    font=dict(color='white')
                                ),
                                tickfont=dict(color='white')
                            ),
                            title=dict(
                                text=f"<b>Total Assets: {len(fixed_assets_df)}</b>",
                                x=0.5,
                                y=0.95,
                                font=dict(size=12, color='white')
                            )
                        )
                        st.plotly_chart(fig_values, use_container_width=True)

                # Score split analysis with enhanced styling
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📈 Score Split by Fixed Asset Status**")

                    # Add statistical annotations
                    fixed_scores = filtered_df[filtered_df['is_fixed_asset'] == True]['score']
                    non_fixed_scores = filtered_df[filtered_df['is_fixed_asset'] == False]['score']

                    fig_score_split = px.violin(
                        filtered_df,
                        x='is_fixed_asset',
                        y='score',
                        color='is_fixed_asset',
                        color_discrete_map={True: '#00D4AA', False: '#FF6B35'},
                        height=380,
                        box=True  # Show box plot inside violin
                    )

                    fig_score_split.update_layout(
                        xaxis_title="Is Fixed Asset",
                        yaxis_title="Score",
                        showlegend=False,
                        margin=dict(t=40, b=40, l=40, r=20),
                        font=dict(color='white'),
                        paper_bgcolor='#1E1E1E',
                        plot_bgcolor='#1E1E1E',
                        xaxis=dict(
                            ticktext=['No', 'Yes'], 
                            tickvals=[False, True],
                            gridcolor='#444'
                        ),
                        yaxis=dict(gridcolor='#444'),
                        title=dict(
                            text=f"<b>Fixed: μ={fixed_scores.mean():.1f} | Non-Fixed: μ={non_fixed_scores.mean():.1f}</b>",
                            x=0.5,
                            y=0.95,
                            font=dict(size=11, color='white')
                        )
                    )
                    st.plotly_chart(fig_score_split, use_container_width=True)

                with col2:
                    st.markdown("**🏷️ Category Distribution (Fixed Assets Only)**")
                    if len(filtered_df[filtered_df['is_fixed_asset'] == True]) > 0:
                        category_counts = filtered_df[filtered_df['is_fixed_asset'] == True]['category'].value_counts()

                        category_icons = {
                            'electronics_it': '💻',
                            'furniture_office': '🪑', 
                            'machinery_equipment': '⚙️',
                            'vehicles_transport': '🚛',
                            'building_improvements': '🏗️',
                            'software_licenses': '💿'
                        }

                        # Create horizontal bar chart instead of pie for better readability
                        category_labels = [f"{category_icons.get(cat, '📦')} {cat.replace('_', ' ').title()}" 
                                         for cat in category_counts.index]

                        fig_categories = px.bar(
                            x=category_counts.values,
                            y=category_labels,
                            color=category_counts.values,
                            color_continuous_scale=['#FF4757', '#FFD23F', '#715DEC', '#00D4AA'],
                            height=380,
                            orientation='h',
                            labels={'x': 'Count', 'y': 'Category', 'color': 'Count'}
                        )

                        # Add count labels
                        fig_categories.update_traces(
                            texttemplate='%{x}',
                            textposition='outside',
                            textfont=dict(color='white', size=10)
                        )

                        fig_categories.update_layout(
                            xaxis_title="Asset Count",
                            yaxis_title="",
                            showlegend=False,
                            margin=dict(t=40, b=20, l=150, r=60),
                            font=dict(color='white'),
                            paper_bgcolor='#1E1E1E',
                            plot_bgcolor='#1E1E1E',
                            xaxis=dict(gridcolor='#444'),
                            yaxis=dict(gridcolor='#444'),
                            title=dict(
                                text=f"<b>Categories: {len(category_counts)}</b>",
                                x=0.5,
                                y=0.95,
                                font=dict(size=12, color='white')
                            )
                        )
                        st.plotly_chart(fig_categories, use_container_width=True)
        
        with tab_results:
            st.markdown("**📋 All Transaction Results**")
            
            # Search functionality
            search_col1, search_col2 = st.columns([3, 1])
            
            with search_col1:
                search_term = st.text_input("🔍 Search transactions", placeholder="Search by vendor, description, or category...")
            
            with search_col2:
                show_only_assets = st.checkbox("🏢 Show only fixed assets", value=False)
            
            # Apply search and filter
            display_df = filtered_df.copy()
            
            if search_term:
                search_mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                display_df = display_df[search_mask]
            
            if show_only_assets:
                display_df = display_df[display_df['is_fixed_asset'] == True]
            
            # Sort by confidence level (highest to lowest)
            confidence_order = {'Very High': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Very Low': 1}
            display_df['confidence_sort'] = display_df['confidence_level'].map(confidence_order)
            display_df = display_df.sort_values(['confidence_sort', 'score'], ascending=[False, False])
            display_df = display_df.drop('confidence_sort', axis=1)
            
            # Pagination
            pagination_col1, pagination_col2 = st.columns([1, 3])
            
            with pagination_col1:
                items_per_page = st.selectbox("Items per page", [10, 25, 50, 100], index=1)
            
            with pagination_col2:
                total_pages = (len(display_df) - 1) // items_per_page + 1 if len(display_df) > 0 else 1
                current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
            
            start_idx = current_page * items_per_page
            end_idx = start_idx + items_per_page
            page_df = display_df.iloc[start_idx:end_idx]
            
            if len(page_df) > 0:
                # Format the dataframe for better display
                display_columns = []
                
                # Essential columns
                if 'Employee' in page_df.columns:
                    display_columns.append('Employee')
                
                if 'vendor' in page_df.columns:
                    display_columns.append('vendor')
                elif 'merchant' in page_df.columns:
                    display_columns.append('merchant')
                elif 'Vendor name' in page_df.columns:
                    display_columns.append('Vendor name')
                
                if 'description' in page_df.columns:
                    display_columns.append('description')
                elif 'memo' in page_df.columns:
                    display_columns.append('memo')
                
                if 'amount' in page_df.columns:
                    display_columns.append('amount')
                
                display_columns.extend(['is_fixed_asset', 'score', 'confidence_level', 'category'])
                
                # Filter columns that actually exist
                available_columns = [col for col in display_columns if col in page_df.columns]
                
                styled_df = page_df[available_columns].copy()
                
                # Format amount column if it exists
                if 'amount' in styled_df.columns:
                    styled_df['amount'] = styled_df['amount'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                
                # Apply styling based on confidence and fixed asset status using theme colors
                def color_row(row):
                    if row['is_fixed_asset']:
                        if row['confidence_level'] in ['Very High', 'High']:
                            return [f'background-color: rgba(113, 93, 236, 0.1); border-left: 3px solid #715DEC'] * len(row)
                        elif row['confidence_level'] == 'Medium':
                            return [f'background-color: rgba(252, 108, 15, 0.1); border-left: 3px solid #FC6C0F'] * len(row)
                        else:
                            return [f'background-color: rgba(44, 33, 111, 0.1); border-left: 3px solid #2C216F'] * len(row)
                    return [f'background-color: rgba(235, 232, 255, 0.05)'] * len(row)
                
                st.dataframe(
                    styled_df.style.apply(color_row, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                st.markdown(f"Showing {start_idx + 1}-{min(end_idx, len(display_df))} of {len(display_df)} transactions")
            else:
                st.info("No transactions match the current filters.")
        
        with tab_export:
            st.markdown("**💾 Export Results**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export filtered results
                if st.button("📊 Export Filtered Results", use_container_width=True):
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"fixed_assets_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                # Export only fixed assets
                fixed_assets_only = filtered_df[filtered_df['is_fixed_asset'] == True]
                if st.button("🏢 Export Fixed Assets Only", use_container_width=True):
                    csv_data = fixed_assets_only.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Fixed Assets CSV",
                        data=csv_data,
                        file_name=f"fixed_assets_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                # Export summary report
                if st.button("📋 Export Summary Report", use_container_width=True):
                    # Create summary report
                    summary_data = {
                        'Total Transactions': len(filtered_df),
                        'Fixed Assets Detected': len(filtered_df[filtered_df['is_fixed_asset'] == True]),
                        'Detection Rate (%)': f"{len(filtered_df[filtered_df['is_fixed_asset'] == True])/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%",
                        'Average Score': f"{filtered_df['score'].mean():.1f}" if len(filtered_df) > 0 else "0",
                        'Total Asset Value': f"${filtered_df[filtered_df['is_fixed_asset'] == True]['amount'].sum():,.2f}" if 'amount' in filtered_df.columns else "$0",
                        'High Confidence Count': len(filtered_df[filtered_df['confidence_level'].isin(['Very High', 'High'])]),
                        'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Threshold Used': fa_threshold
                    }
                    
                    summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
                    csv_data = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=csv_data,
                        file_name=f"fixed_assets_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Export options explanation
            st.markdown("---")
            st.markdown("**📝 Export Options:**")
            st.markdown("""
            <div class="info-box">
            <strong>📊 Filtered Results:</strong> All transactions matching current filters with analysis results<br>
            <strong>🏢 Fixed Assets Only:</strong> Only transactions identified as fixed assets<br>
            <strong>📋 Summary Report:</strong> High-level statistics and metrics from the analysis
            </div>
            """, unsafe_allow_html=True)

            with tab_email:
                # Add email sending section
                st.subheader("📧 Email Fixed Asset Alerts")

                # Check if email credentials are configured
                email_config_complete = all([
                    st.session_state.get('smtp_server', ''),
                    st.session_state.get('smtp_user', ''),
                    st.session_state.get('smtp_pass', '')
                ])

                if not email_config_complete:
                    # Show configuration requirement message
                    st.error("⚠️ Email configuration required before sending alerts")

                    missing_fields = []
                    if not st.session_state.get('smtp_server', ''): missing_fields.append("SMTP Server")
                    if not st.session_state.get('smtp_user', ''): missing_fields.append("Email Username")
                    if not st.session_state.get('smtp_pass', ''): missing_fields.append("Email Password")

                    st.markdown(f"""
                    <div style="background-color: #2C1A7A; border: 1px solid #3A2A8C; border-radius: 5px; padding: 15px; margin: 10px 0;">
                    <h4>🔧 Configuration Required:</h4>
                    <p>Please configure the following in the sidebar before sending emails:</p>
                    <ul>
                    {"".join([f"<li>• {field}</li>" for field in missing_fields])}
                    </ul>
                    """, unsafe_allow_html=True)

                else:
                    # Email configuration is complete - show full email functionality
                    st.success("✅ Email configuration complete - ready to send alerts")

                    # Create SMTP configuration from session state
                    smtp_config = {
                        'smtp_server': st.session_state.smtp_server,
                        'smtp_port': st.session_state.smtp_port,
                        'smtp_user': st.session_state.smtp_user,
                        'smtp_pass': st.session_state.smtp_pass
                    }

                # Show preview and email options regardless of config status
                if len(fixed_assets_only) > 0:
                    if not email_config_complete:
                        st.markdown("---")
                        st.markdown("**📋 Fixed Assets That Would Be Alerted** *(Configuration required to send)*")
                    else:
                        st.markdown("**Send fixed asset alerts to employees:**")

                    # Define the display columns we want to show
                    display_columns = []

                    # Add description column
                    if 'Employee' in fixed_assets_only.columns:
                        display_columns.append('Employee')
                    if 'description' in fixed_assets_only.columns:
                        display_columns.append('description')
                    elif 'Memo' in fixed_assets_only.columns:
                        display_columns.append('Memo')

                    # Add amount column
                    if 'amount' in fixed_assets_only.columns:
                        display_columns.append('amount')
                    elif 'Amount (by category)' in fixed_assets_only.columns:
                        display_columns.append('Amount (by category)')

                    # Add date column
                    if 'date' in fixed_assets_only.columns:
                        display_columns.append('date')
                    elif 'Purchase Date' in fixed_assets_only.columns:
                        display_columns.append('Purchase Date')

                    # Add vendor column if available
                    if 'vendor' in fixed_assets_only.columns:
                        display_columns.append('vendor')
                    elif 'Vendor name' in fixed_assets_only.columns:
                        display_columns.append('Vendor name')

                    # Add analysis results
                    if 'score' in fixed_assets_only.columns:
                        display_columns.append('score')
                    if 'confidence_level' in fixed_assets_only.columns:
                        display_columns.append('confidence_level')
                    if 'category' in fixed_assets_only.columns:
                        display_columns.append('category')

                    # Show the dataframe with available columns
                    if display_columns:
                        st.dataframe(fixed_assets_only[display_columns], use_container_width=True)
                    else:
                        st.warning("⚠️ No suitable columns found to display")
                    
                    # Only show email sending interface if configured
                    if email_config_complete:
                        # Group transactions by employee first
                        employee_col1, employee_col2 = st.columns([2, 1])

                        with employee_col1:
                            # Determine which employees have fixed asset transactions
                            if 'Employee' in fixed_assets_only.columns:
                                employees_with_assets = fixed_assets_only['Employee'].unique().tolist()
                            elif 'employee' in fixed_assets_only.columns:
                                employees_with_assets = fixed_assets_only['employee'].unique().tolist()
                            elif 'employee_name' in fixed_assets_only.columns:
                                employees_with_assets = fixed_assets_only['employee_name'].unique().tolist()
                            else:
                                # If no employee column exists, fallback to all employees
                                employees_with_assets = list(st.session_state.fixed_asset_detector.EMPLOYEE_EMAIL_MAPPING.keys())

                            # Filter to only include employees who exist in the email mapping
                            available_employees = [emp for emp in employees_with_assets 
                                                 if emp in st.session_state.fixed_asset_detector.EMPLOYEE_EMAIL_MAPPING.keys()]

                            selected_employees = st.multiselect(
                                "Select employees to notify:",
                                options=available_employees,
                                default=available_employees[:3] if len(available_employees) <= 3 else [],  # Select first few by default
                                help="Only showing employees with detected fixed asset transactions"
                            )

                        with employee_col2:
                            test_mode = st.checkbox("🧪 Test Mode", value=True, help="Only send emails to lulu@k-id.com")

                        # Only show transaction selection if employees are selected
                        if selected_employees:
                            st.markdown("---")
                            st.markdown("### 📋 Select Transactions to Include in Alerts")
                            st.markdown("Choose which specific transactions should be included in employee notifications:")

                            # Filter transactions for selected employees only
                            selected_employee_transactions = pd.DataFrame()

                            for emp in selected_employees:
                                if 'Employee' in fixed_assets_only.columns:
                                    emp_transactions = fixed_assets_only[fixed_assets_only['Employee'] == emp]
                                elif 'employee' in fixed_assets_only.columns:
                                    emp_transactions = fixed_assets_only[fixed_assets_only['employee'] == emp]
                                elif 'employee_name' in fixed_assets_only.columns:
                                    emp_transactions = fixed_assets_only[fixed_assets_only['employee_name'] == emp]
                                else:
                                    # If no employee mapping exists, assign all transactions to each selected employee
                                    emp_transactions = fixed_assets_only

                                selected_employee_transactions = pd.concat([selected_employee_transactions, emp_transactions], ignore_index=True)

                            if len(selected_employee_transactions) > 0:
                                # Create a transaction selection interface
                                transaction_selection_data = []
                                selected_transaction_indices = []

                                # Add "Select All" / "Deselect All" buttons
                                col1, col2, col3 = st.columns([1, 1, 4])
                                with col1:
                                    if st.button("✅ Select All", key="select_all_transactions"):
                                        st.session_state.selected_all_transactions = True
                                        st.rerun()

                                with col2:
                                    if st.button("❌ Deselect All", key="deselect_all_transactions"):
                                        st.session_state.selected_all_transactions = False
                                        st.rerun()

                                # Initialize selection state
                                if 'selected_all_transactions' not in st.session_state:
                                    st.session_state.selected_all_transactions = True  # Default to all selected

                                # Process transactions for selection interface
                                for idx, (original_idx, transaction) in enumerate(selected_employee_transactions.iterrows()):
                                    employee_name = transaction.get('Employee', transaction.get('employee', 'Unknown Employee'))
                                    vendor = str(transaction.get('vendor', transaction.get('merchant', transaction.get('Vendor name', 'Unknown Vendor'))))
                                    amount = transaction.get('amount', transaction.get('Amount (by category)', 0))
                                    description = str(transaction.get('description', transaction.get('memo', transaction.get('Memo', 'No description'))))
                                    category = str(transaction.get('category', 'Unclassified'))
                                    confidence = str(transaction.get('confidence_level', 'Unknown'))
                                    score = transaction.get('score', 0)

                                    transaction_selection_data.append({
                                        'index': idx,
                                        'original_index': original_idx,
                                        'employee': employee_name,
                                        'vendor': vendor,
                                        'amount': float(amount) if amount else 0,
                                        'description': description,
                                        'category': category,
                                        'confidence': confidence,
                                        'score': float(score) if score else 0,
                                        'original_data': transaction
                                    })

                                # Group transactions by employee for better organization
                                from collections import defaultdict
                                transactions_by_employee = defaultdict(list)
                                for trans_data in transaction_selection_data:
                                    transactions_by_employee[trans_data['employee']].append(trans_data)

                                # Display transaction selection interface
                                st.markdown("**Select specific transactions to include in alerts:**")

                                # Create selection checkboxes organized by employee
                                for employee_name in selected_employees:  # Use the selected employees order
                                    if employee_name in transactions_by_employee:
                                        employee_transactions = transactions_by_employee[employee_name]

                                        with st.expander(f"👤 **{employee_name}** - {len(employee_transactions)} transaction(s)", expanded=True):
                                            st.markdown(f"**Email:** {st.session_state.fixed_asset_detector.EMPLOYEE_EMAIL_MAPPING.get(employee_name, 'No email found')}")

                                            # Employee-level select/deselect
                                            emp_col1, emp_col2 = st.columns([1, 5])
                                            with emp_col1:
                                                select_all_emp = st.checkbox(
                                                    "Select All", 
                                                    key=f"select_all_{employee_name}",
                                                    value=st.session_state.get('selected_all_transactions', True)
                                                )

                                            for trans_data in employee_transactions:
                                                trans_idx = trans_data['index']

                                                # Create a unique key for this transaction
                                                trans_key = f"trans_select_{trans_idx}_{employee_name}"

                                                # Transaction selection checkbox with details
                                                col1, col2 = st.columns([1, 5])

                                                with col1:
                                                    is_selected = st.checkbox(
                                                        "Include",
                                                        key=trans_key,
                                                        value=select_all_emp
                                                    )

                                                with col2:
                                                    # Display transaction details in a compact format
                                                    confidence_color = {
                                                        'Very High': '🟢',
                                                        'High': '🔵', 
                                                        'Medium': '🟡',
                                                        'Low': '🟠',
                                                        'Very Low': '🔴'
                                                    }.get(trans_data['confidence'], '⚪')

                                                    category_icon = {
                                                        'electronics_it': '💻',
                                                        'furniture_office': '🪑',
                                                        'machinery_equipment': '⚙️',
                                                        'vehicles_transport': '🚛',
                                                        'building_improvements': '🏗️',
                                                        'software_licenses': '💿'
                                                    }.get(trans_data['category'], '📦')

                                                    # Fix the description display issue
                                                    desc_text = str(trans_data['description']) if trans_data['description'] else 'No description'
                                                    if desc_text == 'nan':
                                                        desc_text = 'No description'

                                                    truncated_desc = desc_text[:100] + '...' if len(desc_text) > 100 else desc_text

                                                    st.markdown(f"""
                                                    **{trans_data['vendor']}** - ${trans_data['amount']:,.2f}

                                                    {confidence_color} {trans_data['confidence']} ({trans_data['score']:.1f}/100) | {category_icon} {trans_data['category']}

                                                    📝 {truncated_desc}
                                                    """)

                                                    if is_selected and trans_idx not in selected_transaction_indices:
                                                        selected_transaction_indices.append(trans_idx)
                                                    elif not is_selected and trans_idx in selected_transaction_indices:
                                                        selected_transaction_indices.remove(trans_idx)

                                # Filter the dataset to only include selected transactions
                                if selected_transaction_indices:
                                    # Get the original indices for filtering
                                    original_indices_to_keep = [transaction_selection_data[i]['original_index'] for i in selected_transaction_indices]
                                    final_selected_transactions = selected_employee_transactions.loc[original_indices_to_keep].copy()

                                    st.markdown("---")
                                    st.success(f"✅ **{len(final_selected_transactions)} transactions selected** for employee notifications")

                                    # Show summary of selected transactions
                                    if len(final_selected_transactions) > 0:
                                        total_selected_value = final_selected_transactions['amount'].sum() if 'amount' in final_selected_transactions.columns else 0
                                        st.markdown(f"💰 **Total value of selected transactions:** ${total_selected_value:,.2f}")

                                        # Create employee-transaction mapping for the final selected data
                                        employee_transactions_dict = {}
                                        for emp in selected_employees:
                                            if 'Employee' in final_selected_transactions.columns:
                                                emp_transactions = final_selected_transactions[final_selected_transactions['Employee'] == emp]
                                            elif 'employee' in final_selected_transactions.columns:
                                                emp_transactions = final_selected_transactions[final_selected_transactions['employee'] == emp]
                                            elif 'employee_name' in final_selected_transactions.columns:
                                                emp_transactions = final_selected_transactions[final_selected_transactions['employee_name'] == emp]
                                            else:
                                                emp_transactions = final_selected_transactions

                                            if len(emp_transactions) > 0:
                                                employee_transactions_dict[emp] = emp_transactions

                                        # Show preview of what will be sent
                                        with st.expander("📋 Email Preview", expanded=False):
                                            st.markdown(f"**📧 Email Configuration:**")
                                            st.code(f"""
                        SMTP Server: {smtp_config['smtp_server']}:{smtp_config['smtp_port']}
                        From: {smtp_config['smtp_user']}
                        Test Mode: {'Enabled' if test_mode else 'Disabled'}
                                            """)
                                            st.markdown("---")

                                            for emp in employee_transactions_dict:
                                                emp_transactions = employee_transactions_dict[emp]
                                                total_value = emp_transactions['amount'].sum() if 'amount' in emp_transactions.columns else 0
                                                st.markdown(f"**{emp}** ({st.session_state.fixed_asset_detector.EMPLOYEE_EMAIL_MAPPING.get(emp, 'No email')})")
                                                st.markdown(f"- Will receive {len(emp_transactions)} fixed asset transactions")
                                                st.markdown(f"- Total value: ${total_value:,.2f}")
                                                st.markdown("---")

                                # Send emails button
                                if st.button("📧 Schedule Fixed Asset Alerts", type="primary", use_container_width=True):
                                    with st.spinner("📧 Scheduling fixed asset alert emails..."):
                                        result = st.session_state.fixed_asset_detector.send_fixed_asset_emails_with_ui(
                                            employee_transactions_dict=employee_transactions_dict,
                                            smtp_config=smtp_config,
                                            test_mode=test_mode,
                                            delay_minutes=5
                                        )

                                        if result['success']:
                                            # FIXED: Store result in session state for UI tracking
                                            st.session_state.fixed_asset_email_result = {
                                                'success': True,
                                                'scheduled_count': result['scheduled_count']
                                            }

                                            st.success(f"🎉 {result['message']}")
                                            if result['skipped_count'] > 0:
                                                st.warning(f"⚠️ {result['skipped_count']} emails were skipped (no email address found)")

                                            # ADD THIS: Launch static UI directly here
                                            try:
                                                # Start the static UI server
                                                generate_and_serve_static_ui(
                                                    emails_scheduled=result['scheduled_count'],
                                                    delay_minutes=5,
                                                    mode="Fixed Asset Mode", 
                                                    email_type="fixed_asset"
                                                )

                                                # Small delay to ensure server is ready
                                                time.sleep(1)

                                                # Automatically open the browser
                                                try:
                                                    webbrowser.open('http://localhost:5000')
                                                    st.success("🌐 Static Email Management UI launched and opened automatically!")
                                                except Exception as browser_error:
                                                    st.info("🌐 Static Email Management UI launched at http://localhost:5000")
                                                    st.info("📱 Please open the link manually if browser didn't open automatically")

                                            except Exception as ui_error:
                                                st.warning(f"⚠️ Static UI failed to launch: {ui_error}")

                                        else:
                                            st.session_state.fixed_asset_email_result = {
                                                'success': False,
                                                'scheduled_count': 0
                                            }
                                            st.error(f"❌ {result['message']}")

                                # CORRECTED: Cancel button section with proper imports
                                if ('fixed_asset_email_result' in st.session_state and 
                                    st.session_state.fixed_asset_email_result.get('success') and 
                                    st.session_state.fixed_asset_email_result.get('scheduled_count', 0) > 0):

                                    st.markdown("---")

                                    # Show current scheduled emails info
                                    # CORRECTED: Initialize EMAIL_LOCK properly
                                    import threading
                                    EMAIL_LOCK = threading.Lock()

                                    with EMAIL_LOCK:
                                        # CORRECTED: Initialize session state if not exists
                                        if 'FIXED_ASSET_SCHEDULED_EMAILS' not in st.session_state:
                                            st.session_state.FIXED_ASSET_SCHEDULED_EMAILS = {}

                                        total_emails = len(st.session_state.FIXED_ASSET_SCHEDULED_EMAILS)
                                        active_emails = sum(1 for email_data in st.session_state.FIXED_ASSET_SCHEDULED_EMAILS.values() 
                                                          if not email_data.get('cancelled', False) 
                                                          and not email_data.get('sent', False)
                                                          and not email_data.get('failed', False))

                                    st.write(f"📧 Current scheduled fixed asset emails: {total_emails}")
                                    st.write(f"⏳ Active fixed asset emails pending: {active_emails}")

                                    if active_emails > 0:
                                        if st.button("❌ Cancel Fixed Asset Emails", type="secondary", key="cancel_fixed_asset_emails"):
                                            with st.spinner("Cancelling fixed asset emails..."):
                                                try:
                                                    # CORRECTED: Call the global function properly
                                                    cancelled_count = cancel_all_scheduled_fixed_asset_emails()
                                                    time.sleep(2)  # Give threads time to process

                                                    if cancelled_count > 0:
                                                        st.success(f"✅ Successfully cancelled {cancelled_count} scheduled fixed asset emails")
                                                        st.info("📧 Cancelled emails will not be sent when their scheduled time arrives")

                                                        # Clear the result
                                                        if 'fixed_asset_email_result' in st.session_state:
                                                            del st.session_state.fixed_asset_email_result
                                                        st.rerun()
                                                    else:
                                                        st.warning("⚠️ No fixed asset emails were cancelled")

                                                except Exception as e:
                                                    st.error(f"❌ Error cancelling fixed asset emails: {str(e)}")
                                    else:
                                        st.info("ℹ️ No active fixed asset emails to cancel")
                                        # Dismiss button when no active emails
                                        if st.button("✅ Dismiss", key="dismiss_fixed_asset_no_active"):
                                            if 'fixed_asset_email_result' in st.session_state:
                                                del st.session_state.fixed_asset_email_result
                                            st.rerun()
                                else:
                                    st.warning("⚠️ No transactions selected. Please select at least one transaction to send alerts.")
                            else:
                                st.info(f"📭 No fixed asset transactions found for selected employees: {', '.join(selected_employees)}")
                        else:
                            st.info("👆 Select employees to enable transaction selection and email sending")
                                          
                        
                # Email management info (always show)
                with st.expander("⚙️ Email System Integration", expanded=False):
                    st.markdown(f"""
                    <div class="info-box">
                    <h4>📧 Email System Features:</h4>

                    ✅ **Scheduled Delivery**: Emails are scheduled with configurable delay
                    <br>✅ **Test Mode**: Safe testing with redirect to test email
                    <br>✅ **Cancellation**: Emails can be cancelled before sending
                    <br>✅ **Thread Management**: Background processing for better performance
                    <br>✅ **Brand Styling**: Consistent with company email templates
                    <br><br>

                    <strong>📊 Current Status:</strong>
                    <br>• Email Configuration: {'✅ Complete' if email_config_complete else '❌ Incomplete'}
                    <br>• SMTP Server: {st.session_state.get('smtp_server', 'Not configured')}
                    <br>• Email Username: {st.session_state.get('smtp_user', 'Not configured')}
                    <br>• Password: {'✅ Set' if st.session_state.get('smtp_pass', '') else '❌ Not set'}
                    <br><br>

                    <strong>🔧 Configuration Instructions:</strong>
                    <br>1. Go to Email Settings in the sidebar
                    <br>2. Enter your SMTP server details
                    <br>3. Enter your email username and password
                    <br>4. Return to this tab to send alerts
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Show sample data or instructions when no analysis has been run
        st.markdown("---")
        st.subheader("📋 Getting Started")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🚀 How to Use the Fixed Asset Detector:</h4>
            
            <strong>1. Upload Data:</strong> Upload a CSV file containing your transaction data<br>
            <strong>2. Configure Settings:</strong> Adjust the detection threshold and filters above<br>
            <strong>3. Run Analysis:</strong> Click the "Analyze Fixed Assets" button to start detection<br>
            <strong>4. Review Results:</strong> Examine the results in the Analytics Dashboard and All Results tabs<br>
            <strong>5. Export Data:</strong> Export your results for further analysis or reporting<br><br>
            
            <strong>📊 Required CSV Columns:</strong><br>
            • Vendor/Merchant name<br>
            • Transaction description (optional but recommended)<br>
            • Transaction amount<br>
            • Transaction date (optional but helpful)
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>💡 Tips for Better Results:</h4>
            
            • Use descriptive vendor names<br>
            • Include transaction descriptions<br>
            • Ensure amounts are in numeric format<br>
            • Start with threshold = 50 for balanced detection<br>
            • Review high-confidence results first<br>
            • Use category filters to focus analysis
            </div>
            """, unsafe_allow_html=True)
    
    # Help section with improved styling
    with st.expander("❓ Help & Documentation", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🎯 Detection Categories:</h4>
            
            <strong>💻 Electronics & IT:</strong> Computers, servers, network equipment, software<br>
            <strong>🪑 Office Furniture:</strong> Desks, chairs, filing cabinets, storage<br>
            <strong>⚙️ Machinery & Equipment:</strong> Industrial machines, tools, specialized equipment<br>
            <strong>🚛 Vehicles & Transport:</strong> Fleet vehicles, trucks, construction vehicles<br>
            <strong>🏗️ Building Improvements:</strong> HVAC, electrical, structural modifications<br>
            <strong>💿 Software Licenses:</strong> Enterprise software, professional applications
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📈 Confidence Levels:</h4>
            
            <strong>🟢 Very High (80-100):</strong> Strong indicators, exact vendor matches<br>
            <strong>🔵 High (65-79):</strong> Good patterns, business context present<br>
            <strong>🟡 Medium (50-64):</strong> Some indicators, moderate confidence<br>
            <strong>🟠 Low (30-49):</strong> Weak patterns, low confidence<br>
            <strong>🔴 Very Low (0-29):</strong> Very few or no asset indicators
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>🔧 Advanced Features:</h4>
        
        <strong>🧠 Semantic Analysis:</strong> Advanced text analysis for better categorization<br>
        <strong>💰 Amount Thresholds:</strong> Category-specific minimum amounts for classification<br>
        <strong>🏢 Business Context:</strong> Detection of business-related keywords and patterns<br>
        <strong>📊 Batch Processing:</strong> Efficient analysis of large transaction datasets<br>
        <strong>📈 Real-time Filtering:</strong> Dynamic filtering by category, confidence, and amount ranges<br>
        <strong>📋 Comprehensive Reporting:</strong> Detailed analytics with multiple export formats
        </div>
        """, unsafe_allow_html=True)

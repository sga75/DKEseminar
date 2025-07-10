import json
import os
import re
from bs4 import BeautifulSoup 
import config

def load_company_data(data_path=config.DATA_PATH):
    company_data = []
    unique_categories = set()

    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist. Please ensure 'data' folder is present and populated.")
        return [], []

    print(f"Loading data from: {os.path.abspath(data_path)}")
    company_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

    if not company_folders:
        print(f"Warning: No company sub-folders found in '{data_path}'. Please ensure your data is correct.")
        return [], []

    for company_folder_name in company_folders:
        company_folder_path = os.path.join(data_path, company_folder_name)
        company_id = company_folder_name

        company_data_file = os.path.join(company_folder_path, "company_data.json")
        code_desc_file = os.path.join(company_folder_path, "code_desc.json")
        
        
        website_content_file = None
        for fname in os.listdir(company_folder_path):
            if fname.endswith(('.html', '.txt')):
                website_content_file = os.path.join(company_folder_path, fname)
                break 

        company_info = {}
        if os.path.exists(company_data_file):
            try:
                with open(company_data_file, "r", encoding="utf-8") as f:
                    company_info = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON from {company_data_file}. Skipping.")
                continue
        else:
            print(f"Warning: {company_data_file} not found. Skipping company {company_id}.")
            continue 

        code_desc_full = []
        if os.path.exists(code_desc_file):
            try:
                with open(code_desc_file, "r", encoding="utf-8") as f:
                    code_desc_full = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON from {code_desc_file}. Skipping code descriptions for {company_id}.")

        website_text = ""
        if website_content_file and os.path.exists(website_content_file):
            try:
                with open(website_content_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if website_content_file.endswith('.html'):
                        
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        
                        for script_or_style in soup(["script", "style"]):
                            script_or_style.decompose()
                        
                        
                        website_text = soup.get_text(separator=' ', strip=True)
                        website_text = re.sub(r'\s+', ' ', website_text).strip()

                    else: # .txt file
                        website_text = content
            except Exception as e:
                print(f"Warning: Could not read or parse content from {website_content_file}: {e}. Skipping website text for {company_id}.")

        
        combined_text = website_text
        if "website_text" in company_info and company_info["website_text"]: 
            combined_text += " " + company_info["website_text"]
        
        
        if code_desc_full:
            for item in code_desc_full:
                if isinstance(item, dict) and "description" in item and item["description"]:
                    combined_text += " " + item["description"]
                elif isinstance(item, str) and item: 
                    combined_text += " " + item
        
        
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()

        if "category" in company_info and combined_text:
            company_data.append({
                "company_id": company_id,
                "category": company_info["category"], 
                "raw_text": combined_text, 
                "code_desc_full": code_desc_full 
            })
            unique_categories.add(company_info["category"])
        else:
            print(f"Warning: Skipping company {company_id} due to missing category or empty combined text after extraction.")

    return company_data, list(unique_categories)

if __name__ == "__main__":
    companies, categories = load_company_data()
    print(f"Loaded {len(companies)} companies.")
    print(f"Found categories: {categories}")
    if companies:
        print(f"Sample company raw text: {companies[0]['raw_text'][:200]}...")

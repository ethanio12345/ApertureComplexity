import pandas as pd
import re
import os
import fitz  # PyMuPDF
import glob
from pathlib import Path

def extract_treatment_data(pdf_path):
    """
    Extract radiation treatment data from a PDF report using PyMuPDF
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        DataFrame with extracted data including patient name and plan name
    """
    # Open the PDF file
    doc = fitz.open(pdf_path)
    text = ""
    
    # Extract text from all pages
    for page in doc:
        text += page.get_text()
    
    # Extract patient name - looking for patterns in pre-treatment reports
    # First look for standard format at the top of the report
    patient_match = re.search(r"PRE-TREATMENT REPORT\s+([\w\s]+),\s+([\w\s]+)\s+\d{10}", text)
    if patient_match:
        patient_name = f"{patient_match.group(1)}, {patient_match.group(2)}".strip()
        auid = patient_match.group(0).split()[-1]  # Extract AUID from the match
    else:
        # Try alternative formats
        alt_match = re.search(r"(?:Patient|Name):\s*([\w\s]+,\s*[\w\s]+)", text, re.IGNORECASE)
        if alt_match:
            patient_name = alt_match.group(1).strip()
            # Extract AUID from the line directly under the patient name
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if alt_match.group(0) in line:
                    auid_line = lines[i + 1]
                    auid = re.search(r"\d{10}", auid_line)
                    if auid:
                        auid = auid.group()
                    else:
                        auid = "Unknown"
                    break
        else:
            # Last resort - look for a name-like pattern near the ID
            id_match = re.search(r"(\w+,\s+\w+)\s+\d{7,10}", text)
            patient_name = id_match.group(1).strip() if id_match else "Unknown"
            auid = "Unknown"
    
    # Extract plan name
    plan_match = re.search(r"Plan:\s+([\w\s_\.]+)", text)
    if not plan_match:
        # Try alternative patterns for plan name
        plan_match = re.search(r"(?:Treatment Plan|Plan Name):\s*([\w\s_\.]+)", text, re.IGNORECASE)
    
    plan_name = plan_match.group(1).strip() if plan_match else "Unknown"
    
    # Find the treatment data table
    pattern = r"Beam\s+Gantry\s+Energy.*?\n(Composite.*?)(?:\n\s*Histograms|\n\s*Parameter)"
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        raise ValueError(f"Could not find the treatment data table in {pdf_path}")
    
    # Extract the raw table lines
    table_text = match.group(1)
    lines = table_text.strip().split('\n')
    
    data = []
    column_names = ["Patient_Name", "Plan_Name", "Beam", "Rotation", "Gantry", "Angle_Range", "Energy", 
                    "Daily_corr_factor", "Norm_dose", "Det_Dev", "DTA", "γ-index", "Median_dose_dev"]
    
    # Find all energy values from beam lines
    energy_values = set()
    for line in lines:
        # Skip the composite line
        if line.startswith("Composite"):
            continue
        
        # Extract energy from beam lines
        # Looking for patterns like "F1 CW 302 302° to 45° 10 MV 1.000"
        energy_match = re.search(r"(?:\d+°\s+to\s+\d+°)\s+(\d+(?:\.\d+)?\s*[A-Za-z]+)", line)
        if energy_match:
            energy_values.add(energy_match.group(1).strip())
    
    # Combine energy values with "/"
    combined_energy = " / ".join(sorted(energy_values)) if energy_values else "Unknown"

    # Find the composite line in the treatment data table
    composite_pattern = r"Composite\s+(\d+\s+cGy)\s+(\d+\.\d+%)\s+(\d+\.\d+%)\s+(\d+\.\d+%)\s+(\d+\.\d+%)"
    match = re.search(composite_pattern, text)
    
    if not match:
        raise ValueError(f"Could not find composite data in {pdf_path}")
    
    # Extract the composite values
    norm_dose = match.group(1)
    det_dev = match.group(2)
    dta = match.group(3)
    gamma_index = match.group(4)
    median_dose_dev = match.group(5)
    
    # Create a DataFrame with a single row
    data = [{
        "Patient_Name": patient_name,
        "Plan_Name": plan_name,
        "AUID": auid,
        'Energy': combined_energy,
        "Norm_dose": norm_dose,
        "Det_Dev": det_dev,
        "DTA": dta,
        "γ-index": gamma_index,
        "Median_dose_dev": median_dose_dev,
        "Source_File": os.path.basename(pdf_path)
    }]
    
    return pd.DataFrame(data)

def process_pdf_files(folders, output_file):
    """
    Process all PDF files in the specified folders and append results to output file
    
    Args:
        folders: List of folder paths to search for PDFs
        output_file: Path to the output CSV/Excel file
    """
    # Create or load the output file
    if os.path.exists(output_file):
        # Check file extension
        if output_file.endswith('.xlsx'):
            existing_data = pd.read_excel(output_file)
        else:  # Assume CSV
            existing_data = pd.read_csv(output_file)
        print(f"Loaded existing file with {len(existing_data)} entries")
    else:
        existing_data = pd.DataFrame(columns=["Patient_Name", "Plan_Name", "Energy", 
                                              "Det_Dev", "DTA", "γ-index", "Median_dose_dev"])
        print("Creating new output file")
    
    # Count for processed files
    processed_count = 0
    error_count = 0
    
    # Process each folder
    for folder in folders:
        print(f"Processing folder: {folder}")
        pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing: {pdf_file}")
                df = extract_treatment_data(pdf_file)
                
                # Add filename as reference
                df["Source_File"] = os.path.basename(pdf_file)
                
                existing_data = pd.concat([existing_data, df], ignore_index=True)
                processed_count += 1
                
                # Save after each successful processing to avoid losing data
                if output_file.endswith('.xlsx'):
                    existing_data.to_excel(output_file, index=False)
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                error_count += 1
    
    print(f"Processing complete. Processed {processed_count} files with {error_count} errors.")
    print(f"Results saved to {output_file}")

def main():
    # List of folders to search for PDFs
    from pathlib import Path
    
    base_path = Path(r"V:\01 Physics Clinical QA\06 Patient QA\Delta4 Store\PDF Printouts\Archive")
    folders = list(base_path.glob("*/"))
    # Output file path
    output_file = "radiation_treatment_data.xlsx"  # Can also use .xlsx for Excel
    
    process_pdf_files(folders, output_file)

if __name__ == "__main__":
    main()

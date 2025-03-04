import pandas as pd

def read_data(df_file: str, pcm_summary_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads data from an Excel file and a CSV file.

    :param df_file: Path to the Excel file
    :param pcm_summary_file: Path to the CSV file
    :return: A tuple containing two DataFrames, the first from the Excel file and the second from the CSV file
    """
    archcheck_df = pd.read_excel(df_file, header=0)
    pcm_summary_df = pd.read_excel(pcm_summary_file, header=0)
    return archcheck_df, pcm_summary_df

def main(df_file, pcm_summary_file):
    """Main function to read the two input files, merge them based on MRN and AUID, 
    and save the merged DataFrame to a new file."""
    d4_df, pcm_summary_df = read_data(df_file, pcm_summary_file)

    print("Data types of ArchCheck columns:")
    print(d4_df.dtypes)
    print("\nData types of PCM summary columns:")
    print(pcm_summary_df.dtypes)

    # Convert 'AUID' column to integer type
    d4_df['AUID'] = pd.to_numeric(d4_df['AUID'], errors='coerce')

    # Merge the two DataFrames based on MRN and AUID
    merged_df = pd.merge(d4_df, pcm_summary_df, left_on=['AUID'], right_on=['Patient ID'], how='inner')
    # Add a new column 'Pass Rate' with the values from d4_df
    
    columns_to_extract = ['Patient ID', 'Course ID', 'Plan ID', 'Source_File',
                          'Energy', 'Det_Dev', 'DTA', 'γ-index', 'Median_dose_dev',
                           'Avg_MCS', 'Avg_EM', 'Avg_LeafTravel', 'Avg_ArcLength']
    extracted_df = merged_df[columns_to_extract]

    correlated_rows = (extracted_df['γ-index'].notna()) & (extracted_df['Avg_MCS'].notna())
    extracted_df = extracted_df[correlated_rows]

    # Save the merged DataFrame to a new file
    extracted_df.to_excel('matched_rows_d4.xlsx', index=False)
 
if __name__ == "__main__":
    main(r"C:\Users\60208787\Github\D4-PSQA\radiation_treatment_data.xlsx", 
         "parsed_output_datamining_d4patients.xlsx")


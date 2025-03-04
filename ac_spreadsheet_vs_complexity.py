import pandas as pd

def read_data(archcheck_file: str, pcm_summary_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads data from an Excel file and a CSV file.

    :param archcheck_file: Path to the Excel file
    :param pcm_summary_file: Path to the CSV file
    :return: A tuple containing two DataFrames, the first from the Excel file and the second from the CSV file
    """
    archcheck_df = pd.read_excel(archcheck_file, header=0)
    pcm_summary_df = pd.read_csv(pcm_summary_file, header=0)
    return archcheck_df, pcm_summary_df

def main(archcheck_file, pcm_summary_file):
    """Main function to read the two input files, merge them based on MRN and AUID, 
    and save the merged DataFrame to a new file."""
    arccheck_df, pcm_summary_df = read_data(archcheck_file, pcm_summary_file)

    pcm_summary_df[' AUID'] = pd.to_numeric(pcm_summary_df[' AUID'], errors='coerce')
    arccheck_df['MRN'] = pd.to_numeric(arccheck_df['MRN'], errors='coerce')


    # Merge the two DataFrames based on MRN and AUID
    merged_df = pd.merge(arccheck_df, pcm_summary_df, left_on=['MRN'], right_on=[' AUID'], how='inner', indicator=True)


    # Add a new column 'Pass Rate' with the values from arccheck_df
    merged_df['Pass Rate'] = merged_df['Absolute DTA (3%,2mm)']
    
    columns_to_extract = ['Plan Name', 'plan', 'MRN', ' Course ID', 'Clinical Site', ' parent', 
                          ' PyComplexityMetric (CI [mm^-1])', ' MeanAreaMetricEstimator (mm^2)',
                          ' AreaMetricEstimator (mm^2)', ' ApertureIrregularityMetric (dimensionless)', 'Pass Rate']
    
    extracted_df = merged_df[columns_to_extract]

    columns_to_write = ['Plan Name', 'Plan File', 'MRN', 'Course ID', 'Clinical Site', 'Parent Folder', 
                          'PyComplexityMetric', 'MeanAreaMetricEstimator',
                          'AreaMetricEstimator', 'ApertureIrregularityMetric', 'Pass Rate']
    extracted_df.columns = columns_to_write
    extracted_df = extracted_df[~extracted_df['Pass Rate'].isna()]
    # Save the merged DataFrame to a new file
    extracted_df.to_csv('matched_rows_arccheck.csv', index=False)
 
if __name__ == "__main__":
    main("BCHC - ArcCheck Results Summary_cleaned.xlsx", 
         "AC Measurements\plan_complexity_summary.csv")


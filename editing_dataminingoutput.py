import pandas as pd
import re

# Read the CSV file
df = pd.read_csv(r"P:\8. Staff\EB\DataMiner Re-Build\DataMiningOutput.20250228140332.csv")

# Define the regular expression pattern to extract the data
pattern = r"\((F\d+) (CW*|CCW*)\s*\d*:([^,]+),([^,]+),([^,]+),([^,]+)\)"

# Create a new DataFrame to store the parsed data
parsed_df = pd.DataFrame()

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    # Extract the output string from the row
    output = row['PlanComplexity(Beam ID:MCS,EM[mm-1],LeafTravel[mm],ArcLength[deg])']

    # Check if the output is not NaN
    if pd.notna(output):
        # Use the regular expression to extract the data
        matches = re.findall(pattern, output)
        
        # Create a dictionary to store the data
        data = {}
        
        # Iterate over the matches and populate the dictionary
        for match in matches:
            fan = match[0]
            direction = match[1]
            mcs = match[2]
            em = match[3]
            leaf_travel = match[4]
            arc_length = match[5]
            
            # Create a new row in the dictionary
            data[f"{fan} - MCS"] = mcs
            data[f"{fan} - EM"] = em
            data[f"{fan} - LeafTravel"] = leaf_travel
            data[f"{fan} - ArcLength"] = arc_length
        
        # Create a new row in the parsed DataFrame
        parsed_row = pd.DataFrame([data])
        
        # Append the parsed row to the parsed DataFrame
        parsed_df = pd.concat([parsed_df, parsed_row], ignore_index=True)

# Delete the original column from the DataFrame
df.drop(columns=['PlanComplexity(Beam ID:MCS,EM[mm-1],LeafTravel[mm],ArcLength[deg])'], inplace=True)

# Concatenate the parsed DataFrame with the original DataFrame
combined_df = pd.concat([df, parsed_df], axis=1)

# Convert MCS, EM, LeafTravel, and ArcLength columns to numeric values
combined_df['Avg_MCS'] = combined_df.filter(like='MCS').apply(pd.to_numeric).mean(axis=1)
combined_df['Avg_EM'] = combined_df.filter(like='EM').apply(pd.to_numeric).mean(axis=1)
combined_df['Avg_LeafTravel'] = combined_df.filter(like='LeafTravel').apply(pd.to_numeric).mean(axis=1)
combined_df['Avg_ArcLength'] = combined_df.filter(like='ArcLength').apply(pd.to_numeric).mean(axis=1)
# Write the parsed DataFrame to a CSV file
combined_df.to_excel("parsed_output_datamining_d4patients.xlsx", index=False)
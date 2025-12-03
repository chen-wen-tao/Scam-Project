# This file is used to merge the three data files into one single file.
# The complaint id is the unique identifier for each complaint.

import pandas as pd

# Read the two CSV files
ashley_df = pd.read_csv('data-ashley.csv')
jessica_df = pd.read_csv('data-jessica.csv')

# Extract only the required columns from each dataframe
# data-ashley.csv already has the correct columns
ashley_selected = ashley_df[['Complaint ID', 'Consumer complaint narrative']].copy()

# data-jessica.csv has many columns, extract only the required ones
jessica_selected = jessica_df[['Complaint ID', 'Consumer complaint narrative']].copy()

# Merge the two dataframes
merged_df = pd.concat([ashley_selected, jessica_selected], ignore_index=True)

# Remove duplicate rows based on Complaint ID (keep first occurrence)
merged_df = merged_df.drop_duplicates(subset=['Complaint ID'], keep='first')  # type: ignore

# Convert Complaint ID to integer (remove any NaN values first)
merged_df = merged_df.dropna(subset=['Complaint ID'])
merged_df['Complaint ID'] = merged_df['Complaint ID'].astype(int)

# Sort by Complaint ID for better organization
merged_df = merged_df.sort_values('Complaint ID').reset_index(drop=True)

# Save to a new CSV file
merged_df.to_csv('data-merged.csv', index=False)

print(f"Merged data saved to 'data-merged.csv'")
print(f"Total unique complaints: {len(merged_df)}")
print(f"From ashley file: {len(ashley_selected)}")
print(f"From jessica file: {len(jessica_selected)}")
print(f"Duplicates removed: {len(ashley_selected) + len(jessica_selected) - len(merged_df)}")

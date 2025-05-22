import pandas as pd

# Load the CSV data
csv_file = 'manual_annotations_progress.csv'
data = pd.read_csv(csv_file)


data = data.drop(columns=['GT_ID_5', 'GT_ID_6'])

# Display the first few rows to confirm
print(data.head())

# Save the modified DataFrame back to CSV
data.to_csv('classroom4.csv', index=False)
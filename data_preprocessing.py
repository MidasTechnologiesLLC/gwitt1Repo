import pandas as pd

# Define the path to your CSV file
csv_file_path = 'C:/Users/gwitt/MidasTechnologies/API/SPY_3yr_5min_data.csv'  # Replace with your actual file path
df = pd.read_csv(csv_file_path)

# Step 2: Preprocess the data
# Parse the 'Date' column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%Y%m%d %H:%M:%S')
df.set_index('Date', inplace=True)

# Sort data in chronological order
df.sort_index(inplace=True)

# Handle missing data by forward filling
df.ffill(inplace=True)

# Step 3: Save preprocessed data to a new CSV file
preprocessed_file_path = 'SPY_5min_preprocessed.csv'  # Replace with your desired path
df.to_csv(preprocessed_file_path)

print(f"Preprocessed data saved to {preprocessed_file_path}")



import re
import pandas as pd
from PIL import Image

# Step 1: Read log file
file_path = '/volume/kent-ttm-ssd/Symphonies/Symphonies_VP_1526/outputs/version_19/log.txt'  # Update with the correct path to your log file
with open(file_path, 'r') as file:
    log_content = file.read()

# Step 2: Define regex patterns for val and train records
pattern_val = r"(val/.*?epoch: \d+, step: \d+)"
# pattern_train = r"(train/.*?epoch: \d+, step: \d+)"

# Step 3: Extract all matching records
val_records = re.findall(pattern_val, log_content, re.DOTALL)
# train_records = re.findall(pattern_train, log_content, re.DOTALL)

# Step 4: Convert records to DataFrames
def refine_records_to_dataframe(records):
    records_data = []
    for record in records:
        # Extract key-value pairs
        pairs = [item.split(": ") for item in record.replace("\n", " ").split(", ") if ": " in item]
        record_dict = {
            key.strip(): float(value.strip()) if value.strip().replace('.', '', 1).isdigit() else value.strip()
            for key, value in pairs
        }
        records_data.append(record_dict)
    return pd.DataFrame(records_data)

val_df = refine_records_to_dataframe(val_records)
# train_df = refine_records_to_dataframe(train_records)

# Define the column order manually based on the extracted text or visual inspection
desired_column_order = [
    "road", "sidewalk", "parking", "other-ground", "building", "car", "truck",
    "bicycle", "motorcycle", "other-vehicle", "vegetation", "trunk", "terrain",
    "person", "bicyclist", "motorcyclist", "fence", "pole", "traffic-sign"
]

# Step 6: Reorder columns in the DataFrames
def reorder_columns(df, desired_order):
    # Extract matching columns and append others at the end if not in the desired order
    columns_to_reorder = [col for col in desired_order if any(desired in col for desired in df.columns)]
    remaining_columns = [col for col in df.columns if col not in columns_to_reorder]
    final_order = columns_to_reorder + remaining_columns
    return df[final_order]

val_df_reordered = reorder_columns(val_df, desired_column_order)
# train_df_reordered = reorder_columns(train_df, desired_column_order)

# Step 7: Save or display the reordered DataFrames
val_df_reordered.to_csv('reordered_validation_records.csv', index=False)
# train_df_reordered.to_csv('reordered_training_records.csv', index=False)

print("Reordered validation and training records saved as CSV files.")

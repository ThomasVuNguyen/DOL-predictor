import json

import pandas as pd


def create_hf_json_dataset(data_path, output_path):
    """
    Transforms raw WHD CSV data into a structured JSON format
    suitable for Hugging Face (Input/Output mapping).
    """
    # Load the primary dataset
    df = pd.read_csv(data_path)

    # Define our 'Input' features (Predictors)
    input_features = [
        "st_cd",
        "naic_cd",
        "naics_code_description",
        "findings_start_date",
        "findings_end_date",
        "flsa_repeat_violator",
    ]

    # Define our 'Output' targets (Labels)
    output_targets = [
        "case_violtn_cnt",
        "cmp_assd",
        "ee_violtd_cnt",
        "bw_atp_amt",
        "ee_atp_cnt",
    ]

    hf_dataset = []

    for _, row in df.iterrows():
        # Construct the Input object with proper type casting
        entry_input = {}
        for feat in input_features:
            val = row[feat]
            if pd.notnull(val):
                # Convert dates to strings
                if "date" in feat.lower():
                    entry_input[feat] = str(val)
                # Keep string fields as strings
                elif isinstance(val, str):
                    entry_input[feat] = val
                # Convert numeric to proper types
                elif isinstance(val, (int, float)):
                    entry_input[feat] = int(val) if val == int(val) else float(val)
                else:
                    entry_input[feat] = str(val)
            else:
                entry_input[feat] = ""  # Use empty string instead of None

        # Construct the Output object with explicit numeric types
        entry_output = {}
        for target in output_targets:
            val = row[target]
            if pd.notnull(val):
                # All output targets are numeric, ensure proper type
                try:
                    entry_output[target] = float(val)
                except (ValueError, TypeError):
                    entry_output[target] = 0.0
            else:
                entry_output[target] = 0.0  # Use 0.0 instead of None for numeric fields

        # Append as a structured pair
        hf_dataset.append(
            {
                "instruction": "Predict the litigation risk and potential financial penalties for the following business profile.",
                "input": entry_input,
                "output": entry_output,
            }
        )

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hf_dataset, f, indent=2)

    print(f"Successfully generated {len(hf_dataset)} records at {output_path}")


# Run the generator
if __name__ == "__main__":
    create_hf_json_dataset("whd_whisard.csv", "whd_predictive_dataset.json")

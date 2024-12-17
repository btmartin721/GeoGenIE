from pathlib import Path

import pandas as pd


def parse_module_connections(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Extract table rows (ignoring headers, separators)
    table_lines = [line.strip() for line in lines if "|" in line and "---" not in line]

    # Parse rows manually into a list of dictionaries
    parsed_data = []
    for line in table_lines:
        columns = [col.strip() for col in line.split("|")[1:-1]]  # Ignore outer '|'
        if len(columns) == 4:  # Ensure correct number of columns
            parsed_data.append({
                "Module": columns[0],
                "Type": columns[1],
                "Name": columns[2],
                "Details": columns[3],
            })

    # Convert to DataFrame
    return pd.DataFrame(parsed_data)


# Function to extract major steps from train_test_predict flow
def extract_major_steps(connections):
    major_steps = []
    # Start with train_test_predict from geogenie
    main_entry = connections[
        (connections["Module"] == "geogenie") & (connections["Name"] == "train_test_predict")
    ]
    if not main_entry.empty:
        major_steps.append({
            "Module/Method": "geogenie.train_test_predict",
            "Purpose": "Main entry point for the code flow.",
            "Dependencies": "Calls major methods for training, testing, and predictions."
        })

    # Identify functions and methods that are likely major steps
    for _, row in connections.iterrows():
        if row["Type"] in ["Function", "Method"] and "train" in row["Name"].lower():
            major_steps.append({
                "Module/Method": f"{row['Module']}.{row['Name']}",
                "Purpose": "Handles major tasks like training or predictions.",
                "Dependencies": row["Details"]
            })
    return major_steps

# Function to save major steps as markdown
def save_major_steps_as_markdown(major_steps, output_path):
    with open(output_path, "w") as f:
        f.write("# Major Code Flow Steps\n\n")
        f.write("| Module/Method          | Purpose                                | Dependencies              |\n")
        f.write("|------------------------|----------------------------------------|---------------------------|\n")
        for step in major_steps:
            f.write(f"| {step['Module/Method']} | {step['Purpose']} | {step['Dependencies']} |\n")
    print(f"Major steps saved to: {output_path}")

# Main execution
module_connections_file = Path("./code_flow_summaries/module_connections.md")
output_file = Path("./code_flow_summaries/major_code_flow_steps.md")

connections = parse_module_connections(module_connections_file)
major_steps = extract_major_steps(connections)
save_major_steps_as_markdown(major_steps, output_file)

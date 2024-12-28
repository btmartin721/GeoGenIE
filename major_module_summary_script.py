import os
import ast
import pandas as pd
from pathlib import Path
from textwrap import dedent
import networkx as nx
import matplotlib.pyplot as plt
import random

# Function to recursively find all Python files in a directory
def find_python_files(directory):
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

# Function to summarize code flow within a module
def summarize_module(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=filepath)

    imports = []
    functions = []
    classes = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes[node.name] = class_methods

    return {"imports": imports, "functions": functions, "classes": classes}

# Function to generate summaries for all modules
def generate_module_summaries(directory, output_dir):
    python_files = find_python_files(directory)
    master_summary = []

    for filepath in python_files:
        module_name = Path(filepath).stem
        summary = summarize_module(filepath)
        module_file = output_dir / f"{module_name}_summary.md"

        # Write individual module summary in markdown-linter-friendly format
        with open(module_file, "w") as f:
            f.write(f"# Module: {module_name}\n\n")
            f.write("## Imports\n\n")
            f.write(format_list(summary['imports']))
            f.write("\n\n## Functions\n\n")
            f.write(format_list(summary['functions']))
            f.write("\n\n## Classes and Methods\n\n")
            f.write(format_classes(summary['classes']))

        # Add to master summary
        master_summary.append({
            "Module": module_name,
            "Imports": format_list_inline(summary["imports"]),
            "Functions": format_list_inline(summary["functions"]),
            "Classes": format_classes_inline(summary["classes"]),
        })

    # Write master summary with a TOC and summary table
    master_file = output_dir / "master_summary.md"
    with open(master_file, "w") as f:
        f.write("# Master Code Flow Summary\n\n")

        # Table of Contents
        f.write("## Table of Contents\n\n")
        for entry in master_summary:
            f.write(f"- [Module: {entry['Module']}](#{entry['Module']})\n")
        f.write("\n")

        # Summary Table
        f.write("## Summary Table\n\n")
        summary_df = pd.DataFrame(master_summary)
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")

        # Detailed Module Summaries
        for entry in master_summary:
            f.write(f"## Module: {entry['Module']}\n\n")
            f.write(f"### Imports\n\n{entry['Imports']}\n\n")
            f.write(f"### Functions\n\n{entry['Functions']}\n\n")
            f.write(f"### Classes\n\n{entry['Classes']}\n\n")

def generate_major_code_flow_graph(major_steps_file, output_dir):
    """
    Generates a clean NetworkX graph for major code flow visualization with improved aesthetics.
    """
    G = nx.DiGraph()

    # Read the major steps Markdown file
    with open(major_steps_file, "r") as f:
        lines = f.readlines()

    # Parse table rows
    table_lines = [line.strip() for line in lines if "|" in line and "---" not in line]
    steps = []
    for line in table_lines:
        columns = [col.strip() for col in line.split("|")[1:-1]]
        if len(columns) == 3:  # Ensure correct format
            steps.append({
                "Module_Method": columns[0].split('.')[-1],  # Extract method name only
                "Purpose": columns[1],
                "Dependencies": columns[2]
            })

    # Create nodes and edges
    for step in steps:
        method = step["Module_Method"]
        dependencies = step["Dependencies"].split(",") if step["Dependencies"] != "None" else []

        # Add method node
        G.add_node(method)

        # Add edges to dependencies
        for dep in dependencies:
            dep = dep.strip().split('.')[-1]  # Extract method name only for dependencies
            if dep:
                G.add_edge(dep, method)

    # Define fixed colors for nodes and edges
    node_color = "#f8eaad"  # Light blue nodes
    edge_color = "#0091ad"  # Dark blue edges

    # Layout and positioning
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=1.5, seed=42)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_color, edgecolors="black")
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=2, arrows=True, arrowstyle="-|>", connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="black")

    # Add edge labels
    edge_labels = {(u, v): "calls" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="#7B8D93")

    # Save the graph
    output_file = output_dir / "major_code_flow_networkx.png"
    plt.title("Major Code Flow Visualization", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Major code flow graph saved at: {output_file}")

# Helper functions for formatting
def format_list(items):
    if not items:
        return "- None"
    return "\n".join([f"- {item}" for item in items])

def format_list_inline(items):
    if not items:
        return "None"
    return ", ".join(items)

def format_classes(classes):
    if not classes:
        return "- None"
    result = []
    for cls, methods in classes.items():
        result.append(f"- {cls}")
        if methods:
            result.extend([f"  - {method}" for method in methods])
        else:
            result.append("  - None")
    return "\n".join(result)

def format_classes_inline(classes):
    if not classes:
        return "None"
    result = []
    for cls, methods in classes.items():
        methods_str = ", ".join(methods) if methods else "None"
        result.append(f"{cls}({methods_str})")
    return ", ".join(result)

# Main execution
directory = "./geogenie"  # Replace with your actual path
output_dir = Path("./code_flow_summaries")
output_dir.mkdir(exist_ok=True)

major_steps_file = output_dir / "major_code_flow_steps.md"

generate_module_summaries(directory, output_dir)
generate_major_code_flow_graph(major_steps_file, output_dir)
print(f"Code flow summaries and visualization generated in: {output_dir}")

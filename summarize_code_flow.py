import os
import ast
import pandas as pd
from pathlib import Path
from textwrap import dedent
from graphviz import Digraph

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
          
def generate_module_graph(directory, output_dir):
    python_files = find_python_files(directory)
    dot = Digraph(comment="Code Flow Visualization", engine="sfdp")  # Switch engine to 'sfdp'
    dot.attr(ratio="1.5", size="12,12", overlap="scale")  # Adjust size and scaling
    
    for filepath in python_files:
        module_name = Path(filepath).stem
        summary = summarize_module(filepath)

        # Add module node
        dot.node(module_name, module_name, shape="box", style="filled", color="lightblue")

        # Add function nodes
        for func in summary["functions"]:
            dot.node(f"{module_name}.{func}", func, shape="ellipse")
            dot.edge(module_name, f"{module_name}.{func}")

        # Add class nodes and methods
        for cls, methods in summary["classes"].items():
            dot.node(f"{module_name}.{cls}", cls, shape="box", color="lightgreen")
            dot.edge(module_name, f"{module_name}.{cls}")
            for method in methods:
                dot.node(f"{module_name}.{cls}.{method}", method, shape="ellipse", color="lightyellow")
                dot.edge(f"{module_name}.{cls}", f"{module_name}.{cls}.{method}")

    graph_file = output_dir / "code_flow_graph"
    dot.render(graph_file, format="png", cleanup=True)
    print(f"Code flow graph saved at: {graph_file}.png")

def generate_relationships_summary(directory, output_dir):
    python_files = find_python_files(directory)
    relationships = []

    for filepath in python_files:
        module_name = Path(filepath).stem
        summary = summarize_module(filepath)

        # Add module imports
        for imp in summary["imports"]:
            relationships.append({
                "Module": module_name,
                "Type": "Import",
                "Name": imp,
                "Details": "Imported module"
            })

        # Add functions
        for func in summary["functions"]:
            relationships.append({
                "Module": module_name,
                "Type": "Function",
                "Name": func,
                "Details": "Function defined in module"
            })

        # Add classes and methods
        for cls, methods in summary["classes"].items():
            relationships.append({
                "Module": module_name,
                "Type": "Class",
                "Name": cls,
                "Details": "Class defined in module"
            })
            for method in methods:
                relationships.append({
                    "Module": module_name,
                    "Type": "Method",
                    "Name": f"{cls}.{method}",
                    "Details": "Method defined in class"
                })

    # Convert to DataFrame and write to a relationships Markdown file
    relationships_df = pd.DataFrame(relationships)
    relationships_file = output_dir / "module_connections.md"

    with open(relationships_file, "w") as f:
        f.write("# Module Relationships Summary\n\n")
        f.write(relationships_df.to_markdown(index=False))
    print(f"Relationships summary saved to: {relationships_file}")

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

generate_module_summaries(directory, output_dir)
generate_module_graph(directory, output_dir)
generate_relationships_summary(directory, output_dir)
print(f"Code flow summaries and visualization generated in: {output_dir}")

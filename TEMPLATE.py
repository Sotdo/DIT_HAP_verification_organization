"""
[Brief one-line description of the script's purpose]

[Detailed description of what this script does, including:
- Main functionality
- Scientific/biological context if relevant
- Key algorithms or methods used
- Any important assumptions or constraints]

Typical Usage:
    python script_name.py --input data.tsv --output results.tsv

Input: [Description of expected input files and formats]
Output: [Description of output files and formats]
Other information: [Any additional context or notes]
"""

# =============================== Imports ===============================
# Import standard library modules first
import sys
from pathlib import Path
from dataclasses import dataclass

# Import third-party libraries
import pandas as pd

# Import project-specific modules (if needed)
# SCRIPT_DIR = Path(__file__).parent.resolve()
# TARGET_path = str((SCRIPT_DIR / "../../src").resolve())
# sys.path.append(TARGET_path)
# from utils import custom_function


# =============================== Constants ===============================
# Define constants at the top for easy modification
# Use UPPERCASE for constants
EXAMPLE_THRESHOLD = 100
DEFAULT_VALUE = 0.5


# =============================== Configuration & Models ===============================

@dataclass
class Config:
    """One line description of the configuration."""
    parameter1: int = 3
    # Add more configuration parameters as needed

# =============================== Core Functions ===============================
def function1(param1: str, param2: int) -> pd.DataFrame:
    """One line docs of function1."""
    # Function implementation goes here
    data = {'Column1': [1, 2], 'Column2': [3, 4]}
    df = pd.DataFrame(data)
    return df

def function2(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """One line docs of function2."""
    # Function implementation goes here
    df['NewColumn'] = df['Column1'] * config.parameter1
    return df

# =============================== Main Execution Block ===============================
def main():
    """Main function to execute the script logic."""
    # Example usage of the functions defined above
    config = Config(parameter1=2)
    df = function1("example", 10)
    result_df = function2(df, config)
    print(result_df)

if __name__ == "__main__":
    main()
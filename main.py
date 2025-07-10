import os
import sys
import config

script_directory = os.path.join(os.path.dirname(__file__), 'scripts')
if script_directory not in sys.path:
    sys.path.insert(0, script_directory)

from train import training_pipeline

if __name__ == "__main__":
    data_path = config.DATA_PATH
    if not os.path.exists(data_path):
        print(f"Error: The directory '{data_path}' does not exist.")
        print("Please create this directory and place your company data inside it as specified.")
        sys.exit(1) 

    training_pipeline() 
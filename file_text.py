import os

# Define the main folder path
main_folder = 'Capstone data sets'

# Check if the main folder exists
if not os.path.exists(main_folder):
    print(f"The specified folder '{main_folder}' does not exist.")
else:
    # Function to read different types of files
    def read_files(file_path):
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.txt':
            # Read text files
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                print(f"Content of {file_path}:\n", content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            # Placeholder for other file types
            print(f"File at {file_path} is of type {ext}. Add processing logic as needed.")

    # Walk through folders and subfolders
    for root, dirs, files in os.walk(main_folder):
        print(f"Checking folder: {root}")  # Debugging: Print the current folder
        if files:
            print(f"Found files: {files}")  # Debugging: List files found
        else:
            print("No files found in this folder.")
        
        for file_name in files:
            file_path = os.path.join(root, file_name)
            read_files(file_path)

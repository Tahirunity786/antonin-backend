import os
import zipfile

def extracter(zip_file, extract_to, up_name):
    if not os.path.exists(zip_file):
        print('The file path you provided does not exist. Please provide a valid path!')
        return False
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_file, 'r') as ref_zip:
        # Extract all contents
        ref_zip.extractall(extract_to)
        
        # Get the list of files in the ZIP
        all_files = ref_zip.namelist()
        print(all_files)  # To show all the files and folders in the zip

        # Find the main folder in the extracted files
        main_folder = None
        for file in all_files:
            # The main folder would be the first part of the file path
            if '/' in file:
                main_folder = file.split('/')[0]
                break
        
        if main_folder:
            main_folder_path = os.path.join(os.path.abspath(extract_to), main_folder)
           
        else:
            print('No main folder found in the ZIP file.')

        return True

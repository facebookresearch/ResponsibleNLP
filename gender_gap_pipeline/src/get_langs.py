
from pathlib import Path

import os

if __name__ == '__main__':
    
    script_path = Path(__file__).resolve().parent
    base_folder = script_path/'..'/'dataset'/'v1.0'
    
    SUPPORTED_LANGS = [file.stem.split('_')[0] for file in base_folder.iterdir() if file.is_file() if file.suffix == '.json']

    import pdb
    pdb.set_trace()
import re
import os

def next_version_filename(filename: str) -> str:
    """
    Given a filename (e.g. 'data.csv' or 'data_2.csv'),
    return the next versioned filename (e.g. 'data_1.csv' or 'data_3.csv').
    """
    base, ext = os.path.splitext(filename)

    # Look for a trailing underscore + number (e.g. "_2")
    match = re.search(r'_(\d+)$', base)
    
    if match:
        # Increment the number
        number = int(match.group(1)) + 1
        new_base = re.sub(r'_(\d+)$', f'_{number}', base)
    else:
        # No version number found, add "_1"
        new_base = f"{base}_1"

    return f"{new_base}{ext}"

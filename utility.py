import os

def create_dir(directory, is_base_dir=True):
    if is_base_dir:
        if not os.path.exists(directory):
            os.makedirs(directory)
    else:
        base_dir = os.path.dirname(directory)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
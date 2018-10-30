from os import path
import os

def get_dbpath():
    model_dir = os.environ['MODEL_DIR']
    task = os.environ['TASK']
    model_path = path.join(model_dir, task+'_parallel_output')  
    return model_path
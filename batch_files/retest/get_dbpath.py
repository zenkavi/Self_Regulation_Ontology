from sys import path

def get_dbpath():
    model_path = path.join(model_dir, task+'_parallel_output')  
    return model_path
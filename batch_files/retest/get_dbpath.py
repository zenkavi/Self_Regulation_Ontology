from os import path

def get_dbpath():
    global model_dir
    global task
    model_path = path.join(model_dir, task+'_parallel_output')  
    return model_path
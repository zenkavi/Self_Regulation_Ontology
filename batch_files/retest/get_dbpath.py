from sys import path

def get_dbpath():
    if model_dir is None or task is None:
        print('Assigning default values for model_dir and task')
        model_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_complete_output/'
        task = 'test'

    model_path = path.join(model_dir, task+'_parallel_output')
    
    return model_path
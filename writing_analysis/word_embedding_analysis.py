import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_behav_data, get_demographics, get_recent_dataset
from writing_analysis.utils import loadGloveModel, tokenize

def embed_writing(text, model):
    tokenized = tokenize(text)
    vectors = [model[w] for w in tokenized if w in model.keys()]
    mat = np.vstack(vectors)
    # normalize matrix before average
    norms = np.sum(mat**2,1)**.5
    mat = mat/norms[:,None]
    return mat.mean(0)

dataset = get_recent_dataset()
# load writing data
data = get_behav_data(file='Individual_Measures/writing_task.csv.gz', dataset=dataset)
index = data.worker_id
data = data.final_text
data.index = index
# get demographics
demo = get_demographics(dataset=dataset)
reduced_demo = demo.loc[:, ['Age']]
# get ontological scores
results = load_results(datafile=dataset)
scores = pd.DataFrame()
for key, val in results.items():
    results_scores = val.EFA.get_scores()
    scores = pd.concat([scores, results_scores], axis=1)

# currently uses glove embedding taken from here: https://nlp.stanford.edu/projects/glove/
model = loadGloveModel('glove.6B.50d.txt')
embeddings = data.apply(lambda x: embed_writing(x, model))

# do the prediction!
pipe = Pipeline([('scale', StandardScaler()),
                 ('model', RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)))])
X = np.vstack(embeddings.values)
# predict demographics
for name, Y in reduced_demo.iteritems():
    print(name, np.mean(cross_val_score(pipe, X, Y, cv=10)))

# predict ontological scores
for name, Y in scores.iteritems():
    print(name, np.mean(cross_val_score(pipe, X, Y, cv=10)))

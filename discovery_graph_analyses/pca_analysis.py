import numpy as np
from os import path
import pandas as pd
from util import dendroheatmap, get_info
import seaborn as sns
from sklearn import decomposition

#load Data
data_dir=path.join(get_info('base_directory'),'Data/Discovery_9-26-16')

# get DV df
DV_df = pd.read_csv(path.join(data_dir,'meaningful_variables_EZ_contrasts.csv'), index_col = 0)
valence_df = pd.read_json(path.join(data_dir, 'mturk_discovery_DV_valence.json'))


#flip negative signed valence DVs
flip_df = valence_df.replace(to_replace ={'Pos': 1, 'NA': 1, 'Neg': -1}).mean()
for c in DV_df.columns:
    try:
        DV_df.loc[:,c] = DV_df.loc[:,c] * flip_df.loc[c]
    except TypeError:
        continue

# ************************************
# ************ PCA *******************
# ************************************

pca_data = DV_df.corr()
pca = decomposition.PCA()
pca.fit(pca_data)

# plot explained variance vs. components
sns.plt.plot(pca.explained_variance_ratio_.cumsum())

# plot loadings of 1st component
sns.barplot(np.arange(200),pca.components_[0])

# dimensionality reduction
pca.n_components = 2
reduced_df = pca.fit_transform(pca_data)
sns.plt.scatter(reduced_df[:,0], reduced_df[:,1])

def top_variables(pca, labels, n = 5):
    components = pca.components_
    variance = pca.explained_variance_ratio_
    output = []
    for c,v in zip(components,variance):
        order = np.argsort(np.abs(c))[::-1]
        output.append(([labels[o] for o in order[:n]],v))
    return output
    

top = top_variables(pca, pca_data.columns)
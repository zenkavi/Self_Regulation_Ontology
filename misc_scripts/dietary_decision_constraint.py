
import matplotlib.pyplot as plt
import seaborn as sns
from selfregulation.utils.utils import get_recent_dataset, get_behav_data

dataset = get_recent_dataset()
data = get_behav_data()
dietary = get_behav_data(file='Individual_Measures/dietary_decision.csv.gz')

# get groups
restraint_data = data['eating_survey.cognitive_restraint']
high_group = list(restraint_data[restraint_data>restraint_data.median()].index)
low_group = list(restraint_data[restraint_data<restraint_data.median()].index)

# mean responses
mean_responses = dietary.groupby(['health_diff','taste_diff']) \
        .coded_response.mean().reset_index()
full_pivoted = mean_responses.pivot(index='health_diff', 
                    columns='taste_diff', 
                    values='coded_response').iloc[::-1,:]

# high contraint
mean_responses = dietary.query('worker_id in %s' % high_group) \
        .groupby(['health_diff','taste_diff']) \
        .coded_response.mean().reset_index()
high_pivoted = mean_responses.pivot(index='health_diff', 
                    columns='taste_diff', 
                    values='coded_response').iloc[::-1,:]

# low contraint
mean_responses = dietary.query('worker_id in %s' % low_group) \
        .groupby(['health_diff','taste_diff']) \
        .coded_response.mean().reset_index()
low_pivoted = mean_responses.pivot(index='health_diff', 
                    columns='taste_diff', 
                    values='coded_response').iloc[::-1,:]

# plotting
f, axes = plt.subplots(3,1, figsize=(12,36))
sns.heatmap(full_pivoted, square=True, vmin=-2, vmax=2,
            ax=axes[0])
axes[0].tick_params(labelsize=14)
axes[0].set_xlabel('Taste Difference', fontsize=20)
axes[0].set_ylabel('Health Difference', fontsize=20)
axes[0].set_title('Full Dataset', fontsize=30)

# high contraint
sns.heatmap(high_pivoted, square=True, vmin=-2, vmax=2,
            ax=axes[1])
axes[1].tick_params(labelsize=14)
axes[1].set_xlabel('Taste Difference', fontsize=20)
axes[1].set_ylabel('Health Difference', fontsize=20)
axes[1].set_title('High Constraint', fontsize=30)

# low contraint
sns.heatmap(low_pivoted, square=True, vmin=-2, vmax=2,
            ax=axes[2])
axes[2].tick_params(labelsize=14)
axes[2].set_xlabel('Taste Difference', fontsize=20)
axes[2].set_ylabel('Health Difference', fontsize=20)
axes[2].set_title('Low Constraint', fontsize=30)

f.savefig('/home/ian/choice_heatmaps.pdf')

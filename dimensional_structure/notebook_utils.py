from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dimensional_structure.utils import get_factor_groups
from selfregulation.utils.plot_utils import format_variable_names

def plot_factor_df(EFA, rotate='oblimin'):
    c = EFA.get_c()
    loadings = EFA.get_loading(c, rotate=rotate)
    loadings = EFA.reorder_factors(loadings, rotate=rotate)           
    grouping = get_factor_groups(loadings)
    flattened_factor_order = []
    for sublist in [i[1] for i in grouping]:
        flattened_factor_order += sublist
    loadings = loadings.loc[flattened_factor_order]
    loadings.index = format_variable_names(loadings.index)
    loadings.columns = loadings.columns.map(lambda x: str(x).ljust(15))

    # visualization functions
    def magnify():
        return [dict(selector="tr:hover",
                    props=[("border-top", "2pt solid black"),
                           ("border-bottom", "2pt solid black")]),
                dict(selector="th:hover",
                     props=[("font-size", "10pt")]),
                dict(selector="td",
                     props=[('padding', "0em 0em")]),
               # dict(selector="th:hover",
               #      props=[("font-size", "12pt")]),
                dict(selector="tr:hover td:hover",
                     props=[('max-width', '200px'),
                            ('font-size', '16pt')])
    ]
    
    cm =sns.diverging_palette(220,15,n=200)
    def color_red_or_green(val):
        color = to_hex(cm[int(val*100)+100])
        return 'background-color: %s' % color

    
    styler = loadings.style
    styler \
        .applymap(color_red_or_green) \
        .set_properties(**{'max-width': '100px', 'font-size': '0pt', 'border-color': 'white'})\
        .set_precision(2)\
        .set_table_styles(magnify())
    return styler

def plot_EFA_robustness(EFA_robustness):
    EFA_robustness = pd.DataFrame(EFA_robustness).T
    EFA_robustness.index = [' '.join(i.split('_')) for i in EFA_robustness.index]
    min_robustness = EFA_robustness.min().min()
    def color(val):
        return 'color: white' if val <.9 else 'color: black'

    def cell_color(val):
        if val>.9:
            return 'background-color: None'
        else:
            cm =sns.color_palette('Reds', n_colors=100)[::-1]
            color = to_hex(cm[int((val-min_robustness)*50)])
            return 'background-color: %s' % color

    return EFA_robustness.style \
        .applymap(color) \
        .applymap(cell_color) \
        .set_properties(**{'font-size': '12pt', 'border-color': 'white'}) \
            .set_precision(3)

# helper plotting function
def plot_bootstrap_results(boot_stats):
    mean_loading = boot_stats['means']
    std_loading = boot_stats['sds']
    coef_of_variation = std_loading/mean_loading
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.plot(mean_loading.values.flatten(), coef_of_variation.values.flatten(), 'o')
    ax1.set_xlabel('Mean Loading')
    ax1.set_ylabel('Coefficient of Variation')
    ax1.set_ylim([-1,1])
    ax1.grid()

    ax2.plot(mean_loading.values.flatten(), std_loading.values.flatten(),'o')
    ax2.set_xlabel('Mean Loading')
    ax2.set_ylabel('Standard Deviation of Loading')

    plt.subplots_adjust(wspace=.5)
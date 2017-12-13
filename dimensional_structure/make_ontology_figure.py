from cross_results_plots import plot_corr_hist
from HCA_plots import plot_dendrograms
import glob
import os
from utils import load_results
import seaborn as sns
import svgutils.transform as sg

# load data
datafile = 'Complete_12-05-2017'
results = load_results(datafile)

# make histogram plot
colors = [sns.color_palette("Paired")[i] for i in [0, 1, 4]]
f = plot_corr_hist(results,colors, reps=2)

# Create and save dendorgram plots temporarily
scores = {}
for title, subset in [('Behavioral Tasks', 'task'), ('Self-Report Surveys', 'survey')]:
    r = results[subset]
    c = r.EFA.get_metric_cs()['c_metric-BIC']
    inp = 'EFA%s' % c
    plot_dendrograms(r, c, display_labels=False, inp=inp, titles=[title],
                     figsize=(12,8), ext='svg',  plot_dir='/tmp/')
    # get scores
    scores[subset] = r.EFA.get_scores(c)
dendrograms = glob.glob('/tmp/*dendrogram.svg')

# ***************************************************************************
# Ontology Figure
# ***************************************************************************


#plot_clusterings(results['all'], show_clusters=False, figsize=(10,10),
#                 plot_dir='/tmp/', ext='svg')


# combine them into one SVG file
# load matpotlib-generated figures
fig1 = sg.fromfile(dendrograms[0])
fig2 = sg.fromfile(dendrograms[1])
fig3 = sg.from_mpl(f, {})
#fig4 = sg.fromfile('/tmp/clustering_input-data.svg')
#create new SVG figure
# set height and width based on constituent plots
size1 = [int(i[:-2]) for i in fig1.get_size()]
size2 = [int(i[:-2]) for i in fig2.get_size()]

width1 = max([size1[0], size2[0]]) 
width2 = int(fig3.get_size()[0])
wpad = (width1 + width2)*.02
width = width1 + width2 + wpad

height1 = size1[1]
height2 = size2[1]
hpad = (height1+height2)*.02
height = height1 + height2 + hpad


# create svg fig
fig = sg.SVGFigure(width, height)
fig.root.set("viewbox", "0 0 %s %s" % (width, height))

# get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot2.moveto(0, height1+hpad)
plot3 = fig3.getroot()
plot3.moveto(width1+wpad, 0)
#plot4 = fig4.getroot()
#plot4.moveto(int(fig3.get_size()[0]) + wpad, height1+hpad)

# add text labels
txt1 = sg.TextElement(25,30, "A", size=30, weight="bold")
txt2 = sg.TextElement(25, 30+hpad+height1, "B", size=30, weight="bold")
txt3 = sg.TextElement(width1+wpad+25, 30, "C", size=30, weight="bold")
txt4 = sg.TextElement(width1+wpad+25, 
                      30+hpad+height1, "D", size=30, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2, plot3])
fig.append([txt1, txt2, txt3, txt4])

# save generated SVG files
fig.save("/home/ian/tmp/fig_final.svg")

for file in dendrograms:
    os.remove(file)
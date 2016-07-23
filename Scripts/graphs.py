"""
Network analysis
"""
from colorsys import hsv_to_rgb

def gen_color(h):
    # Source: http://www.goldennumber.net/color/
    golden_ratio = (1 + 5 ** 0.5) / 2
    h += golden_ratio
    h %= 1
    return '#%02x%02x%02x' % tuple(int(a*100) for a in hsv_to_rgb(h, 0.55, 2.3))


# Graph Analysis 

import networkx as nx
import numpy as np

G_wgt = nx.Graph()
G_bin = nx.Graph()

DVs = subset
heatmap = DVs.corr()
color_vals = {}
for i,exp in enumerate(heatmap.columns):
    color_vals[exp.split('.')[0]] = gen_color(i/float(len(heatmap)))

for DV in heatmap.columns:
    G_wgt.add_node(DV)
    G_bin.add_node(DV)
    
for i,DV in enumerate(heatmap.columns):
    for j in range(i+1,len(heatmap)):
        target = heatmap[DV].index[j]
        weight = heatmap[DV][j]
        if weight > 0:
            color = 'r'
        else:
            color = 'b'
        # color = color_vals[DV.split('.')[0]]
        if abs(weight)>.15:
            G_wgt.add_edge(DV, target, color = color, weight = weight)
            G_bin.add_edge(DV, target,color = color)

# export
nx.write_graphml(G_wgt,'graph.graphml')

# analysis
between = nx.betweenness_centrality(G_bin).items()
np.argmax([i[1] for i in between])
index=np.argsort([i[1] for i in between])
sort_centrality = [between[i] for i in index]

# visualization
import matplotlib.pyplot as plt

G = G_wgt
nodes = G.degree().keys()
node_size = np.array(G.degree().values())*10
node_colors = [color_vals[x.split('.')[0]] for x  in nodes]
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]


pos = nx.spring_layout(G) #spring or circular

def onpick(event):
    (x,y)   = (event.xdata, event.ydata)
    for node in G.nodes():            
        p = pos[node]
        distance = pow(x-p[0],2)+pow(y-p[1],2)
        if distance < 0.001:
            print 'node:', node, distance
    
fig = plt.figure(figsize = (14,14))
fig.canvas.mpl_connect('button_press_event', onpick)
nx.draw(G, pos = pos, edges = edges, edge_color = colors, 
        nodelist = nodes,  node_size = node_size, node_color = node_colors)





"""
Functions to create a chord grpah
"""

import numpy as np
from math import pi, cos, sin
 
def draw_circle_points(points, radius=1, centerX=0, centerY=0):
    part = 2 * pi / points
    final_points = []
 
    for point in range(points):
        angle = part * point;
        newX = centerX + radius * cos(angle)
        newY = centerY + radius * sin(angle)
        final_points.append((newX, newY))
 
    return final_points
     
def connectome_lines(connectome, hardness=.05):
    number_of_areas = connectome.shape[0]
    positions = draw_circle_points(number_of_areas)
 
    final_positions = []
    area_name = []
 
    for index, area1 in enumerate(connectome):
        commonXY = positions[index]
        for index2, area2 in enumerate(area1):
            newXY = positions[index2]
            if abs(area2) > hardness:
                final_positions.append([[commonXY[0], newXY[0]], [commonXY[1], newXY[1]]])
 
    return final_positions
 
#simulation data
connectome = heatmap.as_matrix()
connectome[connectome==1]=0

lines = np.array(connectome_lines(connectome)).T

from bokeh.plotting import figure
from bokeh.models import Range1d, ColumnDataSource
 
# The Data
beziers = ColumnDataSource({
            'x0' : lines[0][0],
            'y0' : lines[0][1],
            'x1' : lines[1][0],
            'y1' : lines[1][1],
            'cx0' : lines[0][0]/1.5,
            'cy0' : lines[0][1]/1.5,
            'cx1' : lines[1][0]/1.5,
            'cy1' : lines[1][1]/1.5
        })
 
dots = ColumnDataSource(
        data=dict(
            x=lines[0][0],
            y=lines[0][1]
        )
    )
    
# The Plot
TOOLS = "box_select"
p2 = figure(tools = TOOLS, title="Connectomme", toolbar_location="below")
p2.x_range = Range1d(-1.1, 1.1)
p2.y_range = Range1d(-1.1, 1.1)
# The Glyphs
p2.bezier('x0', 'y0', 'x1', 'y1', 'cx0', 'cy0', 'cx1', 'cy1',
          source=beziers,
          line_cap='round',
          line_width=connectome.flatten()) # Add the width

p2.circle('x', 'y', size=8, fill_color="#6D6A75", line_color=None, source=dots)

save(p2,'tmp.plot')


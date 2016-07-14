import matplotlib.pyplot as plt
import numpy as np

def heatmap(mat, filename = 'heatmap.png', labels = False, title = False, value_range = False):
    ''' Plots a heatmap from a matrix and labels = tuple(xlabels, ylabels)
        where xlabels, ylabels are lists '''

    fig,ax = plt.subplots()
    if value_range:
        heatmap = ax.pcolormesh(mat,  vmin=value_range[0], vmax=value_range[1])#,cmap=plt.cm.PRGn)
    else:
        heatmap = ax.pcolormesh(mat)#,cmap=plt.cm.PRGn)
    # want a more natural, table-like display
    cbar = plt.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(mat.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(mat.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xbound(0,mat.shape[1])
    ax.set_ybound(0,mat.shape[0])
    
    if labels:
        ax.set_xticklabels(labels[0], minor=False)
        ax.set_yticklabels(labels[1], minor=False)
        plt.xticks(rotation=90)
        try:
            if len(labels[0][0]) > 1:
                plt.xticks(rotation=-90)
        except (TypeError,IndexError):
            pass
    if title:
        plt.title(title,y=1.08)
    ax.grid(False)
    plt.savefig(filename)

def bar_graph(vector, xaxis, yaxis, title = None):
    fig, ax = plt.subplots()
    rects = ax.bar(np.arange(len(vector)),vector,color='g')
    # add some text for labels, title and axes ticks
    ax.set_ylabel(yaxis)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(vector)) + 0.5)
    ax.set_xticklabels(xaxis)
    ax.set_xlim([0,len(vector)])
    plt.xticks(rotation=90)
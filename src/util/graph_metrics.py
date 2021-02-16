import pandas as pd
import matplotlib.pyplot as plt

# Graphs multiple CloudWatch metric data frames as subplots
def graph_metrics(dfs, xlim=None, ylims=[], ylabels=[], titles=[], figtext=None, suptitle=None):
    fig, axs = plt.subplots(len(dfs), figsize = (30,10), sharex=True)
    fig.set_figheight(15)
    if (suptitle is not None):
        fig.suptitle(suptitle, fontsize=30)
    if (xlim):
        plt.xlim(xlim)
    for i, df in enumerate(dfs):
        axs[i].plot(df["Timestamps"], df["Values"])
        axs[i].tick_params(axis='both', labelsize=20, length=10, width=2)
        if (ylims is not None):
            if (ylims[i] is not None):
                axs[i].set_ylim(ylims[i])
            
        if (ylabels[i] is not None):
            if (ylabels[i] is not None):
                axs[i].set_ylabel(ylabels[i], fontsize=30)
            
        if (titles[i] is not None):
            if (titles[i] is not None):
                axs[i].set_title(titles[i], fontsize=30)
    plt.show()
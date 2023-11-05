import json

import pandas
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

def plot_tradeoffs(json_file, output_file):
    matplotlib.rcParams['figure.dpi'] = 500
    
    with open(json_file) as f:
        js = json.load(f)

    columns=[ 'pr', 'det_scale', 'tra_scale', "stride", 'time(avg)', "mAP" ]
    df = []
    for key in js:
        v = key.split("_")
        df.append([ int(v[1]), int(v[3]), int(v[5]), int(v[7]), js[key]["average_time"], js[key]["results"]["AP"]*100 ])
    df = pandas.DataFrame(np.array(df), columns=columns)

    sns_plot = sns.scatterplot(data=df, x="time(avg)", y="mAP", 
                                hue="tra_scale", size="det_scale",  style="stride",
                                palette="deep")
    # sns_plot = sns.scatterplot(data=df, x="time(avg)", y="mAP", 
    #                             hue="tra_scale", size="det_scale", style="stride")
    plt.axvline(33, 0, 23)

    ax=plt.subplot()
    ax.set_ylabel("mAP", fontsize=18)
    ax.set_xlabel("Time (avg)", fontsize=18)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # ax = plt.gca()
    # ax.set_xscale('log')
    plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)
    
    plt.tight_layout()
    fig = sns_plot.get_figure()
    fig.savefig(output_file)

    # sns_plot.savefig(output_file)

SPINE_COLOR = 'gray'
def latexify(fig_width=1.87, fig_height=1.4, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
           'axes.labelsize': 6, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'figure.titlesize': 8, # was 10
              'figure.dpi': 300,
              'legend.fontsize': 5, # was 10
              'xtick.labelsize': 6,
              'ytick.labelsize': 6,
              'axes.labelpad': 1,
            #   'figure.figsize': [fig_width,fig_height],
            #   'font.family': 'calibri',
              'savefig.pad_inches':0,
              'figure.constrained_layout.h_pad':0,
              'figure.constrained_layout.w_pad':0,
              'axes.xmargin': 0,
              'axes.ymargin': 0,
              'xtick.major.pad':0.2,
              'ytick.major.pad':0.2,
              'pdf.fonttype': 42,
              'ps.fonttype' : 42
            #   'legend.fontsize': 4,
            #   'legend.borderaxespad': 2
            #   'legend.labelspacing':
            #   'legend.loc':
    }

    matplotlib.rcParams.update(params)


if __name__ == "__main__":
    # latexify(None, None, 2)
    plot_tradeoffs(
        "tradeoff_results_argoverse_faster_rcnn_tracktor_faster_rcnn.json", 
        "plot_tradeoff_results_argoverse_faster_rcnn_tracktor_faster_rcnn.png"
    )
    # plot_tradeoffs(
    #     "tradeoff_results_argoverse_fcos_tracktor_faster_rcnn.json", 
    #     "plot_tradeoff_results_argoverse_fcos_tracktor_faster_rcnn.png"
    # )
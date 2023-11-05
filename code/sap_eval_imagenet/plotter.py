import numpy as np
num_heatmap = 9
def fun()
    y_axis = ["Testing Noise " + str(((i)/2)) for i in range(9)]
    x_axis = ["Training Noise " + str(((i)/2)) for i in range(9)]
    print(len(y_axis))


    plt.rcParams['figure.figsize'] = [10, 10]

    fig, ax = plt.subplots()


    import matplotlib
    import matplotlib.pyplot as plt

    avg_heatmap = np.array(avg_heatmap)
    print(avg_heatmap.shape)
    im = ax.imshow(avg_heatmap)
    # ax = sns.heatmap(avg_heatmap)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_yticks(np.arange(len(y_axis)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_axis, fontsize=15)
    ax.set_yticklabels(y_axis, fontsize=15)



    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            text = ax.text(j, i, avg_heatmap[i, j],
                        ha="center", va="center", color="w", fontsize=15)

    # ax.set_title("Heatmap of savings", fontsize=20)

    plt.figure(num=None, figsize=(20,20), dpi=300, facecolor='w', edgecolor='k')

    #plt.rcParams['figure.figsize'] = [20, 20]

    plt.save('heatmap.png')
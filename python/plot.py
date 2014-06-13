import sys
import cPickle as pickle
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


def barch(data):
    '''BARCHARTS'''
    err = defaultdict(dict)
    time = defaultdict(dict)

    for method_and_subsample, all_rms_and_time in data.iteritems():
        method, subsample = method_and_subsample
        all_rms, timee = all_rms_and_time
        # Save data in another way
        err[method][subsample] = sum(all_rms)
        time[method][subsample] = timee

    # Assume subsamples are equal for all methods
    fig = plt.figure()
    ax_err = fig.add_subplot(121)
    ax_time = fig.add_subplot(122)

    X = sorted(list(set(s for m, s in data.keys())))
    during = 'merge_during'
    after = 'merge_after'

    err_during = [err[during][s] if s in err[during] else 0 for s in X]
    err_after = [err[after][s] if s in err[after] else 0 for s in X]

    plot_bars(ax_err, X,
              ((err_after, "merge_after"), (err_during, "merge_during")),
              title="Root mean squared error", xlabel="Subsampling factor",
              ylabel="Root mean squared error")

    time_during = [time[during][s] if s in time[during] else 0 for s in X]
    time_after = [time[after][s] if s in time[after] else 0 for s in X]
    plot_bars(ax_time, X,
              ((time_after, "merge_after"), (time_during, "merge_during")),
              title="Time taken", xlabel="Subsampling factor",
              ylabel="Time (seconds)")


def plot_bars(ax, X, args, title="", xlabel="", ylabel=""):
    ''''''
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    width = 0.9 / len(args)
    ind = np.arange(len(X))

    names, rects = [], []
    colors = ['r', 'y', 'b']
    max_Y = 0
    for i, (Y, color) in enumerate(zip(args, colors)):
        Y, Y_name = Y
        rect = ax.bar(ind + 0.05 + i * width, Y, width, color=color)
        rects.append(rect)
        names.append(Y_name)

        max_Y = max(max_Y, max(Y))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ind + 0.5)
    ax.set_xticklabels([str(x) for x in X])
    ax.legend([r for r in rects], names)
    ax.set_ylim([0, max_Y * 1.1])

    return ax


if __name__ == "__main__":
    barch(pickle.load(open(sys.argv[1], 'rb')))
    plt.show()

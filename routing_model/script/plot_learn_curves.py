import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("result_path",
            help="Path to .csv file containing results and stats gathered" \
            + "while training MARDAM (train.py default to: <outdir>/loss_gap.csv)")
    parser.add_argument("--output-path", "-o", default=None,
            help="Path to the pdf file the plot will be exported to")
    parser.add_argument("--font-size", default=20,
            help="Reference size of fonts for all title/label on the figure")
    return parser.parse_args()

def smooth_and_subsample(y, x=None, win_size=20, skip=10):
    win = np.full((win_size,), 1 / win_size)
    padded_y = np.pad(y, (win_size-1, 0), mode='edge')
    smoothed_y = np.convolve(padded_y, win, mode='valid')
    if x is None:
        return smoothed_y[::skip]
    else:
        return x[::skip], smoothed_y[::skip]

def main(args):
    mpl.rc('font', size=args.font_size)
    with open(args.result_path) as f:
        header = next(f)
        Result = namedtuple("Result", [col.strip(" \n#").lower() for col in header.split() if col])
        results = []
        for row in f:
            results.append(Result(*(float(val) for val in row.split() if val)))
    results = Result(*(np.array(res) for res in zip(*results)))
    
    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(nrows=3, ncols=4)

    ax = fig.add_subplot(grid[:,:3])
#    ax.fill_between(results.ep, -results.test_mu - results.test_std,
#            -results.test_mu + results.test_std, color='g', alpha=0.5)
    ax.axvline(15, color='k', linestyle=':')
    ax.plot(results.ep, results.val, label="observed on train data", color='r')
    ax.plot(results.ep, results.bl, label="estimated by critic on train data", color='b')
    ax.plot(results.ep, -results.test_mu, label="observed on test data", color='g')
    ax.legend(loc="lower right")
    ax.set_ylabel("Mean cumulated reward")
    ax.set_xlabel("Training epoch")

    ax = fig.add_subplot(grid[0, 3])
    ax.axvline(15, color='k', linestyle=':')
    ax.plot(results.ep, results.prob)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Routes prob.")

    ax = fig.add_subplot(grid[1, 3])
    ax.axvline(15, color='k', linestyle=':')
    ax.plot(results.ep, results.loss)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("AC loss")

    ax = fig.add_subplot(grid[2, 3])
    ax.axvline(15, color='k', linestyle=':')
    ax.plot(results.ep, results.norm)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Grad. norm")
    ax.set_xlabel("Train. epoch")

    fig.set_size_inches(16,9)

    if args.output_path is None:
        args.output_path = args.result_path.replace(".csv", ".pdf")
    fig.savefig(args.output_path)
    plt.show()


if __name__ == "__main__":
    main(parse_args())

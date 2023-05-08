import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def simple_plot(x_vals, y_vals, xlabel, ylabel, title, do_plot=True):
    sns.set_style("darkgrid")
    plt.plot(x_vals, y_vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if do_plot:
        plt.show()

def plot_rew_with_gaussian(range, desired_point, k):
    x_vals = np.arange(range[0], range[1], 0.01)
    y_vals = np.exp(-k * np.power(desired_point - x_vals, 2))

    two_superscript = '\u00b2'
    two_subscript = '\u2082'
    ylabel = "exp(-k(||d-x||" + two_subscript + ")" + two_superscript +"),"+f" k={k}"
    simple_plot(x_vals, y_vals, "x", ylabel , "euclidean distance with gaussian kernel", do_plot=False)
    plt.axvline(x=desired_point, color="black")
    plt.text(desired_point+0.01, -0.01, s=f"d={desired_point}")
    #plt.set_xticks([])
    #plt.xticks([])

    plt.show()

if __name__ == "__main__":
    plot_rew_with_gaussian([0, 3], 1.2, 4)

# -*- coding: utf-8 -*-


from functools import partial
import warnings

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolor
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from matplotlib.ticker import StrMethodFormatter, FuncFormatter, ScalarFormatter
from matplotlib import cm, colors
from cycler import cycler

DEFAULT_FONTSIZE = 10


def get_widths_balancer(widths, minmax_ratio=0.02):
    """
    Given a list of positive numbers, find a linear function, such that when applied to the numbers, the maximum value
    remains the same, and the minimum value is minmax_ratio times the maximum value.
    :param widths: list of numbers
    :param minmax_ratio: the desired max / min ratio in the transformed list.
    :return: a linear function with one float argument that has the above property
    """
    if not 0 <= minmax_ratio <= 1:
        raise ValueError("minmax_ratio must be between 0 and 1")
    max_width = max(widths)
    min_width = min(widths)
    if 1.0 * min_width / max_width >= minmax_ratio:
        slope = 1
        intercept = 0
    else:
        slope = max_width * (1.0 - minmax_ratio) / (max_width - min_width)
        intercept = (
            max_width * (max_width * minmax_ratio - min_width) / (max_width - min_width)
        )

    def balancer(width):
        return slope * width + intercept

    return balancer


def get_column_widths(
    equal_col_width, widths_minmax_ratio, min_width_for_annotation, chunk_sizes
):
    """
    Get the column widths based on the provided parameters.
    :param equal_col_width: If True, all columns will have equal width.
    :param widths_minmax_ratio: Preset ratio of the maximum to minimum width for the columns.
    :param min_width_for_annotation: The minimum width for the annotation of columns.
    :param chunk_sizes: A list of sizes for each chunk in the Venn diagram.
    :return: A tuple containing the column widths and the effective minimum width for annotations.
    """

    # Make all columns equal width and display all annotations
    if equal_col_width:
        col_widths = [1] * len(chunk_sizes)
        effective_min_width_for_annotation = 1
    else:
        # Predefined ratio of the maximum to minimum width for the columns
        if widths_minmax_ratio is not None:
            widths_balancer = get_widths_balancer(chunk_sizes, widths_minmax_ratio)
            col_widths = [widths_balancer(chunk_size) for chunk_size in chunk_sizes]
            effective_min_width_for_annotation = widths_balancer(
                min_width_for_annotation
            )
        # If no ratio is provided, use the true chunk sizes
        else:
            col_widths = chunk_sizes
            effective_min_width_for_annotation = min_width_for_annotation

    return col_widths, effective_min_width_for_annotation


def get_ax_size_inches(fig, ax):
    # return the size of the axes in inches
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    return width, height


def rescale_axes():
    # Scaling factor to make the cells square
    xvals, yvals = plt.gca().axes.get_xlim(), plt.gca().axes.get_ylim()
    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    ratio = yrange / xrange

    fig = plt.gcf()
    curr_size = fig.get_size_inches()
    fig.set_size_inches(
        curr_size[0],
        (curr_size[0]) * ratio,
    )


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def remove_spines(ax, spines=None):
    ax.set_axis_off()

    if spines is not None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)


def ticks_format(value, index, minor=False):
    """
    get the value and returns the value as:
    integer: [0,99]
    1 digit float: [0.1, 0.99]
    n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    if minor:
        return ""

    exp = np.floor(np.log10(value))
    base = value / 10**exp
    if exp >= 0 or exp < 1:
        return "${0:d}$".format(int(value))
    if exp == -1:
        return "${0:.1f}$".format(value)
    else:
        return "${0:d}\\times10^{{{1:d}}}$".format(int(base), int(exp))


def colorFader(c1, c2, mix=0):
    """
    fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """
    c1 = np.array(mplcolor.to_rgb(c1))
    c2 = np.array(mplcolor.to_rgb(c2))
    return mplcolor.to_hex((1 - mix) * c1 + mix * c2)


def ucla_colorgradient(n=10):
    # Hardcode UCLA blue gradient
    c1 = "#DAEBFE"
    c2 = "#2774AE"
    c3 = "#005587"

    if n < 2:
        return [c1]
    elif n < 3:
        return [c1, c2]
    elif n < 4:
        return [c1, c2, c3]

    map = []
    for i in range(n // 2 + n % 2):
        map.append(colorFader(c1, c2, i / (n // 2 + n % 2)))

    for i in range(n // 2):
        map.append(colorFader(c2, c3, i / (n // 2 - 1)))

    return map


def get_mpl_cmap(cmap_name, n):
    mpl_cmap = cm.get_cmap(cmap_name, n)
    cmap = [rgb2hex(mpl_cmap(i)) for i in range(mpl_cmap.N)]
    return cmap


def get_cmap_list(cmap_name, composition_array, chunks, log_color=False):
    # sum along axis 0 to get number of sets for each chunk
    sets_per_chunk = composition_array.sum(0)

    # This is to ignore chunks with elements that don't belong to any set (remainder in universe)
    relevant_chunk_sizes = []
    for i, chunk in enumerate(chunks):
        if sets_per_chunk[i] > 0:
            relevant_chunk_sizes.append(len(chunk))
        else:
            # Set these to zero so they get default color
            relevant_chunk_sizes.append(0)

    original_chunk_sizes = np.array(relevant_chunk_sizes)
    if log_color:
        relevant_chunk_sizes = np.empty_like(original_chunk_sizes)
        mask = original_chunk_sizes > 0
        relevant_chunk_sizes[mask] = np.round(
            np.log2(original_chunk_sizes[mask]) + 1, 0
        ).astype(int)
        relevant_chunk_sizes[~mask] = 0

    if cmap_name is None or cmap_name == "ucla":
        cmap = ucla_colorgradient(n=max(relevant_chunk_sizes))
    else:
        cmap = get_mpl_cmap(cmap_name, max(relevant_chunk_sizes))

    color_cycle = [cmap[chunk_size - 1] for chunk_size in relevant_chunk_sizes]
    return color_cycle, get_cbar_mappable(
        cmap, original_chunk_sizes, log_color=log_color
    )


def get_cmap_grid(cmap_name, composition_array):
    # For odds ratios
    original_composition_array = composition_array

    composition_array = np.round(
        composition_array / np.max(composition_array) * 100
    ).astype(int)

    if cmap_name is None or cmap_name == "ucla":
        cmap = ucla_colorgradient(n=int(np.ceil(np.max(composition_array) + 1)))
    else:
        mpl_cmap = cm.get_cmap(cmap_name, int(np.ceil(np.max(composition_array) + 1)))
        cmap = [rgb2hex(mpl_cmap(i)) for i in range(mpl_cmap.N)]

    color_cycle = [
        [None for x in range(composition_array.shape[1])]
        for y in range(composition_array.shape[0])
    ]

    for i in range(composition_array.shape[0]):
        for j in range(composition_array.shape[1]):
            color_cycle[i][j] = cmap[composition_array[i][j]]

    return color_cycle, get_cbar_mappable(cmap, np.max(original_composition_array))


def get_cbar_mappable(cmap, original_chunk_sizes, log_color=False):
    cmap = LinearSegmentedColormap.from_list("cmap", cmap, N=256)

    if log_color:
        norm = colors.LogNorm(1, np.max(original_chunk_sizes))
    else:
        norm = colors.Normalize(1, np.max(original_chunk_sizes))
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def setup_axes(height_ratios=[1.5, 4.5], width_ratios=[5.5, 1.5, 1]):
    """
    Set up axes for plot and return them in a dictionary.
    Params
    ------

    Returns
    -------
    axes (dict): dict with string as keys and axes as values
    """

    # Define and optionally create the encasing axis for plot according to arguments

    supervenn_ax = plt.gca()

    remove_spines(supervenn_ax)

    fig = supervenn_ax.get_figure()
    get_gridspec = partial(
        gridspec.GridSpecFromSubplotSpec,
        subplot_spec=supervenn_ax.get_subplotspec(),
        hspace=0,
        wspace=0,
    )

    gs = get_gridspec(
        len(height_ratios),
        len(width_ratios),
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )
    axes = {}
    for i in range(len(height_ratios)):
        for j in range(len(width_ratios)):

            axes[(i, j)] = fig.add_subplot(gs[i, j])

    # Remove tick from every axis, and set ticks length to 0 (we'll add ticks to the side plots manually later)
    for ax in axes.values():
        remove_ticks(ax)
        ax.tick_params(which="major", length=0)

    # if side plots are used, supervenn_ax isn't included in axes dict
    remove_ticks(supervenn_ax)

    return axes

# -*- coding: utf-8 -*-
"""
Routines for plotting multiple sets.
"""

from functools import partial
import warnings

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolor
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter, FuncFormatter, ScalarFormatter
from matplotlib import cm, colors

from supervenn._algorithms import (
    break_into_chunks,
    get_chunks_and_composition_array,
    get_permutations,
    DEFAULT_MAX_BRUTEFORCE_SIZE,
    DEFAULT_SEEDS,
    DEFAULT_NOISE_PROB,
)


DEFAULT_FONTSIZE = 12


class SupervennPlot(object):
    """
    Attributes
    ----------
    axes: `dict
        a dict containing all the plot's axes under descriptive keys: 'main', 'top_side_plot', 'right_side_plot',
        'unused' (the small empty square in the top right corner)
    figure: matplotlib.figure.Figure
        figure containing the plot.
    chunks: dict
        a dictionary allowing to get the contents of chunks. It has frozensets of key indices as keys and chunks
        as values.

    Methods
    -------
    get_chunk(set_indices)
        get a chunk by the indices of its defining sets without them to a frozenset first
    """

    def __init__(
        self,
        axes,
        figure,
        chunks_dict,
        chunks,
        composition_array,
        set_annotations,
        universe,
    ):
        self.axes = axes
        self.figure = figure
        self.chunks_dict = chunks_dict
        self.chunks = chunks
        self.composition_array = composition_array
        self.set_annotations = set_annotations
        self.universe_size = len(universe) if universe is not None else 1

    def get_chunk(self, set_indices):
        """
        Get the contents of a chunk defined by the sets indicated by sets_indices. Indices refer to the original sets
        order as it was passed to supervenn() function (any reordering of sets due to use of sets_ordering params is
        ignored).
        For example .get_chunk_by_set_indices([1,5]) will return the chunk containing all the items that are in
        subgroup_set_0 and sets[5], but not in any of the other sets.
        supervenn() function, the
        :param set_indices: iterable of integers, referring to positions in sets list, as passed into supervenn().
        :return: chunk with items, that are in each of the sets with indices from set_indices, but not in any of the
        other sets.
        """
        return self.chunks_dict[frozenset(set_indices)]


def get_alternated_ys(ys_count, low, high):
    """
    A helper function generating y-positions for x-axis annotations, useful when some annotations positioned along the
    x axis are too crowded.
    :param ys_count: integer from 1 to 3.
    :param low: lower bound of the area designated for annotations
    :param high: higher bound for thr area designated for annotations.
    :return:
    """
    if ys_count not in [1, 2, 3]:
        raise ValueError("Argument ys_count should be 1, 2 or 3.")
    if ys_count == 1:
        coefs = [0.5]
        vas = ["center"]
    elif ys_count == 2:
        coefs = [0.15, 0.85]
        vas = ["bottom", "top"]
    else:
        coefs = [0.15, 0.5, 0.85]
        vas = ["bottom", "center", "top"]

    ys = [low + coef * (high - low) for coef in coefs]

    return ys, vas


def plot_binary_array(
    arr,
    ax=None,
    col_widths=None,
    row_heights=None,
    min_width_for_annotation=1,
    row_annotations=None,
    row_annotations_y=0.5,
    row_annotations_x=0.5,
    row_annotations_ha="center",
    row_annotations_offset=0.5,
    col_annotations=None,
    col_annotations_area_height=0.75,
    col_annotations_ys_count=1,
    rotate_col_annotations=False,
    color_by="row",
    bar_height=1,
    bar_alpha=0.6,
    bar_align="edge",
    color_cycle=None,
    alternating_background=False,
    fontsize=DEFAULT_FONTSIZE,
):
    """
    Visualize a binary array as a grid with variable sized columns and rows, where cells with 1 are filled using bars
    and cells with 0 are blank.
    :param arr: numpy.array of zeros and ones
    :param ax: axis to plot into (current axis by default)
    :param col_widths: widths for grid columns, must have len equal to arr.shape[1]
    :param row_heights: heights for grid rows, must have len equal to arr.shape[0]
    :param min_width_for_annotation: don't annotate column with its size if size is less than this value (default 1)
    :param row_annotations: annotations for each row, plotted in the middle of the row
    :param row_annotations_y: a number in (0, 1), position for row annotations in the row. Default 0.5 - center of row.
    :param row_annotations_x: a number in (0, 1), position for row annotations in the full width. Default 0.5 - center of total columns.
    :param row_annotations_offset: Add translation to row annotation. Default 0 - no translation.
    :param row_annotations_ha: horizontal alightment (center, left, or right). Default center
    :param col_annotations: annotations for columns, plotted in the bottom, below the x axis.
    :param col_annotations_area_height: height of area for column annotations in inches, 1 by default
    :param col_annotations_ys_count: 1 (default), 2, or 3 - use to reduce clutter in column annotations area
    :param rotate_col_annotations: True / False
    :param color_by: 'row' (default) or 'column'. If 'row', all cells in same row are same color, etc.
    :param bar_height: height of cell fill as a fraction of row height, a number in (0, 1).
    :param bar_alpha: alpha for cell fills.
    :param bar_align: vertical alignment of bars, 'edge' (defaulr) or 'center'. Only matters when bar_height < 1.
    :param color_cycle: a list of colors, given as names of matplotlib named colors, or hex codes (e.g. '#1f77b4')
    :param alternating_background: True / False (default) - give avery second row a slight grey tint
    :param fontsize: font size for annotations (default {}).
    """.format(
        DEFAULT_FONTSIZE
    )
    if row_heights is None:
        row_heights = [1] * arr.shape[0]

    if col_widths is None:
        col_widths = [1] * arr.shape[1]

    if len(row_heights) != arr.shape[0]:
        raise ValueError("len(row_heights) doesnt match number of rows of array")

    if len(col_widths) != arr.shape[1]:
        raise ValueError("len(col_widths) doesnt match number of columns of array")

    allowed_argument_values = {
        "bar_align": ["center", "edge"],
        "color_by": ["row", "column"],
        "col_annotations_ys_count": [1, 2, 3],
    }

    for argument_name, allowed_argument_values in allowed_argument_values.items():
        if locals()[argument_name] not in allowed_argument_values:
            raise ValueError(
                "Argument {} should be one of {}".format(
                    argument_name, allowed_argument_values
                )
            )

    if not 0 <= row_annotations_y <= 1:
        raise ValueError("row_annotations_y should be a number between 0 and 1")

    if color_cycle is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    grid_xs = np.insert(np.cumsum(col_widths), 0, 0)[:-1]
    grid_ys = np.insert(np.cumsum(row_heights), 0, 0)[:-1]

    if ax is not None:
        plt.sca(ax)

    # CELLS
    for row_index, (row, grid_y, row_height) in enumerate(
        zip(arr, grid_ys, row_heights)
    ):

        bar_y = grid_y + 0.5 * row_height if bar_align == "center" else grid_y

        # alternating background
        if alternating_background and row_index % 2:
            plt.barh(
                y=bar_y,
                left=0,
                width=sum(col_widths),
                height=bar_height * row_height,
                align=bar_align,
                color="grey",
                alpha=0.15,
            )

        for col_index, (is_filled, grid_x, col_width) in enumerate(
            zip(row, grid_xs, col_widths)
        ):
            if is_filled:
                color_index = row_index if color_by == "row" else col_index

                if len(np.array(color_cycle).shape) == 1:
                    color = color_cycle[color_index % len(color_cycle)]
                else:
                    color = color_cycle[row_index][col_index]

                plt.barh(
                    y=bar_y,
                    left=grid_x,
                    width=col_width,
                    height=bar_height * row_height,
                    align=bar_align,
                    color=color,
                    alpha=bar_alpha,
                )

    # ROW ANNOTATIONS
    if row_annotations is not None:
        for row_index, (grid_y, row_height, annotation) in enumerate(
            zip(grid_ys, row_heights, row_annotations)
        ):
            annot_y = grid_y + row_annotations_y * row_height
            plt.annotate(
                str(annotation),
                xy=(row_annotations_x * sum(col_widths), annot_y),
                xytext=(
                    row_annotations_x * sum(col_widths)
                    + row_annotations_offset * sum(col_widths),
                    annot_y,
                ),
                ha=row_annotations_ha,
                va="center",
                fontsize=fontsize,
            )

    # COL ANNOTATIONS
    min_y = 0
    if col_annotations is not None:

        min_y = (
            -1.0
            * col_annotations_area_height
            / plt.gcf().get_size_inches()[1]
            * arr.shape[0]
        )
        plt.axhline(0, c="k")

        annot_ys, vas = get_alternated_ys(col_annotations_ys_count, min_y, 0)

        for col_index, (grid_x, col_width, annotation) in enumerate(
            zip(grid_xs, col_widths, col_annotations)
        ):
            annot_y = annot_ys[col_index % len(annot_ys)]
            if col_width >= min_width_for_annotation:
                plt.annotate(
                    str(annotation),
                    xy=(grid_x + col_width * 0.5, annot_y),
                    ha="center",
                    va=vas[col_index % len(vas)],
                    fontsize=fontsize,
                    rotation=90 * rotate_col_annotations,
                )
    else:
        min_y = (
            -0.01
            * col_annotations_area_height
            / plt.gcf().get_size_inches()[1]
            * arr.shape[0]
        )
        plt.axhline(0, c="k")

    plt.xlim(0, sum(col_widths))
    plt.ylim(min_y, sum(row_heights))
    plt.xticks(grid_xs, [])
    plt.yticks(grid_ys, [])
    plt.grid(True)


def side_plot(
    values,
    widths,
    orient,
    fontsize=DEFAULT_FONTSIZE,
    min_width_for_annotation=1,
    rotate_annotations=False,
    color="tab:gray",
):
    """
    Barplot with multiple bars of variable width right next to each other, with an option to rotate the plot 90 degrees.
    :param values: the values to be plotted.
    :param widths: Widths of bars
    :param orient: 'h' / 'horizontal' (default) or 'v' / 'vertical'
    :param fontsize: font size for annotations
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter. Default 1 - annotate all.)
    :param rotate_annotations: True/False, whether to print annotations vertically instead of horizontally
    :param color: color of bars, default 'tab:gray'
    """
    bar_edges = np.insert(np.cumsum(widths), 0, 0)
    annotation_positions = [
        0.5 * (begin + end) for begin, end in zip(bar_edges[:-1], bar_edges[1:])
    ]
    max_value = max(values)
    if orient in ["h", "horizontal"]:
        horizontal = True
        plt.bar(
            x=bar_edges[:-1],
            height=values,
            width=widths,
            align="edge",
            alpha=0.5,
            color=color,
        )
        ticks = plt.xticks
        lim = plt.ylim
    elif orient in ["v", "vertical"]:
        horizontal = False
        plt.barh(
            y=bar_edges[:-1],
            width=values,
            height=widths,
            align="edge",
            alpha=0.5,
            color=color,
        )
        ticks = plt.yticks
        lim = plt.xlim
    else:
        raise ValueError('Unknown orient: {} (should be "h" or "v")'.format(orient))

    for i, (annotation_position, value, width) in enumerate(
        zip(annotation_positions, values, widths)
    ):
        if width < min_width_for_annotation and horizontal:
            continue
        x, y = 0.05 * max_value, annotation_position
        ha = "left"
        va = "center"
        if horizontal:
            x, y = y, x
            ha = "center"
            va = "bottom"
        plt.annotate(
            value,
            xy=(x, y),
            ha=ha,
            va=va,
            rotation=rotate_annotations * 90,
            fontsize=fontsize,
        )

    ticks(bar_edges, [])
    lim(0, max(values))
    plt.grid(True)


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


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def ticks_format(value, index):
    """
    get the value and returns the value as:
    integer: [0,99]
    1 digit float: [0.1, 0.99]
    n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value / 10**exp
    if exp >= 0:
        return "${0:d}$".format(int(value))
    if exp == -1:
        return "${0:.1f}$".format(value)
    else:
        return "${0:d}\\times10^{{{1:d}}}$".format(int(base), int(exp))


def setup_axes(
    side_plots,
    figsize=None,
    dpi=None,
    ax=None,
    side_top_plot_width=1.5,
    side_right_plot_width=1.5,
):
    """
    Set up axes for plot and return them in a dictionary. The dictionary may include the following keys:
    - 'main': always present
    - 'top_side_plot': present if side_plots = True, 'both' or 'top'
    - 'right_side_plot': present if side_plots = True, 'both' or 'right'
    - 'unused': present if side_plots = 'True' or 'both' (unused area in the top right corner)
    :param side_plots: True / False / 'top' / 'right'
    :param figsize: deprecated, will be removed in future versions
    :param dpi: deprecated, will be removed in future versions
    :param ax: optional encasing axis to plot into, default None - plot into current axis.
    :param side_plot_width: side plots width in inches, default 1.5
    :return: dict with string as keys and axes as values, as described above.
    """

    if side_plots not in (True, False, "top", "right"):
        raise ValueError("Incorrect value for side_plots: {}".format(side_plots))

    # Define and optionally create the encasing axis for plot according to arguments
    if ax is None:
        if figsize is not None or dpi is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        supervenn_ax = plt.gca()
    else:
        supervenn_ax = ax

    # if no side plots, there is only one axis
    if not side_plots:
        axes = {"main": supervenn_ax}

    # if side plots are used, break encasing axis into four smaller axis using matplotlib magic and store them in a dict
    else:
        bbox = supervenn_ax.get_window_extent().transformed(
            supervenn_ax.get_figure().dpi_scale_trans.inverted()
        )
        plot_width, plot_height = bbox.width, bbox.height
        width_ratios = [plot_width - side_right_plot_width, side_right_plot_width]
        height_ratios = [side_top_plot_width, plot_height - side_top_plot_width]
        fig = supervenn_ax.get_figure()
        get_gridspec = partial(
            gridspec.GridSpecFromSubplotSpec,
            subplot_spec=supervenn_ax.get_subplotspec(),
            hspace=0,
            wspace=0,
        )

        if side_plots == True:
            gs = get_gridspec(
                2, 2, height_ratios=height_ratios, width_ratios=width_ratios
            )

            axes = {
                "main": fig.add_subplot(gs[1, 0]),
                "top_side_plot": fig.add_subplot(gs[0, 0]),
                "unused": fig.add_subplot(gs[0, 1]),
                "right_side_plot": fig.add_subplot(gs[1, 1]),
            }

        elif side_plots == "top":
            gs = get_gridspec(2, 1, height_ratios=height_ratios)
            axes = {
                "main": fig.add_subplot(gs[1, 0]),
                "top_side_plot": fig.add_subplot(gs[0, 0]),
            }

        elif side_plots == "right":
            gs = get_gridspec(1, 2, width_ratios=width_ratios)
            axes = {
                "main": fig.add_subplot(gs[0, 0]),
                "right_side_plot": fig.add_subplot(gs[0, 1]),
            }

    # Remove tick from every axis, and set ticks length to 0 (we'll add ticks to the side plots manually later)
    for ax in axes.values():
        remove_ticks(ax)
        ax.tick_params(which="major", length=0)
    remove_ticks(
        supervenn_ax
    )  # if side plots are used, supervenn_ax isn't included in axes dict

    return axes


def colorFader(c1, c2, mix=0):
    """
    fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """
    c1 = np.array(mplcolor.to_rgb(c1))
    c2 = np.array(mplcolor.to_rgb(c2))
    return mplcolor.to_hex((1 - mix) * c1 + mix * c2)


def ucla_colorgradient(n=10):
    # Hardcode UCLA blue gradient
    c1 = "#8BB8E8"  #'#DAEBFE' #'#8BB8E8'
    c2 = "#2774AE"  #'#2774AE' #'#2774AE'
    c3 = "#005587"  #'#003B5C' #'#005587'

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


def supervenn(
    sets,
    set_annotations=None,
    figsize=None,
    side_plots=True,
    chunks_ordering="minimize gaps",
    sets_ordering=None,
    reverse_chunks_order=True,
    reverse_sets_order=True,
    universe=None,
    max_bruteforce_size=DEFAULT_MAX_BRUTEFORCE_SIZE,
    seeds=DEFAULT_SEEDS,
    noise_prob=DEFAULT_NOISE_PROB,
    side_top_plot_width=1,
    side_right_plot_width=1,
    min_width_for_annotation=1,
    widths_minmax_ratio=None,
    side_plot_color="gray",
    dpi=None,
    ax=None,
    xlabel="ITEMS",
    ylabel="SETS",
    widths_minmax_equate=True,
    auto_color=True,
    log_color=False,
    square_cell=False,
    side_top_plot_label="Model\nCount",
    side_right_plot_label="Total Errors\nPer Model",
    cbar_label="Number of Incorrectly\nClassified Patients",
    ticks_off=False,
    # col_annotations_area_height=0.75,
    **kw,
):
    """
    Plot a diagram visualizing relationship of multiple sets.
    :param sets: list of sets
    :param set_annotations: list of annotations for the sets
    :param figsize: figure size
    :param side_plots: True / False: add small barplots on top and on the right. On top, for each chunk it is shown,
    how many sets does this chunk lie inslde. On the right, set sizes are shown.
    :param chunks_ordering: method of ordering the chunks (columns of the grid)
        - 'minimize gaps' (default): use a smart algorithm to find an order of columns giving fewer gaps in each row,
            making the plot as readable as possible.
        - 'size': bigger chunks go first (or last if reverse_chunks_order=False)
        - 'occurence': chunks that are in most sets go first (or last if reverse_chunks_order=False)
        - 'random': randomly shuffle the columns
    :param sets_ordering: method of ordering the sets (rows of the grid)
        - None (default): keep the order as it is passed
        - 'minimize gaps': use a smart algorithm to find an order of rows giving fewer gaps in each column
        - 'size': bigger sets go first (or last if reverse_sets_order = False)
        - 'chunk count': sets that contain most chunks go first (or last if reverse_sets_order = False)
        - 'random': randomly shuffle
    :param reverse_chunks_order: True (default) / False when chunks_ordering is "size" or "occurence",
        chunks with bigger corresponding property go first if reverse_chunks_order=True, smaller go first if False.
    :param reverse_sets_order: True / False, works the same way as reverse_chunks_order
    :param universe: set of all possible elements
    :param max_bruteforce_size: maximal number of items for which bruteforce method is applied to find permutation
    :param seeds: number of different random seeds for the randomized greedy algorithm to find permutation
    :param noise_prob: probability of given element being equal to 1 in the noise array for randomized greedy algorithm
    :param side_plot_width: width of side plots in inches (default 1.5)
    :param side_plot_color: color of bars in side plots, default 'gray'
    :param dpi: figure DPI
    :param ax: axis to plot into. If ax is specified, figsize and dpi will be ignored.
    :param xlabel: label of x axis
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter)
    :param widths_minmax_ratio: desired max/min ratio of displayed chunk widths, default None (show actual widths)
    :param widths_minmax_equate: if true, make all chunks the same size. widths_minmax_ratio will be overridden, default True
    :param rotate_col_annotations: True / False, whether to print annotations vertically
    :param fontsize: font size for all text elements
    :param row_annotations_y: a number in (0, 1), position for row annotations in the row. Default 0.5 - center of row.
    :param col_annotations_area_height: height of area for column annotations in inches, 1 by default
    :param col_annotations_ys_count: 1 (default), 2, or 3 - use to reduce clutter in column annotations area
    :param color_by: 'row' (default) or 'column'. If 'row', all cells in same row are same color, etc.
    :param bar_height: height of cell fill as a fraction of row height, a number in (0, 1).
    :param bar_alpha: alpha for cell fills.
    :param bar_align: vertical alignment of bars, 'edge' (default) or 'center'. Only matters when bar_height < 1.
    :param color_cycle: a list of set colors, given as names of matplotlib named colors, or hex codes (e.g. '#1f77b4')
    :param auto_color: automatically generate a list of colors. overwrites color_cycle
    :param alternating_background: True / False (default) - give avery second row a slight grey tint

    :return: SupervennPlot instance with attributes `axes`, `figure`, `chunks`
        and method `get_chunk(set_indices)`. See docstring to returned object.
    """

    if figsize is not None or dpi is not None:
        warnings.warn(
            "Parameters figsize and dpi of supervenn() are deprecated and will be removed in a future version.\n"
            "Instead of this:\n"
            "    supervenn(sets, figsize=(8, 5), dpi=90)"
            "\nPlease either do this:\n"
            "    plt.figure(figsize=(8, 5), dpi=90)\n"
            "    supervenn(sets)\n"
            "or plot into an existing axis by passing it as the ax argument:\n"
            "    supervenn(sets, ax=my_axis)\n"
        )

    axes = setup_axes(
        side_plots, figsize, dpi, ax, side_top_plot_width, side_right_plot_width
    )

    if set_annotations is None:
        set_annotations = ["set_{}".format(i) for i in range(len(sets))]

    chunks, composition_array = get_chunks_and_composition_array(
        sets, universe=universe
    )

    # Find permutations of rows and columns
    permutations_ = get_permutations(
        chunks,
        composition_array,
        chunks_ordering=chunks_ordering,
        sets_ordering=sets_ordering,
        reverse_chunks_order=reverse_chunks_order,
        reverse_sets_order=reverse_sets_order,
        max_bruteforce_size=max_bruteforce_size,
        seeds=seeds,
        noise_prob=noise_prob,
    )

    # Apply permutations
    chunks = [chunks[i] for i in permutations_["chunks_ordering"]]
    composition_array = composition_array[:, permutations_["chunks_ordering"]]
    composition_array = composition_array[permutations_["sets_ordering"], :]
    set_annotations = [set_annotations[i] for i in permutations_["sets_ordering"]]

    # Main plot
    chunk_sizes = [len(chunk) for chunk in chunks]

    if widths_minmax_equate:
        widths_minmax_ratio = 1 - min(chunk_sizes) / max(chunk_sizes)

    if widths_minmax_ratio is not None:
        widths_balancer = get_widths_balancer(chunk_sizes, widths_minmax_ratio)
        col_widths = [widths_balancer(chunk_size) for chunk_size in chunk_sizes]
        effective_min_width_for_annotation = widths_balancer(min_width_for_annotation)
    else:
        col_widths = chunk_sizes
        effective_min_width_for_annotation = min_width_for_annotation

    if auto_color:
        # overwrite kw. not best, but works for now

        # sum along axis 0
        chunks_with_non_empty_sets = composition_array.sum(0)

        # This is to ignore chunks with elements that don't belong to any set
        relevant_chunk_sizes = []
        for i, chunk in enumerate(chunks):
            if chunks_with_non_empty_sets[i] > 0:
                relevant_chunk_sizes.append(len(chunk))
            else:
                relevant_chunk_sizes.append(0)

        if log_color:
            original_chunk_sizes = np.array(relevant_chunk_sizes)

            # relevant_chunk_sizes = np.round(
            #     relevant_chunk_sizes / np.max(relevant_chunk_sizes) * 255
            # ).astype(int)

            relevant_chunk_sizes = np.round(
                np.where(
                    original_chunk_sizes != 0, np.log2(original_chunk_sizes) + 1, 0
                ),
                0,
            ).astype(int)

        cmap = ucla_colorgradient(n=max(relevant_chunk_sizes))
        color_cycle = [cmap[chunk_size - 1] for chunk_size in relevant_chunk_sizes]
        kw["color_cycle"] = color_cycle

    plot_binary_array(
        arr=composition_array,
        row_annotations=set_annotations,
        col_annotations=chunk_sizes,
        ax=axes["main"],
        col_widths=col_widths,
        min_width_for_annotation=effective_min_width_for_annotation,
        **kw,
    )

    xlim = axes["main"].get_xlim()
    ylim = axes["main"].get_ylim()
    fontsize = kw.get("fontsize", DEFAULT_FONTSIZE)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    # Side plots

    if "top_side_plot" in axes:
        plt.sca(axes["top_side_plot"])
        side_plot(
            composition_array.sum(0),
            col_widths,
            "h",
            min_width_for_annotation=effective_min_width_for_annotation,
            rotate_annotations=kw.get("rotate_col_annotations", False),
            color=side_plot_color,
            fontsize=fontsize,
        )
        plt.xlim(xlim)
        plt.ylabel(side_top_plot_label, fontsize=fontsize)

    if "right_side_plot" in axes:
        plt.sca(axes["right_side_plot"])
        side_plot(
            [len(sets[i]) for i in permutations_["sets_ordering"]],
            [1] * len(sets),
            "v",
            color=side_plot_color,
            fontsize=fontsize,
        )
        plt.ylim(ylim)
        axes["right_side_plot"].xaxis.set_label_position("top")
        plt.xlabel(side_right_plot_label, fontsize=fontsize)

    if log_color:
        cmap = LinearSegmentedColormap.from_list("temp", cmap, N=256)
        norm = colors.LogNorm(1, max(original_chunk_sizes))
        plt.tight_layout(pad=0)
        fig = plt.gcf()
        curr_size = fig.get_size_inches()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes(
            [0.9 + 0.25 / curr_size[0], 0.2, 0.1 / curr_size[0], 0.6]
        )
        # cbar_ax.axis("off")

        plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            label=cbar_label,
        )

        cbar_ax.yaxis.set_minor_formatter(FuncFormatter(ticks_format))
        cbar_ax.yaxis.set_major_formatter(FuncFormatter(ticks_format))
        if ticks_off:
            cbar_ax.minorticks_off()

    else:
        cmap = LinearSegmentedColormap.from_list("temp", cmap, N=256)
        norm = colors.Normalize(1, max(relevant_chunk_sizes))
        plt.tight_layout(pad=0)
        fig = plt.gcf()
        curr_size = fig.get_size_inches()

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes(
            [0.9 + 0.25 / curr_size[0], 0.2, 0.1 / curr_size[0], 0.6]
        )

        plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            format=StrMethodFormatter("{x:.0f}"),
            label=cbar_label,
        )
        # cbar_ax.yaxis.set_minor_formatter(FuncFormatter(ticks_format))
        # cbar_ax.yaxis.set_major_formatter(FuncFormatter(ticks_format))

    if square_cell:
        fig = plt.gcf()
        curr_size = fig.get_size_inches()
        ratio = (
            (curr_size[0] * 0.85) * len(sets)
            + len(col_widths) * (side_top_plot_width)
            - len(sets) * (side_right_plot_width)
        ) / ((curr_size[0]) * (len(col_widths)))
        fig.set_size_inches(
            curr_size[0],
            curr_size[0] * ratio,
        )

    plt.sca(axes["main"])
    return SupervennPlot(
        axes,
        plt.gcf(),
        break_into_chunks(sets),
        chunks,
        composition_array,
        set_annotations,
        universe,
    )  # todo: break_into_chunks is called twice, fix


# TODO: UNFINISHED AND MESSY - CURRENTLY PROTOTYPING
def comparevenn(
    complete_set,
    subgroup_set_0,
    subgroup_set_1,
    set_annotations=None,
    figsize=None,
    side_plots=True,
    side_top_plot_width=1,
    side_right_plot_width=1,
    min_width_for_annotation=1,
    widths_minmax_ratio=None,
    side_plot_color="gray",
    dpi=None,
    ax=None,
    xlabel="ITEMS",
    ylabel="SETS",
    widths_minmax_equate=True,
    auto_color=True,
    square_cell=False,
    side_top_plot_label="Model\nCount",
    side_right_plot_label="Odds Ratio\nPer Model",
    cbar_label="Odds Ratio Contribution",
    **kw,
):
    """
    Plot a diagram visualizing relationship of multiple sets.
    :param complete_set: SuperVenn object
    :param subgroup_set_0: SuperVenn object (subset of complete_set)
    :param subgroup_set_1: SuperVenn object (subset of complete_set)
    :param set_annotations: list of annotations for the sets
    :param figsize: figure size
    :param side_plots: True / False: add small barplots on top and on the right. On top, for each chunk it is shown,
    how many sets does this chunk lie inslde. On the right, set sizes are shown.
    :param side_plot_width: width of side plots in inches (default 1.5)
    :param side_plot_color: color of bars in side plots, default 'gray'
    :param dpi: figure DPI
    :param ax: axis to plot into. If ax is specified, figsize and dpi will be ignored.
    :param xlabel: label of x axis
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter)
    :param widths_minmax_ratio: desired max/min ratio of displayed chunk widths, default None (show actual widths)
    :param widths_minmax_equate: if true, make all chunks the same size. widths_minmax_ratio will be overridden, default True
    :param rotate_col_annotations: True / False, whether to print annotations vertically
    :param fontsize: font size for all text elements
    :param row_annotations_y: a number in (0, 1), position for row annotations in the row. Default 0.5 - center of row.
    :param col_annotations_area_height: height of area for column annotations in inches, 1 by default
    :param col_annotations_ys_count: 1 (default), 2, or 3 - use to reduce clutter in column annotations area
    :param color_by: 'row' (default) or 'column'. If 'row', all cells in same row are same color, etc.
    :param bar_height: height of cell fill as a fraction of row height, a number in (0, 1).
    :param bar_alpha: alpha for cell fills.
    :param bar_align: vertical alignment of bars, 'edge' (default) or 'center'. Only matters when bar_height < 1.
    :param color_cycle: a list of set colors, given as names of matplotlib named colors, or hex codes (e.g. '#1f77b4')
    :param auto_color: automatically generate a list of colors. overwrites color_cycle
    :param alternating_background: True / False (default) - give avery second row a slight grey tint

    :return: None
    """

    if figsize is not None or dpi is not None:
        warnings.warn(
            "Parameters figsize and dpi of supervenn() are deprecated and will be removed in a future version.\n"
            "Instead of this:\n"
            "    supervenn(sets, figsize=(8, 5), dpi=90)"
            "\nPlease either do this:\n"
            "    plt.figure(figsize=(8, 5), dpi=90)\n"
            "    supervenn(sets)\n"
            "or plot into an existing axis by passing it as the ax argument:\n"
            "    supervenn(sets, ax=my_axis)\n"
        )

    axes = setup_axes(
        side_plots, figsize, dpi, ax, side_top_plot_width, side_right_plot_width
    )

    # Use to align the subgroup arrays with the original
    composition_array = np.zeros_like(complete_set.composition_array)
    norm_composition_array_0 = np.zeros_like(complete_set.composition_array, float)
    norm_composition_array_1 = np.zeros_like(complete_set.composition_array, float)
    count_composition_array = np.zeros_like(complete_set.composition_array, float)

    # iterate over columns
    for idx in range(complete_set.composition_array.shape[1]):

        # get the sets for the column
        list_chunk_idx = np.nonzero(complete_set.composition_array[:, idx])[0]
        chunk_idx = frozenset(list_chunk_idx)

        # get the number of elements in the set
        if chunk_idx in subgroup_set_0.chunks_dict:
            norm_composition_array_0[list(list_chunk_idx), idx] = len(
                subgroup_set_0.chunks_dict[chunk_idx]
            )

        # get the number of elements in the set
        if chunk_idx in subgroup_set_1.chunks_dict:
            norm_composition_array_1[list(list_chunk_idx), idx] = len(
                subgroup_set_1.chunks_dict[chunk_idx]
            )

        # store the binary set
        composition_array[list(list_chunk_idx), idx] = 1

    # divide each element in "0" by the sum of the row in "1", both normalized to the number of elements
    count_composition_array = np.divide(
        norm_composition_array_0 / subgroup_set_0.universe_size,
        np.sum(norm_composition_array_1, axis=1)[:, None]
        / subgroup_set_1.universe_size,
    )
    # sum the columns
    chunk_sizes = np.round(np.sum(count_composition_array, axis=0), 2)
    # sum the rows
    set_counts = np.round(
        (norm_composition_array_0.sum(1) / subgroup_set_0.universe_size)
        / (norm_composition_array_1.sum(1) / subgroup_set_1.universe_size),
        2,
    )

    if widths_minmax_equate:
        widths_minmax_ratio = 1 - min(chunk_sizes) / max(chunk_sizes)
    if widths_minmax_ratio is not None:
        widths_balancer = get_widths_balancer(chunk_sizes, widths_minmax_ratio)
        col_widths = [widths_balancer(chunk_size) for chunk_size in chunk_sizes]
        effective_min_width_for_annotation = widths_balancer(min_width_for_annotation)
    else:
        col_widths = chunk_sizes
        effective_min_width_for_annotation = min_width_for_annotation

    if auto_color:
        # overwrite kw. not best, but works for now

        original_count_composition_array = count_composition_array

        # Make all rows percentages
        # count_composition_array = np.round(
        #     count_composition_array
        #     / np.sum(count_composition_array, axis=1)[:, None]
        #     * 100
        # ).astype(int)
        # Rescaling
        count_composition_array = np.round(
            count_composition_array / np.max(count_composition_array) * 100
        ).astype(int)

        cmap = ucla_colorgradient(n=int(np.ceil(np.max(count_composition_array) + 1)))

        color_cycle = [
            [None for x in range(count_composition_array.shape[1])]
            for y in range(count_composition_array.shape[0])
        ]

        for i in range(count_composition_array.shape[0]):
            for j in range(count_composition_array.shape[1]):
                color_cycle[i][j] = cmap[count_composition_array[i][j]]

        kw["color_cycle"] = color_cycle

    col_annotations = None  # [len(chunk) for chunk in complete_set.chunks]
    plot_binary_array(
        arr=composition_array,
        row_annotations=set_annotations,
        col_annotations=col_annotations,
        ax=axes["main"],
        col_widths=col_widths,
        min_width_for_annotation=effective_min_width_for_annotation,
        **kw,
    )

    xlim = axes["main"].get_xlim()
    ylim = axes["main"].get_ylim()
    fontsize = kw.get("fontsize", DEFAULT_FONTSIZE)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    # Side plots
    if "top_side_plot" in axes:
        plt.sca(axes["top_side_plot"])
        side_plot(
            composition_array.sum(0),
            col_widths,
            "h",
            min_width_for_annotation=effective_min_width_for_annotation,
            rotate_annotations=kw.get("rotate_col_annotations", False),
            color=side_plot_color,
            fontsize=fontsize,
        )
        plt.xlim(xlim)
        plt.ylabel(side_top_plot_label, fontsize=fontsize)

    if "right_side_plot" in axes:
        plt.sca(axes["right_side_plot"])
        side_plot(
            [i for i in set_counts],
            [1] * len(set_counts),
            "v",
            color=side_plot_color,
            fontsize=fontsize,
        )
        plt.ylim(ylim)
        axes["right_side_plot"].xaxis.set_label_position("top")
        plt.xlabel(side_right_plot_label, fontsize=fontsize)

    cmap = LinearSegmentedColormap.from_list("temp", cmap, N=256)
    norm = colors.Normalize(0, np.max(original_count_composition_array))
    plt.tight_layout(pad=0)
    fig = plt.gcf()
    curr_size = fig.get_size_inches()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.9 + 0.25 / curr_size[0], 0.2, 0.1 / curr_size[0], 0.6])

    plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        format=StrMethodFormatter("{x:.2f}"),
        label=cbar_label,
    )

    if square_cell:
        fig = plt.gcf()
        curr_size = fig.get_size_inches()
        ratio = (
            (curr_size[0] * 0.85) * len(set_counts)
            + len(col_widths) * (side_top_plot_width)
            - len(set_counts) * (side_right_plot_width)
        ) / ((curr_size[0]) * (len(col_widths)))
        fig.set_size_inches(curr_size[0], curr_size[0] * ratio)

    plt.sca(axes["main"])
    return


# TODO: UNFINISHED AND MESSY - CURRENTLY PROTOTYPING
def side_sub_plot(
    values,
    widths,
    orient,
    subgroup_labels,
    fontsize=DEFAULT_FONTSIZE,
    min_width_for_annotation=1,
    rotate_annotations=False,
    color="tab:gray",
    colors=[
        "#FFFF00",
        "#00FF87",
        "#FF00A5",
        "#00FFFF",
        "#8237FF",
        "#000000",
    ],
    alpha=0.5,
):
    """
    Barplot with multiple bars of variable width right next to each other, with an option to rotate the plot 90 degrees.
    :param values: the values to be plotted.
    :param widths: Widths of bars
    :param orient: 'h' / 'horizontal' (default) or 'v' / 'vertical'
    :param fontsize: font size for annotations
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter. Default 1 - annotate all.)
    :param rotate_annotations: True/False, whether to print annotations vertically instead of horizontally
    :param color: color of bars, default 'tab:gray'
    """
    bar_edges = np.insert(np.cumsum(widths), 0, 0)
    annotation_positions = [
        0.5 * (begin + end) for begin, end in zip(bar_edges[:-1], bar_edges[1:])
    ]
    bottom = np.zeros_like(values[0])
    if orient in ["h", "horizontal"]:
        horizontal = True
        for value in values:
            plt.bar(
                x=bar_edges[:-1],
                height=value,
                width=widths,
                bottom=bottom,
                align="edge",
                alpha=alpha,
                color=color,
            )
            bottom = bottom + value
        ticks = plt.xticks
        lim = plt.ylim
        upper_lim = max(np.sum(np.array(values), 0))
    elif orient in ["v", "vertical"]:
        horizontal = False
        for i, value in enumerate(values):
            plt.barh(
                y=bar_edges[:-1],
                width=value,
                height=widths,
                left=bottom,
                align="edge",
                alpha=alpha,
                color=colors[i],
            )
            bottom = bottom + value
        ticks = plt.yticks
        lim = plt.xlim
        upper_lim = max(np.sum(np.array(values), 0))
    elif orient in ["revh"]:
        horizontal = True
        for i, value in enumerate(values):
            plt.bar(
                x=bar_edges[:-1],
                height=-1 * np.array(value),
                width=widths,
                bottom=bottom,
                align="edge",
                alpha=alpha,
                color=colors[i],
                label=f"{subgroup_labels[i]}",
            )
            bottom = bottom - value
        ticks = plt.xticks
        lim = plt.ylim
        upper_lim = -max(np.sum(np.array(values), 0))
    else:
        raise ValueError('Unknown orient: {} (should be "h" or "v")'.format(orient))

    if orient not in ["revh"]:
        for i, (annotation_position, value, width) in enumerate(
            zip(annotation_positions, np.sum(np.array(values), 0), widths)
        ):
            if width < min_width_for_annotation and horizontal:
                continue
            x, y = 0.05 * upper_lim, annotation_position
            ha = "left"
            va = "center"
            if horizontal:
                x, y = y, x
                ha = "center"
                va = "bottom"
            plt.annotate(
                value,
                xy=(x, y),
                ha=ha,
                va=va,
                rotation=rotate_annotations * 90,
                fontsize=fontsize,
            )

    ticks(bar_edges, [])
    lim(0, upper_lim)
    plt.grid(True)


# TODO: UNFINISHED AND MESSY - CURRENTLY PROTOTYPING
def subgroupvenn(
    sets,
    subgroup_sets,
    set_annotations=None,
    figsize=None,
    side_plots=True,
    chunks_ordering="minimize gaps",
    sets_ordering=None,
    reverse_chunks_order=True,
    reverse_sets_order=True,
    universe=None,
    max_bruteforce_size=DEFAULT_MAX_BRUTEFORCE_SIZE,
    seeds=DEFAULT_SEEDS,
    noise_prob=DEFAULT_NOISE_PROB,
    side_top_plot_width=1,
    side_right_plot_width=1,
    min_width_for_annotation=1,
    widths_minmax_ratio=None,
    side_plot_color="gray",
    dpi=None,
    ax=None,
    xlabel="ITEMS",
    ylabel="SETS",
    widths_minmax_equate=True,
    auto_color=True,
    log_color=False,
    square_cell=False,
    side_top_plot_label="Model\nCount",
    side_right_plot_label="Total Errors\nPer Model",
    cbar_label="Number of Incorrectly\nClassified Patients",
    ticks_off=False,
    # col_annotations_area_height=0.75,
    **kw,
):
    """
    Plot a diagram visualizing relationship of multiple sets.
    :param complete_set: SuperVenn object
    :param subgroup_set_0: SuperVenn object (subset of complete_set)
    :param subgroup_set_1: SuperVenn object (subset of complete_set)
    :param set_annotations: list of annotations for the sets
    :param figsize: figure size
    :param side_plots: True / False: add small barplots on top and on the right. On top, for each chunk it is shown,
    how many sets does this chunk lie inslde. On the right, set sizes are shown.
    :param side_plot_width: width of side plots in inches (default 1.5)
    :param side_plot_color: color of bars in side plots, default 'gray'
    :param dpi: figure DPI
    :param ax: axis to plot into. If ax is specified, figsize and dpi will be ignored.
    :param xlabel: label of x axis
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter)
    :param widths_minmax_ratio: desired max/min ratio of displayed chunk widths, default None (show actual widths)
    :param widths_minmax_equate: if true, make all chunks the same size. widths_minmax_ratio will be overridden, default True
    :param rotate_col_annotations: True / False, whether to print annotations vertically
    :param fontsize: font size for all text elements
    :param row_annotations_y: a number in (0, 1), position for row annotations in the row. Default 0.5 - center of row.
    :param col_annotations_area_height: height of area for column annotations in inches, 1 by default
    :param col_annotations_ys_count: 1 (default), 2, or 3 - use to reduce clutter in column annotations area
    :param color_by: 'row' (default) or 'column'. If 'row', all cells in same row are same color, etc.
    :param bar_height: height of cell fill as a fraction of row height, a number in (0, 1).
    :param bar_alpha: alpha for cell fills.
    :param bar_align: vertical alignment of bars, 'edge' (default) or 'center'. Only matters when bar_height < 1.
    :param color_cycle: a list of set colors, given as names of matplotlib named colors, or hex codes (e.g. '#1f77b4')
    :param auto_color: automatically generate a list of colors. overwrites color_cycle
    :param alternating_background: True / False (default) - give avery second row a slight grey tint

    :return: None
    """

    if figsize is not None or dpi is not None:
        warnings.warn(
            "Parameters figsize and dpi of supervenn() are deprecated and will be removed in a future version.\n"
            "Instead of this:\n"
            "    supervenn(sets, figsize=(8, 5), dpi=90)"
            "\nPlease either do this:\n"
            "    plt.figure(figsize=(8, 5), dpi=90)\n"
            "    supervenn(sets)\n"
            "or plot into an existing axis by passing it as the ax argument:\n"
            "    supervenn(sets, ax=my_axis)\n"
        )

    # Define and optionally create the encasing axis for plot according to arguments
    #### TODO: Currently bypasses existing setup_axes. Refactor and integrate
    if ax is None:
        if figsize is not None or dpi is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        supervenn_ax = plt.gca()
    else:
        supervenn_ax = ax

    bbox = supervenn_ax.get_window_extent().transformed(
        supervenn_ax.get_figure().dpi_scale_trans.inverted()
    )
    plot_width, plot_height = bbox.width, bbox.height
    width_ratios = [plot_width - side_right_plot_width, side_right_plot_width]
    height_ratios = [
        side_top_plot_width,
        plot_height - side_top_plot_width,
        side_top_plot_width,
    ]
    fig = supervenn_ax.get_figure()
    get_gridspec = partial(
        gridspec.GridSpecFromSubplotSpec,
        subplot_spec=supervenn_ax.get_subplotspec(),
        hspace=0,
        wspace=0,
    )

    gs = get_gridspec(3, 2, height_ratios=height_ratios, width_ratios=width_ratios)

    axes = {
        "main": fig.add_subplot(gs[1, 0]),
        "top_side_plot": fig.add_subplot(gs[0, 0]),
        "unused": fig.add_subplot(gs[0, 1]),
        "right_side_plot": fig.add_subplot(gs[1, 1]),
        "bottom_side_plot": fig.add_subplot(gs[2, 0]),
        "unused2": fig.add_subplot(gs[2, 1]),
    }

    # Remove tick from every axis, and set ticks length to 0 (we'll add ticks to the side plots manually later)
    for ax in axes.values():
        remove_ticks(ax)
        ax.tick_params(which="major", length=0)
    remove_ticks(
        supervenn_ax
    )  # if side plots are used, supervenn_ax isn't included in axes dict
    #### TODO: Currently bypasses existing setup_axes. Refactor and integrate

    chunks, composition_array = get_chunks_and_composition_array(
        sets, universe=universe
    )

    # Find permutations of rows and columns
    permutations_ = get_permutations(
        chunks,
        composition_array,
        chunks_ordering=chunks_ordering,
        sets_ordering=sets_ordering,
        reverse_chunks_order=reverse_chunks_order,
        reverse_sets_order=reverse_sets_order,
        max_bruteforce_size=max_bruteforce_size,
        seeds=seeds,
        noise_prob=noise_prob,
    )

    # Apply permutations
    chunks = [chunks[i] for i in permutations_["chunks_ordering"]]
    composition_array = composition_array[:, permutations_["chunks_ordering"]]
    composition_array = composition_array[permutations_["sets_ordering"], :]
    set_annotations = [set_annotations[i] for i in permutations_["sets_ordering"]]

    # Main plot
    chunk_sizes = [len(chunk) for chunk in chunks]

    if widths_minmax_equate:
        widths_minmax_ratio = 1 - min(chunk_sizes) / max(chunk_sizes)

    if widths_minmax_ratio is not None:
        widths_balancer = get_widths_balancer(chunk_sizes, widths_minmax_ratio)
        col_widths = [widths_balancer(chunk_size) for chunk_size in chunk_sizes]
        effective_min_width_for_annotation = widths_balancer(min_width_for_annotation)
    else:
        col_widths = chunk_sizes
        effective_min_width_for_annotation = min_width_for_annotation

    if auto_color:
        # overwrite kw. not best, but works for now

        # sum along axis 0
        chunks_with_non_empty_sets = composition_array.sum(0)

        # This is to ignore chunks with elements that don't belong to any set
        relevant_chunk_sizes = []
        for i, chunk in enumerate(chunks):
            if chunks_with_non_empty_sets[i] > 0:
                relevant_chunk_sizes.append(len(chunk))
            else:
                relevant_chunk_sizes.append(0)

        if log_color:
            original_chunk_sizes = np.array(relevant_chunk_sizes)

            # relevant_chunk_sizes = np.round(
            #     relevant_chunk_sizes / np.max(relevant_chunk_sizes) * 255
            # ).astype(int)

            relevant_chunk_sizes = np.round(
                np.where(
                    original_chunk_sizes != 0, np.log2(original_chunk_sizes) + 1, 0
                ),
                0,
            ).astype(int)

        cmap = ucla_colorgradient(n=max(relevant_chunk_sizes))
        color_cycle = [cmap[chunk_size - 1] for chunk_size in relevant_chunk_sizes]
        kw["color_cycle"] = color_cycle

    plot_binary_array(
        arr=composition_array,
        row_annotations=set_annotations,
        col_annotations=chunk_sizes,
        ax=axes["main"],
        col_widths=col_widths,
        min_width_for_annotation=effective_min_width_for_annotation,
        **kw,
    )

    xlim = axes["main"].get_xlim()
    ylim = axes["main"].get_ylim()
    fontsize = kw.get("fontsize", DEFAULT_FONTSIZE)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    # Side plots
    if "bottom_side_plot" in axes:
        plt.sca(axes["bottom_side_plot"])

        values = []
        subgroup_ratios = []
        for subgroup_name, subgroup in subgroup_sets.items():
            values.append(
                [len(chunk.intersection(subgroup)) / len(chunk) for chunk in chunks],
                # [len(chunk.intersection(subgroup)) for chunk in chunks],
                # [len(chunk.intersection(subgroup))/(len(subgroup))/len(chunk) for chunk in chunks],
            )
            non_percent = [len(chunk.intersection(subgroup)) for chunk in chunks]
            subgroup_ratios.append(np.sum(non_percent))
        for i in range(1, len(subgroup_ratios)):
            subgroup_ratios[i] = subgroup_ratios[i] + subgroup_ratios[i - 1]
        for i in range(len(subgroup_ratios)):
            subgroup_ratios[i] = subgroup_ratios[i] / subgroup_ratios[-1]

        side_sub_plot(
            values,
            col_widths,
            "revh",
            list(subgroup_sets.keys()),
            min_width_for_annotation=effective_min_width_for_annotation,
            rotate_annotations=kw.get("rotate_col_annotations", False),
            color="#8237FF",
            fontsize=fontsize,
        )
        plt.xlim(xlim)
        axes["bottom_side_plot"].xaxis.set_label_position("bottom")
        plt.ylabel("Subgroup\nProportions", fontsize=fontsize)
        axes["bottom_side_plot"].invert_yaxis()

        if len(subgroup_ratios) <= 3:
            for i, ratio in enumerate(subgroup_ratios):
                if ratio >= 1:
                    continue
                if i == 0:
                    plt.axhline(
                        y=-ratio,
                        color="gray",
                        linestyle="--",
                        label="Original\nProportion",
                    )
                else:
                    plt.axhline(y=-ratio, color="gray", linestyle="--")

        plt.legend(
            loc="lower left",
            bbox_to_anchor=(1.32, 1),
            frameon=False,
        )

    if "right_side_plot" in axes:
        plt.sca(axes["right_side_plot"])

        values = []
        for subgroup_name, subgroup in subgroup_sets.items():
            values.append(
                [
                    len(sets[i].intersection(subgroup))  # /len(sets[i])
                    for i in permutations_["sets_ordering"]
                ],
            )

        side_sub_plot(
            values,
            [1] * len(sets),
            "v",
            list(subgroup_sets.keys()),
            color=side_plot_color,
            fontsize=fontsize,
        )
        plt.ylim(ylim)
        axes["right_side_plot"].xaxis.set_label_position("top")
        plt.xlabel(side_right_plot_label, fontsize=fontsize)
        # for ratio in subgroup_ratios:
        #     if ratio >= 1:
        #         continue
        #     plt.axvline(x=ratio, color="gray", linestyle="--")

    if "top_side_plot" in axes:
        plt.sca(axes["top_side_plot"])
        side_plot(
            composition_array.sum(0),
            col_widths,
            "h",
            min_width_for_annotation=effective_min_width_for_annotation,
            rotate_annotations=kw.get("rotate_col_annotations", False),
            color=side_plot_color,
            fontsize=fontsize,
        )
        plt.xlim(xlim)
        plt.ylabel(side_top_plot_label, fontsize=fontsize)

    if log_color:
        cmap = LinearSegmentedColormap.from_list("temp", cmap, N=256)
        norm = colors.LogNorm(1, max(original_chunk_sizes))
        plt.tight_layout(pad=0)
        fig = plt.gcf()
        curr_size = fig.get_size_inches()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes(
            [0.9 + 0.25 / curr_size[0], 0.2, 0.1 / curr_size[0], 0.6]
        )

        plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            label=cbar_label,
        )

        cbar_ax.yaxis.set_minor_formatter(FuncFormatter(ticks_format))
        cbar_ax.yaxis.set_major_formatter(FuncFormatter(ticks_format))
        if ticks_off:
            cbar_ax.minorticks_off()

    else:
        cmap = LinearSegmentedColormap.from_list("temp", cmap, N=256)
        norm = colors.Normalize(1, max(relevant_chunk_sizes))
        plt.tight_layout(pad=0)
        fig = plt.gcf()
        curr_size = fig.get_size_inches()

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes(
            [0.9 + 0.25 / curr_size[0], 0.2, 0.1 / curr_size[0], 0.6]
        )

        plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            format=StrMethodFormatter("{x:.0f}"),
            label=cbar_label,
        )

    if square_cell:
        fig = plt.gcf()
        curr_size = fig.get_size_inches()
        ratio = (
            (curr_size[0] * 0.75) * len(sets)
            + len(col_widths) * (side_top_plot_width * 2)
            - len(sets) * (side_right_plot_width)
        ) / ((curr_size[0]) * (len(col_widths)))
        fig.set_size_inches(
            curr_size[0],
            curr_size[0] * ratio,
        )

    plt.sca(axes["main"])
    return SupervennPlot(
        axes,
        plt.gcf(),
        break_into_chunks(sets),
        chunks,
        composition_array,
        set_annotations,
        universe,
    )  # todo: break_into_chunks is called twice, fix

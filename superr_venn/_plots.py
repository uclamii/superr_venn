# -*- coding: utf-8 -*-
"""
Routines for plotting multiple sets.
"""

from functools import partial
from collections import OrderedDict
import warnings

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolor
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter, FuncFormatter, ScalarFormatter
from matplotlib import cm, colors
from cycler import cycler

from superr_venn._algorithms import (
    break_into_chunks,
    get_chunks_and_composition_array,
    get_permutations,
    DEFAULT_MAX_BRUTEFORCE_SIZE,
    DEFAULT_SEEDS,
    DEFAULT_NOISE_PROB,
)

from superr_venn._plotting_utils import (
    DEFAULT_FONTSIZE,
    remove_ticks,
    remove_spines,
    get_widths_balancer,
    ticks_format,
    get_column_widths,
    get_cmap_list,
    setup_axes,
    get_cmap_grid,
    get_ax_size_inches,
    rescale_axes,
    get_mpl_cmap,
)

from superr_venn._error_profiling_utils import compute_sets_from_df


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


def plot_binary_array(
    arr,
    ax=None,
    # Cells
    col_widths=None,
    row_heights=None,
    color_cycle=None,
    color_by="row",
    bar_height=1.0,
    bar_alpha=1.0,
    bar_align="center",
    alternating_background=False,
    # Annotations
    row_annotations=None,
    col_annotations=None,
    min_width_for_annotation=1,
    row_annotations_offset=-0.25,
    col_annotations_x=0.5,
    row_annotations_y=0.5,
    col_annotations_area_height=0.75,
    fontsize=DEFAULT_FONTSIZE,
    xlabel="",
):
    """
    Visualize a binary array as a grid with variable sized columns and rows, where cells with 1 are filled using bars
    and cells with 0 are blank.
    :param arr: numpy.array of zeros and ones
    :param ax: axis to plot into (current axis by default)
    :param col_widths: widths for grid columns, must have len equal to arr.shape[1]
    :param row_heights: heights for grid rows, must have len equal to arr.shape[0]
    :param color_cycle: a list of colors, given as names of matplotlib named colors, or hex codes (e.g. '#1f77b4')
    :param color_by: 'row' (default) or 'column'. If 'row', all cells in same row are same color, etc.
    :param bar_height: height of cell fill as a fraction of row height, a number in (0, 1).
    :param bar_alpha: alpha for cell fills.
    :param bar_align: vertical alignment of bars, 'edge' (defaulr) or 'center'. Only matters when bar_height < 1.
    :param alternating_background: True / False (default) - give avery second row a slight grey tint
    :param row_annotations: annotations for each row, plotted in the middle of the row
    :param col_annotations: annotations for columns, plotted in the bottom, below the x axis.
    :param min_width_for_annotation: don't annotate column with its size if size is less than this value (default 1)
    :param row_annotations_offset: Add translation to row annotation. Default 0 - no translation.
    :param col_annotations_x: a number in (0, 1), position for col annotations in each col. Default 0.5 - center of col.
    :param row_annotations_y: a number in (0, 1), position for row annotations in the row. Default 0.5 - center of row.
    :param col_annotations_area_height: height of area for column annotations in inches, 1 by default
    :param fontsize: font size for annotations (default {}).
    :param xlabel: label for x axis
    """

    grid_xs = np.insert(np.cumsum(col_widths), 0, 0)[:-1]
    grid_ys = np.insert(np.cumsum(row_heights), 0, 0)[:-1]

    if ax is not None:
        plt.sca(ax)

    # Cells
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

    # Row annotations
    if row_annotations is not None:
        for row_index, (grid_y, row_height, annotation) in enumerate(
            zip(grid_ys, row_heights, row_annotations)
        ):
            annot_y = grid_y + row_annotations_y * row_height

            plt.annotate(
                str(annotation),
                xy=(
                    0,
                    annot_y,
                ),
                xytext=(
                    row_annotations_offset,
                    annot_y,
                ),
                ha="right",
                va="center",
                fontsize=fontsize,
            )

    # Column annotations
    if col_annotations is not None:

        min_y = -1.0
        plt.axhline(0, c="k")

        annot_y = row_annotations_y * row_height * min_y

        for col_index, (grid_x, col_width, annotation) in enumerate(
            zip(grid_xs, col_widths, col_annotations)
        ):
            annot_x = grid_x + col_annotations_x * col_width

            if col_width >= min_width_for_annotation:
                plt.annotate(
                    str(annotation),
                    xy=(annot_x, annot_y),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    rotation=0,
                )
    else:
        min_y = -0.01
        plt.axhline(0, c="k", linewidth=0.5)

    plt.xlim(0, sum(col_widths))
    plt.ylim(min_y, sum(row_heights))
    plt.xticks(grid_xs, [])
    plt.yticks(grid_ys, [])
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=fontsize)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    return xlim, ylim


def plot_side(
    values,
    widths,
    orient,
    fontsize=DEFAULT_FONTSIZE,
    min_width_for_annotation=1,
    color="#FFB81C",
    labels=None,
    cmap="Set3",
):
    """
    Barplot with multiple bars of variable width right next to each other, with an option to rotate the plot 90 degrees.
    :param values: the values to be plotted.
    :param widths: Widths of bars
    :param orient: 'h' / 'horizontal' (default) or 'v' / 'vertical'
    :param fontsize: font size for annotations
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter. Default 1 - annotate all.)
    :param color: color of bars
    """

    if len(values) < 2:
        colors = [color] * len(values)
    else:
        subgroup = True
        colors = get_mpl_cmap(cmap, len(values))

    bar_edges = np.insert(np.cumsum(widths), 0, 0)
    annotation_positions = [
        0.5 * (begin + end) for begin, end in zip(bar_edges[:-1], bar_edges[1:])
    ]
    bottom = np.zeros_like(values[0])

    # Plot bars
    if orient in ["h", "horizontal"]:
        horizontal = True
        for i, value in enumerate(values):
            plt.bar(
                x=bar_edges[:-1],
                height=value,
                width=widths,
                bottom=bottom,
                align="edge",
                alpha=0.5,
                color=colors[i],
            )
            bottom = bottom + value
        ticks = plt.xticks
        lim = plt.ylim
        max_value = max(np.sum(np.array(values), 0))

        # tick alignment
        ha = "center"
        va = "bottom"

    elif orient in ["v", "vertical"]:
        horizontal = False
        for i, value in enumerate(values):
            plt.barh(
                y=bar_edges[:-1],
                width=value,
                height=widths,
                left=bottom,
                align="edge",
                alpha=0.5,
                color=colors[i],
            )
            bottom = bottom + value
        ticks = plt.yticks
        lim = plt.xlim
        max_value = max(np.sum(np.array(values), 0))

        # tick alignment
        ha = "left"
        va = "center"

    elif orient in ["h_neg"]:
        horizontal = True
        for i, value in enumerate(values):
            plt.bar(
                x=bar_edges[:-1],
                height=-1 * np.array(value),
                width=widths,
                bottom=bottom,
                align="edge",
                alpha=0.5,
                color=colors[i],
                label=f"{labels[i]}" if subgroup else None,
            )
            bottom = bottom - value
        ticks = plt.xticks
        lim = plt.ylim
        max_value = -max(np.sum(np.array(values), 0))

        # tick alignment
        ha = "center"
        va = "bottom"

    # Annotate bars
    if orient not in ["h_neg"]:
        for i, (annotation_position, value, width) in enumerate(
            zip(annotation_positions, np.sum(np.array(values), 0), widths)
        ):
            if width < min_width_for_annotation:
                continue

            # Slight offset
            x, y = 0.05 * max_value, annotation_position

            if horizontal:
                x, y = y, x

            plt.annotate(
                value,
                xy=(x, y),
                ha=ha,
                va=va,
                fontsize=fontsize,
            )

    ticks(bar_edges, [])
    lim(0, max_value)
    plt.grid(True)


def plot_cbar(cmap, composition_array, chunks, log_color, ax, cbar_label, grid=False):

    # Get color cycle (list) and cbar_mappable for the color bar
    if grid:
        # For odds ratios, use a grid colormap
        color_cycle, cbar_mappable = get_cmap_grid(cmap, composition_array)
        formatter = StrMethodFormatter("{x:.2f}")
    else:
        color_cycle, cbar_mappable = get_cmap_list(
            cmap, composition_array, chunks, log_color
        )
        formatter = StrMethodFormatter("{x:.0f}")

    # Plot color bar
    cbar = plt.colorbar(
        cbar_mappable,
        label=cbar_label,
        format=formatter,
        ax=ax,
        location="right",
        fraction=0.9,
        shrink=0.8,
        anchor=(0.0, 0.8),
    )

    # Remove axis border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Reformat scientific notation
    if log_color:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(ticks_format))
        cbar.ax.yaxis.set_minor_formatter(
            FuncFormatter(partial(ticks_format, minor=True))
        )

    return color_cycle


def generate_plot_data(
    sets,
    universe=None,
    set_annotations=None,
    chunks_ordering="occurrence",
    sets_ordering=None,
    reverse_chunks_order=True,
    reverse_sets_order=True,
    max_bruteforce_size=DEFAULT_MAX_BRUTEFORCE_SIZE,
    seeds=DEFAULT_SEEDS,
    noise_prob=DEFAULT_NOISE_PROB,
):
    """
    Generate the chunks and composition array for the sets.
    :param sets: list of sets
    :param universe: set of all possible elements, if None, universe is set to the union of all sets
    :return: chunks, composition_array
    """
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
    set_annotations = (
        [set_annotations[i] for i in permutations_["sets_ordering"]]
        if set_annotations is not None
        else None
    )
    chunk_sizes = [len(chunk) for chunk in chunks]

    return chunks, chunk_sizes, composition_array, permutations_, set_annotations


def supervenn(
    sets,
    # Plotting
    set_annotations=None,  # label for each row
    height_ratios=[1.5, 4.5],
    width_ratios=[5.5, 1.5, 1],
    ## Column widths
    min_width_for_annotation=1,
    widths_minmax_ratio=None,
    equal_col_width=True,
    square_cell=False,
    ## Color
    side_plot_color="#FFB81C",
    cmap="ucla",
    log_color=False,
    color_by="column",
    ## Labeling
    xlabel="Patients",
    side_top_plot_label="Model\nCount",
    side_right_plot_label="Total Errors\nPer Model",
    cbar_label="Number of Incorrectly\nClassified Patients",
    # Algorithm
    sets_ordering=None,
    chunks_ordering="occurrence",
    reverse_sets_order=False,
    reverse_chunks_order=True,
    max_bruteforce_size=DEFAULT_MAX_BRUTEFORCE_SIZE,
    seeds=DEFAULT_SEEDS,
    noise_prob=DEFAULT_NOISE_PROB,
    fontsize=DEFAULT_FONTSIZE,
    universe=None,
):
    """
    Plot a diagram visualizing relationship of multiple sets.
    :param sets: list of sets
    :param set_annotations: list of annotations for the sets
    :param height_ratios: list of ratios for the heights of the rows in the grid, default [1.5, 4.5]
    :param width_ratios: list of ratios for the widths of the columns in the grid, default [5.5, 1.5, 1]
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter)
    :param widths_minmax_ratio: desired max/min ratio of displayed chunk widths, default None (show actual widths)
    :param equal_col_width: if true, make all chunks the same size. widths_minmax_ratio will be overridden, default True
    :param square_cell: True / False (default) - make the cells square, i.e. the height of each row will be equal to
        the width of each column, so that the cells are squares.
    :param side_plot_color: color of bars in side plots, default 'gray'
    :param cmap: colormap to use for the color bar, default 'ucla'
    :param log_color: True / False (default) - use logarithmic scale for the color bar
    :param color_by: 'row' or 'column'. If 'row', all cells in same row are same color, etc.
    :param xlabel: label of x axis
    :param side_top_plot_label: label for the top side plot
    :param side_right_plot_label: label for the right side plot
    :param cbar_label: label for the color bar
    :param sets_ordering: method of ordering the sets (rows of the grid)
        - None (default): keep the order as it is passed
        - 'minimize gaps': use a smart algorithm to find an order of rows giving fewer gaps in each column
        - 'size': bigger sets go first (or last if reverse_sets_order = False)
        - 'chunk count': sets that contain most chunks go first (or last if reverse_sets_order = False)
        - 'random': randomly shuffle
    :param chunks_ordering: method of ordering the chunks (columns of the grid)
        - 'minimize gaps' (default): use a smart algorithm to find an order of columns giving fewer gaps in each row,
            making the plot as readable as possible.
        - 'size': bigger chunks go first (or last if reverse_chunks_order=False)
        - 'occurence': chunks that are in most sets go first (or last if reverse_chunks_order=False)
        - 'random': randomly shuffle the columns
    :param reverse_sets_order: True / False, works the same way as reverse_chunks_order
    :param reverse_chunks_order: True (default) / False when chunks_ordering is "size" or "occurence",
        chunks with bigger corresponding property go first if reverse_chunks_order=True, smaller go first if False.
    :param max_bruteforce_size: maximal number of items for which bruteforce method is applied to find permutation
    :param seeds: number of different random seeds for the randomized greedy algorithm to find permutation
    :param noise_prob: probability of given element being equal to 1 in the noise array for randomized greedy algorithm
    :param fontsize: font size for all text elements
    :param universe: set of all possible elements

    :return: SupervennPlot instance with attributes `axes`, `figure`, `chunks`
        and method `get_chunk(set_indices)`. See docstring to returned object.
    """

    ### Main algorithm. Chunks := columns. Sets := rows
    chunks, chunk_sizes, composition_array, permutations_, set_annotations = (
        generate_plot_data(
            sets,
            universe,
            set_annotations,
            chunks_ordering=chunks_ordering,
            sets_ordering=sets_ordering,
            reverse_chunks_order=reverse_chunks_order,
            reverse_sets_order=reverse_sets_order,
            max_bruteforce_size=max_bruteforce_size,
            seeds=seeds,
            noise_prob=noise_prob,
        )
    )

    ### Prepare plotting
    axes = setup_axes(height_ratios, width_ratios)
    remove_spines(axes[(0, 2)])

    ### Calculate width scaling factors
    col_widths, effective_min_width_for_annotation = get_column_widths(
        equal_col_width,
        widths_minmax_ratio,
        min_width_for_annotation,
        chunk_sizes,
    )

    ### Color bar
    color_cycle = plot_cbar(
        cmap, composition_array, chunks, log_color, axes[(1, 2)], cbar_label
    )

    ### Main plot
    xlim, ylim = plot_binary_array(
        arr=composition_array,
        ax=axes[(1, 0)],
        row_annotations=set_annotations,
        col_annotations=chunk_sizes,
        col_widths=col_widths,
        row_heights=[1] * composition_array.shape[0],
        min_width_for_annotation=effective_min_width_for_annotation,
        color_cycle=color_cycle,
        fontsize=fontsize,
        color_by=color_by,
        xlabel=xlabel,
    )

    ### Top side plot
    plt.sca(axes[(0, 0)])
    plot_side(
        values=[composition_array.sum(0)],
        widths=col_widths,
        orient="h",
        min_width_for_annotation=effective_min_width_for_annotation,
        color=side_plot_color,
        fontsize=fontsize,
    )
    plt.xlim(xlim)
    plt.ylabel(
        side_top_plot_label, fontsize=fontsize, rotation=0, ha="right", va="center"
    )

    ### Right side plot
    plt.sca(axes[(1, 1)])
    plot_side(
        values=[[len(sets[i]) for i in permutations_["sets_ordering"]]],
        widths=[1] * len(sets),
        orient="v",
        color=side_plot_color,
        fontsize=fontsize,
    )
    plt.ylim(ylim)
    axes[(1, 1)].xaxis.set_label_position("top")
    plt.xlabel(side_right_plot_label, fontsize=fontsize)

    # Final formatting
    if square_cell:
        # Scaling factor to make the cells square
        plt.sca(axes[(1, 0)])
        rescale_axes()

    return SupervennPlot(
        axes,
        plt.gcf(),
        break_into_chunks(sets),
        chunks,
        composition_array,
        set_annotations,
        universe,
    )  # todo: break_into_chunks is called twice, fix


def oddsratio_venn(
    complete_set,
    subgroup_set_0,
    subgroup_set_1,
    # Plotting
    set_annotations=None,  # label for each row
    height_ratios=[1.5, 4.5],
    width_ratios=[5.5, 1.5, 1],
    ## Column widths
    min_width_for_annotation=1,
    widths_minmax_ratio=None,
    equal_col_width=True,
    square_cell=False,
    ## Color
    side_plot_color="#FFB81C",
    cmap="ucla",
    log_color=False,
    color_by="column",
    ## Labeling
    xlabel="Patients",
    side_top_plot_label="Model\nCount",
    side_right_plot_label="Odds Ratio\nPer Model",
    cbar_label="Odds Ratio\nContribution",
    # Algorithm
    fontsize=DEFAULT_FONTSIZE,
):
    """
    Plot a diagram visualizing relationship of multiple sets.
    :param complete_set: SuperVenn object
    :param subgroup_set_0: SuperVenn object (subset of complete_set)
    :param subgroup_set_1: SuperVenn object (subset of complete_set)
    """

    ### Main algorithm

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

    ### Prepare plotting
    axes = setup_axes(height_ratios, width_ratios)
    remove_spines(axes[(0, 2)])

    ### Calculate width scaling factors
    col_widths, effective_min_width_for_annotation = get_column_widths(
        equal_col_width,
        widths_minmax_ratio,
        min_width_for_annotation,
        chunk_sizes,
    )

    ### Color bar
    color_cycle = plot_cbar(
        cmap,
        count_composition_array,
        None,
        log_color,
        axes[(1, 2)],
        cbar_label,
        grid=True,
    )

    ### Main plot
    xlim, ylim = plot_binary_array(
        arr=composition_array,
        ax=axes[(1, 0)],
        row_annotations=set_annotations,
        col_annotations=None,  # [len(chunk) for chunk in complete_set.chunks],
        col_widths=col_widths,
        row_heights=[1] * composition_array.shape[0],
        min_width_for_annotation=effective_min_width_for_annotation,
        color_cycle=color_cycle,
        fontsize=fontsize,
        color_by=color_by,
        xlabel=xlabel,
    )

    ### Top side plot
    plt.sca(axes[(0, 0)])
    plot_side(
        values=[composition_array.sum(0)],
        widths=col_widths,
        orient="h",
        min_width_for_annotation=effective_min_width_for_annotation,
        color=side_plot_color,
        fontsize=fontsize,
    )
    plt.xlim(xlim)
    plt.ylabel(
        side_top_plot_label, fontsize=fontsize, rotation=0, ha="right", va="center"
    )

    ### Right side plot
    plt.sca(axes[(1, 1)])
    plot_side(
        values=[[i for i in set_counts]],
        widths=[1] * len(set_counts),
        orient="v",
        color=side_plot_color,
        fontsize=fontsize,
    )
    plt.ylim(ylim)
    axes[(1, 1)].xaxis.set_label_position("top")
    plt.xlabel(side_right_plot_label, fontsize=fontsize)

    # Final formatting
    if square_cell:
        # Scaling factor to make the cells square
        plt.sca(axes[(1, 0)])
        rescale_axes()

    plt.sca(axes[(1, 0)])
    return


# TODO: UNFINISHED AND MESSY - CURRENTLY PROTOTYPING
def subgroup_venn(
    sets,
    subgroup_sets,
    # Plotting
    set_annotations=None,  # label for each row
    height_ratios=[1.5, 4.5, 1.5],
    width_ratios=[5.5, 1.5, 1],
    ## Column widths
    min_width_for_annotation=1,
    widths_minmax_ratio=None,
    equal_col_width=True,
    square_cell=False,
    ## Color
    side_plot_color="#FFB81C",
    cmap="ucla",
    log_color=False,
    color_by="column",
    ## Labeling
    xlabel="Patients",
    side_top_plot_label="Model\nCount",
    side_right_plot_label="Total Errors\nPer Model",
    cbar_label="Number of Incorrectly\nClassified Patients",
    # Algorithm
    sets_ordering=None,
    chunks_ordering="occurrence",
    reverse_sets_order=False,
    reverse_chunks_order=True,
    max_bruteforce_size=DEFAULT_MAX_BRUTEFORCE_SIZE,
    seeds=DEFAULT_SEEDS,
    noise_prob=DEFAULT_NOISE_PROB,
    fontsize=DEFAULT_FONTSIZE,
    universe=None,
):
    """
    Plot a diagram visualizing relationship of multiple sets.
    :param complete_set: SuperVenn object
    :param subgroup_set_0: SuperVenn object (subset of complete_set)
    :param set_annotations: list of annotations for the sets
    :param height_ratios: list of ratios for the heights of the rows in the grid, default [1.5, 4.5]
    :param width_ratios: list of ratios for the widths of the columns in the grid, default [5.5, 1.5, 1]
    :param min_width_for_annotation: for horizontal plot, don't annotate bars of widths less than this value (to avoid
    clutter)
    :param widths_minmax_ratio: desired max/min ratio of displayed chunk widths, default None (show actual widths)
    :param equal_col_width: if true, make all chunks the same size. widths_minmax_ratio will be overridden, default True
    :param square_cell: True / False (default) - make the cells square, i.e. the height of each row will be equal to
        the width of each column, so that the cells are squares.
    :param side_plot_color: color of bars in side plots, default 'gray'
    :param cmap: colormap to use for the color bar, default 'ucla'
    :param log_color: True / False (default) - use logarithmic scale for the color bar
    :param color_by: 'row' or 'column'. If 'row', all cells in same row are same color, etc.
    :param xlabel: label of x axis
    :param side_top_plot_label: label for the top side plot
    :param side_right_plot_label: label for the right side plot
    :param cbar_label: label for the color bar
    :param sets_ordering: method of ordering the sets (rows of the grid)
        - None (default): keep the order as it is passed
        - 'minimize gaps': use a smart algorithm to find an order of rows giving fewer gaps in each column
        - 'size': bigger sets go first (or last if reverse_sets_order = False)
        - 'chunk count': sets that contain most chunks go first (or last if reverse_sets_order = False)
        - 'random': randomly shuffle
    :param chunks_ordering: method of ordering the chunks (columns of the grid)
        - 'minimize gaps' (default): use a smart algorithm to find an order of columns giving fewer gaps in each row,
            making the plot as readable as possible.
        - 'size': bigger chunks go first (or last if reverse_chunks_order=False)
        - 'occurence': chunks that are in most sets go first (or last if reverse_chunks_order=False)
        - 'random': randomly shuffle the columns
    :param reverse_sets_order: True / False, works the same way as reverse_chunks_order
    :param reverse_chunks_order: True (default) / False when chunks_ordering is "size" or "occurence",
        chunks with bigger corresponding property go first if reverse_chunks_order=True, smaller go first if False.
    :param max_bruteforce_size: maximal number of items for which bruteforce method is applied to find permutation
    :param seeds: number of different random seeds for the randomized greedy algorithm to find permutation
    :param noise_prob: probability of given element being equal to 1 in the noise array for randomized greedy algorithm
    :param fontsize: font size for all text elements
    :param universe: set of all possible elements

    :return: SupervennPlot instance with attributes `axes`, `figure`, `chunks`
        and method `get_chunk(set_indices)`. See docstring to returned object.
    """

    ### Main algorithm. Chunks := columns. Sets := rows
    chunks, chunk_sizes, composition_array, permutations_, set_annotations = (
        generate_plot_data(
            sets,
            universe,
            set_annotations,
            chunks_ordering=chunks_ordering,
            sets_ordering=sets_ordering,
            reverse_chunks_order=reverse_chunks_order,
            reverse_sets_order=reverse_sets_order,
            max_bruteforce_size=max_bruteforce_size,
            seeds=seeds,
            noise_prob=noise_prob,
        )
    )

    ### Prepare plotting
    axes = setup_axes(height_ratios, width_ratios)
    remove_spines(axes[(0, 2)])
    remove_spines(axes[(2, 2)])
    remove_spines(axes[(2, 1)])

    ### Calculate width scaling factors
    col_widths, effective_min_width_for_annotation = get_column_widths(
        equal_col_width,
        widths_minmax_ratio,
        min_width_for_annotation,
        chunk_sizes,
    )

    ### Color bar
    color_cycle = plot_cbar(
        cmap, composition_array, chunks, log_color, axes[(1, 2)], cbar_label
    )

    ### Main plot
    xlim, ylim = plot_binary_array(
        arr=composition_array,
        ax=axes[(1, 0)],
        row_annotations=set_annotations,
        col_annotations=chunk_sizes,
        col_widths=col_widths,
        row_heights=[1] * composition_array.shape[0],
        min_width_for_annotation=effective_min_width_for_annotation,
        color_cycle=color_cycle,
        fontsize=fontsize,
        color_by=color_by,
        xlabel=xlabel,
    )

    ### Bottom side plot
    plt.sca(axes[(2, 0)])

    values = []
    subgroup_ratios = []
    for subgroup_name, subgroup in subgroup_sets.items():
        values.append(
            [len(chunk.intersection(subgroup)) / len(chunk) for chunk in chunks],
        )
        non_percent = [len(chunk.intersection(subgroup)) for chunk in chunks]
        subgroup_ratios.append(np.sum(non_percent))
    for i in range(1, len(subgroup_ratios)):
        subgroup_ratios[i] = subgroup_ratios[i] + subgroup_ratios[i - 1]
    for i in range(len(subgroup_ratios)):
        subgroup_ratios[i] = subgroup_ratios[i] / subgroup_ratios[-1]

    plot_side(
        values=values,
        widths=col_widths,
        orient="h_neg",
        min_width_for_annotation=effective_min_width_for_annotation,
        fontsize=fontsize,
        labels=list(subgroup_sets.keys()),
    )
    plt.xlim(xlim)
    axes[(2, 0)].xaxis.set_label_position("bottom")
    axes[(2, 0)].invert_yaxis()
    axes[(2, 0)].set_ylabel("Subgroup\nProportions", fontsize=fontsize)

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
        loc="upper left",
        bbox_to_anchor=(1.25, 1.0),
        frameon=False,
    )

    ### Right side plot
    plt.sca(axes[(1, 1)])
    values = []
    for subgroup_name, subgroup in subgroup_sets.items():
        values.append(
            [
                len(sets[i].intersection(subgroup))
                for i in permutations_["sets_ordering"]
            ],
        )

    plot_side(
        values=values,
        widths=[1] * len(sets),
        orient="v",
        color=side_plot_color,
        fontsize=fontsize,
        labels=list(subgroup_sets.keys()),
    )
    plt.ylim(ylim)
    axes[(1, 1)].xaxis.set_label_position("top")
    plt.xlabel(side_right_plot_label, fontsize=fontsize)

    ### Top plot
    plt.sca(axes[(0, 0)])
    plot_side(
        values=[composition_array.sum(0)],
        widths=col_widths,
        orient="h",
        min_width_for_annotation=effective_min_width_for_annotation,
        color=side_plot_color,
        fontsize=fontsize,
    )
    plt.xlim(xlim)
    plt.ylabel(
        side_top_plot_label, fontsize=fontsize, rotation=0, ha="right", va="center"
    )

    if square_cell:
        # Scaling factor to make the cells square
        plt.sca(axes[(1, 0)])
        rescale_axes()

    plt.sca(axes[(1, 0)])
    return SupervennPlot(
        axes,
        plt.gcf(),
        break_into_chunks(sets),
        chunks,
        composition_array,
        set_annotations,
        universe,
    )  # todo: break_into_chunks is called twice, fix


def create_error_profile(
    df, id=None, columns=None, set_annotations=None, incorrect_value=0, figsize=(10, 6)
):
    sets, universe = compute_sets_from_df(df, id, columns, incorrect_value)

    plt.figure(figsize=figsize)
    venn = supervenn(
        sets,
        set_annotations=set_annotations if set_annotations else columns,
        log_color=True,
        square_cell=True,
        universe=universe,
    )
    return venn


def create_stratified_error_profile(
    df,
    subgroup,
    id=None,
    columns=None,
    set_annotations=None,
    incorrect_value=0,
    figsize=(10, 6),
):

    sets, universe = compute_sets_from_df(df, id, columns, incorrect_value)

    subgroups = OrderedDict(
        {val: set(df[df[subgroup] == val][id]) for val in df[subgroup].unique()}
    )

    plt.figure(figsize=figsize)
    venn = subgroup_venn(
        sets,
        subgroups,
        set_annotations=set_annotations if set_annotations else columns,
        log_color=True,
        square_cell=True,
        universe=universe,
    )
    return venn


def create_oddratio_profile(
    df,
    subgroup,
    id=None,
    columns=None,
    set_annotations=None,
    incorrect_value=0,
    figsize=(10, 6),
):
    assert (
        len(df["Subgroup"].unique()) == 2
    ), "Subgroup must have exactly two unique values for comparison."

    sets, universe = compute_sets_from_df(df, id, columns, incorrect_value)

    set_annotations = set_annotations if set_annotations is not None else columns

    venn = supervenn(
        sets,
        set_annotations=set_annotations,
        log_color=True,
        square_cell=True,
        universe=universe,
    )
    plt.close()

    subgroup_profiles = []
    for subgroup in df["Subgroup"].unique():
        subgroup_df = df[df["Subgroup"] == subgroup]
        subgroup_profile = create_error_profile(
            subgroup_df,
            id=id,
            columns=columns,
            incorrect_value=incorrect_value,
        )
        plt.close()
        subgroup_profiles.append(subgroup_profile)

    plt.figure(figsize=figsize)
    oddsratio_venn(
        venn,
        subgroup_profiles[0],
        subgroup_profiles[1],
        square_cell=True,
        set_annotations=set_annotations,
    )

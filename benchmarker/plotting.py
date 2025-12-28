import matplotlib.cm
from scipy.stats import rankdata
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Callable, Any, Literal, List
from anndata import AnnData
import pandas as pd
import numpy as np
import scanpy as sc
import math
from matplotlib import rcParams
from plottable import ColumnDefinition, Table
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from .utils import apply_formatter, normed_cmap, get_default_cmap, repeat_to_length, get_scatter_cmap

def bar(
    ax: matplotlib.axes.Axes,
    val: float,
    vmin: float = None,
    vmax: float = None,
    xlim: Tuple[float, float] = (0, 1),
    cmap: matplotlib.colors.Colormap = None,
    plot_bg_bar: bool = False,
    annotate: bool = False,
    textprops: Dict[str, Any] = {},
    formatter: Callable = None,
    **kwargs,
) -> matplotlib.container.BarContainer:
    """Plots a bar on the axes.

    Args:
        ax (matplotlib.axes.Axes):
            Axes
        val (float):
            value
        xlim (Tuple[float, float], optional):
            data limit for the x-axis. Defaults to (0, 1).
        cmap (matplotlib.colors.Colormap, optional):
            colormap. Defaults to None.
        plot_bg_bar (bool, optional):
            whether to plot a background bar. Defaults to False.
        annotate (bool, optional):
            whether to annotate the value. Defaults to False.
        textprops (Dict[str, Any], optional):
            textprops passed to ax.text. Defaults to {}.
        formatter (Callable, optional):
            a string formatter.
            Can either be a string format, ie "{:2f}" for 2 decimal places.
            Or a Callable that is applied to the value. Defaults to None.

    Returns:
        matplotlib.container.BarContainer
    """
    
    if "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = "C1"

    if cmap is not None:
        color = cmap(float(val))

    if plot_bg_bar:
        ax.barh(
            0.5,
            xlim[1],
            left=xlim[0],
            fc="None",
            ec=plt.rcParams["text.color"],
            **kwargs,
            zorder=0.1,
        )
    

    bar = ax.barh(0.4, val, fc=color, ec="None", **kwargs, zorder=0.05)

    if annotate:
        if val < 0.5 * xlim[1]:
            ha = "left"
            x = val + 0.025 * abs(xlim[1] - xlim[0])
        else:
            ha = "right"
            x = val - 0.025 * abs(xlim[1] - xlim[0])

        if formatter is not None:
            text = apply_formatter(formatter, val)
        else:
            text = val
        if(vmin is None or vmax is None):
            ax.text(x, 0.5, text, ha=ha, va="center", **textprops, zorder=0.3)
        else:
            vmin, vmax = float(vmin), float(vmax)
            if((val-vmin) >= ((vmax-vmin)*0.65) and ha=="right"):
                ax.text(x, 0.5, text, ha=ha, va="center", **textprops, zorder=0.3, color="white")
            else:
                ax.text(x, 0.5, text, ha=ha, va="center", **textprops, zorder=0.3, color="black")

    ax.axis("off")
    ax.set_xlim(
        xlim[0] - 0.025 * abs(xlim[1] - xlim[0]),
        xlim[1] + 0.025 * abs(xlim[1] - xlim[0]),
    )
    ax.set_ylim(0, 1)

    return bar

# def plot_results_table(
#         res_df: pd.DataFrame, show: bool = True, save_dir: Optional[str] = None, total_name: str = "Total",
#         cmap1: matplotlib.cm = None,  cmap2: matplotlib.cm = None, cell_color_by: Literal["value", "rank"] = "rank",
#         highlight_methods: List = None, highlight_color: str = None,
#     ) -> Table:

#         _METRIC_TYPE = "Metric Type"
#         _AGGREGATE_SCORE = ["Aggregate score", "Aggregate\nscore"]

#         if(cmap1 is None or cmap2 is None):
#             d_cmap1, d_cmap2 = get_default_cmap()
#             cmap1 = cmap1 or d_cmap1
#             cmap2 = cmap2 or d_cmap2

#         num_embeds = res_df.shape[0] - 1
#         if(cell_color_by=="rank"):
#             cmap_fn1 = lambda col_data: normed_cmap(col_data, cmap=cmap1, num_stds=None)
#             cmap_fn2 = lambda col_data: normed_cmap(col_data, cmap=cmap2, num_stds=None)
#         else:
#             cmap_fn1 = lambda col_data: normed_cmap(col_data, cmap=cmap1)
#             cmap_fn2 = lambda col_data: normed_cmap(col_data, cmap=cmap2)

#         df = res_df.copy()
#         plot_df = df.drop(_METRIC_TYPE, axis=0)
#         plot_df = plot_df.sort_values(by=total_name, ascending=False).astype(np.float64)
#         plot_df["Method"] = plot_df.index

#         score_cols = df.columns[df.loc[_METRIC_TYPE].isin(_AGGREGATE_SCORE)]
#         other_cols = df.columns[~df.loc[_METRIC_TYPE].isin(_AGGREGATE_SCORE)]

#         if(highlight_methods is not None):
#             text_dict = {i: "black" for i in plot_df.index}
#             for i in highlight_methods:
#                 text_dict[i] = highlight_color
#             text_cmap = lambda x: text_dict[x]
#             print(callable(text_cmap))
#         else:
#             text_cmap = None
#         column_definitions = [
#             ColumnDefinition("Method", width=1.2, textprops={"ha": "left", "weight": "bold", "fontsize": 12},
#                              text_cmap=text_cmap)
#         ]
#         column_definitions += [
#             ColumnDefinition(
#                 col,
#                 title=col.replace(" ", "\n", 1),
#                 width=0.95,
#                 textprops={
#                     "ha": "center",
#                     "bbox": {"boxstyle": "circle", "pad": 0.25},
#                     "fontsize": 11,
#                 },
#                 cmap=cmap_fn1(plot_df[col]),
#                 group=df.loc[_METRIC_TYPE, col],
#                 formatter="{:.2f}",
#             )
#             for i, col in enumerate(other_cols)
#         ]

#         column_definitions += [
#             ColumnDefinition(
#                 col,
#                 width=1.1,
#                 title=col.replace(" ", "\n", 1),
#                 plot_fn=bar,
#                 plot_kw={
#                     "cmap": cmap_fn2(plot_df[col]),
#                     "plot_bg_bar": False,
#                     "annotate": True,
#                     "height": 0.9,
#                     "formatter": "{:.2f}",
#                     "vmax": apply_formatter("{:.2f}", max(plot_df[col])),
#                     "vmin": apply_formatter("{:.2f}", min(plot_df[col])),
#                 },
#                 group=df.loc[_METRIC_TYPE, col],
#                 border="both" if i == 0 else None,
#             )
#             for i, col in enumerate(score_cols)
#         ]

#         with matplotlib.rc_context({"svg.fonttype": "none"}):
#             fig, ax = plt.subplots(figsize=(len(df.columns) * 1.4, 3 + 0.55 * num_embeds))
#             tab = Table(
#                 plot_df,
#                 cell_kw={
#                     "linewidth": 0,
#                     "edgecolor": "k",
#                 },
#                 column_definitions=column_definitions,
#                 ax=ax,
#                 row_dividers=True,
#                 footer_divider=True,
#                 textprops={"fontsize": 10.5, "ha": "center"},
#                 row_divider_kw={"linewidth": 0.9, "linestyle":(0, (1, 5))},
#                 col_label_divider_kw={"linewidth": 0.7, "linestyle": "-"},
#                 column_border_kw={"linewidth": 0.7, "linestyle": "-"},
#                 footer_divider_kw={"linewidth": 0.7, "linestyle": "-"},
#                 index_col="Method",
#             ).autoset_fontcolors(colnames=plot_df.columns)
#         if show:
#             plt.show()
#         if save_dir is not None:
#             fig.savefig(save_dir, facecolor=ax.get_facecolor(), dpi=300, bbox_inches='tight')

#         return tab

def plot_results_table(
        res_df: pd.DataFrame, show: bool = True, save_dir: Optional[str] = None, total_name: str = "Total",
        cmap1: matplotlib.cm = None,  cmap2: matplotlib.cm = None, cell_color_by: Literal["value", "rank"] = "rank",
        highlight_methods: List = None, highlight_color: str = None,
        insert_marker_row: Optional[int] = None,
        show_top: int = 5,
        show_bottom: int = 3,
        marker_bg_color: str = "#E8E8E8",
    ) -> Table:

    _METRIC_TYPE = "Metric Type"
    _AGGREGATE_SCORE = ["Aggregate score", "Aggregate\nscore"]

    # -------------------------------------------
    # Colors
    # -------------------------------------------
    if cmap1 is None or cmap2 is None:
        d_cmap1, d_cmap2 = get_default_cmap()
        cmap1 = cmap1 or d_cmap1
        cmap2 = cmap2 or d_cmap2

    if cell_color_by == "rank":
        cmap_fn1 = lambda col_data: normed_cmap(col_data, cmap=cmap1, num_stds=None)
        cmap_fn2 = lambda col_data: normed_cmap(col_data, cmap=cmap2, num_stds=None)
    else:
        cmap_fn1 = lambda col_data: normed_cmap(col_data, cmap=cmap1)
        cmap_fn2 = lambda col_data: normed_cmap(col_data, cmap=cmap2)

    df = res_df.copy()

    # -------------------------------------------
    # Build plot_df
    # -------------------------------------------
    plot_df = df.drop(_METRIC_TYPE, axis=0)
    plot_df = plot_df.sort_values(by=total_name, ascending=False).astype(float)
    plot_df["Method"] = plot_df.index

    idx_all = plot_df.index.tolist()
    marker_label = None

    # -------------------------------------------
    # Row collapsing
    # -------------------------------------------
    if insert_marker_row is not None and len(plot_df) > (show_top + show_bottom):

        top_part = idx_all[:show_top]
        bottom_part = idx_all[-show_bottom:]
        hidden_count = len(idx_all) - show_top - show_bottom

        marker_label = f"1"

        # marker 行，不允许显示 NaN
        # marker 行
        marker_df = pd.DataFrame(
            {col: [np.nan] for col in plot_df.columns},  # 保持 NaN
            index=[marker_label]
        )
        marker_df["Method"] = marker_label


        plot_df = pd.concat([
            plot_df.loc[top_part],
            marker_df,
            plot_df.loc[bottom_part],
        ])

    # -------------------------------------------
    # Column groups
    # -------------------------------------------
    score_cols = df.columns[df.loc[_METRIC_TYPE].isin(_AGGREGATE_SCORE)]
    other_cols = df.columns[~df.loc[_METRIC_TYPE].isin(_AGGREGATE_SCORE)]

    # -------------------------------------------
    # Text highlight
    # -------------------------------------------
    if highlight_methods is not None:
        text_dict = {i: "black" for i in plot_df.index}
        for m in highlight_methods:
            if m in text_dict:
                text_dict[m] = highlight_color
        if marker_label is not None:
            text_dict[marker_label] = "gray"

        text_cmap = lambda x: text_dict.get(x, "black")
    else:
        text_cmap = None

    # -------------------------------------------
    # Column definitions
    # -------------------------------------------
    column_definitions = []

    # Method column —— marker 行居中显示
    column_definitions.append(
        ColumnDefinition(
            "Method",
            width=1.5,
            textprops={
                "ha": "center" if marker_label else "left",
                "weight": "bold",
                "fontsize": 12
            },
            text_cmap=text_cmap,
        )
    )

    # Non-aggregate metrics
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1.0,
            textprops={
                "ha": "center",
                "fontsize": 11,
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn1(plot_df[col]),
            group=df.loc[_METRIC_TYPE, col],
            formatter="{:.2f}",
        )
        for col in other_cols
    ]

    # Score bar columns
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1.1,
            plot_fn=bar,
            plot_kw={
                "cmap": cmap_fn2(plot_df[col]),
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
                "vmax": apply_formatter("{:.2f}", np.nanmax(plot_df[col].values.astype(float))),
                "vmin": apply_formatter("{:.2f}", np.nanmin(plot_df[col].values.astype(float))),
            },
            group=df.loc[_METRIC_TYPE, col],
            border="both",
        )
        for col in score_cols
    ]

    # -------------------------------------------
    # Figure height 根据折叠后行数动态调整
    # -------------------------------------------
    visible_rows = len(plot_df)
    fig_height = 3 + 0.55 * visible_rows   # 原逻辑，但正确使用“折叠后行数”

    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.4, fig_height))

        tab = Table(
            plot_df,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10.5, "ha": "center"},
            row_divider_kw={"linewidth": 0.9, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 0.7, "linestyle": "-"},
            column_border_kw={"linewidth": 0.7, "linestyle": "-"},
            footer_divider_kw={"linewidth": 0.7, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)

    # -------------------------------------------
    # Patch-based background coloring for marker row
    # -------------------------------------------
    if marker_label is not None:
        row_idx = list(plot_df.index).index(marker_label)

        for patch in ax.patches:
            if abs(patch.get_y() - row_idx) < 1e-6:
                patch.set_facecolor(marker_bg_color)

    if show:
        plt.show()

    if save_dir is not None:
        fig.savefig(save_dir, dpi=300, bbox_inches="tight")

    return tab




def _embed_plot(adata: AnnData, embed_list: List, label_list: List, figsize: Tuple, ncol: int,
                palettes: List = [None], cmaps: List = [None],
                outer_col_wspace: float = 0.2, outer_row_hspace: float = 0.1, 
                inner_gs_col: int = 1, inner_gs_row: int = 1,
                inner_col_wspace: float = 0.1, inner_row_hspace: float = 0.05, 
                frameon: bool = False,
                ylabel: List = None, ylabel_pad: float = 0.02, xlabel: List  = None, xlabel_pad: float = 0.02,
                only_show_top: bool = False, only_show_left: bool = False,
                label_fontsize: float = 13, save: str = None, save_dpi: int = 150,
                axis_width: float = None, axis_color: str = None,
                background_color: Callable = None,
                # inner_common_cmap: bool = False,
                sizes: List = None,
                **kwargs):
    
    palettes = repeat_to_length(palettes, len(embed_list))
    cmaps = repeat_to_length(cmaps, len(embed_list))
    
    # xlabel = xlabel if xlabel is None else repeat_to_length(xlabel, math.ceil(len(embed_list)/len(xlabel)), mode="repeat")
    # ylabel = ylabel if ylabel is None else repeat_to_length(ylabel, len(embed_list))

    nrows = math.ceil(len(embed_list) / (ncol*inner_gs_col*inner_gs_row))
    axes_list = []
    fig = plt.figure(figsize=figsize)
    outer_gs = gridspec.GridSpec(nrows=nrows, ncols=ncol, figure=fig, wspace=outer_col_wspace, 
                                 hspace=outer_row_hspace)
    all_groups = []
    for i in range(nrows):
        for col in range(ncol):
            groups = []
            inner_gs = gridspec.GridSpecFromSubplotSpec(nrows=inner_gs_row, ncols=inner_gs_col,
                                                        subplot_spec=outer_gs[i, col], 
                                                        wspace=inner_col_wspace, hspace=inner_row_hspace)
            for icol in range(inner_gs_col):
                for irow  in range(inner_gs_row):
                    if(len(axes_list)==len(embed_list)):
                        break
                    axes_list.append(fig.add_subplot(inner_gs[irow, icol]))
                    groups.append(axes_list[-1])
            if(len(groups)>0):
                all_groups.append(groups)
    
    inner_first_palette = None
    for idx in range(len(embed_list)):
        adata = sc.AnnData(X=np.zeros((embed_list[idx].shape[0], 1)))
        adata.obsm["temp_embed"] = embed_list[idx].copy()
        adata.obs[f"temp_label"] = label_list[idx].copy()
        
        palette = palettes[idx]
        cmap = cmaps[idx]
        if(palette is None):
            # if(inner_common_cmap and idx%(inner_gs_col*inner_gs_row)==0): #first
            #         all_labels = list(adata.obs[f"temp_label"])
            #         palette = get_scatter_cmap(all_labels)
            #         inner_first_palette = palette
            # elif(inner_common_cmap):
            #         palette=  inner_first_palette
            # else:
            all_labels = list(adata.obs[f"temp_label"])
            palette = get_scatter_cmap(all_labels)
            

        ax = axes_list[idx]
        if(sizes is not None):
            size = sizes[idx]
            sc.pl.embedding(adata, basis="temp_embed", color=f"temp_label", ax=ax, title="", show=False, 
                            frameon=False, palette=palette, size=size, cmap=cmap, colorbar_loc=None, **kwargs)
        else:
            sc.pl.embedding(adata, basis="temp_embed", color=f"temp_label", ax=ax, title="", show=False, 
                            frameon=False, palette=palette, cmap=cmap, colorbar_loc=None, **kwargs)
        legend = ax.get_legend()
        if(legend is not None):
            legend.remove()
        ax.set_xlabel("")

        if(ylabel is not None and len(ylabel)>0):
            if(not only_show_left or idx % (ncol*inner_gs_row*inner_gs_col)<inner_gs_row):
                if(idx%inner_gs_col==0):
                    text = ylabel.pop(0)
                    bg_color = background_color(text) if background_color is not None else None

                    if bg_color is not None:
                        rect = mpatches.Rectangle(
                            (-12*ylabel_pad, 0),      
                            10 * ylabel_pad,
                            1, 
                            transform=ax.transAxes,
                            clip_on=False,
                            facecolor=bg_color,
                            edgecolor='none',
                            zorder=0,
                            alpha=0.35
                        )
                        ax.add_patch(rect)
            
                    ax.text(
                        -6.5*ylabel_pad, 0.5, text,
                        fontsize=label_fontsize,
                        ha='center', va='center',
                        transform=ax.transAxes,
                        zorder=1,
                        rotation=90,
                    )
            # ax.set_ylabel(ylabel[idx], rotation=90, labelpad=ylabel_pad, 
            #             va='center', fontdict={"fontsize":label_fontsize})
        ax.set_ylabel("")
        if(xlabel is not None and len(xlabel)>0):
            if(not only_show_top or idx<(ncol*inner_gs_col)):
                if(idx % inner_gs_row==0):

                    text = xlabel.pop(0)
                    bg_color = background_color(text) if background_color is not None else None

                    if bg_color is not None:
                        rect = mpatches.Rectangle(
                            (0, 1 + 2*xlabel_pad),      
                            1,
                            12 * xlabel_pad, 
                            transform=ax.transAxes,
                            clip_on=False,
                            facecolor=bg_color,
                            edgecolor='none',
                            zorder=0,
                            alpha=0.35
                        )
                        ax.add_patch(rect)
                    
                    ax.text(
                        0.5, 1+7*xlabel_pad, text,
                        fontsize=label_fontsize,
                        ha='center', va='center',
                        transform=ax.transAxes,
                        zorder=1
                    )
    if(frameon):
        for group in all_groups:
            boxes = [ax.get_position(fig.transFigure) for ax in group]
            x0 = min([b.x0 for b in boxes])
            y0 = min([b.y0 for b in boxes])
            x1 = max([b.x1 for b in boxes])
            y1 = max([b.y1 for b in boxes])
            rect = patches.Rectangle(
                (x0 , y0),
                x1 - x0,
                y1 - y0,
                transform=fig.transFigure,
                facecolor='none',
                edgecolor=axis_color,
                linewidth=axis_width,
                zorder=1000
            )
            fig.add_artist(rect)

        for idx in range(len(embed_list), len(axes_list)):
            ax = axes_list[idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

    if(save!=None):
        plt.savefig(save, dpi=save_dpi, bbox_inches='tight')   
    plt.show()

def _legend_plot(color_dict: dict, marker: str = "s",  markersize: int = 15, figsize: Tuple = (6,1),
                 ncol: int = 3, label: List = None,
                 handletextpad: float = 0.2, borderpad: float = 0.6,
                 labelspacing: float = 0.6, columnspacing: float = 1,
                 order: List = None, 
                 save=None):
    order = order or sorted(color_dict.keys())
    colors = [color_dict[i] for i in order]
    label = label or order
    legend_patches = [Line2D([0], [0], marker=marker, color=color, label=label[i],
                         markerfacecolor=color, markersize=markersize,  linestyle='None') 
                         for i, color in enumerate(colors)]
    fig, ax = plt.subplots(figsize=figsize) 
    ax.axis('off')  # Hide the axes

    # Create the legend with the specified patches
    legend = ax.legend(handles=legend_patches, frameon=False, loc='center', ncol=ncol, handletextpad=handletextpad, 
                       borderpad=borderpad, labelspacing=labelspacing, columnspacing=columnspacing)

    # Adjust the canvas size to fit the legend only
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(bbox.width, bbox.height)
    if(save!=None):
        plt.savefig(save, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()

def _heatmap_legend_plot(cmap1: matplotlib.cm, cmap2: matplotlib.cm, n_ranks: int = 10, save: str = None):

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.axis('off')

    block_h = 0.35 * 0.618
    block_w = 0.35

    x0, y0 = 0.5, 1.0

    n_block = 5
    color_list1 = [cmap1(i / (n_block - 1)) for i in range(n_block)]
    color_list2 = [cmap2(i / (n_block - 1)) for i in range(n_block)]

    for i in range(n_block):
        rect = patches.Rectangle(
            (x0, y0 + i * block_h),
            block_w,
            block_h,
            facecolor=color_list1[i],
            edgecolor='black',
            linewidth=.8
        )
        ax.add_patch(rect)

    for i in range(n_block):
        rect = patches.Rectangle(
            (x0 + block_w + 0.15, y0 + i * block_h),
            block_w,
            block_h,
            facecolor=color_list2[i],
            edgecolor='black', 
            linewidth=.8
        )
        ax.add_patch(rect)

    ax.text(x0, y0 + n_block * block_h + 0.15, "Ranking", fontsize=17,)

    ax.annotate(
        '', xy=(x0 + block_w*2 + 0.4, y0 + n_block * block_h + 0.04),
        xytext=(x0 + block_w*2 + 0.4, y0),
        arrowprops=dict(
        arrowstyle='->',
        linewidth=1.5,
        color='black'
    )
    )
    ax.text(x0 + block_w*2 + 0.48, y0 + n_block * block_h -0.05, "1", fontsize=15, va='center')
    ax.text(x0 + block_w*2 + 0.48, y0 + 0.05, str(n_ranks), fontsize=15, va='center')

    ax.set_xlim(x0 -0.2, x0 + block_w*2 + 0.6)
    ax.set_ylim(y0 -0.2, y0 + n_block * block_h + 0.5)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()

def _box_plot(df: pd.DataFrame, figsize: Tuple, label_rotation: int = None, ylabel: str = None,
              xlabel: str = None, vert: bool = True, save: str = None, cmap: dict = None,
              show_data_points: bool = False, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    data = [list(df.loc[row,:]) for row in df.index]
    box = ax.boxplot(data, labels=df.index, vert=vert,
                     patch_artist=True, showfliers=False, **kwargs)

    if(cmap is None):
        colors = get_scatter_cmap(list(df.index)[::-1])
    else:
        colors = cmap

    for patch, method in zip(box['boxes'], df.index):
        color = colors[method]
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.9)
        patch.set_linewidth(0.6)

    if(label_rotation is not None):
        if(vert):
            plt.xticks(rotation=label_rotation)
        else:
            plt.yticks(rotation=label_rotation)
    

    for median in box['medians']:
        median.set_color("black")
        median.set_linewidth(0.6)

    if show_data_points:
        for i, vals in enumerate(data):
            if vert:
                ax.scatter([i+1]*len(vals), vals, alpha=0.95, color='black', s=4, zorder=3)
            else:
                ax.scatter(vals, [i+1]*len(vals), alpha=0.95, color='black', s=4, zorder=3)

    if(ylabel is not None):
        plt.ylabel(ylabel, labelpad=15)
    if(xlabel is not None):
        plt.xlabel(xlabel, labelpad=15)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


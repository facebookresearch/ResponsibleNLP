# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.transforms as transforms
import matplotlib

def plot_butterfly(df_butterfly, out_folder):
    # set all text font
    matplotlib.rc("font", size=12)
    
    x_to_eng = df_butterfly["xx_to_en"] * 100
    eng_to_x = df_butterfly["en_to_xx"] * 100

    langs_to_plot = df_butterfly.index

    # bars centered on the y axis
    pos = np.arange(df_butterfly.shape[0]) + .5

    # make the left and right axes for women and men
    fig = plt.figure(facecolor='white', edgecolor='none', figsize=(4,6))
    ax_x_to_eng = fig.add_axes([0.05, 0.1, 0.35, 0.5])
    ax_eng_to_x = fig.add_axes( [0.5, 0.1, 0.35, 0.5]) 

    #(left, bottom, width, height)
    ax_eng_to_x.set_xticks(np.arange(0, 20, 1))
    ax_x_to_eng.set_xticks(np.arange(-50, 20, 1))

    ax_x_to_eng.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax_eng_to_x.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    ax_x_to_eng.xaxis.set_ticks_position('top')
    ax_eng_to_x.xaxis.set_ticks_position('top')

    # make the left graphs
    ax_x_to_eng.barh(pos, x_to_eng, align='center', facecolor='#DDAA33', edgecolor='None')
    ax_x_to_eng.set_yticks([])
    ax_x_to_eng.invert_xaxis()

    # make the right graphs
    ax_eng_to_x.barh(pos, eng_to_x, align='center', facecolor='#DDAA33', edgecolor='None')
    ax_eng_to_x.set_yticks([])

    # we want the cancer labels to be centered in the fig coord system and
    # centered w/ respect to the bars so we use a custom transform
    transform = transforms.blended_transform_factory(
        fig.transFigure, ax_eng_to_x.transData)

    for i, label in enumerate(langs_to_plot):
        ax_eng_to_x.text(0.45, i+0.45, label, ha='center', va='center',
    transform=transform)

    ax_x_to_eng.set_title('EN-to-XX', x=1.7, y=1.1)
    ax_eng_to_x.set_title('XX-to-EN', x=-0.7, y=1.1)

    ax_x_to_eng.set_xlim([2.5, 0])
    ax_eng_to_x.set_xlim([0, 2.5])

    out_path = f'{out_folder}/mutox_butterfly.pdf'
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    print(f"butterfly plot saved to {out_path}")


def plot_axis_distribution(df, df_desc, translation_type, out_folder):
    cols_to_sum = df_desc['axis'].unique().tolist()
    row_sums = df[cols_to_sum].sum(axis=1)
    df[cols_to_sum] = df[cols_to_sum].div(row_sums, axis=0)

    df.set_index('lang', inplace=True)

    df[cols_to_sum].plot(kind='bar', 
                        stacked=True, 
                        colormap='tab20b', 
                        figsize=(10, 6),
                        width=0.7)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("")
    plt.ylabel("Distribution of toxic translations")

    out_path = f'{out_folder}/mutox_demographic_axis_{translation_type}.pdf'
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    print(f"distribution plot saved to {out_path}")
    
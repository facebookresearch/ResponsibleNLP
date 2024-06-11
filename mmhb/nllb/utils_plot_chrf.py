# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def plot_chrf(df, out_folder, translation_type):
    # set all text font
    matplotlib.rc("font", size=18)

    # Set the positions and width for the bars
    pos = list(range(len(df['masculine'])))
    width = 0.25

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # https://personal.sron.nl/~pault/#sec:qualitative
    color_map = {
        "blue": "#004488",
        "red": "#BB5566",
        "yellow": "#DDAA33",
    }
    
    # Create a bar plot for each category
    if translation_type == "en-to-xx":
        columns = [('masculine', 'Masculine Reference', color_map['blue']), ('feminine', 'Feminine Reference', color_map['red']), ('both', 'Multi-Reference', color_map['yellow'])]
    else:
        columns = [('masculine', 'Masculine Source', color_map['blue']), ('feminine', 'Feminine Source', color_map['red'])]
    
    for i, (col, label, color) in enumerate(columns):
        plt.bar([p + width * i for p in pos], df[col], width, color=color, label=label)

    # Set the position of the x ticks
    if translation_type == "en-to-xx":
        ax.set_xticks([p + 1 * width for p in pos])
    else:
        ax.set_xticks([p + 0.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['lang_code'])

    # Adding the legend and showing the plot
    plt.xlabel('Target Language')
    plt.ylabel('chrF score')

    if translation_type == "en-to-xx":
        plt.legend(['Masculine Reference', 'Feminine Reference', 'Multi-Reference'], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    else:
        plt.legend(['Masculine Source', 'Feminine Source'], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    
    plt.grid(axis='y')

    # Show the plot
    out_path = f'{out_folder}/chrf_{translation_type}.pdf'
    plt.savefig(out_path, dpi=500, bbox_inches='tight')


def plot_chrf_demo(df, out_folder, translation_type):
    # set all text font
    matplotlib.rc("font", size=18)
    
    # Set the positions and width for the bars
    pos = list(range(len(df['masculine'])))
    width = 0.25

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # https://personal.sron.nl/~pault/#sec:qualitative
    color_map = {
        "blue": "#004488",
        "red": "#BB5566",
        "yellow": "#DDAA33",
    }
    
    # Create a bar plot for each category
    if translation_type == "en-to-xx":
        columns = [('masculine', 'Masculine Reference', color_map['blue']), ('feminine', 'Feminine Reference', color_map['red']), ('both', 'Multi-Reference', color_map['yellow'])]
    else:
        columns = [('masculine', 'Masculine Source', color_map['blue']), ('feminine', 'Feminine Source', color_map['red'])]
    
    for i, (col, label, color) in enumerate(columns):
        plt.bar([p + width * i for p in pos], df[col], width, color=color, label=label)

    # Set the position of the x ticks
    if translation_type == "en-to-xx":
        ax.set_xticks([p + 1 * width for p in pos])
    else:
        ax.set_xticks([p + 0.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['axis'], rotation=90)

    # Adding the legend and showing the plot
    plt.xlabel('Demographic Axis')
    plt.ylabel('chrF score')

    if translation_type == "en-to-xx":
        plt.legend(['Masculine Reference', 'Feminine Reference', 'Multi-Reference'], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    else:
        plt.legend(['Masculine Source', 'Feminine Source'], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    
    plt.grid(axis='y')

    # Show the plot
    out_path = f'{out_folder}/chrf_demo_{translation_type}.pdf'
    plt.savefig(out_path, dpi=500, bbox_inches='tight')

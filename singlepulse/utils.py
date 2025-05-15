import matplotlib.pyplot as plt
import pandas as pd


def extract_string(text: str, first_character: str, second_character: str, right: bool = True):
    """
    Looks for the index of "first_character" in the string "text" and looks for the index of the first occurence of "second_character" relative to it.
    If right = True (default), the first occurence of "second_character" is searched to the right.
    """
    if right:
        first_index = text.find(first_character)
        second_index = text.find(second_character, first_index+len(first_character), len(text))
        if first_index != -1 and second_index != -1:
            value = text[first_index+len(first_character):second_index]
            return value
        return None
    else:
        first_index = text.find(first_character)
        second_index = text.rfind(second_character, 0, first_index)
        if first_index != -1 and second_index != -1:
            value = text[second_index+len(second_character):first_index]
            return value
        return None


def generate_colors(n, colormap_name: str = 'viridis'):
    colormap = plt.get_cmap(colormap_name)
    colors = [colormap(i/n) for i in range(n)]
    return colors


def generate_marker_list(n, available_markers):
    # Wenn n größer als die Anzahl der verfügbaren Marker ist, wiederhole sie
    markers = [available_markers[i % len(available_markers)] for i in range(n)]
    return markers


def check_nanomodul(file):
    data = pd.read_csv(file, skiprows=0)
    if type(data.iloc[0, 0]) == str:
        return False
    else:
        return True
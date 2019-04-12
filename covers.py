import os

import numpy as np
import scipy
import scipy.cluster
import scipy.misc
from PIL import Image
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from concorde.tsp import TSPSolver


def rgb2hex(r, g, b):
    return f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'


def color_difference(a, b):
    a_rgb = sRGBColor(a[0] / 255, a[1] / 255, a[2] / 255)
    b_rgb = sRGBColor(b[0] / 255, b[1] / 255, b[2] / 255)
    a_lab = convert_color(a_rgb, LabColor)
    b_lab = convert_color(b_rgb, LabColor)
    return delta_e_cie2000(a_lab, b_lab)


def get_colors(img_, num_clusters=5):
    img__ = img_.resize((150, 150))  # optional, to reduce time
    ar = np.asarray(img__)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, num_clusters)
    # print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences

    codes = list(codes)
    counts = list(counts)
    colors = [(code, count) for count, code in sorted(zip(counts, codes), reverse=True)]

    print(colors)


if __name__ == '__main__':
    for f in os.listdir('images'):
        img = Image.open('images/' + f)
        print(f)
        print(get_colors(img))

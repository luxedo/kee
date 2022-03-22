"""
kee

Create ASCII (ass-kee) art as a pdf and print it!

Copyright (C) 2022 Luiz Eduardo Amaral <luizamaral306@gmail.com>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import colorsys
import math
import re
import tempfile
from os import path
from subprocess import run
from xml.etree import ElementTree as ET

import numpy as np
from skimage import color, exposure, io, transform
from sklearn.cluster import KMeans

__version__ = "1.0.0"
MODULE_DIR = path.dirname(__file__)
RGB_PROFILE = "sRGB2014.icc"
CMYK_PROFILE = "USWebCoatedSWOP.icc"
PAPER_SIZES = {
    "A0": (841, 1189),
    "A1": (594, 841),
    "A2": (420, 594),
    "A3": (297, 420),
    "A4": (210, 297),
    "A5": (148, 210),
    "B0": (1000, 1414),
    "B1": (707, 1000),
    "B2": (500, 707),
    "B3": (353, 500),
    "B4": (250, 353),
    "B5": (176, 250),
}
DPI = 300
SATURATION_THRESHOLD = 0.01
VALUE_THRESHOLD = 0.01
DEFAULT_PALETTE = "@%#*+=-:. "


def mm2in(mm):
    return mm / 25.4


def pixel_to_ascii(pixel, palette):
    return palette[round(pixel * (len(palette) - 1))]


def load_image(image_name, black_threshold, white_threshold):
    img = io.imread(image_name) / 256
    img = np.clip(img, black_threshold, white_threshold)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    # img = exposure.equalize_adapthist(img, clip_limit=0.03)
    # img = (img - img.min()) / (img.max() - img.min())
    return img


def build_ascii_art(
    image_name,
    width,
    palette,
    black_threshold,
    white_threshold,
    letter_spacing,
):
    img = load_image(image_name, black_threshold, white_threshold)
    height = round(
        width / img.shape[1] * img.shape[0] / 2 / letter_spacing
    )  # 2 is the height to width ratio of characters
    img = transform.resize(img, (height, width))
    gray = img.copy()
    if gray.ndim == 3:
        gray = color.rgb2gray(gray)
    return (
        "\n".join(
            ["".join([pixel_to_ascii(pixel, palette) for pixel in row]) for row in gray]
        ),
        img,
    )


def rgb2hex(rgb):
    return [
        "#" + "".join([f"{int(c):02x}" for c in (col * 255).round()]) for col in rgb
    ]


def hex2rgb(_hex):
    _hex = _hex.removeprefix("#")
    return tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))


def build_palette(color_array, n_colors, value_offset):
    """
    Return palette in descending order of frequency
    """
    kmeans = KMeans(n_clusters=n_colors)
    X = color_array.reshape((-1, 3))
    kmeans.fit(X)
    colors_ = kmeans.cluster_centers_
    if value_offset != 0:
        hls = [colorsys.rgb_to_hls(*c) for c in colors_]
        colors_ = np.array(
            [
                colorsys.hls_to_rgb(
                    *np.clip(
                        [
                            c[0] + value_offset[0],
                            c[1] + value_offset[1],
                            c[2] + value_offset[2],
                        ],
                        0,
                        1,
                    )
                )
                for c in hls
            ]
        )
    return kmeans, kmeans.cluster_centers_, rgb2hex(colors_)


def split_layers(
    ascii_art,
    foreground_color,
    highlight,
    highlight_color,
    colors,
    colors_palette,
    colors_offset,
    img,
):
    art_split = [list(row) for row in ascii_art.split("\n")]
    rows = len(art_split)
    columns = len(art_split[0])
    text_layers = []
    color_layers = []

    # 1. Split text highlight from art
    for h, col in zip(highlight, highlight_color):
        new_layer = "\n".join([" " * columns for _ in range(rows)])
        for i in re.finditer(h, ascii_art):
            left, right = i.span()
            new_layer = new_layer[:left] + h + new_layer[right:]
            ascii_art = ascii_art[:left] + " " * len(h) + ascii_art[right:]
        text_layers.append(new_layer)
        color_layers.append(col)

    # 2. Split saturation highlight from art
    art_split = [list(row) for row in ascii_art.split("\n")]
    if colors:
        _, saturation, value = color.rgb2hsv(img).T
        saturation, value = saturation.T, value.T
        colors = img[(saturation > SATURATION_THRESHOLD) & (value > VALUE_THRESHOLD)]
        model, centers, colors = build_palette(colors, colors_palette, colors_offset)
        closest = np.linalg.norm([img - center for center in centers], axis=3)
        closest = closest.argmin(axis=0)
        hs_layers = [
            [[" " for _ in range(columns)] for _ in range(rows)] for _ in colors
        ]
        for i, j in np.argwhere(saturation > SATURATION_THRESHOLD):
            hs_layers[closest[i][j]][i][j] = art_split[i][j]
            art_split[i][j] = " "
        hs_layers = ["\n".join(["".join(row) for row in layer]) for layer in hs_layers]
        text_layers.extend(hs_layers)
        color_layers.extend(colors)
        ascii_art = "\n".join(["".join(row) for row in art_split])

    text_layers.append(ascii_art)
    color_layers.append(foreground_color)
    return text_layers, color_layers


def blank_svg(width_mm, height_mm, width, height, background_color):
    svg = ET.Element(
        "svg",
        **{
            "width": f"{width_mm}mm",
            "height": f"{height_mm}mm",
            "version": "1.1",
            "viewBox": f"0 0 {width} {height}",
            "xmlns": "http://www.w3.org/2000/svg",
            "xmlns:svg": "http://www.w3.org/2000/svg",
        },
    )
    # defs = ET.SubElement(svg, "defs")
    # style = ET.SubElement(defs, "style", type="text/css")
    # with open("stylesheet.css", "r") as fp:
    #     style.text = f"\n{fp.read()}\n"
    g = ET.Element("g")
    svg.append(g)

    rect = ET.Element(
        "rect",
        **{
            "x": "0",
            "y": "0",
            "width": f"{width}",
            "height": f"{height}",
            "fill": background_color,
        },
    )
    g.append(rect)
    return svg, g


def write_text(g, text_layers, color_layers, center, font_size, letter_spacing):
    for layer, color_ in zip(text_layers, color_layers):
        text = ET.Element(
            "text",
            **{
                "x": "50%",
                "y": "50%",
                "text-anchor": "middle",
                "xml:space": "preserve",
                "fill": color_,
                "transform": f"translate(0, {center})",
                "style": f"font-size: {font_size};font-family: Courier;font-weight: bold;font-kerning: none;font-variant-ligatures: none;letter-spacing: {letter_spacing - 0.8}em;",
            },
        )
        layer_split = layer.split("\n")
        rows = len(layer_split)
        for i, line in enumerate(layer_split):
            tspan = ET.Element(
                "tspan",
                **{
                    "x": "50%",
                    "y": f"{(i-rows/2) * letter_spacing}em",
                },
            )
            tspan.text = line
            text.append(tspan)
        g.append(text)


def write_header(g, header_text, header_font_size, foreground_color):
    header = ET.Element(
        "text",
        **{
            "x": f"{2*header_font_size}",
            "y": f"{2*header_font_size}",
            "fill": foreground_color,
            "style": f"font-size: {header_font_size};",
        },
    )
    for line in header_text.split("\n"):
        tspan = ET.Element(
            "tspan",
            **{
                "x": f"{2*header_font_size}",
                "dy": "1em",
            },
        )
        tspan.text = line
        header.append(tspan)
    g.append(header)


def print_art(text_layers, color_layers, background_color):
    if len(color_layers) == 1:
        print("".join(text_layers[0]))
        return
    depth = len(text_layers)
    fg_pattern = "\x1b[38;2;{0};{1};{2}m"
    bg_pattern = "\x1b[48;2;{0};{1};{2}m"
    rgb_layers = [fg_pattern.format(*hex2rgb(_hex)) for _hex in color_layers]
    prev_color_idx = -1
    print(bg_pattern.format(*hex2rgb(background_color)), end="")
    for i in range(len(text_layers[0])):
        for d in range(depth):
            cell = text_layers[d][i]
            if cell != " ":
                if prev_color_idx == d:
                    print(cell, end="")
                else:
                    print(rgb_layers[d] + cell, end="")
                    prev_color_idx = d
                break
        else:
            print(" ", end="")


def kee(
    image_name,
    columns,
    palette,
    invert_palette,
    font_size,
    black_threshold,
    white_threshold,
    letter_spacing,
    foreground_color,
    background_color,
    highlight_color,
    highlight_text,
    colors,
    colors_palette,
    colors_offset,
    # background_text,
    # background_text_size,
    # background_text_color,
    header_text,
    paper_size,
    output,
):

    if invert_palette:
        palette = palette[::-1]

    if output is not None and not output.endswith(".svg"):
        tf = tempfile.NamedTemporaryFile(suffix=".svg")
        svg_output = tf.name
    else:
        svg_output = output

    ascii_art, img = build_ascii_art(
        image_name,
        columns,
        palette,
        black_threshold,
        white_threshold,
        letter_spacing,
    )

    text_layers, color_layers = split_layers(
        ascii_art,
        foreground_color,
        highlight_text,
        highlight_color,
        colors,
        colors_palette,
        colors_offset,
        img,
    )

    if output is None:
        print_art(text_layers, color_layers, background_color)
        return

    width_mm, height_mm = PAPER_SIZES[paper_size]
    width, height = DPI * mm2in(width_mm), DPI * mm2in(height_mm)

    svg, g = blank_svg(width_mm, height_mm, width, height, background_color)

    write_text(
        g,
        text_layers,
        color_layers,
        center=height / 2,
        font_size=font_size,
        letter_spacing=letter_spacing,
    )

    if header_text is not None:
        header_font_size = 16
        write_header(g, header_text, header_font_size, foreground_color)

    tree = ET.ElementTree(svg)
    tree.write(svg_output, encoding="utf-8", xml_declaration=True)
    if not output.endswith(".svg"):
        run(
            [
                "convert",
                "-density",
                f"{DPI}",
                "-units",
                "PixelsPerInch",
                "-profile",
                path.join(MODULE_DIR, RGB_PROFILE),
                svg_output,
                "-profile",
                path.join(MODULE_DIR, CMYK_PROFILE),
                output,
            ]
        )

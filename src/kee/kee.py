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
import math
import re
import tempfile
from os import path
from subprocess import run
from xml.etree import ElementTree as ET

# import cairosvg
import numpy as np
from skimage import color, exposure, io, transform

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
__version__ = "1.0.0"


def mm2in(mm):
    return mm / 25.4


def pixel_to_ascii(pixel, palette):
    return palette[round(pixel * (len(palette) - 1))]


def build_ascii_art(
    image_name, width, palette, character_ratio, black_threshold, white_threshold
):
    img = io.imread(image_name)
    height = round(width / img.shape[0] * img.shape[1] / character_ratio)
    img = transform.resize(img, (height, width))
    img = color.rgb2gray(img)
    img = np.clip(img, black_threshold, white_threshold)
    img = (img - img.min()) / (img.max() - img.min())
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    return "\n".join(
        ["".join([pixel_to_ascii(cell, palette) for cell in row]) for row in img]
    )


def main(
    image_name,
    columns,
    palette,
    invert_palette,
    font_size,
    black_threshold,
    white_threshold,
    character_ratio,
    foreground_color,
    background_color,
    highlight_color,
    highlight,
    background_text,
    background_text_size,
    background_text_color,
    header_text,
    paper_size,
    svg_output,
    output,
):

    if invert_palette:
        palette = palette[::-1]

    if svg_output is None:
        tf = tempfile.NamedTemporaryFile(suffix=".svg")
        svg_output = tf.name

    if output is None:
        output = path.splitext(image_name)[0] + ".pdf"

    width_mm, height_mm = PAPER_SIZES[paper_size]
    width, height = (DPI * mm2in(dim) for dim in (width_mm, height_mm))

    ascii_art = build_ascii_art(
        image_name, columns, palette, character_ratio, black_threshold, white_threshold
    )
    rows = len(ascii_art.split("\n"))
    text_layers = []
    color_layers = []
    for h, c in zip(highlight, highlight_color):
        new_layer = "\n".join([" " * columns for _ in range(rows)])
        for i in re.finditer(h, ascii_art):
            left, right = i.span()
            new_layer = new_layer[:left] + h + new_layer[right:]
            ascii_art = ascii_art[:left] + " " * len(h) + ascii_art[right:]
        text_layers.append(new_layer)
        color_layers.append(c)
    text_layers = [ascii_art] + text_layers
    color_layers = [foreground_color] + color_layers

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

    if background_text is not None:
        bgt_space = 0.66
        background_text += " "
        bgt_cols = math.floor(width / background_text_size * character_ratio)
        bgt_rows = math.ceil(height / background_text_size / bgt_space) + 1
        text = ET.Element(
            "text",
            **{
                "x": "0",
                "y": "0",
                "xml:space": "preserve",
                "fill": background_text_color,
                "style": f"font-size: {background_text_size};font-family: Courier;font-weight: bold;font-kerning: none;letter-spacing: -0.05em;",
            },
        )
        i = 0
        for row in range(bgt_rows):
            bg_text = ""
            while len(bg_text) < bgt_cols:
                bg_text += background_text[i]
                i += 1
                i %= len(background_text)
            tspan = ET.Element(
                "tspan",
                **{
                    "x": "0",
                    "y": f"{bgt_space +  row * bgt_space}em",
                },
            )
            tspan.text = bg_text
            text.append(tspan)
        g.append(text)

    for layer, color_ in zip(text_layers, color_layers):
        text = ET.Element(
            "text",
            **{
                "x": "50%",
                "y": "50%",
                "text-anchor": "middle",
                "xml:space": "preserve",
                "fill": color_,
                "transform": f"translate(0, {height/2})",
                "style": f"font-size: {font_size};font-family: Courier;font-weight: bold;font-kerning: none;letter-spacing: -0.05em;",
            },
        )
        for i, line in enumerate(layer.split("\n")):
            tspan = ET.Element(
                "tspan",
                **{
                    "x": "50%",
                    "y": f"{i-rows/2}em",
                },
            )
            tspan.text = line
            text.append(tspan)
        g.append(text)

    if header_text is not None:
        header_font_size = 16
        header = ET.Element(
            "text",
            **{
                "x": f"{header_font_size}",
                "y": f"{header_font_size}",
                "fill": foreground_color,
                "style": f"font-size: {header_font_size};",
            },
        )
        for line in header_text.split("\n"):
            tspan = ET.Element(
                "tspan",
                **{
                    "x": f"{header_font_size}",
                    "dy": "1em",
                },
            )
            tspan.text = line
            header.append(tspan)
        g.append(header)

    tree = ET.ElementTree(svg)
    tree.write(svg_output, encoding="utf-8", xml_declaration=True)
    run(
        # [
        #     "inkscape",
        #     "-o",
        #     output,
        #     svg_output,
        # ]
        [
            "convert",
            "-density",
            "300",
            "-units",
            "PixelsPerInch",
            svg_output,
            output,
        ]
    )
    # cairosvg.svg2pdf(url=svg_output, write_to=output)
#!/usr/bin/env python3
"""
kee

ASCII (ASS-kee) art generator
"""
import argparse
import math
from os import path
import re
from subprocess import run
import sys
import tempfile
from xml.etree import ElementTree as ET

# import cairosvg
import numpy as np
from skimage import io, transform, color, exposure, data, filters

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
    defs = ET.SubElement(svg, "defs")
    style = ET.SubElement(defs, "style", type="text/css")
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
                "x": f"0",
                "y": f"0",
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

    for layer, color in zip(text_layers, color_layers):
        text = ET.Element(
            "text",
            **{
                "x": f"50%",
                "y": f"50%",
                "text-anchor": "middle",
                "xml:space": "preserve",
                "fill": color,
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
                    "dy": f"1em",
                },
            )
            tspan.text = line
            header.append(tspan)
        g.append(header)

    tree = ET.ElementTree(svg)
    tree.write(svg_output, encoding="utf-8", xml_declaration=True)
    run(
        [
            "inkscape",
            "-o",
            output,
            svg_output,
        ]
        # [
        #     "convert",
        #     "-density",
        #     "300",
        #     "-page",
        #     paper_size,
        #     svg_output,
        #     output,
        # ]
    )
    # cairosvg.svg2pdf(url=svg_output, write_to=output)


def hex_string(value):
    if not re.match("^#([0-9a-fA-F]{3}){1,2}", value):
        raise argparse.ArgumentTypeError("Not an hex string")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a pdf document with given picture as ASCII art."
    )
    parser.add_argument("image", help="source image file")
    parser.add_argument(
        "-w", "--width", type=int, default=480, help="Number of characters per row"
    )
    parser.add_argument(
        "-p", "--palette", type=str, default="@%#*+=-:. ", help="Character palette"
    )
    parser.add_argument(
        "-i",
        "--invert-palette",
        action="store_true",
        help="Inverts the color palette",
    )
    parser.add_argument(
        "-f",
        "--font-size",
        type=float,
        default=12,
        help="Font size",
    )
    parser.add_argument(
        "-K",
        "--black-threshold",
        type=float,
        default=0,
        help="Image threshold for black",
    )
    parser.add_argument(
        "-W",
        "--white-threshold",
        type=float,
        default=1,
        help="Image threshold for white",
    )
    parser.add_argument(
        "-c",
        "--character-ratio",
        type=float,
        default=2.0,
        help="Character height to width ratio",
    )
    parser.add_argument(
        "-F",
        "--foreground-color",
        type=hex_string,
        default="#111",
        help="Foreground color",
    )
    parser.add_argument(
        "-B",
        "--background-color",
        type=hex_string,
        default="#fff",
        help="Background color",
    )
    parser.add_argument(
        "-G",
        "--highlight-color",
        type=hex_string,
        default=["#e11"],
        nargs="+",
        help="Highlight colors",
    )
    parser.add_argument(
        "-g",
        "--highlight",
        type=str,
        default=[],
        nargs="*",
        help="Highlight words",
    )
    parser.add_argument(
        "--background-text",
        type=str,
        default=None,
        help="Background text",
    )
    parser.add_argument(
        "--background-text-size",
        type=float,
        default=244,
        help="Background text font size",
    )
    parser.add_argument(
        "--background-text-color",
        type=hex_string,
        default="#BBB",
        help="Background text color",
    )
    parser.add_argument(
        "-H",
        "--write-header",
        action="store_true",
        help="Writes header in the document",
    )
    parser.add_argument(
        "-P",
        "--paper-size",
        choices=PAPER_SIZES.keys(),
        default="A3",
        help="Print paper size",
    )
    parser.add_argument(
        "-s",
        "--svg-output",
        default=None,  # Goes to /tmp
        type=str,
        help="Svg output file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,  # Same name as image
        type=str,
        help="Pdf output file",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args()
    command_str = " ".join([a if a.startswith("-") else f'"{a}"' for a in sys.argv[1:]])
    header_text = (
        None if not args.write_header else f"kee v{__version__}\nkee.py {command_str}"
    )
    main(
        args.image,
        args.width,
        args.palette,
        args.invert_palette,
        args.font_size,
        args.black_threshold,
        args.white_threshold,
        args.character_ratio,
        args.foreground_color,
        args.background_color,
        args.highlight_color,
        args.highlight,
        args.background_text,
        args.background_text_size,
        args.background_text_color,
        header_text,
        args.paper_size,
        args.svg_output,
        args.output,
    )

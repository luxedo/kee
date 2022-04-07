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
from io import StringIO
from os import path
from subprocess import run
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from skimage import color as skcolor
from skimage import exposure, io, transform
from sklearn.cluster import KMeans

from .layer import Layer, StackedLayers, hex2rgb, rgb2hex
from .render import ImageRender, SvgRender, TerminalRender

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


def mm2in(mm):
    return mm / 25.4


def float_to_uint8(array: np.ndarray) -> np.ndarray:
    return (array * 255).astype(np.uint8)


class Kee:
    default_character_palette = "@%#*+=-:. "
    character_ratio = 2
    _default_color = "#202020"
    _default_background_color = "#D0D0D0"

    def __init__(
        self,
        filename: str,
        width: int | None = None,
        height: int | None = None,
        default_color: tuple[int, int, int] | None = None,
        default_background_color: tuple[int, int, int] | None = None,
        black_threshold: float = 0,
        white_threshold: float = 1,
        letter_spacing: float = 0.0,
        line_height: float = 1.25,
        dpi: float = 300,
    ):
        self.filename = filename
        self.default_color = default_color or self._default_color
        self.default_background_color = (
            default_background_color or self._default_background_color
        )
        self.black_threshold = black_threshold
        self.white_threshold = white_threshold
        self.image = self.load_image(
            filename, self.black_threshold, self.white_threshold
        )
        self.letter_spacing = letter_spacing
        self.line_height = line_height
        self.line_ratio = (1 + self.letter_spacing) / (
            self.character_ratio * self.line_height
        )
        self.dpi = dpi

        if width is None and height is None:
            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
        elif width is None:
            self.height = height
            self.width = math.ceil(
                self.height
                * self.image.shape[1]
                / self.image.shape[0]
                / self.line_ratio
            )
        else:
            self.width = width
            self.height = math.ceil(
                self.width * self.image.shape[0] / self.image.shape[1] * self.line_ratio
            )
        self.cols = int(self.width)
        self.rows = int(self.height)
        self.characters = self.rows * self.cols
        self.header_text = None
        self.colored_img = transform.resize(self.image, (self.rows, self.cols)).copy()
        if self.colored_img.ndim == 3:
            self.gray_img = skcolor.rgb2gray(self.colored_img)
        else:
            self.gray_img = colored_img

    def __repr__(self):
        return (
            f'{self.__class__.__name__}("{self.filename}", {self.width}, {self.height})'
        )

    # def __str__(self):
    #     if not hasattr(self, "terminal_output"):
    #         self.build_character_palette()
    #     return self.terminal_output

    def load_image(self, filename: str, black_threshold: float, white_threshold: float):
        img = io.imread(filename) / 256
        img = np.clip(img, black_threshold, white_threshold)
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))
        return img

    def character_palette_text(
        self,
        character_palette: str = default_character_palette,
        invert_palette: bool = False,
    ) -> Layer:
        if invert_palette:
            character_palette = character_palette[::-1]
        text = "\n".join(
            [
                "".join(
                    [self.pixel_to_ascii(pixel, character_palette) for pixel in row]
                )
                for row in self.gray_img
            ]
        )
        return text

    def text_to_layers(
        self,
        text: str,
        color: Optional[tuple[int, int, int]] = None,
        background_color: Optional[tuple[int, int, int]] = None,
        color_palette: Optional[int] = None,
        color_offset: Optional[tuple[float, float, float]] = (0, 0, 0),
    ):

        layer = Layer(
            text,
            color=color,
            background_color=background_color,
        )
        if color_palette is not None:
            return self.colorize_layer(
                layer, self.colored_img, color_palette, color_offset, background_color
            )
        else:
            return layer

    def colorize_layer(
        self,
        layer: Layer,
        image: np.ndarray,
        color_palette: int,
        color_offset: Optional[tuple[float, float, float]] = (0, 0, 0),
        background_color: Optional[tuple[int, int, int]] = None,
    ):
        height, width, _ = image.shape
        color_array = image.reshape((-1, 3))
        model, centers, colors = self.build_palette(
            color_array, color_palette, color_offset
        )
        closest = np.linalg.norm([image - center for center in centers], axis=3)
        closest = closest.argmin(axis=0)
        layers = [
            Layer(
                "\n".join([" " * width for _ in range(height)]),
                color=tuple(c),
                background_color=background_color,
            )
            for c in float_to_uint8(colors)
        ]
        for i, row in enumerate(layer._text):
            for j, char in enumerate(row):
                layers[closest[i][j]][i][j] = char
        return StackedLayers(layers)

    def text_to_highlight_layers(self, text, highlight_text, highlight_color):
        highlight_layers = StackedLayers()
        for h, col in zip(highlight_text, highlight_color):
            new_layer = "\n".join([" " * self.cols for _ in range(self.rows)])
            for i in re.finditer(h, text):
                left, right = i.span()
                new_layer = new_layer[:left] + h + new_layer[right:]
                text = text[:left] + " " * len(h) + text[right:]
            highlight_layers += Layer(new_layer, col)
        return highlight_layers, text

    def add_header(self, header_text: str):
        self.header_text = header_text

    def to_image(
        self,
        layer: Layer,
        filename: Optional[str] = None,
        /,
        shape: tuple[float, float] = PAPER_SIZES["A4"],
        font_size: float = 10,
        translate_x: float = 0,
        translate_y: float = 0,
    ):
        return layer.render(
            ImageRender(
                shape,
                self.default_color,
                self.default_background_color,
                self.dpi,
                font_size,
                self.letter_spacing,
                self.line_height,
                self.header_text,
                translate_x,
                translate_y,
            ),
            filename,
        )

    @staticmethod
    def build_palette(
        color_array: np.ndarray,
        n_colors: int,
        hls_offset: tuple[float, float, float] = (0, 0, 0),
    ):
        """
        Return palette in descending order of frequency
        """
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(color_array)
        colors_ = kmeans.cluster_centers_
        hls = [colorsys.rgb_to_hls(*c) for c in colors_]
        colors_ = np.array(
            [
                colorsys.hls_to_rgb(
                    *np.clip(
                        [
                            c[0] + hls_offset[0],
                            c[1] + hls_offset[1],
                            c[2] + hls_offset[2],
                        ],
                        0,
                        1,
                    )
                )
                for c in hls
            ]
        )
        return kmeans, kmeans.cluster_centers_, colors_

    @staticmethod
    def pixel_to_ascii(pixel, palette):
        return palette[round(pixel * (len(palette) - 1))]


# def pixel_to_ascii(pixel, palette):
#     return palette[round(pixel * (len(palette) - 1))]
#
#
# def load_image(image_name, black_threshold, white_threshold):
#     img = io.imread(image_name) / 256
#     img = np.clip(img, black_threshold, white_threshold)
#     p2, p98 = np.percentile(img, (2, 98))
#     img = exposure.rescale_intensity(img, in_range=(p2, p98))
#     return img
#
#
# def build_ascii_art(
#     image_name,
#     width,
#     palette,
#     black_threshold,
#     white_threshold,
#     letter_spacing,
# ):
#     img = load_image(image_name, black_threshold, white_threshold)
#     height = round(
#         width / img.shape[1] * img.shape[0] / CHARACTER_RATIO / letter_spacing
#     )  # 2 is the height to width ratio of characters
#     img = transform.resize(img, (height, width))
#     gray = img.copy()
#     if gray.ndim == 3:
#         gray = skcolor.rgb2gray(gray)
#     return (
#         "\n".join(
#             ["".join([pixel_to_ascii(pixel, palette) for pixel in row]) for row in gray]
#         ),
#         img,
#     )
#
#
# def build_palette(color_array, n_colors, value_offset):
#     """
#     Return palette in descending order of frequency
#     """
#     kmeans = KMeans(n_clusters=n_colors)
#     X = color_array.reshape((-1, 3 if color_array.ndim == 3 else 1))
#     kmeans.fit(X)
#     colors_ = kmeans.cluster_centers_
#     if color_array.ndim == 2:
#         colors_ = [np.array(c.tolist() * 3) for c in colors_]
#     if value_offset != [0, 0, 0]:
#         hls = [colorsys.rgb_to_hls(*c) for c in colors_]
#         colors_ = np.array(
#             [
#                 colorsys.hls_to_rgb(
#                     *np.clip(
#                         [
#                             c[0] + value_offset[0],
#                             c[1] + value_offset[1],
#                             c[2] + value_offset[2],
#                         ],
#                         0,
#                         1,
#                     )
#                 )
#                 for c in hls
#             ]
#         )
#     return kmeans, kmeans.cluster_centers_, rgb2hex(colors_)
#
#
# def split_layers(
#     ascii_art,
#     foreground_color,
#     highlight,
#     highlight_color,
#     colors,
#     colors_palette,
#     colors_offset,
#     img,
# ):
#     art_split = [list(row) for row in ascii_art.split("\n")]
#     rows = len(art_split)
#     columns = len(art_split[0])
#     text_layers = []
#     color_layers = []
#
#     # 1. Split text highlight from art
#     for h, col in zip(highlight, highlight_color):
#         new_layer = "\n".join([" " * columns for _ in range(rows)])
#         for i in re.finditer(h, ascii_art):
#             left, right = i.span()
#             new_layer = new_layer[:left] + h + new_layer[right:]
#             ascii_art = ascii_art[:left] + " " * len(h) + ascii_art[right:]
#         text_layers.append(new_layer)
#         color_layers.append(col)
#
#     # 2. Split saturation highlight from art
#     art_split = [list(row) for row in ascii_art.split("\n")]
#     if colors:
#         model, centers, colors = build_palette(img, colors_palette, colors_offset)
#         if img.ndim == 2:
#             closest = np.array([(img - center) ** 2 for center in centers])
#         else:
#             closest = np.linalg.norm([img - center for center in centers], axis=3)
#         closest = closest.argmin(axis=0)
#         hs_layers = [
#             [[" " for _ in range(columns)] for _ in range(rows)] for _ in colors
#         ]
#         for i in range(len(art_split)):
#             for j in range(len(art_split[i])):
#                 hs_layers[closest[i][j]][i][j] = art_split[i][j]
#                 art_split[i][j] = " "
#         hs_layers = ["\n".join(["".join(row) for row in layer]) for layer in hs_layers]
#         text_layers.extend(hs_layers)
#         color_layers.extend(colors)
#         ascii_art = "\n".join(["".join(row) for row in art_split])
#
#     text_layers.append(ascii_art)
#     color_layers.append(foreground_color)
#     return text_layers, color_layers
#
#
# def blank_svg(width_mm, height_mm, width, height, background_color):
#     svg = ET.Element(
#         "svg",
#         **{
#             "width": f"{width_mm}mm",
#             "height": f"{height_mm}mm",
#             "version": "1.1",
#             "viewBox": f"0 0 {width} {height}",
#             "xmlns": "http://www.w3.org/2000/svg",
#             "xmlns:svg": "http://www.w3.org/2000/svg",
#         },
#     )
#     # defs = ET.SubElement(svg, "defs")
#     # style = ET.SubElement(defs, "style", type="text/css")
#     # with open("stylesheet.css", "r") as fp:
#     #     style.text = f"\n{fp.read()}\n"
#     g = ET.Element("g")
#     svg.append(g)
#
#     rect = ET.Element(
#         "rect",
#         **{
#             "x": "0",
#             "y": "0",
#             "width": f"{width}",
#             "height": f"{height}",
#             "fill": background_color,
#         },
#     )
#     g.append(rect)
#     return svg, g
#
#
# def write_text(g, text_layers, color_layers, center, font_size, letter_spacing):
#     for layer, color_ in zip(text_layers, color_layers):
#         text = ET.Element(
#             "text",
#             **{
#                 "x": "50%",
#                 "y": "50%",
#                 "text-anchor": "middle",
#                 "xml:space": "preserve",
#                 "fill": color_,
#                 "transform": f"translate(0, {center})",
#                 "style": f"font-size: {font_size};font-family: Courier;font-weight: bold;font-kerning: none;font-variant-ligatures: none;letter-spacing: {letter_spacing - 0.7}em;",
#             },
#         )
#         layer_split = layer.split("\n")
#         rows = len(layer_split)
#         for i, line in enumerate(layer_split):
#             tspan = ET.Element(
#                 "tspan",
#                 **{
#                     "x": "50%",
#                     "y": f"{(i-rows/2) * letter_spacing}em",
#                 },
#             )
#             tspan.text = line
#             text.append(tspan)
#         g.append(text)
#
#
# def write_header(g, header_text, header_font_size, foreground_color):
#     header = ET.Element(
#         "text",
#         **{
#             "x": f"{2*header_font_size}",
#             "y": f"{2*header_font_size}",
#             "fill": foreground_color,
#             "style": f"font-size: {header_font_size};",
#         },
#     )
#     for line in header_text.split("\n"):
#         tspan = ET.Element(
#             "tspan",
#             **{
#                 "x": f"{2*header_font_size}",
#                 "dy": "1em",
#             },
#         )
#         tspan.text = line
#         header.append(tspan)
#     g.append(header)
#
#
# def print_art(text_layers, color_layers, background_color):
#     if len(color_layers) == 1:
#         print("".join(text_layers[0]))
#         return
#     depth = len(text_layers)
#     fg_pattern = "\x1b[38;2;{0};{1};{2}m"
#     bg_pattern = "\x1b[48;2;{0};{1};{2}m"
#     if len(background_color) < 7:
#         background_color = f"#{background_color[1] * 2}{background_color[2] * 2}{background_color[3] * 2}"
#     color_layers = [
#         f"#{c[1] * 2}{c[2] * 2}{c[3] * 2}" if len(c) < 7 else c for c in color_layers
#     ]
#     rgb_layers = [fg_pattern.format(*hex2rgb(_hex)) for _hex in color_layers]
#     prev_color_idx = -1
#     print(bg_pattern.format(*hex2rgb(background_color)), end="")
#     for i in range(len(text_layers[0])):
#         for d in range(depth):
#             cell = text_layers[d][i]
#             if cell != " ":
#                 if prev_color_idx == d:
#                     print(cell, end="")
#                 else:
#                     print(rgb_layers[d] + cell, end="")
#                     prev_color_idx = d
#                 break
#         else:
#             print(" ", end="")
#
#
# def kee(
#     image_name,
#     columns,
#     palette,
#     invert_palette,
#     font_size,
#     black_threshold,
#     white_threshold,
#     letter_spacing,
#     foreground_color,
#     background_color,
#     highlight_color,
#     highlight_text,
#     colors,
#     colors_palette,
#     colors_offset,
#     # background_text,
#     # background_text_size,
#     # background_text_color,
#     header_text,
#     paper_size,
#     landscape,
#     output,
# ):
#
#     if invert_palette:
#         palette = palette[::-1]
#
#     ascii_art, image = build_ascii_art(
#         image_name,
#         columns,
#         palette,
#         black_threshold,
#         white_threshold,
#         letter_spacing,
#     )
#     ascii_to_document(
#         image,
#         ascii_art,
#         font_size,
#         letter_spacing,
#         foreground_color,
#         background_color,
#         highlight_color,
#         highlight_text,
#         colors,
#         colors_palette,
#         colors_offset,
#         # background_text,
#         # background_text_size,
#         # background_text_color,
#         header_text,
#         paper_size,
#         landscape,
#         output,
#     )
#
#
# def ascii_to_document(
#     image,
#     ascii_art,
#     font_size,
#     letter_spacing,
#     foreground_color,
#     background_color,
#     highlight_color,
#     highlight_text,
#     colors,
#     colors_palette,
#     colors_offset,
#     # background_text,
#     # background_text_size,
#     # background_text_color,
#     header_text,
#     paper_size,
#     landscape,
#     output,
# ):
#     if output is not None and not output.endswith(".svg"):
#         tf = tempfile.NamedTemporaryFile(suffix=".svg")
#         svg_output = tf.name
#     else:
#         svg_output = output
#
#     text_layers, color_layers = split_layers(
#         ascii_art,
#         foreground_color,
#         highlight_text,
#         highlight_color,
#         colors,
#         colors_palette,
#         colors_offset,
#         image,
#     )
#
#     if output is None:
#         print_art(text_layers, color_layers, background_color)
#         return
#
#     width_mm, height_mm = PAPER_SIZES[paper_size]
#     if landscape:
#         width_mm, height_mm = height_mm, width_mm
#     width, height = DPI * mm2in(width_mm), DPI * mm2in(height_mm)
#
#     svg, g = blank_svg(width_mm, height_mm, width, height, background_color)
#
#     write_text(
#         g,
#         text_layers,
#         color_layers,
#         center=height / 2,
#         font_size=font_size,
#         letter_spacing=letter_spacing,
#     )
#
#     if header_text is not None:
#         header_font_size = 16
#         write_header(g, header_text, header_font_size, foreground_color)
#
#     tree = ET.ElementTree(svg)
#     tree.write(svg_output, encoding="utf-8", xml_declaration=True)
#     if not output.endswith(".svg"):
#         run(
#             [
#                 "convert",
#                 "-density",
#                 f"{DPI}",
#                 "-units",
#                 "PixelsPerInch",
#                 "-profile",
#                 path.join(MODULE_DIR, RGB_PROFILE),
#                 svg_output,
#                 "-profile",
#                 path.join(MODULE_DIR, CMYK_PROFILE),
#                 output,
#             ]
#         )
#
#
# def build_ascii_art2(
#     image_name,
#     text_name,
#     columns,
#     font_size,
#     black_threshold,
#     white_threshold,
# ):
#     img = load_image(image_name, black_threshold, white_threshold)
#     width = columns
#     height = math.ceil(width * img.shape[0] / img.shape[1])
#     img = transform.resize(img, (height, width))
#     with open(text_name, "r") as fp:
#         text = fp.read()
#
#     text = re.sub("\\n+", " ", text)
#     text = re.sub("[ ]+", " ", text)
#     text = text.replace("\n", "¶ ")
#
#     text = "\n".join(
#         text[i * columns : (i + 1) * columns] for i in range(math.ceil(height))
#     )
#     text = text + " " * ((len(text) % columns) - columns)
#
#     return text, img
#
#
# def kee2(
#     image_name,
#     text_name,
#     width,
#     font_size,
#     black_threshold,
#     white_threshold,
#     letter_spacing,
#     foreground_color,
#     background_color,
#     highlight_color,
#     highlight_text,
#     colors,
#     colors_palette,
#     colors_offset,
#     # background_text,
#     # background_text_size,
#     # background_text_color,
#     header_text,
#     paper_size,
#     landscape,
#     output,
# ):
#     ascii_art, image = build_ascii_art2(
#         image_name,
#         text_name,
#         width,
#         font_size,
#         black_threshold,
#         white_threshold,
#     )
#     if not colors:
#         if image.ndim != 2:
#             image = color.rgb2gray(image)
#         colors = True
#
#     ascii_to_document(
#         image,
#         ascii_art,
#         font_size,
#         letter_spacing,
#         foreground_color,
#         background_color,
#         highlight_color,
#         highlight_text,
#         colors,
#         colors_palette,
#         colors_offset,
#         # background_text,
#         # background_text_size,
#         # background_text_color,
#         header_text,
#         paper_size,
#         landscape,
#         output,
#     )

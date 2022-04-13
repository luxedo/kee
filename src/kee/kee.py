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


def mm2in(mm):
    return mm / 25.4


def float_to_uint8(array: np.ndarray) -> np.ndarray:
    return (array * 255).astype(np.uint8)


class Kee:
    default_character_palette = "@%#*+=-:. "
    character_ratio = 2
    _default_color = "#202020"
    _default_background_color = "#D0D0D0"
    portrait_sizes = {
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
    landscape_sizes = {key: tuple(value[::-1]) for key, value in portrait_sizes.items()}
    # TODO: Figure out terminal and SVG default line height
    real_line_height = 1.14  # Needs citation. Why 1em is not the real line height?

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
            self.character_ratio * self.line_height / self.real_line_height
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
            self.gray_img = self.colored_img

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
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)  # Always 3 channels please
        img = np.clip(img, black_threshold, white_threshold)
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))
        return img

    def character_palette_text(
        self,
        character_palette: str = default_character_palette,
        invert_palette: bool = False,
    ) -> str:
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
        colors_palette: Optional[int] = None,
        hsv_offset: Optional[tuple[float, float, float]] = (0, 0, 0),
    ):
        layer = Layer(
            text,
            color=color,
            background_color=background_color,
        )
        if colors_palette is not None:
            return self.colorize_layer(
                layer, self.colored_img, colors_palette, hsv_offset, background_color
            )
        else:
            return layer

    def colorize_layer(
        self,
        layer: Layer,
        image: np.ndarray,
        colors_palette: int,
        hsv_offset: Optional[tuple[float, float, float]] = (0, 0, 0),
        background_color: Optional[tuple[int, int, int]] = None,
    ):
        height, width, _ = image.shape
        color_array = image.reshape((-1, 3))
        model, centers, colors = self.build_palette(
            color_array, colors_palette, hsv_offset
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
        shape: tuple[float, float] = portrait_sizes["A4"],
        font_size: float = 10,
        font_family: str = "Courier",
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
                font_family,
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
        hsv_offset: tuple[float, float, float] = (0, 0, 0),
    ):
        """
        Return palette in descending order of frequency
        """
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(color_array)
        colors_ = kmeans.cluster_centers_
        hsv = [colorsys.rgb_to_hsv(*c) for c in colors_]
        colors_ = np.array(
            [
                colorsys.hsv_to_rgb(
                    *np.clip(
                        [
                            c[0] + hsv_offset[0],
                            c[1] + hsv_offset[1],
                            c[2] + hsv_offset[2],
                        ],
                        0,
                        1,
                    )
                )
                for c in hsv
            ]
        )
        return kmeans, kmeans.cluster_centers_, colors_

    @staticmethod
    def pixel_to_ascii(pixel, palette):
        return palette[round(pixel * (len(palette) - 1))]

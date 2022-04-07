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
from collections.abc import Iterable
import colorsys
from itertools import zip_longest
import math
import re
import tempfile
from os import path
from subprocess import run
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from sklearn.cluster import KMeans

from .render import Render, TerminalRender


def rgb2hex(red, green, blue):
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"


def hex2rgb(_hex):
    _hex = _hex.removeprefix("#")
    if len(_hex) == 3:
        _hex = f"{_hex[0]}{_hex[0]}{_hex[1]}{_hex[1]}{_hex[2]}{_hex[2]}"
    return tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))


class Layer:
    """
    Abstract Base Class for ASCII art layers.

    All layers must have a text and only one color and one background_color.

    Adding Layers promotes them to StackedLayers
    """

    def __init__(
        self,
        text: str,
        color: Optional[tuple[int, int, int]] = None,
        background_color: Optional[tuple[int, int, int]] = None,
        render: Render = TerminalRender(),
    ):
        self.color = color
        self.background_color = background_color
        self._render = render
        self._text = [list(l) for l in text.split("\n")]
        self.rows = len(self._text)

    def __len__(self):
        return len(self._text)

    def __iter__(self):
        for row in self._text:
            yield "".join(row)

    def __getitem__(self, index):
        return self._text[index]

    def __str__(self):
        return self.render()

    def __add__(self, other):
        return StackedLayers([self, other])

    @property
    def text(self) -> str:
        return "\n".join("".join(l) for l in self._text)

    def render(
        self, optional_render: Optional[Render] = None, filename: Optional[str] = None
    ):
        if optional_render is None:
            return self._render(self, filename)
        else:
            return optional_render(self, filename)


class StackedLayersRow:
    def __init__(self, rows, merge_row):
        self.rows = rows
        self.merge_row = merge_row

    def __repr__(self):
        return str(
            [self.rows[channel][index] for index, channel in enumerate(self.merge_row)]
        )

    def __getitem__(self, index):
        channel = self.merge_row[index]
        return self.rows[channel][index]

    def __setitem__(self, index, value):
        channel = self.merge_row[index]
        self.rows[channel][index] = value


class StackedLayers(Layer):
    def __init__(
        self, layers: Layer | Iterable[Layer] = [], render: Render = TerminalRender()
    ):
        self._render = render
        self.layers = []
        if type(layers) == Layer:
            layers = [layers]
        for l in layers:
            if type(l) == Layer:
                self.layers.append(l)
            elif isinstance(l, StackedLayers):
                self.layers.extend(l.layers)
            else:
                raise ValueError(f"Object {layer} is not a Layer type.")
        self.rows = max([layer.rows for layer in self.layers], default=0)
        self.merged: list[list[int]] = []
        for rows in zip_longest(*self.layers, fillvalue=""):
            self.merged.append([])
            for cell in zip_longest(*rows):
                spaces = [i for i, char in enumerate(cell) if char == " "]
                for i, char in enumerate(cell):
                    if char is not None and char != " ":
                        break
                if i in spaces or char is None:
                    i = spaces[0]
                self.merged[-1].append(i)
        self._text: list[StackedLayersRow] = [
            StackedLayersRow(
                [layer._text[i] if i < layer.rows else None for layer in self.layers],
                merge_row,
            )
            for i, merge_row in enumerate(self.merged)
        ]


class SvgLayer:
    def __init__(self, layers, width_mm, height_mm, background_color):
        self.layers = layers
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.background_color = background_color
        self.svg = self._blank_svg(width_mm, height_mm)
        self.g = ET.Element("g")
        self.svg.append(self.g)
        self.background_rect = self._set_background(self.g, background_color)
        self.text = self._write_text(self.g, self.layers, 16, 1)
        self.tree = ET.ElementTree(self.svg)

    def _blank_svg(self, width_mm, height_mm):
        svg = ET.Element(
            "svg",
            **{
                "width": f"{self.width_mm}mm",
                "height": f"{self.height_mm}mm",
                "version": "1.1",
                "viewBox": f"0 0 {self.width_mm} {self.height_mm}",
                "xmlns": "http://www.w3.org/2000/svg",
                "xmlns:svg": "http://www.w3.org/2000/svg",
            },
        )
        return svg

    def _set_background(self, g, background_color):
        rect = ET.Element(
            "rect",
            **{
                "x": "0",
                "y": "0",
                "width": f"{self.width_mm}",
                "height": f"{self.height_mm}",
                "fill": background_color,
            },
        )
        g.append(rect)
        return rect

    def _write_text(self, g, layer, font_size, letter_spacing):
        text = ET.Element(
            "text",
            **{
                "x": "50%",
                "y": "50%",
                "text-anchor": "middle",
                "xml:space": "preserve",
                "fill": f"{'#AAA'}",
                "transform": f"translate(0, {self.height_mm/2})",
                "style": f"font-size: {font_size};font-family: Courier;font-weight: bold;font-kerning: none;font-variant-ligatures: none;letter-spacing: 0em;",
            },
        )
        prev_color = None
        prev_bg_color = None
        flat = ""
        rows = len(layer.text.split("\n"))
        i = 0
        for char, color, bg_color in zip(
            layer.text, layer.color_list, layer.bg_color_list
        ):
            if i == 0 or char == "\n":
                tspan = ET.Element(
                    "tspan",
                    **{
                        "x": "50%",
                        "y": f"{(i-rows/2)}em",
                    },
                )
                text.append(tspan)
                i += 1
            tspan.text += char
            print(char, end="")
        #     if bg_color != prev_bg_color:
        #         flat += self.bg_colormap[bg_color]
        #         prev_bg_color = bg_color
        #     if color != prev_color:
        #         flat += self.colormap[color]
        #         prev_color = color
        #     flat += char
        # flat += self.terminal_reset
        # return flat

        # for layer, color_ in zip(text_layers, color_layers):
        #     text = ET.Element(
        #         "text",
        #         **{
        #             "x": "50%",
        #             "y": "50%",
        #             "text-anchor": "middle",
        #             "xml:space": "preserve",
        #             "fill": color_,
        #             "transform": f"translate(0, {center})",
        #             "style": f"font-size: {font_size};font-family: Courier;font-weight: bold;font-kerning: none;font-variant-ligatures: none;letter-spacing: {letter_spacing - 0.7}em;",
        #         },
        #     )
        #     layer_split = layer.split("\n")
        #     rows = len(layer_split)
        #     for i, line in enumerate(layer_split):
        #         tspan = ET.Element(
        #             "tspan",
        #             **{
        #                 "x": "50%",
        #                 "y": f"{(i-rows/2) * letter_spacing}em",
        #             },
        #         )
        #         tspan.text = line
        #         text.append(tspan)
        #     g.append(text)

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

    def write(self, filename):
        self.tree.write(filename, encoding="utf-8", xml_declaration=True)

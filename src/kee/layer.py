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
from itertools import zip_longest

import numpy as np

from .render import TerminalRender


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
        color: tuple[int, int, int] | None = None,
        background_color: tuple[int, int, int] | None = None,
    ):
        self.color = color
        self.background_color = background_color
        self._text = [list(line) for line in text.split("\n")]
        self.rows = len(self._text)

    def __len__(self):
        return len(self._text)

    def __iter__(self):
        for row in self._text:
            yield "".join(row)

    def __getitem__(self, index):
        return self._text[index]

    def __str__(self):
        return TerminalRender(self).render()

    def __add__(self, other):
        return StackedLayers([self, other])

    @property
    def text(self) -> str:
        return "\n".join("".join(line) for line in self._text)


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
        self,
        layers: Layer | Iterable[Layer] = [],
    ):
        self.layers = []
        if type(layers) == Layer:
            layers = [layers]
        for layer in layers:
            if type(layer) == Layer:
                self.layers.append(layer)
            elif isinstance(layer, StackedLayers):
                self.layers.extend(layer.layers)
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

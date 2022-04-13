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
import tempfile
from os import path
from subprocess import run
from xml.etree import ElementTree as ET

from . import layer as kee_layer


def mm2in(mm):
    return mm / 25.4


class Render:
    def __call__(self, layers, filename: str | None = None) -> str:
        if type(layers) == kee_layer.Layer:
            layers = kee_layer.StackedLayers([layers])
        if filename:
            with open(filename, "w") as fp:
                fp.write(str(layers))
        return str(layers)


class TerminalRender(Render):
    terminal_reset = "\x1b[0m"
    terminal_default_color = "\x1b[39m"
    terminal_default_bg_color = "\x1b[49m"
    terminal_color = "\x1b[38;2;{red};{green};{blue}m"
    terminal_bg_color = "\x1b[48;2;{red};{green};{blue}m"

    def __call__(self, layers, filename: str | None = None) -> str:
        if type(layers) == kee_layer.Layer:
            layers = kee_layer.StackedLayers([layers])

        layer_colors = [layer.color for layer in layers.layers]
        layer_colors = [
            self.terminal_color.format(red=lc[0], green=lc[1], blue=lc[2])
            if lc is not None
            else self.terminal_default_color
            for lc in layer_colors
        ]
        layer_bg_colors = [layer.background_color for layer in layers.layers]
        layer_bg_colors = [
            self.terminal_bg_color.format(red=lc[0], green=lc[1], blue=lc[2])
            if lc is not None
            else self.terminal_default_bg_color
            for lc in layer_bg_colors
        ]

        text = ""
        for text_row, index_row in zip(layers._text, layers.merged):
            for char, index, prev_index in zip(text_row, index_row, [-1] + index_row):
                if index != prev_index:
                    text += layer_colors[index]
                    text += layer_bg_colors[index]
                text += char
            text += "\n"
        text += self.terminal_reset
        if filename:
            with open(filename, "w") as fp:
                fp.write(text)
        return text


class SvgRender(Render):
    base_dpi = 96

    def __init__(
        self,
        shape: tuple[float, float],
        color: tuple[int, int, int],
        background_color: tuple[int, int, int],
        dpi: float,
        font_size: float,
        font_family: str,
        letter_spacing: float,
        line_height: float,
        header_text: str | None,
        translate_x: float = 0,
        translate_y: float = 0,
    ):
        width_mm, height_mm = shape
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.dpi = dpi
        self.dpi_ratio = dpi / self.base_dpi
        self.font_size = font_size
        self.font_family = font_family
        self.letter_spacing = letter_spacing
        self.line_height = line_height
        self.view_width = int(dpi * mm2in(width_mm))
        self.view_height = int(dpi * mm2in(height_mm))
        self.background_color = background_color
        self.color = color
        self.hex_color = kee_layer.rgb2hex(red=color[0], green=color[1], blue=color[2])
        self.hex_background_color = kee_layer.rgb2hex(
            red=background_color[0], green=background_color[1], blue=background_color[2]
        )
        self.header_text = header_text
        self.translate_x = self.font_size * self.dpi_ratio * translate_x
        self.translate_y = self.font_size * self.dpi_ratio * translate_y

    def __call__(self, layers, filename: str | None = None) -> str:
        if type(layers) == kee_layer.Layer:
            layers = kee_layer.StackedLayers([layers])
        svg = self._blank_svg(
            self.width_mm, self.height_mm, self.view_width, self.view_height
        )
        g = ET.Element("g")
        svg.append(g)
        self._set_background(g, self.hex_background_color)
        text_style = {
            "font-size": f"{self.dpi_ratio * self.font_size}px",
            "font-family": f"{self.font_family}",
            "font-weight": "bold",
            "font-kerning": "none",
            "font-variant-ligatures": "none",
            "letter-spacing": f"{self.letter_spacing}em",
        }
        text_height = self.font_size * self.line_height * self.dpi_ratio * layers.rows
        border_top = (self.view_height - text_height) / 2
        text_el_extra_props = {
            "transform": f"translate({self.translate_x}, {self.translate_y + border_top})"
        }
        self._write_text_layers(g, layers, text_style, text_el_extra_props)

        if self.header_text is not None:
            self._write_text(
                g,
                self.header_text,
                text_el_props={
                    "x": "1em",
                    "y": "1em",
                    "fill": f"{self.hex_color}",
                    "style": self.dict_to_css(text_style),
                },
                tspan_el_props={"x": "1em", "dy": "1em"},
            )

        if filename is not None:
            tree = ET.ElementTree(svg)
            tree.write(filename)
        return ET.tostring(svg, encoding="utf8", method="xml")

    def _blank_svg(self, width_mm, height_mm, view_width, view_height):
        return ET.Element(
            "svg",
            **{
                "width": f"{width_mm}mm",
                "height": f"{height_mm}mm",
                "version": "1.1",
                "viewBox": f"0 0 {view_width} {view_height}",
                "xmlns": "http://www.w3.org/2000/svg",
                "xmlns:svg": "http://www.w3.org/2000/svg",
            },
        )

    def _add_stylesheet(self, svg, stylesheet_str: str):
        style = ET.Element("style", type="text/css")
        style.text = stylesheet_str
        svg.append(style)
        return style

    def _set_background(self, g, background_color):
        rect = ET.Element(
            "rect",
            **{
                "x": "0",
                "y": "0",
                "width": f"{self.view_width}",
                "height": f"{self.view_height}",
                "fill": background_color,
            },
        )
        g.append(rect)
        return rect

    def _write_text_layers(
        self,
        g,
        layers,
        text_extra_style={},
        text_el_extra_props={},
        tspan_el_extra_props={},
    ):

        for layer in layers.layers:
            layer_color = kee_layer.rgb2hex(
                red=layer.color[0], green=layer.color[1], blue=layer.color[2]
            )
            text_style = dict(**{}, **text_extra_style)
            text_el_props = dict(
                **{
                    "x": "50%",
                    "y": "0",
                    "text-anchor": "middle",
                    "xml:space": "preserve",
                    "fill": f"{layer_color}",
                    "style": self.dict_to_css(text_style),
                },
                **text_el_extra_props,
            )
            tspan_el_props = {"x": "50%", "dy": f"{self.line_height}em"}
            self._write_text(g, layer.text, text_el_props, tspan_el_props)

    def _write_text(
        self,
        g,
        text,
        text_el_props={},
        tspan_el_props={},
    ):
        text_el = ET.Element(
            "text",
            **text_el_props,
        )
        for line in text.split("\n"):
            tspan = ET.Element(
                "tspan",
                **tspan_el_props,
            )
            tspan.text = line
            text_el.append(tspan)
        g.append(text_el)

    @staticmethod
    def dict_to_css(d):
        return "; ".join([f"{key}: {value}" for key, value in d.items()])


class ImageRender(SvgRender):
    module_dir = path.dirname(__file__)
    rgb_profile = path.join(module_dir, "sRGB2014.icc")
    cmyk_profile = path.join(module_dir, "USWebCoatedSWOP.icc")

    def __call__(self, layers, filename: str | None = None) -> str | bytes | None:
        if filename is not None and filename.endswith(".svg"):
            return super().__call__(layers, filename)

        svg_tf = tempfile.NamedTemporaryFile(suffix=".svg")
        svg_output = svg_tf.name
        super().__call__(layers, svg_output)
        if filename is None:
            pdf_tf = tempfile.NamedTemporaryFile(suffix=".pdf")
            filename = pdf_tf.name
        run(
            [
                "convert",
                "-density",
                f"{self.dpi}",
                "-units",
                "PixelsPerInch",
                "-profile",
                self.rgb_profile,
                svg_output,
                "-profile",
                self.cmyk_profile,
                filename,
            ]
        )
        if filename is None:
            with open(filename, "rb") as fp:
                return fp.read()
        return None

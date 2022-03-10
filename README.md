# kee

> Create ASCII (ass-kee) art as a pdf and print it!

<img src="examples/dst/ada.jpg" width=200 />

# Install

`kee` uses [ImageMagick](https://imagemagick.org/index.php) to create
images so please install the most recent version. Then
install `kee` with [pip](https://pip.pypa.io/en/stable/).

```bash
pip install kee
```

# Usage

```python
import kee

art = kee.ascii("Hello kee!")
```

# Command line

```bash
usage: kee [-h] [-w WIDTH] [-p PALETTE] [-i] [-f FONT_SIZE] [-K BLACK_THRESHOLD] [-W WHITE_THRESHOLD]
           [-c CHARACTER_RATIO] [-F FOREGROUND_COLOR] [-B BACKGROUND_COLOR]
           [-G HIGHLIGHT_COLOR [HIGHLIGHT_COLOR ...]] [-g [HIGHLIGHT ...]] [--background-text BACKGROUND_TEXT]
           [--background-text-size BACKGROUND_TEXT_SIZE] [--background-text-color BACKGROUND_TEXT_COLOR] [-H]
           [-P {A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5}] [-s SVG_OUTPUT] [-o OUTPUT] [-v]
           image

Creates a pdf document with given picture as ASCII art.

positional arguments:
  image                 source image file

optional arguments:
  -h, --help            show this help message and exit
  -w WIDTH, --width WIDTH
                        Number of characters per row
  -p PALETTE, --palette PALETTE
                        Character palette
  -i, --invert-palette  Inverts the color palette
  -f FONT_SIZE, --font-size FONT_SIZE
                        Font size
  -K BLACK_THRESHOLD, --black-threshold BLACK_THRESHOLD
                        Image threshold for black
  -W WHITE_THRESHOLD, --white-threshold WHITE_THRESHOLD
                        Image threshold for white
  -c CHARACTER_RATIO, --character-ratio CHARACTER_RATIO
                        Character height to width ratio
  -F FOREGROUND_COLOR, --foreground-color FOREGROUND_COLOR
                        Foreground color
  -B BACKGROUND_COLOR, --background-color BACKGROUND_COLOR
                        Background color
  -G HIGHLIGHT_COLOR [HIGHLIGHT_COLOR ...], --highlight-color HIGHLIGHT_COLOR [HIGHLIGHT_COLOR ...]
                        Highlight colors
  -g [HIGHLIGHT ...], --highlight [HIGHLIGHT ...]
                        Highlight words
  --background-text BACKGROUND_TEXT
                        Background text
  --background-text-size BACKGROUND_TEXT_SIZE
                        Background text font size
  --background-text-color BACKGROUND_TEXT_COLOR
                        Background text color
  -H, --write-header    Writes header in the document
  -P {A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5}, --paper-size {A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5}
                        Print paper size
  -s SVG_OUTPUT, --svg-output SVG_OUTPUT
                        Svg output file
  -o OUTPUT, --output OUTPUT
                        Pdf output file
  -v, --version         show program's version number and exit
´´´
```

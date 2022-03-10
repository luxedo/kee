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
from os import path

import setuptools

with open("README.md", "r") as fp:
    long_description = fp.read()

rootdir = path.abspath(path.dirname(__file__))
with open(path.join(rootdir, "src", "kee", "__init__.py"), "r") as fp:
    version = (
        [line for line in fp.read().split("\n") if line.startswith("__version__")][0]
        .split("=")[1]
        .strip()
        .strip('"')
    )

setuptools.setup(
    name="kee",
    version=version,
    description="Create ASCII (ass-kee) art as a pdf and print it!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Luiz Eduardo Amaral",
    author_email="luizamaral306@gmail.com",
    url="https://github.com/luxedo/kee",
    license="LICENSE.md",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    scripts=["bin/kee"],
    python_requires=">=3.7",
    install_requires=["numpy>=1.19.5" "scikit-image>=0.18.1"],
    keywords="ASCII art print pdf",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Artistic Software",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

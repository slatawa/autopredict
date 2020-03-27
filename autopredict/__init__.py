# encoding utf-8
# Copyright: (c) 2016 autopredict
# License:   Apache License Version 2.0 (see LICENSE for details)
"""
:mod:`autopredict` -- module for using autopredict

"""

import os


here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, 'buildinfo.txt'), encoding='utf-8') as f:
        __buildinfo__ = f.read()
except:
    __buildinfo__ = "unknown"

try:
    with open(os.path.join(here, 'version.txt'), encoding='utf-8') as f:
        __version__ = f.read()
except:
    __version__ = "0.0.local"

__all__ = ('grid','classification','features','scorers','base')




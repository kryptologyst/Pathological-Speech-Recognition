"""Pathological Speech Recognition Package.

This package provides tools for recognizing speech affected by various disorders
such as Parkinson's disease, stroke, or ALS. It includes models, data processing,
training, and evaluation components specifically designed for pathological speech.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .models import *
from .data import *
from .features import *
from .losses import *
from .metrics import *
from .decoding import *
from .train import *
from .eval import *
from .utils import *

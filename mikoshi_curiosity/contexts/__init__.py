"""Built-in state space contexts."""

from .dataset import DatasetSpace
from .text import TextSpace
from .graph import GraphSpace
from .numeric import NumericSpace
from .api import APISpace

__all__ = ['DatasetSpace', 'TextSpace', 'GraphSpace', 'NumericSpace', 'APISpace']

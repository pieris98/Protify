import os
import sys

# Protify - Protein embedding and probing framework
# When using as a submodule, import from Protify.src.protify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from protify import *

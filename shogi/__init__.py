"""
Shogi Game Package

A complete Shogi (Japanese Chess) implementation with GUI using pygame.

Modules:
- piece: Piece classes and movement patterns
- board: Board representation and drawing
- rules: Game rules and move validation  
- game: Game state and main game logic
- utils: Utilities, constants, and UI helpers
- main: Entry point for the application
"""

from .piece import Piece, AnimatedPiece
from .board import Board, standard_setup, HANDICAPS
from .utils import *

__version__ = "1.0.0"
__all__ = [
    'Piece', 'AnimatedPiece', 'Board', 'standard_setup', 'HANDICAPS'
]
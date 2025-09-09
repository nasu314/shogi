"""
Basic unit tests for core Shogi game functionality after refactoring.

Tests essential components to ensure the refactoring preserved game logic:
- Piece movement and properties
- Board setup and representation
- Move generation and validation
- Game state management
"""

import unittest
import sys
import os

# Add the parent directory to sys.path to import shogi modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shogi.piece import Piece, demote_kind, STEP_MOVES, PROMOTE_MAP
from shogi.board import standard_setup, clone_board
from shogi.rules import (
    generate_all_moves_no_check, find_king, is_in_check, 
    generate_all_moves, PROMOTION_ZONE
)
from shogi.game import GameState, apply_move
from shogi.utils import BOARD_SIZE


class TestPiece(unittest.TestCase):
    """Test piece-related functionality"""
    
    def test_piece_creation(self):
        """Test piece creation and basic properties"""
        piece = Piece('K', 0)
        self.assertEqual(piece.kind, 'K')
        self.assertEqual(piece.owner, 0)
        self.assertFalse(piece.promoted)
    
    def test_piece_promotion_property(self):
        """Test promoted piece detection"""
        normal_piece = Piece('P', 0)
        promoted_piece = Piece('P+', 0)
        self.assertFalse(normal_piece.promoted)
        self.assertTrue(promoted_piece.promoted)
    
    def test_piece_clone(self):
        """Test piece cloning"""
        original = Piece('R', 1)
        cloned = original.clone()
        self.assertEqual(original.kind, cloned.kind)
        self.assertEqual(original.owner, cloned.owner)
        self.assertIsNot(original, cloned)  # Different objects
    
    def test_demote_kind(self):
        """Test piece demotion"""
        self.assertEqual(demote_kind('P+'), 'P')
        self.assertEqual(demote_kind('R+'), 'R')
        self.assertEqual(demote_kind('K'), 'K')  # Already demoted


class TestBoard(unittest.TestCase):
    """Test board-related functionality"""
    
    def test_standard_setup(self):
        """Test standard board setup"""
        board = standard_setup()
        
        # Check board dimensions
        self.assertEqual(len(board), BOARD_SIZE)
        self.assertEqual(len(board[0]), BOARD_SIZE)
        
        # Check specific pieces are in correct positions
        # King positions
        self.assertEqual(board[4][0].kind, 'K')  # Gote king
        self.assertEqual(board[4][0].owner, 1)
        self.assertEqual(board[4][8].kind, 'K')  # Sente king  
        self.assertEqual(board[4][8].owner, 0)
        
        # Pawns
        for x in range(BOARD_SIZE):
            self.assertEqual(board[x][2].kind, 'P')  # Gote pawns
            self.assertEqual(board[x][2].owner, 1)
            self.assertEqual(board[x][6].kind, 'P')  # Sente pawns
            self.assertEqual(board[x][6].owner, 0)
    
    def test_clone_board(self):
        """Test board cloning"""
        original = standard_setup()
        cloned = clone_board(original)
        
        # Check they have same content
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                orig_piece = original[x][y]
                clone_piece = cloned[x][y]
                if orig_piece is None:
                    self.assertIsNone(clone_piece)
                else:
                    self.assertEqual(orig_piece.kind, clone_piece.kind)
                    self.assertEqual(orig_piece.owner, clone_piece.owner)
                    self.assertIsNot(orig_piece, clone_piece)  # Different objects


class TestRules(unittest.TestCase):
    """Test game rules and move generation"""
    
    def setUp(self):
        """Set up test board"""
        self.board = standard_setup()
    
    def test_find_king(self):
        """Test finding king positions"""
        sente_king = find_king(self.board, 0)
        gote_king = find_king(self.board, 1)
        
        self.assertEqual(sente_king, (4, 8))
        self.assertEqual(gote_king, (4, 0))
    
    def test_pawn_moves(self):
        """Test pawn movement generation"""
        # Sente pawn at (0, 6) should move to (0, 5)
        moves = generate_all_moves_no_check(self.board, 0, 6, 0)
        self.assertIn((0, 5), moves)
        self.assertEqual(len(moves), 1)  # Pawn can only move forward
        
        # Gote pawn at (0, 2) should move to (0, 3)
        moves = generate_all_moves_no_check(self.board, 0, 2, 1)
        self.assertIn((0, 3), moves)
        self.assertEqual(len(moves), 1)
    
    def test_king_moves(self):
        """Test king movement (in empty board area)"""
        # Create an empty board and place a king
        board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[4][4] = Piece('K', 0)  # King in center
        
        moves = generate_all_moves_no_check(board, 4, 4, 0)
        # King should have 8 possible moves from center
        expected_moves = [
            (3, 3), (4, 3), (5, 3),
            (3, 4),         (5, 4),
            (3, 5), (4, 5), (5, 5)
        ]
        for move in expected_moves:
            self.assertIn(move, moves)
    
    def test_check_detection(self):
        """Test check detection"""
        # Standard setup should not be in check
        self.assertFalse(is_in_check(self.board, 0))
        self.assertFalse(is_in_check(self.board, 1))
        
        # Create a simple check scenario
        board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[4][4] = Piece('K', 0)  # Sente king
        board[4][2] = Piece('R', 1)  # Gote rook attacking king
        
        self.assertTrue(is_in_check(board, 0))
        self.assertFalse(is_in_check(board, 1))
    
    def test_promotion_zones(self):
        """Test promotion zone definitions"""
        # Sente promotion zone is ranks 0, 1, 2
        self.assertEqual(list(PROMOTION_ZONE[0]), [0, 1, 2])
        # Gote promotion zone is ranks 6, 7, 8  
        self.assertEqual(list(PROMOTION_ZONE[1]), [6, 7, 8])


class TestGameState(unittest.TestCase):
    """Test game state management"""
    
    def test_game_state_creation(self):
        """Test game state initialization"""
        state = GameState()
        
        # Check initial state
        self.assertEqual(state.turn, 0)  # Sente starts
        self.assertFalse(state.game_over)
        self.assertIsNone(state.winner)
        self.assertEqual(len(state.kifu), 0)
        self.assertEqual(len(state.hands[0]), 0)  # No pieces in hand initially
        self.assertEqual(len(state.hands[1]), 0)
    
    def test_move_application_simple(self):
        """Test applying a simple move"""
        state = GameState()
        
        # Move sente pawn from (0, 6) to (0, 5)
        move = (0, 6, 0, 5)
        initial_kifu_len = len(state.kifu)
        
        apply_move(state, move, is_cpu=True, update_history=False)
        
        # Check move was applied
        self.assertIsNone(state.board[0][6])  # Original position empty
        self.assertIsNotNone(state.board[0][5])  # New position has piece
        self.assertEqual(state.board[0][5].kind, 'P')
        self.assertEqual(state.board[0][5].owner, 0)
        
        # Check turn switched
        self.assertEqual(state.turn, 1)
        
        # Check kifu was updated
        self.assertEqual(len(state.kifu), initial_kifu_len + 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for combined functionality"""
    
    def test_complete_game_flow(self):
        """Test a simple game flow with multiple moves"""
        state = GameState()
        
        # Move 1: Sente pawn forward
        apply_move(state, (0, 6, 0, 5), is_cpu=True, update_history=False)
        self.assertEqual(state.turn, 1)
        
        # Move 2: Gote pawn forward
        apply_move(state, (8, 2, 8, 3), is_cpu=True, update_history=False)
        self.assertEqual(state.turn, 0)
        
        # Check both moves recorded in kifu
        self.assertEqual(len(state.kifu), 2)
        
        # Check board state
        self.assertIsNone(state.board[0][6])  # Sente pawn moved
        self.assertEqual(state.board[0][5].kind, 'P')
        self.assertIsNone(state.board[8][2])  # Gote pawn moved  
        self.assertEqual(state.board[8][3].kind, 'P')
    
    def test_legal_move_generation_integration(self):
        """Test that legal moves work with game state"""
        state = GameState()
        
        # Get all legal moves for sente at start
        from shogi.rules import get_legal_moves_all
        legal_moves = get_legal_moves_all(state, 0)
        
        # Should have some legal moves (pawns can move forward)
        self.assertGreater(len(legal_moves), 0)
        
        # All pawn moves should be legal
        pawn_moves = [(x, 6, x, 5) for x in range(BOARD_SIZE)]
        for move in pawn_moves:
            self.assertIn(move, legal_moves)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
"""
Piece classes, movement patterns, and piece-related constants for Shogi game.

This module defines:
- Piece class representing individual game pieces
- AnimatedPiece class for falling animation effects  
- Movement patterns and constants for all piece types
- Piece value mappings and utility functions
"""

from typing import Optional, Dict, List, Tuple
import random


class Piece:
    """軽量な駒表現 (__slots__ でオブジェクト生成・GC負荷を削減)。"""
    __slots__ = ("kind", "owner")
    
    def __init__(self, kind: str, owner: int):
        self.kind = kind
        self.owner = owner

    def clone(self) -> 'Piece':
        # clone が多用される箇所は最適化で極力排除するが、互換性のため残す
        return Piece(self.kind, self.owner)

    @property
    def promoted(self) -> bool:
        return self.kind.endswith('+')

    def __repr__(self) -> str:
        return f"{self.kind}{'S' if self.owner==0 else 'G'}"


class AnimatedPiece:
    """投了時の駒落下アニメーション用クラス"""
    __slots__ = ("piece", "x", "y", "owner", "vx", "vy", "angle", 
                 "angular_velocity", "gravity")
    
    def __init__(self, piece: Piece, x: float, y: float, owner: int):
        self.piece, self.x, self.y, self.owner = piece, x, y, owner
        self.vx = random.uniform(-10, 10)
        self.vy = random.uniform(-15, -5)
        self.angle = 0.0
        self.angular_velocity = random.uniform(-10, 10)
        self.gravity = 0.5

    def update(self, height: int, piece_size: int) -> bool:
        """位置を更新。戻り値は画面内にあるかどうか"""
        self.x += self.vx
        self.vy += self.gravity
        self.y += self.vy
        self.angle += self.angular_velocity
        return self.y <= height + piece_size


# 駒の移動パターン定義
STEP_MOVES = {
    'K': [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)],
    'G': [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],
    'S': [(-1, -1), (0, -1), (1, -1), (-1, 1), (1, 1)],
    'N': [(-1, -2), (1, -2)],
    'L': [(0, -1)],
    'P': [(0, -1)],
    'P+': [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],
    'L+': [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],
    'N+': [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],
    'S+': [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)],
    'B+': [(-1, 0), (1, 0), (0, -1), (0, 1)],
    'R+': [(-1, -1), (1, -1), (-1, 1), (1, 1)],
}

# スライド移動する駒
SLIDERS = {'R', 'B', 'L'}

# 成りと戻しのマッピング
DEMOTE_MAP = {'P+': 'P', 'L+': 'L', 'N+': 'N', 'S+': 'S', 'B+': 'B', 'R+': 'R'}
PROMOTE_MAP = {'P': 'P+', 'L': 'L+', 'N': 'N+', 'S': 'S+', 'B': 'B+', 'R': 'R+'}

# スライド移動の方向
SLIDER_DIRS = {
    'R': [(0, -1), (0, 1), (-1, 0), (1, 0)],
    'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    'L': [(0, -1)],
}

# 駒の日本語名
JAPANESE_PIECE_NAMES = {
    'K': '玉', 'R': '飛', 'B': '角', 'G': '金', 'S': '銀', 'N': '桂', 'L': '香', 'P': '歩',
    'R+': '龍', 'B+': '馬', 'S+': '成銀', 'N+': '成桂', 'L+': '成香', 'P+': 'と',
}

# 日本語名から駒種への逆マップ
KIND_FROM_JP = {v: k for k, v in JAPANESE_PIECE_NAMES.items()}

# 駒の価値
PIECE_VALUES: Dict[str, int] = {
    'K': 10000, 'R': 500, 'B': 500, 'G': 300, 'S': 300, 'N': 200, 'L': 200, 'P': 100,
    'R+': 700, 'B+': 700, 'S+': 400, 'N+': 300, 'L+': 300, 'P+': 200
}


def demote_kind(kind: str) -> str:
    """駒種を元の形に戻す（成り駒→成る前の駒）"""
    return DEMOTE_MAP.get(kind, kind)
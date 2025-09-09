"""
Game rules, move validation, and move generation for Shogi.

This module provides:
- Legal move generation for all piece types
- Check and checkmate detection
- Move validation (including drops and promotion rules)
- Fast move generation for AI search
- Game rule utilities (promotion zones, drop restrictions, etc.)
"""

from typing import List, Tuple, Optional, Iterable
from copy import deepcopy
import random

from .piece import (
    Piece, STEP_MOVES, SLIDERS, SLIDER_DIRS, PROMOTE_MAP, DEMOTE_MAP, 
    demote_kind, PIECE_VALUES
)
from .board import Board, clone_board
from .utils import BOARD_SIZE, in_bounds

# 型エイリアス
BoardMove = Tuple[int, int, int, int]  # (sx, sy, tx, ty)
DropMove = Tuple[None, int, int, int]  # (None, hand_index, tx, ty)
Move = BoardMove | DropMove

# 成り域の定義
PROMOTION_ZONE = {0: range(0, 3), 1: range(6, 9)}


def _sign_for_owner(owner: int) -> int:
    """プレイヤーの向きに応じた符号を返す"""
    return 1 if owner == 0 else -1


def _add_step_moves(board: Board, x: int, y: int, owner: int, 
                   kind_key: str, moves: List[Tuple[int, int]]) -> None:
    """ステップ移動の手を追加"""
    for dx, dy in STEP_MOVES.get(kind_key, []):
        actual_dy = dy * _sign_for_owner(owner)
        nx, ny = x + dx, y + actual_dy
        if not in_bounds(nx, ny):
            continue
        target = board[nx][ny]
        if target is None or target.owner != owner:
            moves.append((nx, ny))


def _add_slider_moves(board: Board, x: int, y: int, owner: int, 
                     dirs: Iterable[Tuple[int, int]], 
                     moves: List[Tuple[int, int]]) -> None:
    """スライド移動の手を追加"""
    for dx, dy in dirs:
        actual_dy = dy * _sign_for_owner(owner)
        nx, ny = x + dx, y + actual_dy
        while in_bounds(nx, ny):
            target = board[nx][ny]
            if target is None:
                moves.append((nx, ny))
            else:
                if target.owner != owner:
                    moves.append((nx, ny))
                break
            nx, ny = nx + dx, ny + actual_dy


def _generate_drop_moves(board: Board, p_kind: str, p_owner: int) -> List[Tuple[int, int]]:
    """打つ手の合法な着手先を生成"""
    moves: List[Tuple[int, int]] = []
    for nx in range(BOARD_SIZE):
        for ny in range(BOARD_SIZE):
            if _is_valid_drop(board, p_kind, p_owner, nx, ny):
                moves.append((nx, ny))
    return moves


def _is_valid_drop(board: Board, p_kind: str, p_owner: int, nx: int, ny: int) -> bool:
    """打つ手が有効かチェック"""
    if board[nx][ny]:
        return False
    if p_kind == 'P' and _has_pawn_in_file(board, nx, p_owner):
        return False
    if _drop_into_forbidden_rank(p_kind, p_owner, ny):
        return False
    if p_kind == 'P' and _pawn_drop_would_result_in_checkmate(board, nx, ny, p_owner):
        return False
    return True


def _has_pawn_in_file(board: Board, file_x: int, owner: int) -> bool:
    """指定筋に既に歩があるかチェック（二歩禁止）"""
    for i in range(BOARD_SIZE):
        piece = board[file_x][i]
        if piece is not None and piece.kind == 'P' and piece.owner == owner:
            return True
    return False


def _drop_into_forbidden_rank(p_kind: str, owner: int, ny: int) -> bool:
    """行き所のない駒の判定"""
    if p_kind in ['P', 'L']:
        return (owner == 0 and ny == 0) or (owner == 1 and ny == 8)
    if p_kind == 'N':
        return (owner == 0 and ny <= 1) or (owner == 1 and ny >= 7)
    return False


def _pawn_drop_would_result_in_checkmate(board: Board, nx: int, ny: int, p_owner: int) -> bool:
    """歩打ち詰めの判定"""
    temp_board = deepcopy(board)
    temp_board[nx][ny] = Piece('P', p_owner)
    if not is_in_check(temp_board, 1 - p_owner):
        return False
    # 相手に一手でも合法手があれば詰みではない
    for opp_x in range(BOARD_SIZE):
        for opp_y in range(BOARD_SIZE):
            opp_piece = temp_board[opp_x][opp_y]
            if opp_piece is not None and opp_piece.owner == 1 - p_owner:
                if generate_all_moves(temp_board, opp_x, opp_y, 1 - p_owner):
                    return False
    return True


def _generate_promoted_moves(board: Board, x: int, y: int, p: Piece) -> List[Tuple[int, int]]:
    """成り駒（龍・馬）の移動手を生成"""
    moves: List[Tuple[int, int]] = []
    _add_step_moves(board, x, y, p.owner, p.kind, moves)
    base_kind = demote_kind(p.kind)
    if base_kind == 'R':
        _add_slider_moves(board, x, y, p.owner, [(0, -1), (0, 1), (-1, 0), (1, 0)], moves)
    elif base_kind == 'B':
        _add_slider_moves(board, x, y, p.owner, [(-1, -1), (-1, 1), (1, -1), (1, 1)], moves)
    return moves


def find_king(board: Board, owner: int) -> Optional[Tuple[int, int]]:
    """王の位置を探す"""
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            p = board[x][y]
            if p and p.kind == 'K' and p.owner == owner:
                return (x, y)
    return None


def is_in_check(board: Board, owner: int) -> bool:
    """王手状態かチェック"""
    king_pos = find_king(board, owner)
    if not king_pos:
        return False
    kx, ky = king_pos
    opponent = 1 - owner
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            p = board[x][y]
            if p and p.owner == opponent:
                for tx, ty in generate_all_moves_no_check(board, x, y, p.owner):
                    if (tx, ty) == (kx, ky):
                        return True
    return False


def generate_all_moves_no_check(board: Board, x: Optional[int], y: int, owner: int, 
                               kind: Optional[str] = None) -> List[Tuple[int, int]]:
    """王手チェックなしの移動可能先を生成"""
    moves: List[Tuple[int, int]] = []
    if x is None:
        # 手駒打ち
        if kind is None:
            return []
        return _generate_drop_moves(board, kind, owner)
    
    p = board[x][y]
    if p is None:
        return []
    
    if p.kind in ['B+', 'R+']:
        return _generate_promoted_moves(board, x, y, p)
    
    if p.kind in STEP_MOVES:
        _add_step_moves(board, x, y, p.owner, p.kind, moves)
    
    dirs = SLIDER_DIRS.get(p.kind)
    if dirs:
        _add_slider_moves(board, x, y, p.owner, dirs, moves)
    
    return moves


def generate_all_moves(board: Board, x: Optional[int], y: int, owner: int, 
                      kind: Optional[str] = None, check_rule: bool = True) -> List[Tuple[int, int]]:
    """合法手を生成（王手放置チェック付き）"""
    moves = generate_all_moves_no_check(board, x, y, owner, kind)
    if not check_rule:
        return moves
    
    valid_moves: List[Tuple[int, int]] = []
    for tx, ty in moves:
        temp_board = deepcopy(board)
        if x is None and kind is not None:
            temp_board[tx][ty] = Piece(kind, owner)
        elif x is not None:
            piece = temp_board[x][y]
            if piece is None:
                continue
            temp_board[tx][ty], temp_board[x][y] = piece.clone(), None
        else:
            continue
        if not is_in_check(temp_board, owner):
            valid_moves.append((tx, ty))
    return valid_moves


def get_legal_moves_all(state, owner: int) -> List[Move]:
    """全ての合法手を取得（汎用版）"""
    # 探索中は高速版を優先
    if getattr(state, 'fast_mode', False):
        return get_legal_moves_all_fast(state, owner)
    
    all_moves: List[Move] = []
    board = state.board
    
    # 盤上の駒の移動
    for x in range(BOARD_SIZE):
        col = board[x]
        for y in range(BOARD_SIZE):
            p = col[y]
            if p and p.owner == owner:
                moves = generate_all_moves(board, x, y, owner)
                all_moves.extend([(x, y, tx, ty) for tx, ty in moves])
    
    # 手駒の打つ手
    for idx, kind in enumerate(list(state.hands[owner])):
        moves = generate_all_moves(board, None, idx, owner, kind=kind)
        all_moves.extend([(None, idx, tx, ty) for tx, ty in moves])
    
    return all_moves


# ----------------------
# 高速探索用 合法手生成
# ----------------------
def _pseudo_moves_for_piece(board: Board, x: int, y: int, owner: int, piece: Piece) -> List[Tuple[int, int]]:
    """盤上駒の擬似(王手放置考慮なし)移動先を列挙"""
    if piece.kind in ['B+', 'R+']:
        return _generate_promoted_moves(board, x, y, piece)
    
    moves = []
    if piece.kind in STEP_MOVES:
        _add_step_moves(board, x, y, owner, piece.kind, moves)
    dirs = SLIDER_DIRS.get(piece.kind)
    if dirs:
        _add_slider_moves(board, x, y, owner, dirs, moves)
    return moves


def get_legal_moves_all_fast(state, owner: int) -> List[Move]:
    """deepcopy を使わず簡易シミュレーションで合法手判定（高速版）"""
    board = state.board
    legal: List[Move] = []
    
    # 盤上駒
    for x in range(BOARD_SIZE):
        col = board[x]
        for y in range(BOARD_SIZE):
            p = col[y]
            if not p or p.owner != owner:
                continue
            pmoves = _pseudo_moves_for_piece(board, x, y, owner, p)
            for tx, ty in pmoves:
                src_piece = p
                tgt_piece = board[tx][ty]
                # 成り処理(必須/任意)を仮適用
                original_kind = src_piece.kind
                promoted = False
                if (not src_piece.promoted) and src_piece.kind in PROMOTE_MAP:
                    if _must_promote(src_piece.kind, owner, ty) or _can_promote(src_piece.kind, owner, y, ty):
                        src_piece.kind = PROMOTE_MAP[src_piece.kind]
                        promoted = True
                # 仮移動
                board[tx][ty] = src_piece
                board[x][y] = None
                illegal = is_in_check(board, owner)
                # 差し戻し
                board[x][y] = src_piece
                board[tx][ty] = tgt_piece
                if promoted:
                    src_piece.kind = original_kind
                if not illegal:
                    legal.append((x, y, tx, ty))
    
    # 打つ手
    hand_list = state.hands[owner]
    for idx, kind in enumerate(hand_list):
        drop_targets = []
        for nx in range(BOARD_SIZE):
            for ny in range(BOARD_SIZE):
                if _is_valid_drop(board, kind, owner, nx, ny):
                    drop_targets.append((nx, ny))
        for tx, ty in drop_targets:
            board[tx][ty] = Piece(kind, owner)
            illegal = is_in_check(board, owner)
            board[tx][ty] = None
            if not illegal:
                legal.append((None, idx, tx, ty))
    
    return legal


def _must_promote(piece_kind: str, owner: int, to_y: int) -> bool:
    """必須成りの判定"""
    if piece_kind in ['P', 'L']:
        return (owner == 0 and to_y == 0) or (owner == 1 and to_y == 8)
    if piece_kind == 'N':
        return (owner == 0 and to_y <= 1) or (owner == 1 and to_y >= 7)
    return False


def _can_promote(piece_kind: str, owner: int, from_y: int, to_y: int) -> bool:
    """任意成りの判定"""
    return (piece_kind in PROMOTE_MAP and 
            (to_y in PROMOTION_ZONE[owner] or from_y in PROMOTION_ZONE[owner]))


def check_mate(state) -> bool:
    """詰みの判定"""
    # 王がいない場合は詰み扱い
    king_pos = find_king(state.board, state.turn)
    if not king_pos:
        return True
    
    # 王手でない場合は詰みではない
    if not is_in_check(state.board, state.turn):
        return False
    
    # 合法手があれば詰みではない
    if not get_legal_moves_all(state, state.turn):
        return True
    
    return False


def check_sennichite(state) -> bool:
    """千日手の判定（現在は未実装）"""
    return False
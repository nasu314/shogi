"""
Game state management, AI, and game loop logic for Shogi.

This module provides:
- GameState class for managing game state
- AI search algorithms and evaluation functions
- Move application and game logic
- Game history management
- CPU player implementations with different difficulty levels
"""

import time
import random
import csv
from typing import List, Dict, Optional, Any, Union, TypedDict, Tuple, cast
from copy import deepcopy

from .piece import Piece, AnimatedPiece, PIECE_VALUES, demote_kind, PROMOTE_MAP, DEMOTE_MAP
from .board import Board, standard_setup, HANDICAPS, clone_board, pst_value
from .rules import (
    Move, BoardMove, DropMove, PROMOTION_ZONE, find_king, is_in_check, 
    generate_all_moves, get_legal_moves_all, get_legal_moves_all_fast,
    check_mate, check_sennichite, _must_promote, _can_promote
)
from .utils import (
    BOARD_SIZE, SUDDEN_DEATH_TIME, kifu_path, coords_to_kifu, 
    JAPANESE_TURN_SYMBOL, JAPANESE_PIECE_NAMES, KIND_FROM_JP,
    Y_COORD_FROM_JP, ZEN_TO_HAN_TABLE, JAPANESE_Y_COORDS
)

# CPU難易度設定
CPU_DIFFICULTIES = {'入門': 'beginner', '初級': 'easy', '中級': 'medium', '上級': 'hard', '達人': 'master'}


class HistoryItem(TypedDict):
    """対局状態スナップショット"""
    board: Board
    hands: Dict[int, List[str]]
    turn: int
    kifu: List[str]
    sente_time: float
    gote_time: float
    last_move_target: Optional[Tuple[int, int]]
    in_sudden_death: Dict[int, bool]


class GameState:
    """ゲーム状態を管理するクラス"""
    
    def __init__(self, handicap: str = '平手', mode: str = '2P', 
                 cpu_difficulty: str = 'easy', time_limit: Optional[int] = None):
        self.board = standard_setup()
        if handicap in HANDICAPS:
            for pos in HANDICAPS[handicap]():
                self.board[pos[0]][pos[1]] = None
        
        # hands: 各手番の持ち駒種類(kind)文字列
        self.hands = {0: [], 1: []}
        self.turn = 0
        self.selected = None
        self.legal_moves: List[Tuple[int, int]] = []
        self.selected_hand = None
        self.kifu = []
        self.game_over = False
        self.winner = None
        self.last_move_target = None

        self.mode = mode
        self.cpu_difficulty = cpu_difficulty
        self.time_limit = time_limit
        base_time = float(time_limit) if time_limit is not None else 0.0
        self.sente_time = base_time
        self.gote_time = base_time
        self.last_move_time = time.time()
        self.timer_paused = False
        self.in_sudden_death = {0: False, 1: False}
        self.kifu_scroll_offset = 0
        self.saved_message_time = None
        self.resigning_animation = False
        self.animated_pieces: List[AnimatedPiece] = []
        self.animation_finished = False
        self.cpu_thinking = False
        self.hand_scroll_offset = {0: 0, 1: 0}
        self.hand_scrollbar_rect = {0: None, 1: None}
        self.check_display_time = 0
        self.checkmate_display_time = 0
        
        # 履歴とタイマー表示キャッシュ
        self.history = []
        self.timer_display_cache = {
            'sente': '00:00:00',
            'gote': '00:00:00',
            'last_update': 0.0,
        }
        
        # 追加初期化 (GUI フィールド)
        self.resign_button_rect = None
        self.save_button_rect = None
        self.matta_button_rect = None
        self.timer_button_rect = None
        self.scrollbar_rect = None
        self.scroll_y_start = 0
        self.scroll_offset_start = 0
        self.scroll_x_start = 0
        self.end_processed = False
        self.zobrist = 0
        self._last_search_value = 0.0
        self.fast_mode = False

    def save_history(self) -> None:
        """現在局面を履歴へ保存"""
        history_item: HistoryItem = {
            'board': clone_board(self.board),
            'hands': {0: self.hands[0][:], 1: self.hands[1][:]},
            'turn': self.turn,
            'kifu': list(self.kifu),
            'sente_time': self.sente_time,
            'gote_time': self.gote_time,
            'last_move_target': self.last_move_target,
            'in_sudden_death': {0: self.in_sudden_death[0], 1: self.in_sudden_death[1]},
        }
        self.history.append(history_item)

    def load_history(self, item: HistoryItem) -> None:
        """履歴から状態を復元"""
        self.board = item['board']
        self.hands = item['hands']
        self.turn = item['turn']
        self.kifu = item['kifu']
        self.sente_time = item['sente_time']
        self.gote_time = item['gote_time']
        self.last_move_time = time.time()
        self.last_move_target = item['last_move_target']
        self.in_sudden_death = item['in_sudden_death']
        self.selected, self.selected_hand, self.legal_moves = None, None, []


# ========================================
# 評価関数とAIロジック
# ========================================

EVAL_CACHE: Dict[int, float] = {}
EVAL_CACHE_MAX = 200000


def _eval_material_and_positional(state: GameState):
    """駒得と位置価値を評価"""
    material: Dict[int, float] = {0: 0.0, 1: 0.0}
    positional: Dict[int, float] = {0: 0.0, 1: 0.0}
    king_pos: Dict[int, Optional[Tuple[int, int]]] = {0: None, 1: None}
    board = state.board
    
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            p = board[x][y]
            if not p:
                continue
            v = PIECE_VALUES.get(p.kind, 0)
            # 昇級域(敵陣)進入ボーナス
            if y in PROMOTION_ZONE[p.owner]:
                v += 8
            if p.kind == 'K':
                king_pos[p.owner] = (x, y)
            yf = y if p.owner == 0 else 8 - y
            positional[p.owner] += pst_value(p.kind, yf)
            material[p.owner] += v
    
    # 手駒: ゲーム進行度で重み変化
    base_non_king = sum(v for v in material.values()) - 2*PIECE_VALUES['K']
    phase = max(0.0, min(1.0, base_non_king / 6000.0))
    hand_scale = 0.9 + 0.1 * phase
    for o in (0, 1):
        for k in state.hands[o]:
            material[o] += float(PIECE_VALUES.get(k, 0)) * hand_scale
    
    return material, positional, king_pos, phase


def _eval_mobility_fast(state: GameState):
    """概算モビリティ評価"""
    board = state.board
    mob: Dict[int, float] = {0: 0.0, 1: 0.0}
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            p = board[x][y]
            if not p:
                continue
            o = p.owner
            if p.kind[0] in ('R', 'B') or p.kind in ('R+', 'B+'):
                # 対応する方向集合
                dirs = []
                base = demote_kind(p.kind)
                if base == 'R':
                    dirs.extend([(0, 1), (0, -1), (1, 0), (-1, 0)])
                if base == 'B':
                    dirs.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
                for dx, dy in dirs:
                    nx, ny = x+dx, y+dy
                    step_score = 0
                    while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        if not board[nx][ny]:
                            step_score += 1.0
                        else:
                            step_score += 0.5 if board[nx][ny].owner != o else 0.0
                            break
                        nx += dx
                        ny += dy
                        mob[o] += float(step_score) * 0.6
            else:
                from .piece import STEP_MOVES
                key = p.kind if p.kind in STEP_MOVES else demote_kind(p.kind)
                for dx, dy in STEP_MOVES.get(key, []):
                    dy = dy * (1 if p.owner == 0 else -1)
                    nx, ny = x+dx, y+dy
                    if (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
                        (not board[nx][ny] or board[nx][ny].owner != p.owner)):
                        mob[o] += 0.8
    return mob


def _eval_king_safety(state: GameState, king_pos):
    """王の安全度評価"""
    board = state.board
    safety = {0: 0, 1: 0}
    for o in (0, 1):
        kp = king_pos[o]
        if not kp:
            continue
        kx, ky = kp
        shield = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = kx+dx, ky+dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    pc = board[nx][ny]
                    if pc:
                        if pc.owner == o:
                            shield += 0.6
                        else:
                            shield -= 0.7
        # 王の段(相手からの距離)でボーナス: 自陣深いほど +
        depth_bonus = (ky if o == 1 else 8-ky) * 0.2
        safety[o] = shield + depth_bonus
    return safety


def evaluate_board(state: GameState, owner: int) -> float:
    """局面評価関数"""
    # キャッシュ利用
    zob = getattr(state, 'zobrist', None)
    if zob is not None:
        cached = EVAL_CACHE.get(zob)
        if cached is not None:
            return cached if owner == 0 else -cached

    material, positional, king_pos, phase = _eval_material_and_positional(state)
    mobility = _eval_mobility_fast(state)
    king_safety = _eval_king_safety(state, king_pos)

    # 重み (終盤で王安全性よりモビリティ/位置を強調)
    w_mat = 1.0
    w_pos = 0.9 + 0.2 * phase
    w_mob = 0.5 + 0.5 * phase
    w_king = 0.8 - 0.3 * phase

    score_side0 = (
        w_mat * (material[0]-material[1]) +
        w_pos * (positional[0]-positional[1]) +
        w_mob * (mobility[0]-mobility[1]) +
        w_king * (king_safety[0]-king_safety[1])
    )

    # キャッシュ保存
    if zob is not None:
        if len(EVAL_CACHE) > EVAL_CACHE_MAX:
            # ランダム要素で eviction
            for _ in range(1000):
                try:
                    EVAL_CACHE.pop(next(iter(EVAL_CACHE)))
                except StopIteration:
                    break
                if len(EVAL_CACHE) <= EVAL_CACHE_MAX:
                    break
        EVAL_CACHE[zob] = score_side0
    
    return score_side0 if owner == 0 else -score_side0


# ========================================
# CPU移動関数
# ========================================

def get_cpu_move_beginner(state: GameState) -> Optional[Move]:
    """入門レベル: ランダム手"""
    legal_moves = get_legal_moves_all(state, state.turn)
    return random.choice(legal_moves) if legal_moves else None


def get_cpu_move_easy(state: GameState) -> Optional[Move]:
    """初級レベル: 取れる駒があれば優先"""
    legal_moves = get_legal_moves_all(state, state.turn)
    if not legal_moves:
        return None
    capture_moves = [m for m in legal_moves if m[0] is not None and state.board[m[2]][m[3]]]
    return random.choice(capture_moves) if capture_moves else random.choice(legal_moves)


# ========================================
# 高速探索用軽量 move/unmove
# ========================================

def apply_move_fast(state: GameState, move: Move):
    """探索専用: O(1) で指し手適用し undo 情報を返す"""
    sx, sy_or_idx, tx, ty = move
    undo = {"captured": None, "piece": None, "prev_kind": None, "drop": False}
    
    if sx is not None:  # 盤上の移動
        piece = state.board[sx][sy_or_idx]
        undo["piece"] = piece
        undo["prev_kind"] = piece.kind
        captured = state.board[tx][ty]
        if captured:
            undo["captured"] = captured
            state.hands[state.turn].append(demote_kind(captured.kind))
        
        # 成り判定
        if not piece.promoted and piece.kind in PROMOTE_MAP:
            if _must_promote(piece.kind, piece.owner, ty):
                piece.kind = PROMOTE_MAP[piece.kind]
            elif _can_promote(piece.kind, piece.owner, sy_or_idx, ty):
                piece.kind = PROMOTE_MAP[piece.kind]  # CPU は常成り
        
        state.board[tx][ty] = piece
        state.board[sx][sy_or_idx] = None
    else:  # 打つ
        kind = state.hands[state.turn].pop(sy_or_idx)
        piece = Piece(kind, state.turn)
        undo["piece"] = piece
        undo["drop"] = True
        state.board[tx][ty] = piece
    
    state.turn = 1 - state.turn
    return undo


def undo_move_fast(state: GameState, move: Move, undo):
    """apply_move_fast の逆操作"""
    state.turn = 1 - state.turn
    sx, sy_or_idx, tx, ty = move
    
    if sx is not None:  # 差し戻し
        piece = undo["piece"]
        state.board[sx][sy_or_idx] = piece
        state.board[tx][ty] = undo["captured"]
        
        # 成りを戻す
        if undo["prev_kind"] and piece.kind != undo["prev_kind"]:
            piece.kind = undo["prev_kind"]
        
        # 取った駒を手駒から除去
        if undo["captured"]:
            demoted = demote_kind(undo["captured"].kind)
            for i in range(len(state.hands[state.turn]) - 1, -1, -1):
                if state.hands[state.turn][i] == demoted:
                    state.hands[state.turn].pop(i)
                    break
    else:  # 打った駒を手駒に戻す
        piece = undo["piece"]
        state.board[tx][ty] = None
        state.hands[state.turn].append(piece.kind)


# ========================================
# Zobrist ハッシュ & 置換表
# ========================================

ZOBRIST_PIECE_KEYS = {}
_ALL_BOARD_KINDS = ['K', 'R', 'B', 'G', 'S', 'N', 'L', 'P', 'R+', 'B+', 'S+', 'N+', 'L+', 'P+']
for x in range(BOARD_SIZE):
    for y in range(BOARD_SIZE):
        for k in _ALL_BOARD_KINDS:
            for owner in (0, 1):
                ZOBRIST_PIECE_KEYS[(x, y, k, owner)] = random.getrandbits(64)
ZOBRIST_SIDE_KEY = random.getrandbits(64)


def recompute_zobrist(state: GameState):
    """Zobrist ハッシュを再計算"""
    h = 0
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            p = state.board[x][y]
            if p:
                h ^= ZOBRIST_PIECE_KEYS[(x, y, p.kind, p.owner)]
    
    # 持ち駒をカウント反映
    for owner in (0, 1):
        counts = {}
        for k in state.hands[owner]:
            counts[k] = counts.get(k, 0) + 1
        for k, c in counts.items():
            mix = hash((k, owner, c)) & 0xFFFFFFFFFFFFFFFF
            h ^= mix
    
    if state.turn == 1:
        h ^= ZOBRIST_SIDE_KEY
    
    state.zobrist = h
    return h


class TTEntry:
    """置換表エントリ"""
    __slots__ = ("zobrist", "depth", "score", "flag", "best_move", "alpha", "beta")
    
    def __init__(self, zobrist, depth, score, flag, best_move, alpha, beta):
        self.zobrist = zobrist
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move
        self.alpha = alpha
        self.beta = beta


TT: Dict[int, TTEntry] = {}
TT_MAX_SIZE = 50000
KILLER_MOVES: Dict[int, List[Move]] = {}
HISTORY_TABLE: Dict[Tuple[int, Tuple[Optional[int], int, int, int]], int] = {}


def move_key_for_history(move: Move):
    """履歴テーブル用のキー生成"""
    sx, sy, tx, ty = move
    return (sx, sy, tx, ty)


def store_killer(depth: int, move: Move):
    """キラー手を記録"""
    if move is None:
        return
    arr = KILLER_MOVES.get(depth, [])
    if move in arr:
        return
    arr = [move] + arr
    if len(arr) > 2:
        arr = arr[:2]
    KILLER_MOVES[depth] = arr


def add_history(turn: int, move: Move, depth: int):
    """履歴ヒューリスティックに追加"""
    if move is None:
        return
    key = (turn, move_key_for_history(move))
    HISTORY_TABLE[key] = HISTORY_TABLE.get(key, 0) + depth * depth


def history_score(turn: int, move: Move) -> int:
    """履歴スコアを取得"""
    return HISTORY_TABLE.get((turn, move_key_for_history(move)), 0)


def ordered_moves(state: GameState, legal_moves: List[Move], tt_move: Optional[Move], depth: int):
    """手の並び替え"""
    scored = []
    killers = KILLER_MOVES.get(depth, [])
    for m in legal_moves:
        if m == tt_move:
            priority = (-1000000, 0)
        elif m in killers:
            priority = (-900000, 0)
        else:
            sx, _, tx, ty = m
            captured_value = PIECE_VALUES.get(state.board[tx][ty].kind, 0) if (sx is not None and state.board[tx][ty]) else 0
            cap_flag = 1 if captured_value > 0 else 0
            priority = (-cap_flag*500 - captured_value, -history_score(state.turn, m))
        scored.append((priority, m))
    scored.sort(key=lambda x: x[0])
    return [m for _, m in scored]


class SearchContext:
    """探索コンテキスト"""
    __slots__ = ("start_time", "time_limit", "nodes", "timeout")
    
    def __init__(self, time_limit_ms):
        self.start_time = time.time()
        self.time_limit = time_limit_ms/1000.0 if time_limit_ms else None
        self.nodes = 0
        self.timeout = False


INF = 10**9

# 探索パラメータ
SEARCH_PARAMS = {
    'aspiration_window': 60,
    'aspiration_step': 120,
    'null_move_min_depth': 3,
    'null_move_deep_threshold': 6,
    'null_move_base_reduction': 2,
    'null_move_deep_bonus': 1,
    'lmr_min_depth': 3,
    'lmr_first_late_index': 4,
    'lmr_deeper_index': 8,
    'lmr_reduction_deep': 1,
    'lmr_reduction_deeper': 2
}


def _generate_capture_moves(state: GameState):
    """駒取り手のみを列挙"""
    moves = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            p = state.board[x][y]
            if p and p.owner == state.turn:
                for tx, ty in generate_all_moves(state.board, x, y, state.turn):
                    if state.board[tx][ty]:  # capture
                        moves.append((x, y, tx, ty))
    return moves


def _is_favorable_or_equal_capture(board: Board, move: Move):
    """簡易静的交換評価"""
    sx, sy, tx, ty = move
    if sx is None:
        return True
    attacker = board[sx][sy]
    target = board[tx][ty]
    if not attacker or not target:
        return True
    gain = PIECE_VALUES.get(target.kind, 0) - PIECE_VALUES.get(attacker.kind, 0)
    return gain >= -100


def quiescence_search(state: GameState, alpha: int, beta: int, owner: int, ctx: SearchContext, ply: int):
    """静止探索"""
    if ctx.time_limit and (time.time() - ctx.start_time) > ctx.time_limit:
        ctx.timeout = True
        return evaluate_board(state, owner), None
    ctx.nodes += 1

    stand_pat = evaluate_board(state, owner)
    if stand_pat >= beta:
        return beta, None
    if alpha < stand_pat:
        alpha = stand_pat

    in_check_flag = is_in_check(state.board, state.turn)
    if in_check_flag:
        legal_moves = get_legal_moves_all(state, state.turn)
    else:
        raw_caps = _generate_capture_moves(state)
        legal_moves = [m for m in raw_caps if _is_favorable_or_equal_capture(state.board, m)] or raw_caps[:1]

    if not legal_moves:
        return stand_pat, None

    # 簡易オーダリング：捕獲価値降順
    def cap_score(m):
        sx, _, tx, ty = m
        if sx is not None and state.board[tx][ty]:
            return -PIECE_VALUES.get(state.board[tx][ty].kind, 0)
        return 0
    legal_moves.sort(key=cap_score)

    best_move = legal_moves[0]
    for move in legal_moves:
        undo = apply_move_fast(state, move)
        recompute_zobrist(state)
        score, _ = quiescence_search(state, alpha, beta, owner, ctx, ply+1)
        undo_move_fast(state, move, undo)
        recompute_zobrist(state)
        if ctx.timeout:
            return alpha, best_move
        if score > alpha:
            alpha = score
            best_move = move
        if alpha >= beta:
            break
    return alpha, best_move


def _tt_probe(zob: int, depth: int, alpha: int, beta: int):
    """置換表参照"""
    entry = TT.get(zob)
    if not entry or entry.depth < depth:
        return False, alpha, beta, entry
    
    if entry.flag == 'EXACT':
        return True, alpha, beta, entry
    if entry.flag == 'LOWER' and entry.score > alpha:
        alpha = entry.score
    elif entry.flag == 'UPPER' and entry.score < beta:
        beta = entry.score
    if alpha >= beta:
        return True, alpha, beta, entry
    return False, alpha, beta, entry


def _tt_store(zob: int, depth: int, value: float, flag: str, best_move: Optional[Move], alpha: int, beta: int):
    """置換表に保存"""
    if len(TT) > TT_MAX_SIZE:
        for _ in range(1000):
            TT.pop(next(iter(TT)))
            if len(TT) <= TT_MAX_SIZE:
                break
    TT[zob] = TTEntry(zob, depth, value, flag, best_move, alpha, beta)


class SearchParams:
    """探索パラメータ束"""
    __slots__ = ('depth', 'alpha', 'beta', 'maximizing', 'owner', 'ply', 'ctx', 'original_alpha', 'original_beta')
    
    def __init__(self, depth, alpha, beta, maximizing, owner, ply, ctx):
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        self.maximizing = maximizing
        self.owner = owner
        self.ply = ply
        self.ctx = ctx
        self.original_alpha = alpha
        self.original_beta = beta


def _search_child(state: GameState, move: Move, params: SearchParams):
    """1手指して子ノードを評価"""
    undo = apply_move_fast(state, move)
    recompute_zobrist(state)
    child_val, _ = alpha_beta_search(state, params.depth-1, params.alpha, params.beta,
                                   not params.maximizing, params.owner, params.ply+1, params.ctx)
    undo_move_fast(state, move, undo)
    recompute_zobrist(state)
    if params.ctx.timeout:
        return child_val, True
    return child_val, False


def _search_child_reduced(state: GameState, move: Move, params: SearchParams, reduction_depth: int):
    """LMR 用: 減深で1手探索"""
    undo = apply_move_fast(state, move)
    recompute_zobrist(state)
    child_val, _ = alpha_beta_search(state, params.depth-1-reduction_depth, params.alpha, params.beta,
                                   not params.maximizing, params.owner, params.ply+1, params.ctx)
    undo_move_fast(state, move, undo)
    recompute_zobrist(state)
    if params.ctx.timeout:
        return child_val, True
    return child_val, False


def _record_cut_or_history(first_move: Move, current_move: Move, maximizing: bool, owner: int, depth: int, ply: int):
    """カット時の履歴記録"""
    if current_move != first_move:
        store_killer(ply, current_move)
    else:
        add_history(owner if maximizing else (1-owner), current_move, depth)


def _final_flag(value: float, original_alpha: int, original_beta: int) -> str:
    """最終フラグ決定"""
    if value <= original_alpha:
        return 'UPPER'
    if value >= original_beta:
        return 'LOWER'
    return 'EXACT'


def _timeout_or_leaf(state: GameState, depth: int, alpha: int, beta: int, owner: int, ply: int, ctx: SearchContext):
    """時間切れ/葉ノード判定"""
    if ctx.time_limit and (time.time() - ctx.start_time) > ctx.time_limit:
        ctx.timeout = True
        return True, (evaluate_board(state, owner), None)
    ctx.nodes += 1
    if depth == 0 or state.game_over:
        return True, quiescence_search(state, alpha, beta, owner, ctx, ply)
    return False, (0, None)


def _get_ordered_moves_with_tt(state: GameState, entry: Optional[TTEntry], ply: int):
    """置換表を考慮した手順生成"""
    legal = get_legal_moves_all(state, state.turn)
    if not legal:
        score = -100000 if is_in_check(state.board, state.turn) else 0
        return [], score
    tt_move = entry.best_move if entry else None
    ordered = ordered_moves(state, legal, tt_move, ply)
    return ordered, None


def _search_loop(state: GameState, ordered: List[Move], params: SearchParams):
    """最大化/最小化共通ループ"""
    first_move = ordered[0]
    best_move = first_move
    value = -INF if params.maximizing else INF
    
    for idx, move in enumerate(ordered):
        # LMR 判定
        do_lmr = False
        reduction = 0
        if params.depth >= 3 and idx >= 4:
            sx, _, tx, ty = move
            is_capture = (sx is not None and state.board[tx][ty] is not None)
            if not is_capture and not is_in_check(state.board, state.turn):
                do_lmr = True
                reduction = 1 if params.depth >= 4 and idx >= 8 else 0
        
        if do_lmr and reduction > 0:
            child_val, timed_out = _search_child_reduced(state, move, params, reduction)
            # fail-high なら再探索
            if not timed_out and params.maximizing and child_val > params.alpha or (not params.maximizing and child_val < params.beta):
                child_val, timed_out = _search_child(state, move, params)
        else:
            child_val, timed_out = _search_child(state, move, params)
        
        if timed_out:
            if (params.maximizing and value == -INF) or (not params.maximizing and value == INF):
                value = evaluate_board(state, params.owner)
            return value, best_move, 'EXACT'
        
        if (params.maximizing and child_val > value) or ((not params.maximizing) and child_val < value):
            value = child_val
            best_move = move
        
        if params.maximizing:
            if value > params.alpha:
                params.alpha = value
        else:
            if value < params.beta:
                params.beta = value
        
        if params.alpha >= params.beta:  # cut
            _record_cut_or_history(first_move, move, params.maximizing, params.owner, params.depth, params.ply)
            break
    
    flag = _final_flag(value, params.original_alpha, params.original_beta)
    return value, best_move, flag


def alpha_beta_search(state: GameState, depth: int, alpha: int, beta: int, maximizing_player: bool, 
                     owner: int, ply: int, ctx: SearchContext):
    """Alpha-Beta探索"""
    done, res = _timeout_or_leaf(state, depth, alpha, beta, owner, ply, ctx)
    if done:
        return res
    
    zob = state.zobrist
    hit, alpha2, beta2, entry = _tt_probe(zob, depth, alpha, beta)
    alpha, beta = alpha2, beta2
    if hit and entry:
        return entry.score, entry.best_move
    
    # Null Move Pruning
    if depth >= 3 and not is_in_check(state.board, state.turn):
        non_king_count = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                p = state.board[x][y]
                if p and p.owner == state.turn and p.kind != 'K':
                    non_king_count += 1
                    if non_king_count > 1:
                        break
            if non_king_count > 1:
                break
        
        if non_king_count > 1:
            state.turn = 1 - state.turn
            recompute_zobrist(state)
            R = 2 if depth < 6 else 3
            score, _ = alpha_beta_search(state, depth-1-R, -beta, -beta+1, not maximizing_player, owner, ply+1, ctx)
            state.turn = 1 - state.turn
            recompute_zobrist(state)
            if ctx.timeout:
                return (alpha, None)
            if score >= beta:
                return beta, None
    
    ordered, empty_score = _get_ordered_moves_with_tt(state, entry, ply)
    if not ordered:
        return empty_score, None
    
    params = SearchParams(depth, alpha, beta, maximizing_player, owner, ply, ctx)
    value, best_move, flag = _search_loop(state, ordered, params)
    _tt_store(zob, depth, value, flag, best_move, params.alpha, params.beta)
    return value, best_move


def iterative_deepening_best_move(state: GameState, max_depth: int, time_limit_ms: int):
    """反復深化探索"""
    recompute_zobrist(state)
    state.fast_mode = True
    ctx = SearchContext(time_limit_ms)
    best_move = None
    best_value = -INF
    prev_value = 0
    
    for d in range(1, max_depth+1):
        window = SEARCH_PARAMS['aspiration_window'] if d > 1 else INF
        alpha = -INF if d == 1 else prev_value - window
        beta = INF if d == 1 else prev_value + window
        
        while True:
            val, move = alpha_beta_search(state, d, alpha, beta, True, state.turn, 0, ctx)
            if ctx.timeout:
                break
            if d > 1 and val <= alpha:
                alpha -= SEARCH_PARAMS['aspiration_step']
                continue
            if d > 1 and val >= beta:
                beta += SEARCH_PARAMS['aspiration_step']
                continue
            # 成功
            if move is not None:
                best_move = move
                best_value = val
                prev_value = val
            break
        if ctx.timeout:
            break
    
    state._last_search_value = best_value
    state.fast_mode = False
    return best_move


def get_cpu_move_medium(state: GameState) -> Optional[Move]:
    """中級レベル"""
    return iterative_deepening_best_move(state, max_depth=3, time_limit_ms=250) or get_cpu_move_easy(state)


def get_cpu_move_hard(state: GameState) -> Optional[Move]:
    """上級レベル"""
    return iterative_deepening_best_move(state, max_depth=4, time_limit_ms=600)


def get_cpu_move_master(state: GameState) -> Optional[Move]:
    """達人レベル"""
    return iterative_deepening_best_move(state, max_depth=6, time_limit_ms=1500)


# ========================================
# 手の適用と棋譜管理
# ========================================

def apply_move(state: GameState, move: Move, is_cpu: bool = False, screen=None, 
              force_promotion: Optional[bool] = None, update_history: bool = True) -> None:
    """手を適用する"""
    if update_history:
        state.save_history()

    if not state.timer_paused:
        elapsed = time.time() - state.last_move_time
        if state.time_limit is not None:
            if state.turn == 0:
                state.sente_time = max(0, state.sente_time - elapsed)
            else:
                state.gote_time = max(0, state.gote_time - elapsed)
        else:
            if state.turn == 0:
                state.sente_time += elapsed
            else:
                state.gote_time += elapsed

    sx, sy_or_idx, tx, ty = move
    if sx is not None:
        piece, captured = state.board[sx][sy_or_idx], state.board[tx][ty]
        if piece is None:
            return
        dest = "同" if state.last_move_target == (tx, ty) else coords_to_kifu(tx, ty)
        kifu_text = f"{JAPANESE_TURN_SYMBOL[state.turn]}{dest}{JAPANESE_PIECE_NAMES[demote_kind(piece.kind)]}"
        promo = False
        
        if not piece.promoted and piece.kind in PROMOTE_MAP:
            must = (piece.kind in ['P', 'L'] and (ty == 0 if piece.owner == 0 else ty == 8)) or \
                   (piece.kind == 'N' and (ty <= 1 if piece.owner == 0 else ty >= 7))
            can = (ty in PROMOTION_ZONE[piece.owner] or sy_or_idx in PROMOTION_ZONE[piece.owner])
            if must:
                promo = True
            elif can:
                if force_promotion is not None:
                    promo = force_promotion
                elif not is_cpu:
                    from .utils import ask_promotion
                    promo = ask_promotion(screen)
                else:
                    promo = True
            if promo:
                piece.kind = PROMOTE_MAP[piece.kind]
                kifu_text += "成"
            elif can:
                kifu_text += "不成"
            if captured:
                state.hands[state.turn].append(demote_kind(captured.kind))
        
        state.board[tx][ty], state.board[sx][sy_or_idx] = piece, None
        state.selected, state.legal_moves = None, []
    else:
        kind = state.hands[state.turn].pop(sy_or_idx)
        state.board[tx][ty] = Piece(kind, state.turn)
        kifu_text = f"{JAPANESE_TURN_SYMBOL[state.turn]}{coords_to_kifu(tx, ty)}{JAPANESE_PIECE_NAMES[kind]}打"
        state.selected_hand, state.legal_moves = None, []

    state.kifu.append(kifu_text)
    state.last_move_target = (tx, ty)
    state.turn = 1 - state.turn
    state.cpu_thinking = False
    state.last_move_time = time.time()
    state.timer_display_cache['last_update'] = 0.0

    if is_in_check(state.board, state.turn):
        state.check_display_time = time.time()

    # サウンド再生（実際の実装では音声ファイルが必要）
    # if sound_itte and not is_cpu: sound_itte.play()
    
    if check_sennichite(state):
        return
    if check_mate(state):
        state.game_over, state.winner = True, 1-state.turn
        state.checkmate_display_time = time.time()
        # if sound_end and not is_cpu: sound_end.play()


def save_kifu_to_csv(kifu: List[str]) -> bool:
    """棋譜をCSVファイルに保存"""
    try:
        with open(kifu_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['手数', '棋譜'])
            for i, move in enumerate(kifu):
                writer.writerow([i+1, move])
        return True
    except IOError as e:
        print(f"ファイル保存失敗: {e}")
        return False


def parse_kifu_to_move(state: GameState, kifu_str: str) -> Tuple[Move, Optional[bool]]:
    """棋譜文字列を手に変換"""
    kifu_str = kifu_str.strip().translate(ZEN_TO_HAN_TABLE)
    turn = 0 if kifu_str[0] == '▲' else 1
    if turn != state.turn:
        raise ValueError("Turn mismatch")
    
    drop, promo, no_promo = "打" in kifu_str, "成" in kifu_str, "不成" in kifu_str
    promo_flag = promo if (promo or no_promo) else None
    
    if kifu_str[1] == "同":
        if state.last_move_target is None:
            raise ValueError("'同' 指しの参照先が存在しません")
        tx, ty = state.last_move_target
        p_name = kifu_str[2:].replace("成", "").replace("不成", "").replace("打", "")
    else:
        tx, ty = 9-int(kifu_str[1]), Y_COORD_FROM_JP[kifu_str[2]]
        p_name = kifu_str[3:].replace("成", "").replace("不成", "").replace("打", "")
    
    p_kind = KIND_FROM_JP[p_name]
    
    if drop:
        try:
            idx = state.hands[turn].index(p_kind)
            return ((None, idx, tx, ty), promo_flag)
        except ValueError:
            raise ValueError(f"{p_kind} not in hand")
    else:
        base_kind = demote_kind(p_kind)
        sources = []
        for sx in range(BOARD_SIZE):
            for sy in range(BOARD_SIZE):
                p = state.board[sx][sy]
                if p and p.owner == turn and demote_kind(p.kind) == base_kind:
                    if (tx, ty) in generate_all_moves(state.board, sx, sy, turn, check_rule=False):
                        temp_b = deepcopy(state.board)
                        temp_b[tx][ty], temp_b[sx][sy] = p, None
                        if not is_in_check(temp_b, turn):
                            sources.append((sx, sy))
        if not sources:
            raise ValueError(f"No valid source for {kifu_str}")
        if len(sources) > 1:
            print(f"Ambiguous move: {kifu_str}")
        return ((sources[0][0], sources[0][1], tx, ty), promo_flag)


def load_kifu_and_setup_state(handicap: str, mode: str, cpu_difficulty: str, 
                              time_limit: Optional[int] = None) -> GameState:
    """棋譜ファイルから状態を復元"""
    state = GameState(handicap, mode, cpu_difficulty, time_limit)
    try:
        with open(kifu_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                try:
                    move, promo = parse_kifu_to_move(state, row[1])
                    apply_move(state, move, is_cpu=True, force_promotion=promo, update_history=False)
                except Exception as e:
                    print(f"Kifu parse error '{row[1]}': {e}")
                    return state
    except FileNotFoundError:
        print("kifu file not found.")
    state.history = []
    return state
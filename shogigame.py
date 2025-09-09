import pygame
import sys
from copy import deepcopy
import time
import random
import csv
import os
    
# --- パス設定 ---
# スクリプト自身の場所を基準にする
try:
    # PyInstallerでexe化した場合
    base_path = sys._MEIPASS
except AttributeError:
    base_path = os.path.dirname(os.path.abspath(__file__))
    
# 画像と音声フォルダへのパスを作成
assets_path = os.path.join(base_path, "assets")
# assets内のimagesおよびsoundを参照する
image_path = os.path.join(assets_path, "images")
sound_path = os.path.join(assets_path, "sound")
# 棋譜ファイルはスクリプト（またはパッケージ展開先）の直下に置く
# PyInstallerなどで配布された場合でも base_path を基準とする
kifu_path = os.path.join(base_path, "shogi_kifu.csv")
    
    
pygame.init()
pygame.mixer.init()
    
BOARD_SIZE = 9
SQUARE = 64
BOARD_PIXEL_WIDTH = SQUARE * BOARD_SIZE
BOARD_PIXEL_HEIGHT = SQUARE * BOARD_SIZE
    
COORD_MARGIN = 30
    
WINDOW_PADDING_X = 50
WINDOW_PADDING_Y = 50
HAND_AREA_HEIGHT = 80
    
KIFU_WINDOW_WIDTH = 300
INFO_PANEL_HEIGHT = 100
KIFU_ITEM_HEIGHT = 20
RESIGN_BUTTON_HEIGHT = 40
SAVE_BUTTON_HEIGHT = 40
MATTA_BUTTON_HEIGHT = 40
TIMER_BUTTON_HEIGHT = 40
KIFU_LIST_PADDING = 10
    
WIDTH = BOARD_PIXEL_WIDTH + WINDOW_PADDING_X * 2 + KIFU_WINDOW_WIDTH + WINDOW_PADDING_X + COORD_MARGIN * 2
HEIGHT = WINDOW_PADDING_Y + HAND_AREA_HEIGHT + BOARD_PIXEL_HEIGHT + HAND_AREA_HEIGHT + WINDOW_PADDING_Y + COORD_MARGIN * 2
        
BOARD_START_X = WINDOW_PADDING_X + COORD_MARGIN
BOARD_START_Y = WINDOW_PADDING_Y + HAND_AREA_HEIGHT + COORD_MARGIN
        
PIECE_SIZE = 60
PIECE_OFFSET = (SQUARE - PIECE_SIZE) // 2
FPS = 60
try:
    FONT = pygame.font.Font(os.path.join(assets_path, "MPLUS1p-Regular.ttf"), 18)
    LARGE_FONT = pygame.font.Font(os.path.join(assets_path, "MPLUS1p-Regular.ttf"), 24)
    MONO_FONT = pygame.font.Font(os.path.join(assets_path, "MPLUS1p-Regular.ttf"), 14)
    TITLE_FONT = pygame.font.Font(os.path.join(assets_path, "MPLUS1p-Regular.ttf"), 80)
    CHECK_FONT = pygame.font.Font(os.path.join(assets_path, "MPLUS1p-Regular.ttf"), 100)
    GREETING_FONT = pygame.font.Font(os.path.join(assets_path, "MPLUS1p-Regular.ttf"), 70)
    RESULT_FONT = pygame.font.Font(os.path.join(assets_path, "MPLUS1p-Regular.ttf"), 90)
    # set bold where it was previously requested
    CHECK_FONT.set_bold(True)
    GREETING_FONT.set_bold(True)
    RESULT_FONT.set_bold(True)
except Exception:
    # fallback to system fonts if the TTF cannot be loaded
    FONT = pygame.font.SysFont("MS Mincho", 18)
    LARGE_FONT = pygame.font.SysFont("MS Mincho", 24)
    MONO_FONT = pygame.font.SysFont("MS Mincho", 14)
    TITLE_FONT = pygame.font.SysFont("MS Mincho", 80)
    CHECK_FONT = pygame.font.SysFont("MS Mincho", 100, bold=True)
    GREETING_FONT = pygame.font.SysFont("MS Mincho", 70, bold=True)
    RESULT_FONT = pygame.font.SysFont("MS Mincho", 90, bold=True)
    
# 色の定義
WHITE = (223, 235, 234)
BLACK = (20, 20, 20)
GRAY = (171, 214, 211)
LIGHT_GRAY = (220, 220, 200)
GREEN = (120, 255, 120)
RED = (255, 80, 80)
BLUE = (120, 160, 255)
BUTTON_BLUE = (28, 93, 158)
BUTTON_BLUE_HOVER = (48, 113, 178)
ORANGE = (255, 165, 0)
TATAMI_GREEN = (140, 164, 138)
DARK_BROWN = (50, 44, 40)
BOARD_COLOR = (187, 155, 82)
TITLE_COLOR = (230, 190, 130)
    
PROMOTION_ZONE = {0: range(0, 3), 1: range(6, 9)}
        
# サウンドファイルの読み込み (パス指定を修正)
sound_start, sound_end, sound_itte, sound_se1, sound_think = [None] * 5
try:
    sound_start = pygame.mixer.Sound(os.path.join(sound_path, "start.mp3"))
    sound_end = pygame.mixer.Sound(os.path.join(sound_path, "end.mp3"))
    sound_itte = pygame.mixer.Sound(os.path.join(sound_path, "itte.mp3"))
    sound_se1 = pygame.mixer.Sound(os.path.join(sound_path, "se1.mp3"))
    sound_think = pygame.mixer.Sound(os.path.join(sound_path, "think.mp3"))
except pygame.error as e:
    print(f"サウンドファイルの読み込みに失敗しました: {e}")
        
# 画像の読み込み (パス指定を修正)
fusuma_left_img, fusuma_right_img, start_bg_img = [None] * 3
try:
    fusuma_left_img = pygame.image.load(os.path.join(image_path, "fusuma_left.png"))
    fusuma_right_img = pygame.image.load(os.path.join(image_path, "fusuma_right.png"))
    start_bg_img = pygame.image.load(os.path.join(image_path, "board.png"))
except pygame.error as e:
    print(f"画像ファイルの読み込みに失敗しました: {e}")
    
class Piece:
    def __init__(self, kind, owner):
        self.kind, self.owner = kind, owner
        
    def clone(self):
        return Piece(self.kind, self.owner)
        
    @property
    def promoted(self):
        return self.kind.endswith('+')
    
    def __repr__(self):
        return f"{self.kind}{'S' if self.owner==0 else 'G'}"
    
class AnimatedPiece:
    def __init__(self, piece, x, y, owner):
        self.piece, self.x, self.y, self.owner = piece, x, y, owner
        self.vx, self.vy = random.uniform(-10, 10), random.uniform(-15, -5)
        self.angle, self.angular_velocity = 0, random.uniform(-10, 10)
        self.gravity = 0.5
    
    def update(self):
        """Update position/angle. Return True if still visible (alive), False if finished."""
        self.x += self.vx
        self.vy += self.gravity
        self.y += self.vy
        self.angle += self.angular_velocity
        return self.y <= HEIGHT + PIECE_SIZE
        
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
SLIDERS = {'R','B','L'}
DEMOTE_MAP = {'P+':'P','L+':'L','N+':'N','S+':'S','B+':'B','R+':'R'}
PROMOTE_MAP = {'P':'P+','L':'L+','N':'N+','S':'S+','B':'B+','R':'R+'}
SLIDER_DIRS = {
    'R': [(0, -1), (0, 1), (-1, 0), (1, 0)],
    'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    'L': [(0, -1)],
}
JAPANESE_PIECE_NAMES = {
    'K':'玉','R':'飛','B':'角','G':'金','S':'銀','N':'桂','L':'香','P':'歩',
    'R+':'龍','B+':'馬','S+':'成銀','N+':'成桂','L+':'成香','P+':'と',
}
KIND_FROM_JP = {v: k for k, v in JAPANESE_PIECE_NAMES.items()}
JAPANESE_Y_COORDS = ['一','二','三','四','五','六','七','八','九']
Y_COORD_FROM_JP = {v: i for i, v in enumerate(JAPANESE_Y_COORDS)}
ZENKAKU_NUM, HANKAKU_NUM = "１２３４５６７８９", "123456789"
ZEN_TO_HAN_TABLE = str.maketrans(ZENKAKU_NUM, HANKAKU_NUM)
JAPANESE_TURN_SYMBOL = {0:'▲', 1:'△'}
JAPANESE_TURN_NAME = {0:'先手', 1:'後手'}
VICTORY_MESSAGE = {0: '学生軍の勝利', 1: '教員軍の勝利'}
PIECE_VALUES = {'K':10000,'R':500,'B':500,'G':300,'S':300,'N':200,'L':200,'P':100,
                'R+':700,'B+':700,'S+':400,'N+':300,'L+':300,'P+':200}
TIME_SETTINGS = {"なし": None, "15分": 15*60, "30分": 30*60, "1時間": 60*60, "設定": -1}
SUDDEN_DEATH_TIME = 45
                
def coords_to_kifu(x, y): return f"{9-x}{JAPANESE_Y_COORDS[y]}"
def in_bounds(x,y): return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE
def demote_kind(kind): return DEMOTE_MAP.get(kind, kind)

# Helper utilities for move generation extracted to module scope to reduce function complexity
def _sign_for_owner(owner):
    return 1 if owner == 0 else -1

def _add_step_moves(board, x, y, owner, kind_key, moves):
    for dx, dy in STEP_MOVES.get(kind_key, []):
        actual_dy = dy * _sign_for_owner(owner)
        nx, ny = x + dx, y + actual_dy
        if in_bounds(nx, ny) and (not board[nx][ny] or board[nx][ny].owner != owner):
            moves.append((nx, ny))

def _add_slider_moves(board, x, y, owner, dirs, moves):
    for dx, dy in dirs:
        actual_dy = dy * _sign_for_owner(owner)
        nx, ny = x + dx, y + actual_dy
        while in_bounds(nx, ny):
            if not board[nx][ny]:
                moves.append((nx, ny))
            elif board[nx][ny].owner != owner:
                moves.append((nx, ny))
                break
            else:
                break
            nx, ny = nx + dx, ny + actual_dy

def _generate_drop_moves(board, p_kind, p_owner):
    """Generate legal drop destinations for a piece kind (used when x is None)."""
    moves = []
    for nx in range(BOARD_SIZE):
        for ny in range(BOARD_SIZE):
            if _is_valid_drop(board, p_kind, p_owner, nx, ny):
                moves.append((nx, ny))
    return moves

def _is_valid_drop(board, p_kind, p_owner, nx, ny):
    """Return True if dropping p_kind by p_owner at (nx,ny) is allowed."""
    if board[nx][ny]:
        return False
    if p_kind == 'P' and _has_pawn_in_file(board, nx, p_owner):
        return False
    if _drop_into_forbidden_rank(p_kind, p_owner, ny):
        return False
    if p_kind == 'P' and _pawn_drop_would_result_in_checkmate(board, nx, ny, p_owner):
        return False
    return True

def _has_pawn_in_file(board, file_x, owner):
    return any(board[file_x][i] and board[file_x][i].kind == 'P' and board[file_x][i].owner == owner for i in range(BOARD_SIZE))

def _drop_into_forbidden_rank(p_kind, owner, ny):
    # forbidden ranks for pawn/knight/kyosha drops
    if p_kind in ['P', 'L']:
        return (owner == 0 and ny == 0) or (owner == 1 and ny == 8)
    if p_kind == 'N':
        return (owner == 0 and ny <= 1) or (owner == 1 and ny >= 7)
    return False

def _pawn_drop_would_result_in_checkmate(board, nx, ny, p_owner):
    temp_board = deepcopy(board)
    temp_board[nx][ny] = Piece('P', p_owner)
    if not is_in_check(temp_board, 1 - p_owner):
        return False
    opponent_has_legal_move = any(
        generate_all_moves(temp_board, opp_x, opp_y, 1 - p_owner)
        for opp_x in range(BOARD_SIZE)
        for opp_y in range(BOARD_SIZE)
        if temp_board[opp_x][opp_y] and temp_board[opp_x][opp_y].owner == 1 - p_owner
    )
    return not opponent_has_legal_move

def _generate_promoted_moves(board, x, y, p):
    """Generate moves for promoted R/B (they have step moves + base sliders)."""
    moves = []
    _add_step_moves(board, x, y, p.owner, p.kind, moves)
    base_kind = demote_kind(p.kind)
    if base_kind == 'R':
        _add_slider_moves(board, x, y, p.owner, [(0, -1), (0, 1), (-1, 0), (1, 0)], moves)
    elif base_kind == 'B':
        _add_slider_moves(board, x, y, p.owner, [(-1, -1), (-1, 1), (1, -1), (1, 1)], moves)
    return moves
        
def standard_setup():
    board = [[None for _ in range(BOARD_SIZE)] for __ in range(BOARD_SIZE)]
    back = ['L', 'N', 'S', 'G', 'K', 'G', 'S', 'N', 'L']
    for x, k in enumerate(back):
        board[x][0] = Piece(k, 1)
    board[1][1], board[7][1] = Piece('R', 1), Piece('B', 1)
    for x in range(BOARD_SIZE):
        board[x][2] = Piece('P', 1)
    for x in range(BOARD_SIZE):
        board[x][6] = Piece('P', 0)
    board[1][7], board[7][7] = Piece('B', 0), Piece('R', 0)
    for x, k in enumerate(back):
        board[x][8] = Piece(k, 0)
    return board
        
HANDICAPS = {
    '通常': [],
    '角落ち': [('B', (1, 1))],
    '飛車落ち': [('R', (7, 1))],
    '二枚落ち': [('R', (7, 1)), ('B', (1, 1))],
    '四枚落ち': [('R', (7, 1)), ('B', (1, 1)), ('L', (0, 0)), ('L', (8, 0))],
    '六枚落ち': [('R', (7, 1)), ('B', (1, 1)), ('L', (0, 0)), ('L', (8, 0)), ('N', (1, 0)), ('N', (7, 0))],
}
CPU_DIFFICULTIES = {'入門': 'beginner', '初級': 'easy', '中級': 'medium', '上級': 'hard', '達人': 'master'}
            
class GameState:
    def __init__(self, handicap='通常', mode='2P', cpu_difficulty='easy', time_limit=None):
        self.board = standard_setup()
        for _, pos in HANDICAPS.get(handicap, []):
            self.board[pos[0]][pos[1]] = None

        self.hands = {0: [], 1: []}
        self.turn = 0
        self.selected = None
        self.legal_moves = []
        self.selected_hand = None
        self.kifu = []
        self.game_over = False
        self.winner = None
        self.last_move_target = None

        self.mode = mode
        self.cpu_difficulty = cpu_difficulty
        self.time_limit = time_limit
        self.sente_time = time_limit if time_limit else 0.0
        self.gote_time = time_limit if time_limit else 0.0
        self.last_move_time = time.time()
        self.timer_paused = False
        self.in_sudden_death = {0: False, 1: False}

        self.kifu_scroll_offset = 0
        self.saved_message_time = None
        self.resigning_animation = False
        self.animated_pieces = []
        self.animation_finished = False
        self.cpu_thinking = False
        self.hand_scroll_offset = {0: 0, 1: 0}
        self.hand_scrollbar_rect = {0: None, 1: None}
        self.check_display_time = 0
        self.checkmate_display_time = 0
        self.history = []

    def save_history(self):
        history_item = {
            'board': deepcopy(self.board), 'hands': deepcopy(self.hands),
            'turn': self.turn, 'kifu': list(self.kifu),
            'sente_time': self.sente_time, 'gote_time': self.gote_time,
            'last_move_target': self.last_move_target,
            'in_sudden_death': deepcopy(self.in_sudden_death)
        }
        self.history.append(history_item)

    def load_history(self, item):
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


def find_king(board, owner):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            p = board[x][y]
            if p and p.kind == 'K' and p.owner == owner:
                return (x, y)
    return None

def is_in_check(board, owner):
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

def generate_all_moves_no_check(board, x, y, owner, kind=None):
    moves = []
    if x is None:
        # delegate drop-generation rules to helper
        return _generate_drop_moves(board, kind, owner)

    p = board[x][y]
    if not p:
        return []
    # Use module-level helpers for step and slider move generation
    if p.kind in ['B+', 'R+']:
        return _generate_promoted_moves(board, x, y, p)
    else:
        if p.kind in STEP_MOVES:
            _add_step_moves(board, x, y, p.owner, p.kind, moves)
        dirs = SLIDER_DIRS.get(p.kind)
        if dirs:
            _add_slider_moves(board, x, y, p.owner, dirs, moves)
    return moves

def generate_all_moves(board, x, y, owner, kind=None, check_rule=True):
    moves = generate_all_moves_no_check(board, x, y, owner, kind)
    if not check_rule: return moves
    valid_moves = []
    for tx, ty in moves:
        temp_board = deepcopy(board)
        if x is None: temp_board[tx][ty] = Piece(kind, owner)
        else:
            piece = temp_board[x][y]
            temp_board[tx][ty], temp_board[x][y] = piece.clone(), None
        if not is_in_check(temp_board, owner): valid_moves.append((tx, ty))
    return valid_moves

def get_legal_moves_all(state, owner):
    all_moves = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            p = state.board[x][y]
            if p and p.owner == owner:
                all_moves.extend([(x,y,tx,ty) for tx,ty in generate_all_moves(state.board, x, y, owner)])
    # enumerate actual hand indices to preserve duplicate pieces and correct indices
    for idx, kind in enumerate(list(state.hands[owner])):
        all_moves.extend([(None, idx, tx, ty) for tx, ty in generate_all_moves(state.board, None, idx, owner, kind=kind)])
    return all_moves

def check_mate(state):
    # If the king for the side to move is missing (captured), treat as mate/game over
    king_pos = find_king(state.board, state.turn)
    if not king_pos:
        return True
    # Otherwise, standard mate detection: in-check and no legal moves
    if not is_in_check(state.board, state.turn):
        return False
    if not get_legal_moves_all(state, state.turn):
        return True
    return False

def check_sennichite(state):
    return False

def load_piece_images():
    pieces = {}
    names = ["K","G","S","N","L","P","B","R","S+","N+","L+","P+","B+","R+"]
    for n in names:
        try:
            img_sente = pygame.image.load(os.path.join(image_path, f"{n}_S.png"))
            img_sente = pygame.transform.smoothscale(img_sente, (PIECE_SIZE, PIECE_SIZE))
            img_gote = pygame.image.load(os.path.join(image_path, f"{n}_G.png"))
            img_gote = pygame.transform.smoothscale(img_gote, (PIECE_SIZE, PIECE_SIZE))
            pieces[(n,0)], pieces[(n,1)] = img_sente, img_gote
        except pygame.error as e: print(f"駒画像ファイルの読み込みに失敗しました: {n} - {e}")
    return pieces

def draw_board_programmatically(screen):
    margin_rect = pygame.Rect(WINDOW_PADDING_X, BOARD_START_Y - COORD_MARGIN,
                              BOARD_PIXEL_WIDTH + COORD_MARGIN*2, BOARD_PIXEL_HEIGHT + COORD_MARGIN*2)
    pygame.draw.rect(screen, BOARD_COLOR, margin_rect)
    board_outer_rect = pygame.Rect(BOARD_START_X, BOARD_START_Y, BOARD_PIXEL_WIDTH, BOARD_PIXEL_HEIGHT)
    pygame.draw.rect(screen, BOARD_COLOR, board_outer_rect)
    for i in range(BOARD_SIZE + 1):
        pygame.draw.line(screen, BLACK, (BOARD_START_X+i*SQUARE, BOARD_START_Y), (BOARD_START_X+i*SQUARE, BOARD_START_Y+BOARD_PIXEL_HEIGHT), 2 if i in [0,BOARD_SIZE] else 1)
        pygame.draw.line(screen, BLACK, (BOARD_START_X, BOARD_START_Y+i*SQUARE), (BOARD_START_X+BOARD_PIXEL_WIDTH, BOARD_START_Y+i*SQUARE), 2 if i in [0,BOARD_SIZE] else 1)
    for i in range(BOARD_SIZE):
        num_text_top = LARGE_FONT.render(str(9-i), True, BLACK)
        screen.blit(num_text_top, (BOARD_START_X+i*SQUARE+(SQUARE-num_text_top.get_width())//2, BOARD_START_Y - COORD_MARGIN))
        num_text_bottom = LARGE_FONT.render(str(9-i), True, BLACK)
        screen.blit(num_text_bottom, (BOARD_START_X+i*SQUARE+(SQUARE-num_text_bottom.get_width())//2, BOARD_START_Y+BOARD_PIXEL_HEIGHT+5))
        kanji_text_right = LARGE_FONT.render(JAPANESE_Y_COORDS[i], True, BLACK)
        screen.blit(kanji_text_right, (BOARD_START_X+BOARD_PIXEL_WIDTH+5, BOARD_START_Y+i*SQUARE+(SQUARE-kanji_text_right.get_height())//2))
        kanji_text_left = LARGE_FONT.render(JAPANESE_Y_COORDS[i], True, BLACK)
        screen.blit(kanji_text_left, (BOARD_START_X-COORD_MARGIN+5, BOARD_START_Y+i*SQUARE+(SQUARE-kanji_text_left.get_height())//2))

def draw_game_elements(screen, state, piece_images):
    # Top-level drawing flow delegated to helpers for readability
    screen.fill(TATAMI_GREEN)
    hand_y_gote = BOARD_START_Y - COORD_MARGIN - HAND_AREA_HEIGHT
    hand_y_sente = BOARD_START_Y + BOARD_PIXEL_HEIGHT + COORD_MARGIN
    hand_x = BOARD_START_X
    # draw hand backgrounds
    pygame.draw.rect(screen, DARK_BROWN, (hand_x, hand_y_gote, BOARD_PIXEL_WIDTH, HAND_AREA_HEIGHT))
    pygame.draw.rect(screen, DARK_BROWN, (hand_x, hand_y_sente, BOARD_PIXEL_WIDTH, HAND_AREA_HEIGHT))
    draw_board_programmatically(screen)

    if state.resigning_animation:
        # animated pieces falling
        remaining = []
        for ap in state.animated_pieces:
            alive = ap.update()
            img = piece_images.get((ap.piece.kind, ap.owner))
            if img and alive:
                rotated_img = pygame.transform.rotate(img, ap.angle)
                screen.blit(rotated_img, rotated_img.get_rect(center=(ap.x, ap.y)))
            if alive:
                remaining.append(ap)
        state.animated_pieces = remaining
        if not state.animated_pieces:
            state.animation_finished = True
    else:
        # selection and legal move highlights
        if state.selected:
            pygame.draw.rect(screen, BLUE, (BOARD_START_X+state.selected[0]*SQUARE, BOARD_START_Y+state.selected[1]*SQUARE, SQUARE, SQUARE), 3)
        for _,_,tx,ty in state.legal_moves:
            pygame.draw.rect(screen, RED if state.board[tx][ty] else GREEN, (BOARD_START_X+tx*SQUARE, BOARD_START_Y+ty*SQUARE, SQUARE, SQUARE), 3)
        # draw pieces on board
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                p = state.board[x][y]
                if p:
                    img = piece_images.get((p.kind, p.owner))
                    if img:
                        screen.blit(img, (BOARD_START_X+x*SQUARE+PIECE_OFFSET, BOARD_START_Y+y*SQUARE+PIECE_OFFSET))
        # draw hands and scrollbars
        max_visible_hand = BOARD_PIXEL_WIDTH // SQUARE
        for owner in [0, 1]:
            hand, offset = state.hands[owner], state.hand_scroll_offset[owner]
            hand_y = hand_y_sente if owner == 0 else hand_y_gote
            for i, p_kind in enumerate(hand[offset:offset+max_visible_hand]):
                img = piece_images.get((p_kind, owner))
                if img:
                    screen.blit(img, (hand_x+i*SQUARE+PIECE_OFFSET, hand_y+PIECE_OFFSET))
            if len(hand) > max_visible_hand:
                scrollbar_width = BOARD_PIXEL_WIDTH * (max_visible_hand / len(hand))
                scroll_ratio = offset / (len(hand)-max_visible_hand) if (len(hand)-max_visible_hand) > 0 else 0
                scrollbar_x = hand_x + (BOARD_PIXEL_WIDTH-scrollbar_width)*scroll_ratio
                rect = pygame.Rect(scrollbar_x, hand_y+HAND_AREA_HEIGHT-5, scrollbar_width, 5)
                pygame.draw.rect(screen, LIGHT_GRAY, rect)
                state.hand_scrollbar_rect[owner] = rect
            else:
                state.hand_scrollbar_rect[owner] = None

        if state.selected_hand is not None:
            owner, offset = state.turn, state.hand_scroll_offset[state.turn]
            display_idx = state.selected_hand - offset
            if 0 <= display_idx < max_visible_hand:
                hand_y = hand_y_sente if owner==0 else hand_y_gote
                pygame.draw.rect(screen, BLUE, (hand_x+display_idx*SQUARE, hand_y, SQUARE, SQUARE), 3)

    draw_kifu(screen, state)

    if state.check_display_time > 0 and time.time() - state.check_display_time < 1.5:
        check_text = CHECK_FONT.render("王手", True, RED)
        text_rect = check_text.get_rect(center=(BOARD_START_X + BOARD_PIXEL_WIDTH//2, BOARD_START_Y + BOARD_PIXEL_HEIGHT//2))
        shadow_text = CHECK_FONT.render("王手", True, BLACK)
        screen.blit(shadow_text, (text_rect.x + 3, text_rect.y + 3))
        screen.blit(check_text, text_rect)
    else:
        state.check_display_time = 0

    if state.saved_message_time and time.time() - state.saved_message_time < 2:
        saved_text = LARGE_FONT.render("棋譜を保存しました！", True, BUTTON_BLUE)
        screen.blit(saved_text, saved_text.get_rect(center=(WIDTH//2, HEIGHT//2-200)))

    pygame.display.flip()

def draw_kifu(screen, state):
    kifu_area_x = BOARD_START_X + BOARD_PIXEL_WIDTH + COORD_MARGIN + WINDOW_PADDING_X
    kifu_area_y = WINDOW_PADDING_Y
    kifu_area_rect = pygame.Rect(kifu_area_x, kifu_area_y, KIFU_WINDOW_WIDTH, HEIGHT - WINDOW_PADDING_Y*2)
    pygame.draw.rect(screen, WHITE, kifu_area_rect); pygame.draw.rect(screen, BLACK, kifu_area_rect, 2)
    info_panel_rect = pygame.Rect(kifu_area_x, kifu_area_y, KIFU_WINDOW_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, GRAY, info_panel_rect); pygame.draw.rect(screen, BLACK, info_panel_rect, 2)

    # delegate sub-sections
    draw_kifu_info(screen, state, kifu_area_x, kifu_area_y)
    draw_kifu_list(screen, state, kifu_area_x, kifu_area_y)
    draw_kifu_buttons(screen, state, kifu_area_x)

def draw_kifu_info(screen, state, kifu_area_x, kifu_area_y):
    sente_t, gote_t = state.sente_time, state.gote_time
    if not state.timer_paused and not state.game_over:
        elapsed = time.time() - state.last_move_time
        if state.time_limit is not None:
            if state.turn == 0: sente_t = max(0, state.sente_time - elapsed)
            else: gote_t = max(0, state.gote_time - elapsed)
        else:
            if state.turn == 0: sente_t += elapsed
            else: gote_t += elapsed

    sente_time_str = time.strftime('%H:%M:%S', time.gmtime(sente_t))
    gote_time_str = time.strftime('%H:%M:%S', time.gmtime(gote_t))
    info_texts = [f"手数: {len(state.kifu)}", f"手番: {JAPANESE_TURN_NAME[state.turn]}",
                  f"▲先手: {sente_time_str}", f"△後手: {gote_time_str}"]
    for i, text in enumerate(info_texts):
        screen.blit(FONT.render(text, True, BLACK), (kifu_area_x + 10, kifu_area_y + 5 + i*22))

def draw_kifu_list(screen, state, kifu_area_x, kifu_area_y):
    kifu_list_y = kifu_area_y + INFO_PANEL_HEIGHT + KIFU_LIST_PADDING
    kifu_list_height = HEIGHT-kifu_list_y-WINDOW_PADDING_Y-RESIGN_BUTTON_HEIGHT-SAVE_BUTTON_HEIGHT-MATTA_BUTTON_HEIGHT-TIMER_BUTTON_HEIGHT-40
    max_lines = kifu_list_height // KIFU_ITEM_HEIGHT if KIFU_ITEM_HEIGHT > 0 else 0
    start_index = state.kifu_scroll_offset
    for i, move in enumerate(state.kifu[start_index:start_index+max_lines]):
        screen.blit(MONO_FONT.render(f"{start_index+i+1}. {move}", True, BLACK), (kifu_area_x+15, kifu_list_y+i*KIFU_ITEM_HEIGHT))
    if len(state.kifu) > max_lines:
        bar_h = kifu_list_height * (max_lines/len(state.kifu)) if len(state.kifu) > 0 else kifu_list_height
        ratio = start_index / (len(state.kifu)-max_lines) if (len(state.kifu)-max_lines) > 0 else 0
        bar_y = kifu_list_y + (kifu_list_height-bar_h)*ratio
        state.scrollbar_rect = pygame.Rect(kifu_area_x+KIFU_WINDOW_WIDTH-15, bar_y, 10, bar_h)
        pygame.draw.rect(screen, LIGHT_GRAY, state.scrollbar_rect)
    else:
        state.scrollbar_rect = None

def draw_kifu_buttons(screen, state, kifu_area_x):
    resign_button_y = HEIGHT-WINDOW_PADDING_Y-RESIGN_BUTTON_HEIGHT-10
    button_x = kifu_area_x+(KIFU_WINDOW_WIDTH-260)/2
    state.resign_button_rect = pygame.Rect(button_x, resign_button_y, 260, RESIGN_BUTTON_HEIGHT)
    pygame.draw.rect(screen, RED, state.resign_button_rect, 0, 10)
    resign_surf = LARGE_FONT.render("投了", True, WHITE)
    screen.blit(resign_surf, resign_surf.get_rect(center=state.resign_button_rect.center))

    save_button_y = resign_button_y - SAVE_BUTTON_HEIGHT - 10
    state.save_button_rect = pygame.Rect(button_x, save_button_y, 260, SAVE_BUTTON_HEIGHT)
    pygame.draw.rect(screen, BUTTON_BLUE, state.save_button_rect, 0, 10)
    save_surf = LARGE_FONT.render("棋譜を保存", True, WHITE)
    screen.blit(save_surf, save_surf.get_rect(center=state.save_button_rect.center))
    
    matta_button_y = save_button_y - MATTA_BUTTON_HEIGHT - 10
    state.matta_button_rect = pygame.Rect(button_x, matta_button_y, 260, MATTA_BUTTON_HEIGHT)
    color = ORANGE if len(state.history) > 0 else LIGHT_GRAY
    pygame.draw.rect(screen, color, state.matta_button_rect, 0, 10)
    matta_surf = LARGE_FONT.render("待った", True, WHITE)
    screen.blit(matta_surf, matta_surf.get_rect(center=state.matta_button_rect.center))

    if state.time_limit:
        timer_button_y = matta_button_y - TIMER_BUTTON_HEIGHT - 10
        state.timer_button_rect = pygame.Rect(button_x, timer_button_y, 260, TIMER_BUTTON_HEIGHT)
        text = "タイマー再開" if state.timer_paused else "タイマーストップ"
        pygame.draw.rect(screen, GREEN, state.timer_button_rect,0,10)
        screen.blit(LARGE_FONT.render(text, True, WHITE), LARGE_FONT.render(text, True, WHITE).get_rect(center=state.timer_button_rect.center))

def ask_promotion(screen):
    dialog = pygame.Surface((300,150)); dialog.fill(GRAY)
    dialog_rect = dialog.get_rect(center=(WIDTH//2, HEIGHT//2)); pygame.draw.rect(dialog, BLACK, dialog.get_rect(), 3)
    dialog.blit(LARGE_FONT.render("成りますか?", True, BLACK), LARGE_FONT.render("成りますか?", True, BLACK).get_rect(center=(dialog.get_width()//2, 40)))
    yes_rect, no_rect = pygame.Rect(50,80,80,40), pygame.Rect(170,80,80,40)
    pygame.draw.rect(dialog, GREEN, yes_rect); pygame.draw.rect(dialog, RED, no_rect)
    dialog.blit(LARGE_FONT.render("はい",True,BLACK),(65,85)); dialog.blit(LARGE_FONT.render("いいえ",True,BLACK),(175,85))
    screen.blit(dialog, dialog_rect.topleft); pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_rect.move(dialog_rect.topleft).collidepoint(event.pos): return True
                elif no_rect.move(dialog_rect.topleft).collidepoint(event.pos): return False

def ask_rematch(screen, message="再対局しますか？"):
    dialog = pygame.Surface((400,200)); dialog.fill(GRAY)
    dialog_rect = dialog.get_rect(center=(WIDTH//2, HEIGHT//2)); pygame.draw.rect(dialog, BLACK, dialog.get_rect(), 3)
    dialog.blit(LARGE_FONT.render(message, True, BLACK), LARGE_FONT.render(message, True, BLACK).get_rect(center=(dialog.get_width()//2, 60)))
    yes_rect_text, no_rect_text = ("はい", "いいえ") if message == "再対局しますか？" else ("続ける", "終了")
    yes_rect, no_rect = pygame.Rect(50,120,120,50), pygame.Rect(230,120,120,50)
    pygame.draw.rect(dialog, GREEN, yes_rect); pygame.draw.rect(dialog, RED, no_rect)
    dialog.blit(LARGE_FONT.render(yes_rect_text,True,BLACK),LARGE_FONT.render(yes_rect_text,True,BLACK).get_rect(center=yes_rect.center))
    dialog.blit(LARGE_FONT.render(no_rect_text,True,BLACK),LARGE_FONT.render(no_rect_text,True,BLACK).get_rect(center=no_rect.center))
    screen.blit(dialog, dialog_rect.topleft); pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_rect.move(dialog_rect.topleft).collidepoint(event.pos): return True
                elif no_rect.move(dialog_rect.topleft).collidepoint(event.pos): return False

def show_greeting(screen, text, animation_type):
    clock = pygame.time.Clock()
    font_to_use = CHECK_FONT if text == "詰み" else GREETING_FONT
    text_surf = font_to_use.render(text, True, TITLE_COLOR)
    shadow_surf = font_to_use.render(text, True, DARK_BROWN)
    
    duration = 1500

    if animation_type == 'split-in':
        end_y, start_y = HEIGHT // 2, -text_surf.get_height()
        anim_start_time, anim_duration = pygame.time.get_ticks(), 500
        while pygame.time.get_ticks() - anim_start_time < anim_duration:
            ratio = (pygame.time.get_ticks() - anim_start_time) / anim_duration
            current_y = start_y + (end_y - start_y) * ratio
            screen.fill(TATAMI_GREEN)
            rect = text_surf.get_rect(center=(WIDTH//2, int(current_y)))
            screen.blit(shadow_surf, (rect.x + 4, rect.y + 4))
            screen.blit(text_surf, rect)
            pygame.display.flip(); clock.tick(FPS)
    
    elif animation_type == 'fade-in':
        for alpha in range(0, 256, 15):
            screen.fill(TATAMI_GREEN)
            text_surf.set_alpha(alpha); shadow_surf.set_alpha(alpha)
            rect = text_surf.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(shadow_surf, (rect.x + 4, rect.y + 4))
            screen.blit(text_surf, rect)
            pygame.display.flip(); clock.tick(FPS)
    
    pygame.time.wait(duration)

    for alpha in range(255, -1, -15):
        screen.fill(TATAMI_GREEN)
        text_surf.set_alpha(alpha); shadow_surf.set_alpha(alpha)
        rect = text_surf.get_rect(center=(WIDTH//2, HEIGHT//2))
        screen.blit(shadow_surf, (rect.x + 4, rect.y + 4))
        screen.blit(text_surf, rect)
        pygame.display.flip(); clock.tick(FPS)

def start_screen(screen):
    clock=pygame.time.Clock(); anim_speed=12
    buttons=[{'rect':pygame.Rect(WIDTH//2-125,HEIGHT//2-50,250,60),'text':"二人で対局",'action':'2P'},
             {'rect':pygame.Rect(WIDTH//2-125,HEIGHT//2+40,250,60),'text':"一人で対局",'action':'CPU'},
             {'rect':pygame.Rect(WIDTH//2-125,HEIGHT//2+130,250,60),'text':"終了",'action':'QUIT'}]
    bg = pygame.Surface((WIDTH, HEIGHT))
    if start_bg_img:
        scaled=pygame.transform.scale(start_bg_img,(WIDTH,HEIGHT)); scaled.set_alpha(80); bg.blit(scaled,(0,0))
    else: bg.fill(TATAMI_GREEN)
    
    scaled_fusuma_l, scaled_fusuma_r = None, None
    if fusuma_left_img and fusuma_right_img:
        scaled_fusuma_l = pygame.transform.scale(fusuma_left_img, (WIDTH // 2 + 5, HEIGHT))
        scaled_fusuma_r = pygame.transform.scale(fusuma_right_img, (WIDTH // 2 + 5, HEIGHT))
    
    fusuma_x_l, fusuma_x_r = 0, WIDTH // 2
    animation_started = False; action_to_return = None
    
    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type==pygame.QUIT: pygame.quit(); sys.exit()
            if event.type==pygame.MOUSEBUTTONDOWN and not animation_started:
                for btn in buttons:
                    if btn['rect'].collidepoint(mouse_pos):
                        if sound_se1: sound_se1.play()
                        if btn['action']=='QUIT': pygame.quit(); sys.exit()
                        animation_started = True
                        action_to_return = btn['action']
        
        screen.blit(bg, (0,0))
        
        if scaled_fusuma_l and scaled_fusuma_r:
            screen.blit(scaled_fusuma_l, (fusuma_x_l, 0))
            screen.blit(scaled_fusuma_r, (fusuma_x_r, 0))

        title_surf = TITLE_FONT.render("将棋道場", True, TITLE_COLOR)
        title_rect = title_surf.get_rect(center=(WIDTH//2, HEIGHT//2-150))
        shadow_surf = TITLE_FONT.render("将棋道場", True, DARK_BROWN)
        screen.blit(shadow_surf, (title_rect.x + 4, title_rect.y + 4))
        screen.blit(title_surf, title_rect)
        for btn in buttons:
            color = BUTTON_BLUE_HOVER if btn['rect'].collidepoint(mouse_pos) else BUTTON_BLUE
            pygame.draw.rect(screen, color, btn['rect'],0,10)
            screen.blit(LARGE_FONT.render(btn['text'],True,WHITE), LARGE_FONT.render(btn['text'],True,WHITE).get_rect(center=btn['rect'].center))

        if animation_started:
            fusuma_x_l -= anim_speed; fusuma_x_r += anim_speed
            if fusuma_x_l < -WIDTH // 2:
                return action_to_return

        pygame.display.flip(); clock.tick(FPS)

def selection_screen(screen, title, items):
    screen.fill(TATAMI_GREEN)
    title_surf = LARGE_FONT.render(title,True,BLACK)
    screen.blit(title_surf, (WIDTH//2-title_surf.get_width()//2, 50))
    buttons = []
    for i, name in enumerate(items):
        rect = pygame.Rect(WIDTH//2-100, 120+i*(50+10), 200, 50)
        buttons.append((rect, name))
        pygame.draw.rect(screen, BUTTON_BLUE, rect, 0, 10)
        text_surf = FONT.render(name,True,WHITE)
        screen.blit(text_surf, text_surf.get_rect(center=rect.center))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, name in buttons:
                    if rect.collidepoint(event.pos): return name

def ask_continue_screen(screen):
    screen.fill(TATAMI_GREEN)
    title_surf = LARGE_FONT.render("前回の棋譜の続きからしますか？",True,BLACK)
    screen.blit(title_surf, (WIDTH//2-title_surf.get_width()//2, 100))
    kifu_exists = os.path.exists(kifu_path)
    buttons = {'new':{'rect':pygame.Rect(WIDTH//2-120,200,240,60),'text':'初めから'},
               'continue':{'rect':pygame.Rect(WIDTH//2-120,280,240,60),'text':'続きから'}}
    clock = pygame.time.Clock()
    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if buttons['new']['rect'].collidepoint(mouse_pos): return 'new'
                if kifu_exists and buttons['continue']['rect'].collidepoint(mouse_pos): return 'continue'
        color = BUTTON_BLUE_HOVER if buttons['new']['rect'].collidepoint(mouse_pos) else BUTTON_BLUE
        pygame.draw.rect(screen,color,buttons['new']['rect'],0,10)
        text = LARGE_FONT.render(buttons['new']['text'],True,WHITE)
        screen.blit(text,text.get_rect(center=buttons['new']['rect'].center))
        color = (150,150,150); text_color = LIGHT_GRAY
        if kifu_exists:
            color = BUTTON_BLUE_HOVER if buttons['continue']['rect'].collidepoint(mouse_pos) else BUTTON_BLUE
            text_color = WHITE
        pygame.draw.rect(screen,color,buttons['continue']['rect'],0,10)
        text = LARGE_FONT.render(buttons['continue']['text'],True,text_color)
        screen.blit(text,text.get_rect(center=buttons['continue']['rect'].center))
        pygame.display.flip(); clock.tick(FPS)

def evaluate_board(state, owner):
    score = 0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            piece = state.board[x][y]
            if piece:
                value = PIECE_VALUES.get(piece.kind, 0)
                if piece.owner == owner:
                    if y in PROMOTION_ZONE[owner]: value += 10
                    score += value
                else:
                    if y in PROMOTION_ZONE[1-owner]: value += 10
                    score -= value
    for p_kind in state.hands[owner]: score += PIECE_VALUES.get(p_kind, 0)
    for p_kind in state.hands[1-owner]: score -= PIECE_VALUES.get(p_kind, 0)
    return score

def get_cpu_move_beginner(state):
    legal_moves = get_legal_moves_all(state, state.turn)
    return random.choice(legal_moves) if legal_moves else None

def get_cpu_move_easy(state):
    legal_moves = get_legal_moves_all(state, state.turn)
    if not legal_moves: return None
    capture_moves = [m for m in legal_moves if m[0] is not None and state.board[m[2]][m[3]]]
    return random.choice(capture_moves) if capture_moves else random.choice(legal_moves)

def get_cpu_move_medium(state):
    legal_moves = get_legal_moves_all(state, state.turn)
    if not legal_moves: return None
    best_move, best_score = random.choice(legal_moves), -float('inf')
    for move in legal_moves:
        temp_state = deepcopy(state); apply_move(temp_state, move, is_cpu=True, update_history=False)
        score = evaluate_board(temp_state, state.turn)
        if score > best_score: best_score, best_move = score, move
    return best_move

def alpha_beta_search(state, depth, alpha, beta, maximizing_player, owner):
    if depth == 0 or state.game_over:
        return evaluate_board(state, owner), None

    legal_moves = get_legal_moves_all(state, state.turn)
    if not legal_moves:
        return (-100000 if is_in_check(state.board, state.turn) else 0), None
    
    best_move = random.choice(legal_moves)
    if maximizing_player:
        max_eval = -float('inf')
        for move in legal_moves:
            temp_state = deepcopy(state)
            apply_move(temp_state, move, is_cpu=True, update_history=False)
            val, _ = alpha_beta_search(temp_state, depth-1, alpha, beta, False, owner)
            if val > max_eval:
                max_eval = val
                best_move = move
            alpha = max(alpha, val)
            if beta <= alpha: break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            temp_state = deepcopy(state)
            apply_move(temp_state, move, is_cpu=True, update_history=False)
            val, _ = alpha_beta_search(temp_state, depth-1, alpha, beta, True, owner)
            if val < min_eval:
                min_eval = val
                best_move = move
            beta = min(beta, val)
            if beta <= alpha: break
        return min_eval, best_move

def get_cpu_move_hard(state):
    _, move = alpha_beta_search(state, 2, -float('inf'), float('inf'), True, state.turn)
    return move

def get_cpu_move_master(state):
    _, move = alpha_beta_search(state, 3, -float('inf'), float('inf'), True, state.turn)
    return move

def apply_move(state, move, is_cpu=False, screen=None, force_promotion=None, update_history=True):
    if update_history: state.save_history()
    
    if not state.timer_paused:
        elapsed = time.time() - state.last_move_time
        if state.time_limit is not None:
            if state.turn == 0: state.sente_time = max(0, state.sente_time - elapsed)
            else: state.gote_time = max(0, state.gote_time - elapsed)
        else:
            if state.turn == 0: state.sente_time += elapsed
            else: state.gote_time += elapsed

    sx, sy_or_idx, tx, ty = move
    if sx is not None:
        piece, captured = state.board[sx][sy_or_idx], state.board[tx][ty]
        dest = "同" if state.last_move_target==(tx,ty) else coords_to_kifu(tx,ty)
        kifu_text = f"{JAPANESE_TURN_SYMBOL[state.turn]}{dest}{JAPANESE_PIECE_NAMES[demote_kind(piece.kind)]}"
        promo = False
        if not piece.promoted and piece.kind in PROMOTE_MAP:
            must = (piece.kind in ['P','L'] and (ty==0 if piece.owner==0 else ty==8)) or \
                   (piece.kind=='N' and (ty<=1 if piece.owner==0 else ty>=7))
            can = (ty in PROMOTION_ZONE[piece.owner] or sy_or_idx in PROMOTION_ZONE[piece.owner])
            if must: promo = True
            elif can:
                if force_promotion is not None: promo = force_promotion
                elif not is_cpu: promo = ask_promotion(screen)
                else: promo = True
            if promo: piece.kind = PROMOTE_MAP[piece.kind]; kifu_text += "成"
            elif can: kifu_text += "不成"
        if captured:
            state.hands[state.turn].append(demote_kind(captured.kind)); state.hands[state.turn].sort()
        state.board[tx][ty], state.board[sx][sy_or_idx] = piece, None
        state.selected, state.legal_moves = None, []
    else:
        kind = state.hands[state.turn].pop(sy_or_idx)
        state.board[tx][ty] = Piece(kind, state.turn)
        kifu_text = f"{JAPANESE_TURN_SYMBOL[state.turn]}{coords_to_kifu(tx,ty)}{JAPANESE_PIECE_NAMES[kind]}打"
        state.selected_hand, state.legal_moves = None, []

    state.kifu.append(kifu_text)
    state.last_move_target = (tx, ty)
    state.turn = 1 - state.turn
    state.cpu_thinking = False
    state.last_move_time = time.time()
    
    if is_in_check(state.board, state.turn):
        state.check_display_time = time.time()

    if sound_itte and not is_cpu: sound_itte.play()
    if check_sennichite(state): return
    if check_mate(state):
        state.game_over, state.winner = True, 1-state.turn
        state.checkmate_display_time = time.time()
        if sound_end and not is_cpu: sound_end.play()

def save_kifu_to_csv(kifu):
    try:
        with open(kifu_path, 'w', newline='', encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['手数','棋譜'])
            for i, move in enumerate(kifu): writer.writerow([i+1, move])
        return True
    except IOError as e: print(f"ファイル保存失敗: {e}"); return False

def parse_kifu_to_move(state, kifu_str):
    kifu_str = kifu_str.strip().translate(ZEN_TO_HAN_TABLE)
    turn = 0 if kifu_str[0]=='▲' else 1
    if turn != state.turn: raise ValueError("Turn mismatch")
    drop, promo, no_promo = "打" in kifu_str, "成" in kifu_str, "不成" in kifu_str
    promo_flag = promo if (promo or no_promo) else None
    if kifu_str[1] == "同":
        tx,ty = state.last_move_target
        p_name = kifu_str[2:].replace("成","").replace("不成","").replace("打","")
    else:
        tx,ty = 9-int(kifu_str[1]), Y_COORD_FROM_JP[kifu_str[2]]
        p_name = kifu_str[3:].replace("成","").replace("不成","").replace("打","")
    p_kind = KIND_FROM_JP[p_name]
    if drop:
        try: idx = state.hands[turn].index(p_kind); return ((None,idx,tx,ty), promo_flag)
        except ValueError: raise ValueError(f"{p_kind} not in hand")
    else:
        base_kind = demote_kind(p_kind)
        sources = []
        for sx in range(BOARD_SIZE):
            for sy in range(BOARD_SIZE):
                p = state.board[sx][sy]
                if p and p.owner==turn and demote_kind(p.kind)==base_kind:
                    if (tx,ty) in generate_all_moves(state.board, sx, sy, turn, check_rule=False):
                        temp_b = deepcopy(state.board); temp_b[tx][ty],temp_b[sx][sy]=p,None
                        if not is_in_check(temp_b, turn): sources.append((sx, sy))
        if not sources: raise ValueError(f"No valid source for {kifu_str}")
        if len(sources)>1: print(f"Ambiguous move: {kifu_str}")
        return ((sources[0][0],sources[0][1],tx,ty), promo_flag)

def load_kifu_and_setup_state(handicap, mode, cpu_difficulty, time_limit=None):
    state = GameState(handicap, mode, cpu_difficulty, time_limit)
    try:
        with open(kifu_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f); next(reader)
            for row in reader:
                if not row: continue
                try:
                    move, promo = parse_kifu_to_move(state, row[1])
                    apply_move(state, move, is_cpu=True, force_promotion=promo, update_history=False)
                except Exception as e: print(f"Kifu parse error '{row[1]}': {e}"); return state
    except FileNotFoundError: print("kifu file not found.")
    state.history = []
    return state

def game_over_screen(screen, state):
    buttons = [
        {'rect': pygame.Rect(WIDTH//2-150, HEIGHT//2+50, 300, 50), 'text': "棋譜を保存", 'action': 'SAVE'},
        {'rect': pygame.Rect(WIDTH//2-150, HEIGHT//2+120, 300, 50), 'text': "再対局", 'action': 'REMATCH'},
        {'rect': pygame.Rect(WIDTH//2-150, HEIGHT//2+190, 300, 50), 'text': "終了", 'action': 'QUIT'}
    ]
    saved_message_time = 0

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'QUIT'
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn['rect'].collidepoint(mouse_pos):
                        if btn['action'] == 'SAVE':
                            if save_kifu_to_csv(state.kifu):
                                saved_message_time = time.time()
                        else:
                            return btn['action']
        
        screen.fill(TATAMI_GREEN)
        msg = f"{JAPANESE_TURN_NAME[state.winner]}の勝利"
        text_surf = RESULT_FONT.render(msg, True, TITLE_COLOR)
        shadow_surf = RESULT_FONT.render(msg, True, DARK_BROWN)
        text_rect = text_surf.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
        screen.blit(shadow_surf, (text_rect.x + 4, text_rect.y + 4))
        screen.blit(text_surf, text_rect)

        for btn in buttons:
            color = BUTTON_BLUE_HOVER if btn['rect'].collidepoint(mouse_pos) else BUTTON_BLUE
            pygame.draw.rect(screen, color, btn['rect'], 0, 10)
            btn_text = LARGE_FONT.render(btn['text'], True, WHITE)
            screen.blit(btn_text, btn_text.get_rect(center=btn['rect'].center))
        
        if saved_message_time > 0 and time.time() - saved_message_time < 2:
            saved_text = LARGE_FONT.render("棋譜を保存しました！", True, BUTTON_BLUE)
            screen.blit(saved_text, saved_text.get_rect(center=(WIDTH // 2, HEIGHT - 50)))

        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

def main(initial_settings=None, skip_start=False):
    screen = pygame.display.set_mode((WIDTH, HEIGHT)); pygame.display.set_caption("将棋道場")
    clock = pygame.time.Clock(); piece_images = load_piece_images()

    # If initial_settings provided and skip_start True, start a fresh game with those settings
    if initial_settings and skip_start:
        game_mode = initial_settings.get('mode', '2P')
        handicap = initial_settings.get('handicap', '通常')
        cpu_diff = initial_settings.get('cpu_diff', 'easy')
        time_limit = initial_settings.get('time_limit', None)
        # Always start fresh (do not load previous kifu)
        state = GameState(handicap, game_mode, cpu_diff, time_limit)
    else:
        game_mode = start_screen(screen)
        handicap, cpu_diff, time_limit = '通常', 'easy', None
        if game_mode == '2P':
            handicap = selection_screen(screen, "ハンディキャップを選択", HANDICAPS.keys())
            time_choice = selection_screen(screen, "持ち時間モードを選択", TIME_SETTINGS.keys())
            time_limit = TIME_SETTINGS[time_choice]
        else:
            cpu_diff = CPU_DIFFICULTIES[selection_screen(screen, "CPUの強さを選択", CPU_DIFFICULTIES.keys())]

        if ask_continue_screen(screen) == 'continue':
            state = load_kifu_and_setup_state(handicap, game_mode, cpu_diff, time_limit)
        else:
            state = GameState(handicap, game_mode, cpu_diff, time_limit)
    
    show_greeting(screen, "よろしくお願いします", "split-in")
    state.last_move_time = time.time()
    if sound_start: sound_start.play()
    drag_kifu, drag_hand = False, {0:False, 1:False}
    
    # Run game loop until one game ends; return action 'REMATCH' or 'QUIT'
    running = True
    while running:
        _handle_cpu_turn(screen, state, piece_images)
        mx, my = pygame.mouse.get_pos()
        running, drag_kifu, drag_hand = _handle_events(screen, state, mx, my, drag_kifu, drag_hand)
        _update_timers(screen, state)
        draw_game_elements(screen, state, piece_images)
    action = _handle_game_over(screen, state, piece_images)
    if action:
        settings = {'handicap': handicap, 'mode': game_mode, 'cpu_diff': cpu_diff, 'time_limit': time_limit}
        return action, settings
        clock.tick(FPS)

    # If loop exits normally, return 'QUIT' plus settings
    settings = {'handicap': handicap, 'mode': game_mode, 'cpu_diff': cpu_diff, 'time_limit': time_limit}
    return 'QUIT', settings


def _handle_cpu_turn(screen, state, piece_images):
    """Handle CPU thinking and move application when in CPU mode."""
    if state.mode == 'CPU' and state.turn == 1 and not state.game_over:
        if not state.cpu_thinking:
            if sound_think: sound_think.play(-1)
            state.cpu_thinking = True
        draw_game_elements(screen, state, piece_images); pygame.time.wait(100)
        cpu_move_func = {'beginner': get_cpu_move_beginner, 'easy': get_cpu_move_easy, 'medium': get_cpu_move_medium,
                         'hard': get_cpu_move_hard, 'master': get_cpu_move_master}[state.cpu_difficulty]
        cpu_move = cpu_move_func(state)
        if sound_think: sound_think.stop()
        if cpu_move:
            apply_move(state, cpu_move, is_cpu=True, screen=screen)


def _handle_events(screen, state, mx, my, drag_kifu, drag_hand):
    """Process pygame events extracted from main. Returns (running, drag_kifu, drag_hand)."""
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            return running, drag_kifu, drag_hand
        if state.game_over:
            continue

        if event.type == pygame.MOUSEBUTTONDOWN:
            if getattr(state, 'resign_button_rect', None) and state.resign_button_rect.collidepoint(mx, my):
                state.game_over = True; state.winner = 1 - state.turn
                state.last_move_time = time.time()
                state.resigning_animation = True
                for x in range(BOARD_SIZE):
                    for y in range(BOARD_SIZE):
                        p = state.board[x][y]
                        if p: state.animated_pieces.append(AnimatedPiece(p, BOARD_START_X + x * SQUARE + SQUARE // 2, BOARD_START_Y + y * SQUARE + SQUARE // 2, p.owner))
                if sound_end: sound_end.play()
            elif getattr(state, 'save_button_rect', None) and state.save_button_rect.collidepoint(mx, my):
                if save_kifu_to_csv(state.kifu): state.saved_message_time = time.time()
            elif getattr(state, 'matta_button_rect', None) and state.matta_button_rect.collidepoint(mx, my):
                if len(state.history) > 0:
                    steps = 2 if state.mode == 'CPU' and len(state.history) > 1 and state.turn == 0 else 1
                    for _ in range(steps):
                        if state.history: last_state = state.history.pop()
                    state.load_history(last_state)
            elif getattr(state, 'timer_button_rect', None) and state.timer_button_rect.collidepoint(mx, my):
                state.timer_paused = not state.timer_paused
                if not state.timer_paused: state.last_move_time = time.time()

            elif getattr(state, 'scrollbar_rect', None) and state.scrollbar_rect.collidepoint(mx, my):
                drag_kifu = True; state.scroll_y_start, state.scroll_offset_start = my, state.kifu_scroll_offset
            elif any(r and r.collidepoint(mx, my) for r in state.hand_scrollbar_rect.values()):
                owner = 0 if state.hand_scrollbar_rect[0] and state.hand_scrollbar_rect[0].collidepoint(mx, my) else 1
                drag_hand[owner] = True; state.scroll_x_start, state.scroll_offset_start = mx, state.hand_scroll_offset[owner]

            elif not state.timer_paused:
                gx, gy = (mx - BOARD_START_X) // SQUARE, (my - BOARD_START_Y) // SQUARE
                hand_y = BOARD_START_Y + BOARD_PIXEL_HEIGHT + COORD_MARGIN if state.turn == 0 else BOARD_START_Y - COORD_MARGIN - HAND_AREA_HEIGHT
                if hand_y <= my < hand_y + HAND_AREA_HEIGHT:
                    offset = state.hand_scroll_offset[state.turn]
                    idx = (mx - BOARD_START_X) // SQUARE + offset
                    if 0 <= idx < len(state.hands[state.turn]):
                        state.selected, state.selected_hand = None, idx
                        kind = state.hands[state.turn][idx]
                        state.legal_moves = [(None, idx, tx, ty) for tx, ty in generate_all_moves(state.board, None, idx, state.turn, kind=kind)]
                    return running, drag_kifu, drag_hand
                if in_bounds(gx, gy):
                    move = next((m for m in state.legal_moves if m[2] == gx and m[3] == gy and
                                 ((state.selected and m[0] is not None and state.selected == (m[0], m[1])) or
                                  (state.selected_hand is not None and m[0] is None and m[1] == state.selected_hand))), None)
                    if move:
                        apply_move(state, move, screen=screen)
                    else:
                        state.selected_hand = None
                        p = state.board[gx][gy]
                        if p and p.owner == state.turn:
                            state.selected, state.legal_moves = (gx, gy), [(gx, gy, tx, ty) for tx, ty in generate_all_moves(state.board, gx, gy, state.turn)]
                        else:
                            state.selected, state.legal_moves = None, []
                else:
                    state.selected, state.selected_hand, state.legal_moves = None, None, []
        elif event.type == pygame.MOUSEBUTTONUP:
            drag_kifu, drag_hand[0], drag_hand[1] = False, False, False
        elif event.type == pygame.MOUSEMOTION:
            if drag_kifu and state.scrollbar_rect:
                k_h = HEIGHT - (WINDOW_PADDING_Y * 2 + INFO_PANEL_HEIGHT + KIFU_LIST_PADDING + RESIGN_BUTTON_HEIGHT + SAVE_BUTTON_HEIGHT + MATTA_BUTTON_HEIGHT + TIMER_BUTTON_HEIGHT + 40)
                max_l = k_h // KIFU_ITEM_HEIGHT
                if len(state.kifu) > max_l:
                    s_h = k_h - state.scrollbar_rect.height
                    if s_h > 0:
                        ratio = (my - state.scroll_y_start) / s_h
                        new_off = int(state.scroll_offset_start + ratio * (len(state.kifu) - max_l))
                        state.kifu_scroll_offset = max(0, min(new_off, len(state.kifu) - max_l))
            for owner in [0, 1]:
                if drag_hand[owner] and state.hand_scrollbar_rect[owner]:
                    max_v = BOARD_PIXEL_WIDTH // SQUARE; h_size = len(state.hands[owner])
                    if h_size > max_v:
                        s_w = BOARD_PIXEL_WIDTH - state.hand_scrollbar_rect[owner].width
                        if s_w > 0:
                            ratio = (mx - state.scroll_x_start) / s_w; max_off = h_size - max_v
                            new_off = int(state.scroll_offset_start + ratio * max_off)
                            state.hand_scroll_offset[owner] = max(0, min(new_off, max_off))
        elif event.type == pygame.MOUSEWHEEL:
            hand_rects = {0: pygame.Rect(BOARD_START_X, BOARD_START_Y + BOARD_PIXEL_HEIGHT + COORD_MARGIN, BOARD_PIXEL_WIDTH, HAND_AREA_HEIGHT),
                          1: pygame.Rect(BOARD_START_X, BOARD_START_Y - COORD_MARGIN - HAND_AREA_HEIGHT, BOARD_PIXEL_WIDTH, HAND_AREA_HEIGHT)}
            owner = next((o for o, r in hand_rects.items() if r.collidepoint(mx, my)), -1)
            if owner != -1:
                max_v = BOARD_PIXEL_WIDTH // SQUARE; h_size = len(state.hands[owner])
                if h_size > max_v:
                    max_off = h_size - max_v
                    state.hand_scroll_offset[owner] = max(0, min(state.hand_scroll_offset[owner] - event.y, max_off))
            else:
                k_h = HEIGHT - (WINDOW_PADDING_Y * 2 + INFO_PANEL_HEIGHT + KIFU_LIST_PADDING + RESIGN_BUTTON_HEIGHT + SAVE_BUTTON_HEIGHT + MATTA_BUTTON_HEIGHT + TIMER_BUTTON_HEIGHT + 40)
                if k_h > 0:
                    max_l = k_h // KIFU_ITEM_HEIGHT
                    if len(state.kifu) > max_l:
                        state.kifu_scroll_offset = max(0, min(state.kifu_scroll_offset - event.y * 3, len(state.kifu) - max_l))
    return running, drag_kifu, drag_hand


def _update_timers(screen, state):
    """Update timers and handle sudden death or timeouts."""
    if state.time_limit and not state.game_over and not state.timer_paused:
        current_time = time.time()
        elapsed = current_time - state.last_move_time
        turn_player = state.turn
        current_player_time = state.sente_time if turn_player == 0 else state.gote_time

        if current_player_time - elapsed <= 0:
            if not state.in_sudden_death[turn_player]:
                if turn_player == 0:
                    state.sente_time = SUDDEN_DEATH_TIME
                else:
                    state.gote_time = SUDDEN_DEATH_TIME
                state.in_sudden_death[turn_player] = True
                state.last_move_time = current_time
            else:
                if ask_rematch(screen, "持ち時間が切れました。続けますか？"):
                    if turn_player == 0:
                        state.sente_time = SUDDEN_DEATH_TIME
                    else:
                        state.gote_time = SUDDEN_DEATH_TIME
                    state.last_move_time = current_time
                else:
                    state.game_over = True; state.winner = 1 - turn_player


def _handle_game_over(screen, state, piece_images):
    """Handle game over display and actions."""
    if state.game_over and not hasattr(state, 'end_processed'):
        if state.checkmate_display_time > 0:
            show_greeting(screen, "詰み", "fade-in")
        elif state.resigning_animation:
            while not state.animation_finished:
                draw_game_elements(screen, state, piece_images)

        show_greeting(screen, "ありがとうございました", "fade-in")
        action = game_over_screen(screen, state)
        if action == 'REMATCH':
            state.end_processed = True
            return 'REMATCH'
        elif action == 'QUIT':
            state.end_processed = True
            return 'QUIT'
        state.end_processed = True
    

if __name__=="__main__":
    # Run games until user chooses to quit
    # Run games until user chooses to quit
    prev_settings = None
    while True:
        result = main(initial_settings=prev_settings, skip_start=bool(prev_settings))
        if isinstance(result, tuple):
            action, prev_settings = result
        else:
            action, prev_settings = result, None
        if action == 'REMATCH' and prev_settings:
            # start a fresh game with same settings
            continue
        else:
            break
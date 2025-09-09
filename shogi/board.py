"""
Board representation, setup, and rendering for Shogi game.

This module provides:
- Board type definitions and utilities
- Initial board setup functions (standard and handicap games)
- Board and game element drawing functions
- Piece-square tables for AI evaluation
"""

import pygame
import os
from typing import List, Optional, Callable, Dict, Tuple, Any
from .piece import Piece
from .utils import (
    BOARD_SIZE, SQUARE, BOARD_PIXEL_WIDTH, BOARD_PIXEL_HEIGHT, COORD_MARGIN,
    WINDOW_PADDING_X, WINDOW_PADDING_Y, HAND_AREA_HEIGHT, BOARD_START_X, 
    BOARD_START_Y, PIECE_SIZE, PIECE_OFFSET, WIDTH, HEIGHT,
    KIFU_WINDOW_WIDTH, INFO_PANEL_HEIGHT, KIFU_ITEM_HEIGHT, KIFU_LIST_PADDING,
    RESIGN_BUTTON_HEIGHT, SAVE_BUTTON_HEIGHT, MATTA_BUTTON_HEIGHT, TIMER_BUTTON_HEIGHT,
    WHITE, BLACK, GRAY, LIGHT_GRAY, GREEN, RED, BLUE, ORANGE, TATAMI_GREEN,
    DARK_BROWN, BOARD_COLOR, JAPANESE_Y_COORDS, JAPANESE_TURN_NAME,
    image_path, assets_path, initialize_fonts
)

# 型エイリアス
Board = List[List[Optional[Piece]]]


def clone_board(board: Board) -> Board:
    """Board の軽量クローン (Piece を新規生成)"""
    return [[(Piece(p.kind, p.owner) if p else None) for p in col] for col in board]


def standard_setup() -> Board:
    """標準的な初期配置を作成"""
    board: Board = [[None for _ in range(BOARD_SIZE)] for __ in range(BOARD_SIZE)]
    back = ['L', 'N', 'S', 'G', 'K', 'G', 'S', 'N', 'L']
    
    # 後手配置
    for x, k in enumerate(back):
        board[x][0] = Piece(k, 1)
    board[1][1], board[7][1] = Piece('R', 1), Piece('B', 1)
    for x in range(BOARD_SIZE):
        board[x][2] = Piece('P', 1)
    
    # 先手配置
    for x in range(BOARD_SIZE):
        board[x][6] = Piece('P', 0)
    board[1][7], board[7][7] = Piece('B', 0), Piece('R', 0)
    for x, k in enumerate(back):
        board[x][8] = Piece(k, 0)
    
    return board


def handicap_setup(remove_kinds: List[str]) -> List[Tuple[int, int]]:
    """指定された種類(kind)の駒を後手(上手 side=1)から取り除く位置リストを返す。
    remove_kinds: 取り除きたい駒のリスト ['R','B',...] のような形式。
    戻り値: [(x,y), ...]
    """
    positions = []
    temp = standard_setup()
    # 指定の種類ごとに盤を走査し最初に見つかった後手駒を除去対象とする
    for kind in remove_kinds:
        found = False
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                p = temp[x][y]
                if p and p.owner == 1 and p.kind == kind:
                    positions.append((x, y))
                    temp[x][y] = None
                    found = True
                    break
            if found:
                break
    return positions


HANDICAPS: Dict[str, Callable[[], List[Tuple[int, int]]]] = {
    '平手': lambda: [],
    '香落ち': lambda: handicap_setup(['L']),
    '角落ち': lambda: handicap_setup(['B']),
    '飛車落ち': lambda: handicap_setup(['R']),
    '飛香落ち': lambda: handicap_setup(['R', 'L']),
    '二枚落ち': lambda: handicap_setup(['R', 'B']),
}

# Piece-square tables (owner=0 perspective)
PST_TABLES = {
    'P': [0, 1, 2, 3, 3, 2, 1, 0, 0],
    'S': [0, 1, 2, 2, 2, 2, 1, 0, 0],
    'G': [0, 1, 2, 3, 3, 2, 1, 0, 0],
    'L': [0, 1, 2, 3, 4, 3, 2, 1, 0],
    'N': [0, 0, 1, 2, 3, 2, 1, 0, 0],
    'B': [0, 1, 1, 2, 2, 2, 1, 1, 0],
    '+B': [0, 1, 1, 2, 2, 2, 1, 1, 0],
    'R': [0, 0, 1, 2, 2, 2, 1, 0, 0],
    '+R': [0, 0, 1, 2, 2, 2, 1, 0, 0],
    '+P': [0, 1, 2, 3, 3, 2, 1, 0, 0],
    '+S': [0, 1, 2, 3, 3, 2, 1, 0, 0],
    '+L': [0, 1, 2, 3, 3, 2, 1, 0, 0],
    '+N': [0, 1, 2, 3, 3, 2, 1, 0, 0],
    'K': [2, 3, 2, 1, 0, 1, 2, 3, 2],
}


def pst_value(kind: str, y_from_owner0: int) -> int:
    """Piece-square table から位置価値を取得"""
    tbl = PST_TABLES.get(kind)
    if tbl is None:
        return 0
    if y_from_owner0 < 0:
        y_from_owner0 = 0
    elif y_from_owner0 > 8:
        y_from_owner0 = 8
    return tbl[y_from_owner0]


def load_piece_images() -> Dict[Tuple[str, int], pygame.Surface]:
    """駒画像を読み込む"""
    pieces = {}
    names = ["K", "G", "S", "N", "L", "P", "B", "R", "S+", "N+", "L+", "P+", "B+", "R+"]
    for n in names:
        try:
            img_sente = pygame.image.load(os.path.join(image_path, f"{n}_S.png"))
            img_sente = pygame.transform.smoothscale(img_sente, (PIECE_SIZE, PIECE_SIZE))
            img_gote = pygame.image.load(os.path.join(image_path, f"{n}_G.png"))
            img_gote = pygame.transform.smoothscale(img_gote, (PIECE_SIZE, PIECE_SIZE))
            pieces[(n, 0)], pieces[(n, 1)] = img_sente, img_gote
        except pygame.error as e:
            print(f"駒画像ファイルの読み込みに失敗しました: {n} - {e}")
    return pieces


def draw_board_programmatically(screen):
    """盤面をプログラムで描画"""
    from . import utils  # 実行時にフォントを取得
    
    margin_rect = pygame.Rect(WINDOW_PADDING_X, BOARD_START_Y - COORD_MARGIN,
                              BOARD_PIXEL_WIDTH + COORD_MARGIN*2, 
                              BOARD_PIXEL_HEIGHT + COORD_MARGIN*2)
    pygame.draw.rect(screen, BOARD_COLOR, margin_rect)
    
    board_outer_rect = pygame.Rect(BOARD_START_X, BOARD_START_Y, 
                                   BOARD_PIXEL_WIDTH, BOARD_PIXEL_HEIGHT)
    pygame.draw.rect(screen, BOARD_COLOR, board_outer_rect)
    
    # 盤面の線を描画
    for i in range(BOARD_SIZE + 1):
        line_width = 2 if i in [0, BOARD_SIZE] else 1
        # 縦線
        pygame.draw.line(screen, BLACK, 
                        (BOARD_START_X+i*SQUARE, BOARD_START_Y), 
                        (BOARD_START_X+i*SQUARE, BOARD_START_Y+BOARD_PIXEL_HEIGHT), 
                        line_width)
        # 横線
        pygame.draw.line(screen, BLACK, 
                        (BOARD_START_X, BOARD_START_Y+i*SQUARE), 
                        (BOARD_START_X+BOARD_PIXEL_WIDTH, BOARD_START_Y+i*SQUARE), 
                        line_width)
    
    # 座標表示
    for i in range(BOARD_SIZE):
        # 上下の数字
        num_text_top = utils.LARGE_FONT.render(str(9-i), True, BLACK)
        screen.blit(num_text_top, 
                   (BOARD_START_X+i*SQUARE+(SQUARE-num_text_top.get_width())//2, 
                    BOARD_START_Y - COORD_MARGIN))
        
        num_text_bottom = utils.LARGE_FONT.render(str(9-i), True, BLACK)
        screen.blit(num_text_bottom, 
                   (BOARD_START_X+i*SQUARE+(SQUARE-num_text_bottom.get_width())//2, 
                    BOARD_START_Y+BOARD_PIXEL_HEIGHT+5))
        
        # 左右の漢字
        kanji_text_right = utils.LARGE_FONT.render(JAPANESE_Y_COORDS[i], True, BLACK)
        screen.blit(kanji_text_right, 
                   (BOARD_START_X+BOARD_PIXEL_WIDTH+5, 
                    BOARD_START_Y+i*SQUARE+(SQUARE-kanji_text_right.get_height())//2))
        
        kanji_text_left = utils.LARGE_FONT.render(JAPANESE_Y_COORDS[i], True, BLACK)
        screen.blit(kanji_text_left, 
                   (BOARD_START_X-COORD_MARGIN+5, 
                    BOARD_START_Y+i*SQUARE+(SQUARE-kanji_text_left.get_height())//2))


def draw_kifu_info(screen, state, kifu_area_x: int, kifu_area_y: int) -> None:
    """棋譜パネルの情報部分を描画"""
    from . import utils  # 実行時にフォントを取得
    
    # タイマー表示を更新
    _maybe_update_timer_display_cache(state)
    sente_time_str = state.timer_display_cache['sente']
    gote_time_str = state.timer_display_cache['gote']
    
    info_texts = [
        f"手数: {len(state.kifu)}", 
        f"手番: {JAPANESE_TURN_NAME[state.turn]}",
        f"▲先手: {sente_time_str}", 
        f"△後手: {gote_time_str}"
    ]
    
    for i, text in enumerate(info_texts):
        screen.blit(utils.FONT.render(text, True, BLACK), 
                   (kifu_area_x + 10, kifu_area_y + 5 + i*22))


def draw_kifu_list(screen, state, kifu_area_x: int, kifu_area_y: int) -> None:
    """棋譜リストを描画"""
    from . import utils  # 実行時にフォントを取得
    
    kifu_list_y = kifu_area_y + INFO_PANEL_HEIGHT + KIFU_LIST_PADDING
    kifu_list_height = (HEIGHT - kifu_list_y - WINDOW_PADDING_Y - RESIGN_BUTTON_HEIGHT - 
                       SAVE_BUTTON_HEIGHT - MATTA_BUTTON_HEIGHT - TIMER_BUTTON_HEIGHT - 40)
    max_lines = kifu_list_height // KIFU_ITEM_HEIGHT if KIFU_ITEM_HEIGHT > 0 else 0
    start_index = state.kifu_scroll_offset
    
    for i, move in enumerate(state.kifu[start_index:start_index+max_lines]):
        screen.blit(utils.MONO_FONT.render(f"{start_index+i+1}. {move}", True, BLACK), 
                   (kifu_area_x+15, kifu_list_y+i*KIFU_ITEM_HEIGHT))
    
    # スクロールバー
    if len(state.kifu) > max_lines:
        bar_h = kifu_list_height * (max_lines/len(state.kifu)) if len(state.kifu) > 0 else kifu_list_height
        ratio = start_index / (len(state.kifu)-max_lines) if (len(state.kifu)-max_lines) > 0 else 0
        bar_y = kifu_list_y + (kifu_list_height-bar_h)*ratio
        state.scrollbar_rect = pygame.Rect(kifu_area_x+KIFU_WINDOW_WIDTH-15, bar_y, 10, bar_h)
        pygame.draw.rect(screen, LIGHT_GRAY, state.scrollbar_rect)
    else:
        state.scrollbar_rect = None


def draw_kifu_buttons(screen, state, kifu_area_x: int) -> None:
    """棋譜パネルのボタンを描画"""
    from . import utils  # 実行時にフォントを取得
    
    resign_button_y = HEIGHT-WINDOW_PADDING_Y-RESIGN_BUTTON_HEIGHT-10
    button_x = kifu_area_x+(KIFU_WINDOW_WIDTH-260)/2
    
    # 投了ボタン
    state.resign_button_rect = pygame.Rect(button_x, resign_button_y, 260, RESIGN_BUTTON_HEIGHT)
    pygame.draw.rect(screen, RED, state.resign_button_rect, 0, 10)
    resign_surf = utils.LARGE_FONT.render("投了", True, WHITE)
    screen.blit(resign_surf, resign_surf.get_rect(center=state.resign_button_rect.center))

    # 保存ボタン
    save_button_y = resign_button_y - SAVE_BUTTON_HEIGHT - 10
    state.save_button_rect = pygame.Rect(button_x, save_button_y, 260, SAVE_BUTTON_HEIGHT)
    pygame.draw.rect(screen, (28, 93, 158), state.save_button_rect, 0, 10)
    save_surf = utils.LARGE_FONT.render("棋譜を保存", True, WHITE)
    screen.blit(save_surf, save_surf.get_rect(center=state.save_button_rect.center))

    # 待ったボタン
    matta_button_y = save_button_y - MATTA_BUTTON_HEIGHT - 10
    state.matta_button_rect = pygame.Rect(button_x, matta_button_y, 260, MATTA_BUTTON_HEIGHT)
    color = ORANGE if len(state.history) > 0 else LIGHT_GRAY
    pygame.draw.rect(screen, color, state.matta_button_rect, 0, 10)
    matta_surf = utils.LARGE_FONT.render("待った", True, WHITE)
    screen.blit(matta_surf, matta_surf.get_rect(center=state.matta_button_rect.center))

    # タイマーボタン
    if state.time_limit:
        timer_button_y = matta_button_y - TIMER_BUTTON_HEIGHT - 10
        state.timer_button_rect = pygame.Rect(button_x, timer_button_y, 260, TIMER_BUTTON_HEIGHT)
        text = "タイマー再開" if state.timer_paused else "タイマーストップ"
        pygame.draw.rect(screen, GREEN, state.timer_button_rect, 0, 10)
        screen.blit(utils.LARGE_FONT.render(text, True, WHITE), 
                   utils.LARGE_FONT.render(text, True, WHITE).get_rect(center=state.timer_button_rect.center))


def draw_kifu(screen, state) -> None:
    """棋譜全体を描画"""
    kifu_area_x = BOARD_START_X + BOARD_PIXEL_WIDTH + COORD_MARGIN + WINDOW_PADDING_X
    kifu_area_y = WINDOW_PADDING_Y
    kifu_area_rect = pygame.Rect(kifu_area_x, kifu_area_y, KIFU_WINDOW_WIDTH, 
                                HEIGHT - WINDOW_PADDING_Y*2)
    pygame.draw.rect(screen, WHITE, kifu_area_rect)
    pygame.draw.rect(screen, BLACK, kifu_area_rect, 2)
    
    info_panel_rect = pygame.Rect(kifu_area_x, kifu_area_y, KIFU_WINDOW_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, GRAY, info_panel_rect)
    pygame.draw.rect(screen, BLACK, info_panel_rect, 2)

    # delegate sub-sections
    draw_kifu_info(screen, state, kifu_area_x, kifu_area_y)
    draw_kifu_list(screen, state, kifu_area_x, kifu_area_y)
    draw_kifu_buttons(screen, state, kifu_area_x)


def _maybe_update_timer_display_cache(state) -> None:
    """タイマー表示キャッシュを更新"""
    import time
    now = time.time()
    # 0.25 秒間隔で更新
    if now - state.timer_display_cache['last_update'] < 0.25:
        return
    
    sente_t, gote_t = state.sente_time, state.gote_time
    if not state.timer_paused and not state.game_over:
        elapsed = now - state.last_move_time
        if state.time_limit is not None:
            if state.turn == 0:
                sente_t = max(0, state.sente_time - elapsed)
            else:
                gote_t = max(0, state.gote_time - elapsed)
        else:
            if state.turn == 0:
                sente_t += elapsed
            else:
                gote_t += elapsed
    
    state.timer_display_cache['sente'] = time.strftime('%H:%M:%S', time.gmtime(sente_t))
    state.timer_display_cache['gote'] = time.strftime('%H:%M:%S', time.gmtime(gote_t))
    state.timer_display_cache['last_update'] = now
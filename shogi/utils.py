"""
Utility functions, constants, and UI helpers for Shogi game.

This module provides:
- Display constants and color definitions
- Font loading functionality
- Path setup for assets
- UI dialog functions
- Coordinate conversion utilities
- Common helper functions
"""

import pygame
import sys
import os
import time
from typing import Tuple, Dict, Optional, List

# --- パス設定 ---
# スクリプト自身の場所を基準にする
if hasattr(sys, "_MEIPASS"):
    # PyInstaller 展開ディレクトリ (動的属性取得) 安全に文字列化
    from typing import cast
    base_path = cast(str, getattr(sys, "_MEIPASS"))
else:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 画像と音声フォルダへのパスを作成
assets_path = os.path.join(base_path, "assets")
image_path = os.path.join(assets_path, "images")
sound_path = os.path.join(assets_path, "sound")
# 棋譜ファイルはスクリプト（またはパッケージ展開先）の直下に置く
kifu_path = os.path.join(base_path, "shogi_kifu.csv")

# 盤面とウィンドウの基本設定
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

WIDTH = (BOARD_PIXEL_WIDTH + WINDOW_PADDING_X * 2 + KIFU_WINDOW_WIDTH + 
         WINDOW_PADDING_X + COORD_MARGIN * 2)
HEIGHT = (WINDOW_PADDING_Y + HAND_AREA_HEIGHT + BOARD_PIXEL_HEIGHT + 
          HAND_AREA_HEIGHT + WINDOW_PADDING_Y + COORD_MARGIN * 2)

BOARD_START_X = WINDOW_PADDING_X + COORD_MARGIN
BOARD_START_Y = WINDOW_PADDING_Y + HAND_AREA_HEIGHT + COORD_MARGIN

PIECE_SIZE = 60
PIECE_OFFSET = (SQUARE - PIECE_SIZE) // 2
FPS = 60

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

# 座標系と表示関連
JAPANESE_Y_COORDS = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
Y_COORD_FROM_JP = {v: i for i, v in enumerate(JAPANESE_Y_COORDS)}
ZENKAKU_NUM, HANKAKU_NUM = "１２３４５６７８９", "123456789"
ZEN_TO_HAN_TABLE = str.maketrans(ZENKAKU_NUM, HANKAKU_NUM)
JAPANESE_TURN_SYMBOL = {0: '▲', 1: '△'}
JAPANESE_TURN_NAME = {0: '先手', 1: '後手'}
VICTORY_MESSAGE = {0: '学生軍の勝利', 1: '教員軍の勝利'}

# 駒の日本語名
JAPANESE_PIECE_NAMES = {
    'K': '玉', 'R': '飛', 'B': '角', 'G': '金', 'S': '銀', 'N': '桂', 'L': '香', 'P': '歩',
    'R+': '龍', 'B+': '馬', 'S+': '成銀', 'N+': '成桂', 'L+': '成香', 'P+': 'と',
}

# 日本語名から駒種への逆マップ
KIND_FROM_JP = {v: k for k, v in JAPANESE_PIECE_NAMES.items()}

# 時間設定
TIME_SETTINGS: Dict[str, Optional[int]] = {
    "なし": None, "15分": 15*60, "30分": 30*60, "1時間": 60*60, "設定": -1
}
SUDDEN_DEATH_TIME = 45


def _load_fonts() -> Tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font, 
                          pygame.font.Font, pygame.font.Font, pygame.font.Font, 
                          pygame.font.Font]:
    """フォントを読み込む。失敗時はシステムフォントにフォールバック"""
    try:
        font_path = os.path.join(assets_path, "YujiSyuku-Regular.ttf")
        font = pygame.font.Font(font_path, 18)
        large = pygame.font.Font(font_path, 24)
        mono = pygame.font.Font(font_path, 14)
        title = pygame.font.Font(font_path, 80)
        check = pygame.font.Font(font_path, 100)
        greeting = pygame.font.Font(font_path, 70)
        result = pygame.font.Font(font_path, 90)
        check.set_bold(True)
        greeting.set_bold(True)
        result.set_bold(True)
        return font, large, mono, title, check, greeting, result
    except Exception:
        # System fallback
        font = pygame.font.SysFont("MS Mincho", 18)
        large = pygame.font.SysFont("MS Mincho", 24)
        mono = pygame.font.SysFont("MS Mincho", 14)
        title = pygame.font.SysFont("MS Mincho", 80)
        check = pygame.font.SysFont("MS Mincho", 100, bold=True)
        greeting = pygame.font.SysFont("MS Mincho", 70, bold=True)
        result = pygame.font.SysFont("MS Mincho", 90, bold=True)
        return font, large, mono, title, check, greeting, result


# グローバルフォントオブジェクト（遅延初期化）
FONT = LARGE_FONT = MONO_FONT = TITLE_FONT = CHECK_FONT = GREETING_FONT = RESULT_FONT = None


def initialize_fonts():
    """フォントを初期化する（pygame.init()後に呼び出す）"""
    global FONT, LARGE_FONT, MONO_FONT, TITLE_FONT, CHECK_FONT, GREETING_FONT, RESULT_FONT
    if FONT is None:  # まだ初期化されていない場合のみ
        FONT, LARGE_FONT, MONO_FONT, TITLE_FONT, CHECK_FONT, GREETING_FONT, RESULT_FONT = _load_fonts()


def coords_to_kifu(x: int, y: int) -> str:
    """盤上座標を棋譜記法に変換"""
    return f"{9-x}{JAPANESE_Y_COORDS[y]}"


def in_bounds(x: int, y: int) -> bool:
    """座標が盤面内かチェック"""
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def ask_promotion(screen) -> bool:
    """成りの選択ダイアログを表示"""
    dialog = pygame.Surface((300, 150))
    dialog.fill(GRAY)
    dialog_rect = dialog.get_rect(center=(WIDTH//2, HEIGHT//2))
    pygame.draw.rect(dialog, BLACK, dialog.get_rect(), 3)
    
    dialog.blit(LARGE_FONT.render("成りますか?", True, BLACK), 
                LARGE_FONT.render("成りますか?", True, BLACK).get_rect(center=(dialog.get_width()//2, 40)))
    
    yes_rect, no_rect = pygame.Rect(50, 80, 80, 40), pygame.Rect(170, 80, 80, 40)
    pygame.draw.rect(dialog, GREEN, yes_rect)
    pygame.draw.rect(dialog, RED, no_rect)
    dialog.blit(LARGE_FONT.render("はい", True, BLACK), (65, 85))
    dialog.blit(LARGE_FONT.render("いいえ", True, BLACK), (175, 85))
    
    screen.blit(dialog, dialog_rect.topleft)
    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_rect.move(dialog_rect.topleft).collidepoint(event.pos):
                    return True
                elif no_rect.move(dialog_rect.topleft).collidepoint(event.pos):
                    return False


def ask_rematch(screen, message="再対局しますか？") -> bool:
    """再戦確認ダイアログを表示"""
    dialog = pygame.Surface((400, 200))
    dialog.fill(GRAY)
    dialog_rect = dialog.get_rect(center=(WIDTH//2, HEIGHT//2))
    pygame.draw.rect(dialog, BLACK, dialog.get_rect(), 3)
    
    dialog.blit(LARGE_FONT.render(message, True, BLACK), 
                LARGE_FONT.render(message, True, BLACK).get_rect(center=(dialog.get_width()//2, 60)))
    
    yes_rect_text, no_rect_text = ("はい", "いいえ") if message == "再対局しますか？" else ("続ける", "終了")
    yes_rect, no_rect = pygame.Rect(50, 120, 120, 50), pygame.Rect(230, 120, 120, 50)
    pygame.draw.rect(dialog, GREEN, yes_rect)
    pygame.draw.rect(dialog, RED, no_rect)
    
    dialog.blit(LARGE_FONT.render(yes_rect_text, True, BLACK),
                LARGE_FONT.render(yes_rect_text, True, BLACK).get_rect(center=yes_rect.center))
    dialog.blit(LARGE_FONT.render(no_rect_text, True, BLACK),
                LARGE_FONT.render(no_rect_text, True, BLACK).get_rect(center=no_rect.center))
    
    screen.blit(dialog, dialog_rect.topleft)
    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_rect.move(dialog_rect.topleft).collidepoint(event.pos):
                    return True
                elif no_rect.move(dialog_rect.topleft).collidepoint(event.pos):
                    return False


def show_greeting(screen, text: str, animation_type: str) -> None:
    """アニメーション付きメッセージ表示"""
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
            pygame.display.flip()
            clock.tick(FPS)

    elif animation_type == 'fade-in':
        for alpha in range(0, 256, 15):
            screen.fill(TATAMI_GREEN)
            text_surf.set_alpha(alpha)
            shadow_surf.set_alpha(alpha)
            rect = text_surf.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(shadow_surf, (rect.x + 4, rect.y + 4))
            screen.blit(text_surf, rect)
            pygame.display.flip()
            clock.tick(FPS)

    pygame.time.wait(duration)

    for alpha in range(255, -1, -15):
        screen.fill(TATAMI_GREEN)
        text_surf.set_alpha(alpha)
        shadow_surf.set_alpha(alpha)
        rect = text_surf.get_rect(center=(WIDTH//2, HEIGHT//2))
        screen.blit(shadow_surf, (rect.x + 4, rect.y + 4))
        screen.blit(text_surf, rect)
        pygame.display.flip()
        clock.tick(FPS)


def selection_screen(screen, title: str, items: List[str]) -> str:
    """選択画面を表示"""
    screen.fill(TATAMI_GREEN)
    title_surf = LARGE_FONT.render(title, True, BLACK)
    screen.blit(title_surf, (WIDTH//2-title_surf.get_width()//2, 50))
    
    buttons = []
    for i, name in enumerate(items):
        rect = pygame.Rect(WIDTH//2-100, 120+i*(50+10), 200, 50)
        buttons.append((rect, name))
        pygame.draw.rect(screen, BUTTON_BLUE, rect, 0, 10)
        text_surf = FONT.render(name, True, WHITE)
        screen.blit(text_surf, text_surf.get_rect(center=rect.center))
    
    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, name in buttons:
                    if rect.collidepoint(event.pos):
                        return name


def ask_continue_screen(screen) -> str:
    """棋譜継続確認画面"""
    screen.fill(TATAMI_GREEN)
    title_surf = LARGE_FONT.render("前回の棋譜の続きからしますか？", True, BLACK)
    screen.blit(title_surf, (WIDTH//2-title_surf.get_width()//2, 100))
    
    kifu_exists = os.path.exists(kifu_path)
    buttons = {
        'new': {'rect': pygame.Rect(WIDTH//2-120, 200, 240, 60), 'text': '初めから'},
        'continue': {'rect': pygame.Rect(WIDTH//2-120, 280, 240, 60), 'text': '続きから'}
    }
    
    clock = pygame.time.Clock()
    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if buttons['new']['rect'].collidepoint(mouse_pos):
                    return 'new'
                if kifu_exists and buttons['continue']['rect'].collidepoint(mouse_pos):
                    return 'continue'
        
        color = BUTTON_BLUE_HOVER if buttons['new']['rect'].collidepoint(mouse_pos) else BUTTON_BLUE
        pygame.draw.rect(screen, color, buttons['new']['rect'], 0, 10)
        text = LARGE_FONT.render(buttons['new']['text'], True, WHITE)
        screen.blit(text, text.get_rect(center=buttons['new']['rect'].center))
        
        color = (150, 150, 150)
        text_color = LIGHT_GRAY
        if kifu_exists:
            color = BUTTON_BLUE_HOVER if buttons['continue']['rect'].collidepoint(mouse_pos) else BUTTON_BLUE
            text_color = WHITE
        pygame.draw.rect(screen, color, buttons['continue']['rect'], 0, 10)
        text = LARGE_FONT.render(buttons['continue']['text'], True, text_color)
        screen.blit(text, text.get_rect(center=buttons['continue']['rect'].center))
        
        pygame.display.flip()
        clock.tick(FPS)
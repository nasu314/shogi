"""
Main entry point and game loop for Shogi application.

This module provides:
- Main game entry point
- Start screen and selection screens
- Game drawing and rendering
- Event handling and user interaction
- Sound and image loading
- Complete game loop integration
"""

import pygame
import sys
import os
import time
import random
from typing import Dict, Any, Optional, List, Tuple

from .piece import AnimatedPiece
from .board import (
    draw_board_programmatically, draw_kifu, load_piece_images, 
    Board, HANDICAPS
)
from .rules import generate_all_moves, is_in_check, Move
from .game import (
    GameState, CPU_DIFFICULTIES, get_cpu_move_beginner, get_cpu_move_easy,
    get_cpu_move_medium, get_cpu_move_hard, get_cpu_move_master,
    apply_move, save_kifu_to_csv, load_kifu_and_setup_state
)
from .utils import (
    WIDTH, HEIGHT, FPS, BOARD_START_X, BOARD_START_Y, BOARD_PIXEL_WIDTH,
    BOARD_PIXEL_HEIGHT, SQUARE, PIECE_OFFSET, PIECE_SIZE, HAND_AREA_HEIGHT,
    COORD_MARGIN, WINDOW_PADDING_Y, TATAMI_GREEN, DARK_BROWN, BLUE, GREEN,
    RED, WHITE, BLACK, BUTTON_BLUE, BUTTON_BLUE_HOVER, TITLE_COLOR,
    DARK_BROWN, BOARD_SIZE, SUDDEN_DEATH_TIME, TIME_SETTINGS,
    sound_path, image_path, initialize_fonts, ask_rematch, show_greeting,
    selection_screen, ask_continue_screen,
    FONT, LARGE_FONT, TITLE_FONT, CHECK_FONT, RESULT_FONT
)

# サウンドファイルの読み込み
sound_start = sound_end = sound_itte = sound_se1 = sound_think = None
try:
    sound_start = pygame.mixer.Sound(os.path.join(sound_path, "start.mp3"))
    sound_end = pygame.mixer.Sound(os.path.join(sound_path, "end.mp3"))
    sound_itte = pygame.mixer.Sound(os.path.join(sound_path, "itte.mp3"))
    sound_se1 = pygame.mixer.Sound(os.path.join(sound_path, "se1.mp3"))
    sound_think = pygame.mixer.Sound(os.path.join(sound_path, "think.mp3"))
except pygame.error as e:
    print(f"サウンドファイルの読み込みに失敗しました: {e}")

# 画像の読み込み
fusuma_left_img = fusuma_right_img = start_bg_img = None
try:
    fusuma_left_img = pygame.image.load(os.path.join(image_path, "fusuma_left.png"))
    fusuma_right_img = pygame.image.load(os.path.join(image_path, "fusuma_right.png"))
    start_bg_img = pygame.image.load(os.path.join(image_path, "board.png"))
except pygame.error as e:
    print(f"画像ファイルの読み込みに失敗しました: {e}")


def draw_game_elements(screen, state: GameState, piece_images: Dict[Tuple[str, int], pygame.Surface]):
    """ゲーム要素を描画"""
    screen.fill(TATAMI_GREEN)
    hand_y_gote = BOARD_START_Y - COORD_MARGIN - HAND_AREA_HEIGHT
    hand_y_sente = BOARD_START_Y + BOARD_PIXEL_HEIGHT + COORD_MARGIN
    hand_x = BOARD_START_X
    
    # 持ち駒背景を描画
    pygame.draw.rect(screen, DARK_BROWN, (hand_x, hand_y_gote, BOARD_PIXEL_WIDTH, HAND_AREA_HEIGHT))
    pygame.draw.rect(screen, DARK_BROWN, (hand_x, hand_y_sente, BOARD_PIXEL_WIDTH, HAND_AREA_HEIGHT))
    draw_board_programmatically(screen)

    if state.resigning_animation:
        # 投了アニメーション
        remaining = []
        for ap in state.animated_pieces:
            alive = ap.update(HEIGHT, PIECE_SIZE)
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
        # 選択マスと合法手のハイライト
        if state.selected:
            pygame.draw.rect(screen, BLUE, 
                           (BOARD_START_X+state.selected[0]*SQUARE, 
                            BOARD_START_Y+state.selected[1]*SQUARE, SQUARE, SQUARE), 3)
        
        for move in state.legal_moves:
            if len(move) >= 4:
                _, _, tx, ty = move
                color = RED if state.board[tx][ty] else GREEN
                pygame.draw.rect(screen, color, 
                               (BOARD_START_X+tx*SQUARE, BOARD_START_Y+ty*SQUARE, SQUARE, SQUARE), 3)
        
        # 盤上の駒を描画
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                p = state.board[x][y]
                if p:
                    img = piece_images.get((p.kind, p.owner))
                    if img:
                        screen.blit(img, (BOARD_START_X+x*SQUARE+PIECE_OFFSET, 
                                        BOARD_START_Y+y*SQUARE+PIECE_OFFSET))
        
        # 持ち駒と手駒スクロールバーを描画
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
                pygame.draw.rect(screen, (220, 220, 200), rect)
                state.hand_scrollbar_rect[owner] = rect
            else:
                state.hand_scrollbar_rect[owner] = None

        if state.selected_hand is not None:
            owner, offset = state.turn, state.hand_scroll_offset[state.turn]
            display_idx = state.selected_hand - offset
            if 0 <= display_idx < max_visible_hand:
                hand_y = hand_y_sente if owner == 0 else hand_y_gote
                pygame.draw.rect(screen, BLUE, (hand_x+display_idx*SQUARE, hand_y, SQUARE, SQUARE), 3)

    draw_kifu(screen, state)

    # 王手表示
    if state.check_display_time > 0 and time.time() - state.check_display_time < 1.5:
        check_text = CHECK_FONT.render("王手", True, RED)
        text_rect = check_text.get_rect(center=(BOARD_START_X + BOARD_PIXEL_WIDTH//2, 
                                               BOARD_START_Y + BOARD_PIXEL_HEIGHT//2))
        shadow_text = CHECK_FONT.render("王手", True, BLACK)
        screen.blit(shadow_text, (text_rect.x + 3, text_rect.y + 3))
        screen.blit(check_text, text_rect)
    else:
        state.check_display_time = 0

    # 保存メッセージ
    if state.saved_message_time and time.time() - state.saved_message_time < 2:
        saved_text = LARGE_FONT.render("棋譜を保存しました！", True, BUTTON_BLUE)
        screen.blit(saved_text, saved_text.get_rect(center=(WIDTH//2, HEIGHT//2-200)))

    pygame.display.flip()


def start_screen(screen):
    """スタート画面"""
    clock = pygame.time.Clock()
    anim_speed = 12
    buttons = [
        {'rect': pygame.Rect(WIDTH//2-125, HEIGHT//2-50, 250, 60), 'text': "二人で対局", 'action': '2P'},
        {'rect': pygame.Rect(WIDTH//2-125, HEIGHT//2+40, 250, 60), 'text': "一人で対局", 'action': 'CPU'},
        {'rect': pygame.Rect(WIDTH//2-125, HEIGHT//2+130, 250, 60), 'text': "終了", 'action': 'QUIT'}
    ]
    
    bg = pygame.Surface((WIDTH, HEIGHT))
    if start_bg_img:
        scaled = pygame.transform.scale(start_bg_img, (WIDTH, HEIGHT))
        scaled.set_alpha(80)
        bg.blit(scaled, (0, 0))
    else:
        bg.fill(TATAMI_GREEN)

    scaled_fusuma_l = scaled_fusuma_r = None
    if fusuma_left_img and fusuma_right_img:
        scaled_fusuma_l = pygame.transform.scale(fusuma_left_img, (WIDTH // 2 + 5, HEIGHT))
        scaled_fusuma_r = pygame.transform.scale(fusuma_right_img, (WIDTH // 2 + 5, HEIGHT))

    fusuma_x_l, fusuma_x_r = 0, WIDTH // 2
    animation_started = False
    action_to_return = None

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and not animation_started:
                for btn in buttons:
                    if btn['rect'].collidepoint(mouse_pos):
                        if sound_se1:
                            sound_se1.play()
                        if btn['action'] == 'QUIT':
                            pygame.quit()
                            sys.exit()
                        animation_started = True
                        action_to_return = btn['action']

        screen.blit(bg, (0, 0))

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
            pygame.draw.rect(screen, color, btn['rect'], 0, 10)
            text_surf = LARGE_FONT.render(btn['text'], True, WHITE)
            screen.blit(text_surf, text_surf.get_rect(center=btn['rect'].center))

        if animation_started:
            fusuma_x_l -= anim_speed
            fusuma_x_r += anim_speed
            if fusuma_x_l < -WIDTH // 2:
                return action_to_return

        pygame.display.flip()
        clock.tick(FPS)


def game_over_screen(screen, state: GameState) -> str:
    """ゲーム終了画面"""
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
        
        from .utils import JAPANESE_TURN_NAME
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


def _handle_cpu_turn(screen, state: GameState, piece_images: Dict):
    """CPU の手番処理"""
    if state.mode == 'CPU' and state.turn == 1 and not state.game_over:
        if not state.cpu_thinking:
            if sound_think:
                sound_think.play(-1)
            state.cpu_thinking = True
        
        draw_game_elements(screen, state, piece_images)
        pygame.time.wait(100)
        
        cpu_move_func = {
            'beginner': get_cpu_move_beginner, 
            'easy': get_cpu_move_easy, 
            'medium': get_cpu_move_medium,
            'hard': get_cpu_move_hard, 
            'master': get_cpu_move_master
        }[state.cpu_difficulty]
        
        cpu_move = cpu_move_func(state)
        if sound_think:
            sound_think.stop()
        
        if cpu_move:
            apply_move(state, cpu_move, is_cpu=True, screen=screen)


def _handle_events(screen, state: GameState, mx: int, my: int, drag_kifu: bool, 
                  drag_hand: Dict[int, bool]):
    """イベント処理"""
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            return running, drag_kifu, drag_hand
        
        if state.game_over:
            continue

        if event.type == pygame.MOUSEBUTTONDOWN:
            # ボタンクリック処理
            if getattr(state, 'resign_button_rect', None) and state.resign_button_rect.collidepoint(mx, my):
                state.game_over = True
                state.winner = 1 - state.turn
                state.last_move_time = time.time()
                state.resigning_animation = True
                # 駒落下アニメーション
                for x in range(BOARD_SIZE):
                    for y in range(BOARD_SIZE):
                        p = state.board[x][y]
                        if p:
                            state.animated_pieces.append(AnimatedPiece(
                                p, BOARD_START_X + x * SQUARE + SQUARE // 2,
                                BOARD_START_Y + y * SQUARE + SQUARE // 2, p.owner))
                if sound_end:
                    sound_end.play()
            
            elif getattr(state, 'save_button_rect', None) and state.save_button_rect.collidepoint(mx, my):
                if save_kifu_to_csv(state.kifu):
                    state.saved_message_time = time.time()
            
            elif getattr(state, 'matta_button_rect', None) and state.matta_button_rect.collidepoint(mx, my):
                if len(state.history) > 0:
                    steps = 2 if state.mode == 'CPU' and len(state.history) > 1 and state.turn == 0 else 1
                    last_state = None
                    for _ in range(steps):
                        if state.history:
                            last_state = state.history.pop()
                    if last_state is not None:
                        state.load_history(last_state)
            
            elif getattr(state, 'timer_button_rect', None) and state.timer_button_rect.collidepoint(mx, my):
                state.timer_paused = not state.timer_paused
                if not state.timer_paused:
                    state.last_move_time = time.time()

            elif getattr(state, 'scrollbar_rect', None) and state.scrollbar_rect.collidepoint(mx, my):
                drag_kifu = True
                state.scroll_y_start, state.scroll_offset_start = my, state.kifu_scroll_offset
            
            elif any(r and r.collidepoint(mx, my) for r in state.hand_scrollbar_rect.values()):
                owner = 0 if state.hand_scrollbar_rect[0] and state.hand_scrollbar_rect[0].collidepoint(mx, my) else 1
                drag_hand[owner] = True
                state.scroll_x_start, state.scroll_offset_start = mx, state.hand_scroll_offset[owner]

            elif not state.timer_paused:
                # 盤面クリック処理
                gx, gy = (mx - BOARD_START_X) // SQUARE, (my - BOARD_START_Y) // SQUARE
                hand_y = BOARD_START_Y + BOARD_PIXEL_HEIGHT + COORD_MARGIN if state.turn == 0 else BOARD_START_Y - COORD_MARGIN - HAND_AREA_HEIGHT
                
                if hand_y <= my < hand_y + HAND_AREA_HEIGHT:
                    # 持ち駒クリック
                    offset = state.hand_scroll_offset[state.turn]
                    idx = (mx - BOARD_START_X) // SQUARE + offset
                    if 0 <= idx < len(state.hands[state.turn]):
                        state.selected, state.selected_hand = None, idx
                        kind = state.hands[state.turn][idx]
                        state.legal_moves = [(None, idx, tx, ty) for tx, ty in generate_all_moves(state.board, None, idx, state.turn, kind=kind)]
                    return running, drag_kifu, drag_hand
                
                if 0 <= gx < BOARD_SIZE and 0 <= gy < BOARD_SIZE:
                    # 盤面クリック
                    move = None
                    for m in state.legal_moves:
                        if len(m) >= 4 and m[2] == gx and m[3] == gy:
                            if ((state.selected and m[0] is not None and state.selected == (m[0], m[1])) or
                                (state.selected_hand is not None and m[0] is None and m[1] == state.selected_hand)):
                                move = m
                                break
                    
                    if move:
                        apply_move(state, move, screen=screen)
                    else:
                        state.selected_hand = None
                        p = state.board[gx][gy]
                        if p and p.owner == state.turn:
                            state.selected = (gx, gy)
                            state.legal_moves = [(gx, gy, tx, ty) for tx, ty in generate_all_moves(state.board, gx, gy, state.turn)]
                        else:
                            state.selected, state.legal_moves = None, []
                else:
                    state.selected, state.selected_hand, state.legal_moves = None, None, []

        elif event.type == pygame.MOUSEBUTTONUP:
            drag_kifu = False
            drag_hand[0] = drag_hand[1] = False

        elif event.type == pygame.MOUSEMOTION:
            # スクロール処理
            if drag_kifu and state.scrollbar_rect:
                from .utils import KIFU_ITEM_HEIGHT, INFO_PANEL_HEIGHT, KIFU_LIST_PADDING, RESIGN_BUTTON_HEIGHT, SAVE_BUTTON_HEIGHT, MATTA_BUTTON_HEIGHT, TIMER_BUTTON_HEIGHT
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
                    max_v = BOARD_PIXEL_WIDTH // SQUARE
                    h_size = len(state.hands[owner])
                    if h_size > max_v:
                        s_w = BOARD_PIXEL_WIDTH - state.hand_scrollbar_rect[owner].width
                        if s_w > 0:
                            ratio = (mx - state.scroll_x_start) / s_w
                            max_off = h_size - max_v
                            new_off = int(state.scroll_offset_start + ratio * max_off)
                            state.hand_scroll_offset[owner] = max(0, min(new_off, max_off))

        elif event.type == pygame.MOUSEWHEEL:
            # マウスホイールでスクロール
            hand_rects = {
                0: pygame.Rect(BOARD_START_X, BOARD_START_Y + BOARD_PIXEL_HEIGHT + COORD_MARGIN, BOARD_PIXEL_WIDTH, HAND_AREA_HEIGHT),
                1: pygame.Rect(BOARD_START_X, BOARD_START_Y - COORD_MARGIN - HAND_AREA_HEIGHT, BOARD_PIXEL_WIDTH, HAND_AREA_HEIGHT)
            }
            owner = next((o for o, r in hand_rects.items() if r.collidepoint(mx, my)), -1)
            if owner != -1:
                max_v = BOARD_PIXEL_WIDTH // SQUARE
                h_size = len(state.hands[owner])
                if h_size > max_v:
                    max_off = h_size - max_v
                    state.hand_scroll_offset[owner] = max(0, min(state.hand_scroll_offset[owner] - event.y, max_off))
            else:
                from .utils import KIFU_ITEM_HEIGHT, INFO_PANEL_HEIGHT, KIFU_LIST_PADDING, RESIGN_BUTTON_HEIGHT, SAVE_BUTTON_HEIGHT, MATTA_BUTTON_HEIGHT, TIMER_BUTTON_HEIGHT
                k_h = HEIGHT - (WINDOW_PADDING_Y * 2 + INFO_PANEL_HEIGHT + KIFU_LIST_PADDING + RESIGN_BUTTON_HEIGHT + SAVE_BUTTON_HEIGHT + MATTA_BUTTON_HEIGHT + TIMER_BUTTON_HEIGHT + 40)
                if k_h > 0:
                    max_l = k_h // KIFU_ITEM_HEIGHT
                    if len(state.kifu) > max_l:
                        state.kifu_scroll_offset = max(0, min(state.kifu_scroll_offset - event.y * 3, len(state.kifu) - max_l))
    
    return running, drag_kifu, drag_hand


def _update_timers(screen, state: GameState):
    """タイマー更新と時間切れ処理"""
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
                    state.game_over = True
                    state.winner = 1 - turn_player


def _handle_game_over(screen, state: GameState, piece_images: Dict):
    """ゲーム終了処理"""
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


def main(initial_settings: Optional[Dict[str, Any]] = None, skip_start: bool = False):
    """メイン関数"""
    pygame.init()
    try:
        pygame.mixer.init()
    except pygame.error:
        print("Audio not available, continuing without sound")
    initialize_fonts()  # フォント初期化
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("将棋道場")
    piece_images = load_piece_images()

    # 初期設定がある場合はそれを使用、そうでなければスタート画面から
    if initial_settings and skip_start:
        game_mode = initial_settings.get('mode', '2P')
        handicap = initial_settings.get('handicap', '平手')
        cpu_diff = initial_settings.get('cpu_diff', 'easy')
        time_limit = initial_settings.get('time_limit', None)
        state = GameState(handicap, game_mode, cpu_diff, time_limit)
    else:
        game_mode = start_screen(screen)
        handicap, cpu_diff, time_limit = '平手', 'easy', None
        
        if game_mode == '2P':
            handicap = selection_screen(screen, "ハンディキャップを選択", list(HANDICAPS.keys()))
            time_choice = selection_screen(screen, "持ち時間モードを選択", list(TIME_SETTINGS.keys()))
            time_limit = TIME_SETTINGS[time_choice]
        else:
            cpu_diff = CPU_DIFFICULTIES[selection_screen(screen, "CPUの強さを選択", list(CPU_DIFFICULTIES.keys()))]

        if ask_continue_screen(screen) == 'continue':
            state = load_kifu_and_setup_state(handicap, game_mode, cpu_diff, time_limit)
        else:
            state = GameState(handicap, game_mode, cpu_diff, time_limit)

    show_greeting(screen, "よろしくお願いします", "split-in")
    state.last_move_time = time.time()
    if sound_start:
        sound_start.play()
    
    drag_kifu = False
    drag_hand = {0: False, 1: False}

    # ゲームループ
    running = True
    while running:
        _handle_cpu_turn(screen, state, piece_images)
        mx, my = pygame.mouse.get_pos()
        running, drag_kifu, drag_hand = _handle_events(screen, state, mx, my, drag_kifu, drag_hand)
        _update_timers(screen, state)
        draw_game_elements(screen, state, piece_images)
        
        if state.game_over:
            action = _handle_game_over(screen, state, piece_images)
            if action:
                settings = {'handicap': handicap, 'mode': game_mode, 'cpu_diff': cpu_diff, 'time_limit': time_limit}
                return action, settings

    # ループが正常終了した場合
    settings = {'handicap': handicap, 'mode': game_mode, 'cpu_diff': cpu_diff, 'time_limit': time_limit}
    return 'QUIT', settings


if __name__ == "__main__":
    # ゲームを実行し、再戦要求まで繰り返す
    prev_settings = None
    while True:
        result = main(initial_settings=prev_settings, skip_start=bool(prev_settings))
        if isinstance(result, tuple):
            action, prev_settings = result
        else:
            action, prev_settings = result, None
        
        if action == 'REMATCH' and prev_settings:
            # 同じ設定で新しいゲームを開始
            continue
        else:
            break
"""
250422_test.py
----------------------------------------------------
손가락 제스처 인식 + Animal Crossing 느낌의 미니 hub
(come / away / spin / wave / heart / sleep)

- 원본 260422_test.py 의 인식 파이프라인은 그대로 유지
- 예측된 제스처에 따라 오른쪽 hub 패널의 캐릭터가 애니메이션
"""

import os
import math
import time
import random

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model


# ========================================================================
#  Config
# ========================================================================
ACTIONS = ['come', 'away', 'spin', 'wave', 'heart', 'sleep']
SEQ_LENGTH = 30
CONF_THRESHOLD = 0.9
MODEL_PATH = os.path.join('models', '250422_model.h5')

HUB_W, HUB_H = 360, 480          # hub 패널 기본 해상도
ANIM_DURATION = 1.6              # 제스처 1회 당 애니메이션 시간 (초)
RETRIGGER_COOLDOWN = 0.8         # 같은 제스처 재트리거 쿨다운 (초)

# Animal Crossing 느낌의 파스텔 팔레트 (BGR)
C_BG_TOP     = (250, 242, 225)   # 하늘
C_BG_BOTTOM  = (210, 240, 220)   # 풀
C_GROUND     = (170, 220, 190)
C_PANEL      = (215, 240, 250)
C_PANEL_LINE = (120, 150, 180)
C_TEXT       = (80, 100, 130)
C_TEXT_SOFT  = (130, 155, 185)
C_BODY       = (140, 185, 230)
C_BODY_DARK  = (95, 135, 180)
C_EARS_IN    = (165, 195, 240)
C_CHEEK      = (150, 175, 245)
C_EYE        = (55, 70, 95)
C_HEART      = (130, 120, 240)
C_ZZZ        = (170, 140, 110)


# ========================================================================
#  Easing helpers
# ========================================================================
def _lerp(a, b, t):
    return a + (b - a) * t

def _ease_out_back(t):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

def _ease_in_cubic(t):
    return t * t * t

def _ease_in_out_sine(t):
    return -(math.cos(math.pi * t) - 1) / 2


# ========================================================================
#  Hub panel (Animal Crossing feel)
# ========================================================================
class Hub:
    """오른쪽 사이드 패널에 캐릭터 + 애니메이션을 그리는 뷰."""

    SPEECH = {
        'come':  '~ come here! ~',
        'away':  '~ bye bye ~',
        'spin':  '~ weee~ ~',
        'wave':  '~ hi there! ~',
        'heart': '<3 <3 <3',
        'sleep': '~ zzz... ~',
        None:    'show me a gesture!',
    }

    def __init__(self):
        self.current_action = None
        self.anim_start = 0.0
        self.idle_t0 = time.time()
        self.hearts = []
        self.zzz = []

    # ------------------------------------------------------------------
    def trigger(self, action):
        """외부에서 제스처가 확정되면 호출."""
        now = time.time()
        # 같은 제스처가 연속으로 들어오면 짧은 쿨다운 유지
        if action == self.current_action and (now - self.anim_start) < RETRIGGER_COOLDOWN:
            return

        self.current_action = action
        self.anim_start = now
        self.hearts.clear()
        self.zzz.clear()

        if action == 'heart':
            for _ in range(7):
                self.hearts.append({
                    'x':   random.randint(-60, 60),
                    'y0':  random.randint(10, 50),
                    'vy':  random.uniform(55, 85),
                    'start': now + random.uniform(0.0, 0.6),
                    'size': random.randint(10, 16),
                })
        elif action == 'sleep':
            for i in range(4):
                self.zzz.append({
                    'start': now + i * 0.45,
                    'scale': 0.8 + i * 0.15,
                })

    # ------------------------------------------------------------------
    def draw(self):
        canvas = np.zeros((HUB_H, HUB_W, 3), dtype=np.uint8)
        self._draw_background(canvas)
        self._draw_character(canvas)
        self._draw_particles(canvas)
        self._draw_ui(canvas)
        cv2.rectangle(canvas, (0, 0), (HUB_W - 1, HUB_H - 1), C_PANEL_LINE, 2)
        return canvas

    # ------------------------------------------------------------------
    def _draw_background(self, canvas):
        # 하늘 → 풀 그라데이션
        for y in range(HUB_H):
            t = y / HUB_H
            col = tuple(int(_lerp(a, b, t)) for a, b in zip(C_BG_TOP, C_BG_BOTTOM))
            cv2.line(canvas, (0, y), (HUB_W, y), col, 1)

        # 둥근 풀 언덕
        cv2.ellipse(canvas, (HUB_W // 2, HUB_H - 30),
                    (HUB_W, 90), 0, 0, 360, C_GROUND, -1)

        # 잎사귀 스팟 (포인트)
        for (dx, dy) in [(40, 90), (310, 120), (70, 300), (305, 290), (180, 70)]:
            cv2.circle(canvas, (dx, dy), 5, (200, 230, 215), -1)

    # ------------------------------------------------------------------
    def _compute_state(self):
        """현재 시각 기준의 scale / offset / rot / arm / eyes 반환."""
        now = time.time()
        bob = math.sin((now - self.idle_t0) * 2.0) * 4  # idle 호흡

        scale   = 1.0
        ox, oy  = 0, int(bob)
        rot     = 0.0
        arm_t   = 0.0   # 팔 흔드는 정도 0~1
        eyes_c  = False

        if self.current_action is None:
            return scale, ox, oy, rot, arm_t, eyes_c

        t = (now - self.anim_start) / ANIM_DURATION
        t = max(0.0, min(1.0, t))

        a = self.current_action
        if a == 'come':
            e = _ease_out_back(t)
            scale = _lerp(0.3, 1.0, e)
            oy    = int(_lerp(220, 0, e)) + int(bob * 0.3)

        elif a == 'away':
            e = _ease_in_cubic(t)
            scale = _lerp(1.0, 0.05, e)
            oy    = int(_lerp(0, -160, e))

        elif a == 'spin':
            rot = 360.0 * _ease_in_out_sine(t)

        elif a == 'wave':
            # 손을 크게 4번 흔들다가 멈춤
            arm_t = math.sin(t * math.pi * 4) * (1.0 - t * 0.3)

        elif a == 'heart':
            oy += int(math.sin(t * math.pi * 2) * 6)

        elif a == 'sleep':
            eyes_c = True
            oy += int(math.sin((now - self.anim_start) * 2.2) * 3)

        return scale, ox, oy, rot, arm_t, eyes_c

    # ------------------------------------------------------------------
    def _draw_character(self, canvas):
        scale, ox, oy, rot, arm_t, eyes_c = self._compute_state()
        if scale <= 0.02:
            return

        # 별도 surface 에 캐릭터를 그린 뒤 회전/스케일 후 합성
        SZ = 280
        surf = np.zeros((SZ, SZ, 4), dtype=np.uint8)
        cx, cy = SZ // 2, SZ // 2 + 10

        # ---- 몸통 ----
        cv2.ellipse(surf, (cx, cy + 55), (55, 45), 0, 0, 360, (*C_BODY, 255), -1)
        cv2.ellipse(surf, (cx, cy + 55), (55, 45), 0, 0, 360, (*C_BODY_DARK, 255), 2)

        # ---- 팔 (왼쪽은 흔드는 팔) ----
        base = math.radians(210)
        wave = math.radians(-55) * arm_t
        lx = cx - 40 + int(math.cos(base + wave) * 32)
        ly = cy + 60 + int(math.sin(base + wave) * 32)
        cv2.line(surf, (cx - 40, cy + 55), (lx, ly), (*C_BODY_DARK, 255), 10)
        cv2.circle(surf, (lx, ly), 9, (*C_BODY, 255), -1)
        cv2.circle(surf, (lx, ly), 9, (*C_BODY_DARK, 255), 2)
        # 오른쪽 팔
        cv2.line(surf, (cx + 40, cy + 55), (cx + 58, cy + 88),
                 (*C_BODY_DARK, 255), 10)
        cv2.circle(surf, (cx + 58, cy + 88), 9, (*C_BODY, 255), -1)
        cv2.circle(surf, (cx + 58, cy + 88), 9, (*C_BODY_DARK, 255), 2)

        # ---- 머리 ----
        cv2.circle(surf, (cx, cy), 65, (*C_BODY, 255), -1)
        cv2.circle(surf, (cx, cy), 65, (*C_BODY_DARK, 255), 3)

        # ---- 귀 ----
        for sign in (-1, 1):
            ex = cx + sign * 42
            ey = cy - 50
            cv2.circle(surf, (ex, ey), 18, (*C_BODY, 255), -1)
            cv2.circle(surf, (ex, ey), 18, (*C_BODY_DARK, 255), 2)
            cv2.circle(surf, (ex, ey), 10, (*C_EARS_IN, 255), -1)

        # ---- 볼 ----
        cv2.circle(surf, (cx - 30, cy + 12), 10, (*C_CHEEK, 255), -1)
        cv2.circle(surf, (cx + 30, cy + 12), 10, (*C_CHEEK, 255), -1)

        # ---- 눈 ----
        if eyes_c:
            cv2.line(surf, (cx - 24, cy - 8), (cx - 10, cy - 8), (*C_EYE, 255), 3)
            cv2.line(surf, (cx + 10, cy - 8), (cx + 24, cy - 8), (*C_EYE, 255), 3)
        else:
            cv2.circle(surf, (cx - 17, cy - 8), 5, (*C_EYE, 255), -1)
            cv2.circle(surf, (cx + 17, cy - 8), 5, (*C_EYE, 255), -1)
            cv2.circle(surf, (cx - 15, cy - 10), 2, (255, 255, 255, 255), -1)
            cv2.circle(surf, (cx + 19, cy - 10), 2, (255, 255, 255, 255), -1)

        # ---- 입 ----
        cv2.ellipse(surf, (cx, cy + 10), (6, 4), 0, 0, 180, (*C_EYE, 255), 2)

        # ---- 변환 후 합성 ----
        M = cv2.getRotationMatrix2D((cx, cy), -rot, scale)
        surf = cv2.warpAffine(surf, M, (SZ, SZ), borderValue=(0, 0, 0, 0))

        target_cx = HUB_W // 2 + ox
        target_cy = HUB_H // 2 - 20 + oy
        x0 = target_cx - SZ // 2
        y0 = target_cy - SZ // 2

        sx0 = max(0, -x0); sy0 = max(0, -y0)
        dx0 = max(0, x0);  dy0 = max(0, y0)
        dx1 = min(HUB_W, x0 + SZ); dy1 = min(HUB_H, y0 + SZ)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        w, h = dx1 - dx0, dy1 - dy0
        sub = surf[sy0:sy0 + h, sx0:sx0 + w]
        alpha = sub[:, :, 3:4] / 255.0
        canvas[dy0:dy1, dx0:dx1] = (
            sub[:, :, :3] * alpha + canvas[dy0:dy1, dx0:dx1] * (1 - alpha)
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_heart(canvas, cx, cy, size, color):
        r = max(2, size // 2)
        cv2.circle(canvas, (cx - r // 2, cy), r // 2 + 1, color, -1)
        cv2.circle(canvas, (cx + r // 2, cy), r // 2 + 1, color, -1)
        pts = np.array([[cx - r, cy], [cx + r, cy], [cx, cy + r + 2]], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], color)

    def _draw_particles(self, canvas):
        now = time.time()

        # 하트
        for h in self.hearts:
            age = now - h['start']
            if age < 0:
                continue
            y = int(HUB_H // 2 - 20 + h['y0'] - h['vy'] * age)
            x = int(HUB_W // 2 + h['x'])
            if y < -20:
                continue
            alpha = max(0.0, 1.0 - age / 2.2)
            col = tuple(int(c * alpha + 240 * (1 - alpha)) for c in C_HEART)
            self._draw_heart(canvas, x, y, h['size'], col)

        # Z (잠자기)
        for i, z in enumerate(self.zzz):
            age = now - z['start']
            if age < 0 or age > 2.5:
                continue
            y = int(HUB_H // 2 - 90 - age * 40)
            x = int(HUB_W // 2 + 55 + i * 12)
            cv2.putText(canvas, 'Z', (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        z['scale'], C_ZZZ, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_ui(self, canvas):
        # 상단 제목 버블
        cv2.rectangle(canvas, (20, 20), (HUB_W - 20, 80), C_PANEL, -1)
        cv2.rectangle(canvas, (20, 20), (HUB_W - 20, 80), C_PANEL_LINE, 2)
        cv2.putText(canvas, 'Finger Forest', (36, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, '* show your hand! *', (36, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT_SOFT, 1, cv2.LINE_AA)

        # 하단 액션 버블
        label = self.current_action.upper() if self.current_action else '...'
        speech = self.SPEECH.get(self.current_action, '...')
        y0, y1 = HUB_H - 70, HUB_H - 20
        cv2.rectangle(canvas, (20, y0), (HUB_W - 20, y1), C_PANEL, -1)
        cv2.rectangle(canvas, (20, y0), (HUB_W - 20, y1), C_PANEL_LINE, 2)
        cv2.putText(canvas, label, (36, y0 + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, speech, (36, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT_SOFT, 1, cv2.LINE_AA)


# ========================================================================
#  Main
# ========================================================================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"[!] 모델 파일이 없습니다: {MODEL_PATH}\n"
            "    먼저 250422_create_dataset.py 로 데이터를 모으고\n"
            "    250422_train.ipynb 로 학습해 주세요."
        )

    model = load_model(MODEL_PATH)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    seq = []
    action_seq = []
    hub = Hub()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # 조인트 간 벡터
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 각도 (15개)
                angle = np.arccos(np.einsum(
                    'nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
                ))
                angle = np.degrees(angle)

                d = np.concatenate([joint.flatten(), angle])
                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < SEQ_LENGTH:
                    continue

                input_data = np.expand_dims(
                    np.array(seq[-SEQ_LENGTH:], dtype=np.float32), axis=0
                )
                y_pred = model.predict(input_data, verbose=0).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = float(y_pred[i_pred])
                if conf < CONF_THRESHOLD:
                    continue

                action = ACTIONS[i_pred]
                action_seq.append(action)
                if len(action_seq) < 3:
                    continue

                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    hub.trigger(action)

                cv2.putText(
                    img, action.upper(),
                    (int(res.landmark[0].x * img.shape[1]),
                     int(res.landmark[0].y * img.shape[0] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )

        # webcam + hub 패널 합치기
        cam_h = img.shape[0]
        hub_img = hub.draw()
        hub_resized = cv2.resize(hub_img, (int(HUB_W * cam_h / HUB_H), cam_h))
        combined = np.hstack([img, hub_resized])

        cv2.imshow('Finger Forest', combined)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

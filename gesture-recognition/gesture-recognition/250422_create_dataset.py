import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['come', 'away', 'spin', 'wave', 'heart', 'sleep']
seq_length = 30
secs_for_action = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

for idx, action in enumerate(actions):
    data = []

    print(f"\n=== {action.upper()} 동작 수집 시작 ===")
    time.sleep(1)

    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.putText(img, f'Waiting for collecting {action.upper()} action...',
                org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow('img', img)
    cv2.waitKey(2000)

    start_time = time.time()

    while time.time() - start_time < secs_for_action:
        ret, img = cap.read()
        if not ret:
            print("프레임 수신 실패")
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

                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
                angle = np.degrees(angle)

                angle_label = np.array([angle], dtype=np.float32)
                angle_label = np.append(angle_label, idx)

                d = np.concatenate([joint.flatten(), angle_label])
                data.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            print("수집 중단됨.")
            break

    data = np.array(data)
    print(f"{action} 데이터 수집 완료: {data.shape}")
    np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

    # 시퀀스 생성
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])
    full_seq_data = np.array(full_seq_data)

    print(f"{action} 시퀀스 데이터 생성 완료: {full_seq_data.shape}")
    np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)

cap.release()
cv2.destroyAllWindows()

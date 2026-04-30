# 🌲 Hand Forest

> 동물의 숲 감성을 담은 손동작 인식 프로젝트입니다.  
> **웹캠 → MediaPipe → LSTM → 손동작에 반응하는 작은 파스텔 허브**로 구성되어 있습니다.

<p align="center">
  <img src="assets/hero.png" alt="Hand Forest — 6가지 손동작 미리보기" width="720"/>
</p>

공중에 손동작을 그리면, 작은 숲속 친구가 실시간으로 반응합니다.  
부르면 아래에서 튀어나오고, 밀어내면 작아지며 멀어지고, 빙글 돌거나 손을 흔들고, 하트를 날리거나 잠드는 동작까지 보여줍니다.

이 프로젝트는 기존의  
[kairess/gesture-recognition](https://github.com/kairess/gesture-recognition)  
파이프라인을 기반으로 제작되었습니다.  
MediaPipe의 손 21개 랜드마크를 사용해 30프레임 시퀀스를 만들고, LSTM 분류기로 손동작을 인식하는 구조입니다.  
기존 3개 손동작을 6개로 확장하고, 동물의 숲 느낌의 귀여운 반응 패널을 더했습니다.

---

## ✨ 주요 기능

| 손동작 | 의미 | 허브 반응 |
| :---: | :--- | :--- |
| `come` | 이리 와! | 캐릭터가 **아래에서 통통 튀어나옵니다** |
| `away` | 안녕, 잘 가 | 캐릭터가 **작아지며 멀어집니다** |
| `spin` | 빙글빙글 | 캐릭터가 잔디 위에서 **360° 회전합니다** |
| `wave` | 안녕! | 캐릭터가 **좌우로 손을 흔듭니다** |
| `heart` | 사랑해 | 캐릭터 주변으로 **하트가 떠오릅니다** |
| `sleep` | 쿨쿨 | 눈을 감고, 머리 위로 **Z Z Z**가 올라갑니다 |

- 🎥 **실시간 인식** — CPU만 사용하는 노트북에서도 웹캠 기준 약 30fps로 동작합니다.
- 🧠 **MediaPipe 랜드마크 기반 LSTM 모델** — 가볍고 빠르게 재학습할 수 있습니다.
- 🎨 **파스텔 톤 허브 UI** — 별도의 이미지 파일 없이 OpenCV 기본 도형만으로 그려집니다.
- 📦 **재현하기 쉬운 구조** — `requirements.txt` 하나로 환경을 맞출 수 있고, 상대 경로를 사용해 Windows/macOS/Linux에서 실행할 수 있습니다.

---

## 🗂 프로젝트 구조

```text
gesture-recognition/
├── 250422_create_dataset.py   # ① 손동작별로 웹캠 데이터 30초 수집
├── 250422_train.ipynb         # ② LSTM 학습, glob으로 모든 seq_*.npy 파일 로드
├── 250422_test.py             # ③ 동물의 숲 감성 허브가 포함된 실시간 웹캠 데모
├── requirements.txt
├── assets/                    # README에 사용되는 스크린샷
├── dataset/                   # raw_*.npy + seq_*.npy, ① 단계에서 생성됨
├── models/                    # 250422_model.h5, ② 단계에서 생성됨
└── README.md
```

이전 참고용 스크립트인 `create_dataset.py`, `260422_test.py`,  
`models/260422_model.h5`는 기준 코드로 저장되어 있습니다.

---

## 🚀 빠른 시작

### 1. 저장소 클론 및 새 가상환경 만들기

```bash
git clone <your-fork-url>
cd gesture-recognition

python -m venv .venv
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
# macOS / Linux:
#   source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Python 3.10 / 3.11** 환경에서 테스트했습니다.  
> 노트북 코드가 Keras 2 스타일 API를 사용하기 때문에 TensorFlow 버전은 `2.15.0`으로 고정했습니다.

### 2. 학습 데이터 수집하기

```bash
python 250422_create_dataset.py
```

실행하면 창이 열리고, 6가지 손동작을 순서대로 진행합니다.  
각 손동작마다 **30초씩** 수집하므로, 웹캠 앞에서 손을 보이게 한 뒤 해당 동작을 계속 수행하면 됩니다.  
중간에 종료하고 싶다면 `q` 키를 누르면 됩니다.

수집된 데이터는 `dataset/` 폴더에 저장됩니다.

```text
dataset/
├── raw_come_<timestamp>.npy   seq_come_<timestamp>.npy
├── raw_away_<timestamp>.npy   seq_away_<timestamp>.npy
├── raw_spin_<timestamp>.npy   seq_spin_<timestamp>.npy
├── raw_wave_<timestamp>.npy   seq_wave_<timestamp>.npy
├── raw_heart_<timestamp>.npy  seq_heart_<timestamp>.npy
└── raw_sleep_<timestamp>.npy  seq_sleep_<timestamp>.npy
```

더 다양한 데이터를 넣고 싶다면 이 스크립트를 여러 번 실행하면 됩니다.  
학습 노트북은 `dataset/` 폴더 안의 모든 `seq_*.npy` 파일을 자동으로 불러옵니다.

### 3. 모델 학습하기

아래 명령어로 학습 노트북을 열고, 위에서부터 모든 셀을 순서대로 실행합니다.

```bash
jupyter notebook 250422_train.ipynb
```

노트북은 `glob`을 사용해 모든 `seq_*.npy` 파일을 불러옵니다.  
이후 데이터를 80:20 비율로 나누고,  
`LSTM(64) → Dense(32) → Dense(6, softmax)` 구조의 작은 모델을 200 epoch 동안 학습합니다.  
가장 성능이 좋은 체크포인트는 `models/250422_model.h5`에 저장됩니다.

### 4. 실시간 데모 실행하기

```bash
python 250422_test.py
```

실행하면 왼쪽에는 웹캠 화면이, 오른쪽에는 파스텔 톤 허브 화면이 나타납니다.

<p align="center">
  <img src="assets/idle.png" alt="대기 상태 허브" width="260"/>
  &nbsp;
  <img src="assets/come.png" alt="come 손동작" width="260"/>
  &nbsp;
  <img src="assets/heart.png" alt="heart 손동작" width="260"/>
</p>

종료하려면 `q` 키를 누르면 됩니다.

---

## 🧩 동작 원리

```text
웹캠 프레임 ─▶ MediaPipe Hands, 손 21개 랜드마크 × (x, y, z, visibility)
                    │
                    ▼
          손 관절 사이의 각도 계산, 15개 값
                    │
                    ▼
          최근 30프레임을 rolling buffer로 저장
                    │
                    ▼
          LSTM(64) ─▶ Dense(32) ─▶ softmax(6)
                    │
                    ▼
   같은 클래스가 3프레임 연속으로 예측될 때만 손동작 확정
                    │
                    ▼
          Hub.trigger(action) ─▶ 애니메이션 실행
```

동물의 숲 감성 허브는 상태를 복잡하게 저장하지 않는 `Hub` 클래스로 구성되어 있습니다.  
매 프레임마다 360×480 크기의 캔버스를 새로 그리고,  
`cv2.ellipse`, `cv2.circle`, `cv2.warpAffine` 같은 OpenCV 기본 기능을 사용합니다.  
또한 `ease_out_back`, `ease_in_cubic`, `ease_in_out_sine` 같은 easing 함수를 이용해  
현재 애니메이션의 크기, 위치, 회전값을 자연스럽게 조절합니다.  
자세한 구현은 `250422_test.py`에서 확인할 수 있습니다.

---

## 🛠 문제 해결

<details>
<summary><b>❌ 테스트 실행 시 <code>FileNotFoundError: models/250422_model.h5</code>가 발생하는 경우</b></summary>

아직 모델 학습을 진행하지 않은 상태입니다.  
먼저 2단계에서 데이터를 수집하고, 3단계에서 모델을 학습해 주세요.
</details>

<details>
<summary><b>❌ Python 3.12 이상에서 MediaPipe 설치가 실패하는 경우</b></summary>

MediaPipe는 최신 Python 버전에 대한 prebuilt wheel 지원이 조금 늦을 수 있습니다.  
Python 3.10 또는 3.11 환경을 사용하는 것을 권장합니다.
</details>

<details>
<summary><b>❌ <code>cv2.VideoCapture(0)</code>에서 카메라를 열 수 없는 경우</b></summary>

다른 프로그램이 웹캠을 사용 중이거나, 운영체제에서 터미널 또는 IDE에 카메라 권한을 허용하지 않았을 수 있습니다.  
Zoom, Teams 같은 프로그램을 종료한 뒤 다시 실행해 보세요.
</details>

<details>
<summary><b>⚠️ 직접 만든 손동작의 정확도가 낮은 경우</b></summary>

- 데이터를 더 많이 수집해 보세요. `250422_create_dataset.py`를 여러 번 실행하면 되고, 학습 노트북은 모든 `seq_*.npy` 파일을 자동으로 불러옵니다.
- 손동작을 더 명확하게 구분해 주세요. 이 모델은 마지막 손 모양만 보는 것이 아니라 30프레임의 움직임을 함께 보기 때문에, 단순히 최종 손 모양만 다르게 하는 것보다 움직이는 경로가 다를수록 좋습니다.
- 노트북 하단의 confusion matrix 셀을 확인해 어떤 손동작끼리 자주 헷갈리는지 살펴보세요.
</details>

---

## 🙏 출처 및 참고

- 기본 손동작 인식 파이프라인:  
  [`kairess/gesture-recognition`](https://github.com/kairess/gesture-recognition) (MIT)
- 손 랜드마크 모델:  
  [Google MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- 분위기 참고: *Animal Crossing* 🍃

## 📄 라이선스

MIT — 자세한 내용은 [`LICENCE`](LICENCE)를 참고해 주세요.

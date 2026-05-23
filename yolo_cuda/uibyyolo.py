import sys
import os
import cv2
import shutil
import torch
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# =========================
# 1. 클래스 카드 위젯 (Step 1)
# =========================
class ClassCard(QFrame):
    def __init__(self, class_name, parent=None):
        super().__init__(parent)
        self.class_name = class_name
        self.is_capturing = False
        self.setFixedWidth(350)
        self.setFixedHeight(200)
        self.setStyleSheet("""
            QFrame { background-color: white; border-radius: 15px; border: 1px solid #E0E0E0; }
            QLabel { border: none; color: #333; }
        """)
        
        layout = QVBoxLayout(self)
        
        # 헤더: 클래스명
        header = QHBoxLayout()
        self.name_label = QLabel(f"Class: {class_name}")
        self.name_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(self.name_label)
        header.addStretch()
        layout.addLayout(header)
        
        self.info_label = QLabel("0 이미지 샘플")
        self.info_label.setStyleSheet("color: #666; font-size: 13px;")
        layout.addWidget(self.info_label)
        
        # 버튼 영역
        btn_layout = QHBoxLayout()
        self.webcam_btn = QPushButton("📷 웹캠 (누르기)")
        self.upload_btn = QPushButton("📤 업로드")
        
        for btn in [self.webcam_btn, self.upload_btn]:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton { 
                    background-color: #F1F3F4; border: none; border-radius: 8px; 
                    padding: 10px; color: #1A73E8; font-weight: bold;
                }
                QPushButton:hover { background-color: #E8F0FE; }
            """)
        
        btn_layout.addWidget(self.webcam_btn)
        btn_layout.addWidget(self.upload_btn)
        layout.addLayout(btn_layout)

# =========================
# 2. 메인 어플리케이션
# =========================
class TeachableApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTX 5060 AI Studio - shw")
        self.setMinimumSize(1300, 900)
        self.setStyleSheet("background-color: #F8F9FA;")
        
        # 데이터 경로 설정
        self.base_path = os.path.join(os.getcwd(), "dataset")
        os.makedirs(self.base_path, exist_ok=True)
        
        self.cards = []
        self.init_ui()
        
        # 카메라 설정 (통합 관리)
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.camera_loop)
        self.timer.start(30)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # --- [STEP 1: 클래스 카드 리스트] ---
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setStyleSheet("border: none; background: transparent;")
        
        self.left_container = QWidget()
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.add_class_btn = QPushButton("+ 클래스 추가")
        self.add_class_btn.setFixedHeight(80)
        self.add_class_btn.setStyleSheet("""
            QPushButton {
                border: 2px dashed #BDC1C6; border-radius: 15px;
                background: transparent; color: #5F6368; font-size: 16px;
            }
            QPushButton:hover { border: 2px dashed #1A73E8; color: #1A73E8; background: #F1F3F4; }
        """)
        self.add_class_btn.clicked.connect(self.create_new_class)
        
        self.left_layout.addWidget(self.add_class_btn)
        left_scroll.setWidget(self.left_container)
        main_layout.addWidget(left_scroll, 5) # Stretch 정수값으로 수정

        # --- [STEP 2: 학습 설정 및 실행] ---
        center_panel = QFrame()
        center_panel.setFixedWidth(320)
        center_panel.setStyleSheet("background: white; border-radius: 15px; border: 1px solid #E0E0E0;")
        center_layout = QVBoxLayout(center_panel)
        
        center_layout.addWidget(QLabel("<h2>학습</h2>"))
        
        self.train_btn = QPushButton("모델 학습 시작")
        self.train_btn.setFixedHeight(50)
        self.train_btn.setStyleSheet("""
            QPushButton { background-color: #1A73E8; color: white; border-radius: 8px; font-weight: bold; font-size: 16px; }
            QPushButton:disabled { background-color: #E0E0E0; color: #9AA0A6; }
        """)
        center_layout.addWidget(self.train_btn)
        
        # 파라미터 제어 (Epoch, Batch)
        param_group = QGroupBox("고급 설정")
        param_group.setStyleSheet("border: none; color: #5F6368; font-weight: bold;")
        param_layout = QFormLayout(param_group)
        
        self.epoch_spin = QSpinBox(); self.epoch_spin.setRange(1, 1000); self.epoch_spin.setValue(10)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 256); self.batch_spin.setValue(32)
        
        param_layout.addRow("에포크:", self.epoch_spin)
        param_layout.addRow("배치 크기:", self.batch_spin)
        center_layout.addWidget(param_group)
        
        # 터미널 스타일 로그 영역
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background: #1E1E2E; color: #A6E3A1; font-family: 'Consolas'; font-size: 11px; border-radius: 8px;")
        center_layout.addWidget(QLabel("<b>로그</b>"))
        center_layout.addWidget(self.log_area)
        
        main_layout.addWidget(center_panel, 0)

        # --- [STEP 3: 미리보기 및 지표] ---
        right_panel = QFrame()
        right_panel.setStyleSheet("background: white; border-radius: 15px; border: 1px solid #E0E0E0;")
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel("<h2>미리보기</h2>"))
        self.cam_feed = QLabel()
        self.cam_feed.setFixedSize(320, 240)
        self.cam_feed.setStyleSheet("background: black; border-radius: 10px;")
        right_layout.addWidget(self.cam_feed, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 분석 지표 (정확도 테이블 등)
        self.analysis_scroll = QScrollArea()
        self.analysis_scroll.setWidgetResizable(True)
        self.analysis_scroll.setStyleSheet("border: none;")
        analysis_content = QWidget()
        self.analysis_layout = QVBoxLayout(analysis_content)
        
        self.acc_label = QLabel("학습 결과가 여기에 표시됩니다.")
        self.confusion_label = QLabel() # 혼동 행렬 이미지
        
        self.analysis_layout.addWidget(QLabel("<b>클래스별 정확도</b>"))
        self.analysis_layout.addWidget(self.acc_label)
        self.analysis_layout.addWidget(QLabel("<b>혼동 행렬</b>"))
        self.analysis_layout.addWidget(self.confusion_label)
        
        self.analysis_scroll.setWidget(analysis_content)
        right_layout.addWidget(self.analysis_scroll)
        
        main_layout.addWidget(right_panel, 6) # Stretch 정수값으로 수정

    # --- 기능 로직 ---
    def camera_loop(self):
        ret, frame = self.cap.read()
        if ret:
            # 연사 저장 (웹캠 버튼 누르고 있을 때)
            for card in self.cards:
                if card.is_capturing:
                    save_dir = os.path.join(self.base_path, card.class_name)
                    img_path = os.path.join(save_dir, f"cap_{len(os.listdir(save_dir))}.jpg")
                    cv2.imwrite(img_path, frame)
                    card.info_label.setText(f"{len(os.listdir(save_dir))} 이미지 샘플")

            # 메인 피드 업데이트
            display_frame = cv2.flip(frame, 1)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = display_frame.shape
            qt_img = QImage(display_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.cam_feed.setPixmap(QPixmap.fromImage(qt_img).scaled(320, 240, Qt.AspectRatioMode.KeepAspectRatio))

    def create_new_class(self):
        name, ok = QInputDialog.getText(self, "클래스 생성", "이름:")
        if ok and name:
            os.makedirs(os.path.join(self.base_path, name), exist_ok=True)
            card = ClassCard(name)
            # 연사 버튼 이벤트 (눌렀을 때/뗐을 때)
            card.webcam_btn.pressed.connect(lambda: self.set_capture(card, True))
            card.webcam_btn.released.connect(lambda: self.set_capture(card, False))
            # 업로드 버튼
            card.upload_btn.clicked.connect(lambda: self.handle_upload(name, card))
            
            self.left_layout.insertWidget(self.left_layout.count() - 1, card)
            self.cards.append(card)

    def set_capture(self, card, state):
        card.is_capturing = state
        card.webcam_btn.setText("🔴 저장 중..." if state else "📷 웹캠 (누르기)")

    def handle_upload(self, name, card):
        files, _ = QFileDialog.getOpenFileNames(self, "이미지 업로드", "", "Images (*.jpg *.png *.jpeg)")
        if files:
            save_dir = os.path.join(self.base_path, name)
            for f in files:
                shutil.copy(f, save_dir)
            card.info_label.setText(f"{len(os.listdir(save_dir))} 이미지 샘플")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = TeachableApp()
    ex.show()
    sys.exit(app.exec())
import sys
import torch
import pickle
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtTextToSpeech import QTextToSpeech

from torchvision import transforms, models
import torch.nn as nn

# ----------------------------
# Load Model
# ----------------------------
NUM_CLASSES = 13
MODEL_PATH = "final_trained_model.pth"
CLASS_NAMES_PATH = "class_names.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with open(CLASS_NAMES_PATH, "rb") as f:
    class_names = pickle.load(f)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# Themes (Improved)
# ----------------------------
LIGHT_THEME = """
QWidget {
    background-color: #f4f8f6;
    color: #2d3436;
}
QFrame {
    background-color: #ffffff;
    border-radius: 18px;
}
QPushButton {
    background-color: #3fa37c;
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2e8b68;
}
"""

DARK_THEME = """
QWidget {
    background-color: #1f2a2e;
    color: #ecf0f1;
}
QFrame {
    background-color: #2c3e43;
    border-radius: 18px;
}
QPushButton {
    background-color: #4cd39b;
    color: #102a23;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3cbf8a;
}
"""

# ----------------------------
# Hover Speaker Helper
# ----------------------------
class HoverSpeaker(QObject):
    def __init__(self, tts, enabled=True):
        super().__init__()
        self.tts = tts
        self.enabled = enabled

    def eventFilter(self, obj, event):
        if self.enabled and event.type() == QEvent.Enter:
            if isinstance(obj, QLabel) or isinstance(obj, QPushButton):
                text = obj.text().replace("🌿", "").replace("🔊", "")
                if text:
                    self.tts.say(text)
        return super().eventFilter(obj, event)

# ----------------------------
# Main App
# ----------------------------
class LeafLens(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LeafLens - Plant Disease Detection")
        self.setFixedSize(540, 720)

        self.image_path = None
        self.dark_mode = False
        self.voice_enabled = True

        self.tts = QTextToSpeech()
        self.speaker = HoverSpeaker(self.tts)

        self.build_ui()
        self.apply_theme()

    def build_ui(self):
        main = QVBoxLayout()

        # Header
        title = QLabel("🌿 LeafLens")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Smart Tomato & Potato Leaf Disease Detector")
        subtitle.setAlignment(Qt.AlignCenter)

        main.addWidget(title)
        main.addWidget(subtitle)

        # Image Card
        self.card = QFrame()
        card_layout = QVBoxLayout()

        self.image_label = QLabel("Drop Image Here\nor Click Browse")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(440, 280)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #3fa37c;
                border-radius: 14px;
                font-size: 14px;
            }
        """)
        self.image_label.setAcceptDrops(True)

        card_layout.addWidget(self.image_label)
        self.card.setLayout(card_layout)
        main.addWidget(self.card, alignment=Qt.AlignCenter)

        # Buttons
        btns = QHBoxLayout()

        browse_btn = QPushButton("Browse Image")
        browse_btn.clicked.connect(self.browse_image)

        predict_btn = QPushButton("Predict Disease")
        predict_btn.clicked.connect(self.predict_image)

        btns.addWidget(browse_btn)
        btns.addWidget(predict_btn)
        main.addLayout(btns)

        # Result
        self.result_label = QLabel("Prediction: -")
        self.result_label.setFont(QFont("Arial", 15, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)

        self.conf_label = QLabel("Confidence: -")
        self.conf_label.setAlignment(Qt.AlignCenter)

        main.addWidget(self.result_label)
        main.addWidget(self.conf_label)

        # Accessibility Controls
        acc = QHBoxLayout()

        self.voice_btn = QPushButton("🔊 Voice ON")
        self.voice_btn.clicked.connect(self.toggle_voice)

        theme_btn = QPushButton("🌙 Dark / Light")
        theme_btn.clicked.connect(self.toggle_theme)

        acc.addWidget(self.voice_btn)
        acc.addWidget(theme_btn)
        main.addLayout(acc)

        # Status
        self.status = QLabel("Status: Ready")
        self.status.setAlignment(Qt.AlignCenter)
        main.addWidget(self.status)

        self.setLayout(main)

        # Install hover speaker
        for label in self.findChildren(QLabel):
            label.installEventFilter(self.speaker)

        for button in self.findChildren(QPushButton):
            button.installEventFilter(self.speaker)

    # ----------------------------
    # Theme
    # ----------------------------
    def apply_theme(self):
        self.setStyleSheet(DARK_THEME if self.dark_mode else LIGHT_THEME)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    # ----------------------------
    # Voice
    # ----------------------------
    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        self.speaker.enabled = self.voice_enabled
        self.voice_btn.setText("🔊 Voice ON" if self.voice_enabled else "🔇 Voice OFF")

    def speak(self, text):
        if self.voice_enabled:
            self.tts.say(text)

    # ----------------------------
    # Drag & Drop
    # ----------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()

    def dropEvent(self, event):
        self.load_image(event.mimeData().urls()[0].toLocalFile())

    # ----------------------------
    # Browse
    # ----------------------------
    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        self.image_path = path
        pixmap = QPixmap(path).scaled(440,280, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.status.setText("Status: Image Loaded")
        self.speak("Image loaded successfully")

    # ----------------------------
    # Prediction
    # ----------------------------
    def predict_image(self):
        if not self.image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        image = Image.open(self.image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        disease = class_names[pred.item()]
        conf = confidence.item() * 100

        self.result_label.setText(f"Prediction: {disease}")
        self.conf_label.setText(f"Confidence: {conf:.2f}%")
        self.status.setText("Status: Prediction Completed")

        self.speak(f"The leaf shows {disease}. Confidence {int(conf)} percent")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LeafLens()
    window.show()
    sys.exit(app.exec())
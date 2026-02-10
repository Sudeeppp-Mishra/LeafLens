import sys
import os
import json
import pickle
import io
from datetime import datetime

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from rembg import remove  # Background removal

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame,
    QListWidget, QProgressBar, QStackedWidget, QCheckBox
)
from PySide6.QtGui import QPixmap, QColor, QIcon
from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtTextToSpeech import QTextToSpeech

# ───────────────────────────────────────────────
# ⚙️ CONFIG & PATHS
# ───────────────────────────────────────────────

NUM_CLASSES = 13
MODEL_PATH = "final_trained_model.pth"
CLASS_NAMES_PATH = "class_names.pkl"
HISTORY_PATH = "prediction_history.json"
SETTINGS_PATH = "app_settings.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────
# 🎨 DYNAMIC STYLESHEETS (IMPROVED)
# ───────────────────────────────────────────────

def get_stylesheet(dark_mode=False):
    if dark_mode:
        return """
        /* DARK MODE */
        QWidget { font-family: 'Segoe UI', sans-serif; font-size: 14px; color: #cbd5e1; }
        QWidget#RootWidget { background-color: #0f172a; } /* Main Window Background */

        /* Sidebar */
        QFrame#Sidebar { background-color: #1e293b; min-width: 240px; border-right: 1px solid #334155; }
        QLabel#SidebarTitle { color: #34d399; font-size: 26px; font-weight: bold; padding: 30px 20px; }

        /* Navigation Buttons */
        QPushButton#NavBtn { background-color: transparent; color: #94a3b8; text-align: left; padding: 12px 20px; border: none; border-radius: 10px; margin: 4px 15px; font-weight: 600; }
        QPushButton#NavBtn:hover { background-color: #334155; color: white; }
        QPushButton#NavBtn:checked { background-color: #34d399; color: #0f172a; }

        /* Cards */
        QFrame#Card { background-color: #1e293b; border-radius: 16px; border: 1px solid #334155; }
        
        /* Text & Headers */
        QLabel#Header { font-size: 24px; font-weight: bold; color: #f1f5f9; margin-bottom: 10px; }
        QLabel#SubHeader { font-size: 14px; color: #94a3b8; }
        QLabel#ResultText { font-size: 20px; font-weight: bold; color: #34d399; }
        QLabel#DropZone { background-color: #0f172a; border: 2px dashed #475569; border-radius: 16px; color: #64748b; }

        /* Buttons */
        QPushButton#PrimaryBtn { background-color: #34d399; color: #0f172a; font-weight: bold; border-radius: 10px; padding: 12px; border: none; }
        QPushButton#PrimaryBtn:hover { background-color: #10b981; }
        QPushButton#SecondaryBtn { background-color: #334155; color: white; border-radius: 10px; padding: 12px; border: 1px solid #475569; }
        QPushButton#SecondaryBtn:hover { background-color: #475569; }
        
        QListWidget { background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; color: #cbd5e1; }
        """
    else:
        return """
        /* LIGHT MODE - CLEAN & MODERN */
        QWidget { font-family: 'Segoe UI', sans-serif; font-size: 14px; color: #334155; }
        QWidget#RootWidget { background-color: #f8fafc; } /* Slate 50 Background */

        /* Sidebar */
        QFrame#Sidebar { background-color: #ffffff; min-width: 240px; border-right: 1px solid #e2e8f0; }
        QLabel#SidebarTitle { color: #059669; font-size: 26px; font-weight: bold; padding: 30px 20px; }

        /* Navigation Buttons */
        QPushButton#NavBtn { background-color: transparent; color: #64748b; text-align: left; padding: 12px 20px; border: none; border-radius: 10px; margin: 4px 15px; font-weight: 600; }
        QPushButton#NavBtn:hover { background-color: #f1f5f9; color: #0f172a; }
        QPushButton#NavBtn:checked { background-color: #d1fae5; color: #065f46; } /* Soft Green Pill */

        /* Cards */
        QFrame#Card { background-color: #ffffff; border-radius: 16px; border: 1px solid #cbd5e1; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }

        /* Text & Headers */
        QLabel#Header { font-size: 24px; font-weight: bold; color: #1e293b; margin-bottom: 10px; }
        QLabel#SubHeader { font-size: 14px; color: #64748b; }
        QLabel#ResultText { font-size: 20px; font-weight: bold; color: #059669; }
        QLabel#DropZone { background-color: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 16px; color: #64748b; font-weight: 500; }

        /* Buttons */
        QPushButton#PrimaryBtn { background-color: #059669; color: white; font-weight: bold; border-radius: 10px; padding: 12px; border: none; }
        QPushButton#PrimaryBtn:hover { background-color: #047857; }
        QPushButton#SecondaryBtn { background-color: #ffffff; color: #475569; border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px; font-weight: 600; }
        QPushButton#SecondaryBtn:hover { background-color: #f1f5f9; border-color: #94a3b8; }
        
        QListWidget { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px; outline: none; }
        QListWidget::item { padding: 10px; border-bottom: 1px solid #f1f5f9; }
        QListWidget::item:selected { background-color: #f0fdf4; color: #166534; }
        """

# ───────────────────────────────────────────────
# 🧵 SEGMENTATION & PREDICTION THREAD
# ───────────────────────────────────────────────

class AnalysisThread(QThread):
    finished = Signal(object)

    def __init__(self, image_path, model, class_names):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def run(self):
        try:
            # 1. Background Removal
            raw_image = Image.open(self.image_path).convert("RGB")
            img_byte_arr = io.BytesIO()
            raw_image.save(img_byte_arr, format='PNG')
            output_bytes = remove(img_byte_arr.getvalue())
            
            segmented_rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
            white_bg = Image.new("RGBA", segmented_rgba.size, "WHITE")
            white_bg.paste(segmented_rgba, (0, 0), segmented_rgba)
            processed_rgb = white_bg.convert("RGB")

            # 2. Prediction
            img_tensor = self.transform(processed_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                top_probs, top_idxs = torch.topk(probs, 3)
            
            results = {
                'disease': self.class_names[top_idxs[0][0].item()],
                'confidence': top_probs[0][0].item() * 100,
                'top_3': [(self.class_names[top_idxs[0][i].item()], top_probs[0][i].item() * 100) for i in range(3)],
                'processed_pil': processed_rgb
            }
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit(e)

# ───────────────────────────────────────────────
# 🖥️ MAIN APPLICATION
# ───────────────────────────────────────────────

class LeafLens(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("RootWidget") # <--- OPTIONAL FIX: Ensures background color applies correctly
        self.setWindowTitle("LeafLens AI Pro")
        self.setMinimumSize(1200, 800)
        self.setAcceptDrops(True)
        
        # Data & Settings
        self.settings = self.load_settings()
        self.prediction_history = self.load_history()
        self.image_path = None
        self.tts = QTextToSpeech()
        
        self.init_model()
        self.init_ui()
        self.apply_current_theme()
        self.update_history_list()

    def load_settings(self):
        default = {"dark_mode": False, "voice_enabled": True}
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, 'r') as f: return json.load(f)
            except: return default
        return default

    def save_settings(self):
        with open(SETTINGS_PATH, 'w') as f: json.dump(self.settings, f)

    def load_history(self):
        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, 'r') as f: return json.load(f)
            except: return []
        return []

    def save_history(self):
        with open(HISTORY_PATH, 'w') as f: json.dump(self.prediction_history[:30], f)

    def init_model(self):
        self.model = models.efficientnet_b0()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, NUM_CLASSES)
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            except: print("Warning: Model weights not loaded.")
        self.model.to(device).eval()
        
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, "rb") as f: self.class_names = pickle.load(f)
        else: self.class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

    def apply_current_theme(self):
        self.setStyleSheet(get_stylesheet(self.settings.get("dark_mode", False)))

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Sidebar
        sidebar = QFrame(); sidebar.setObjectName("Sidebar")
        side_ly = QVBoxLayout(sidebar)
        title = QLabel("🌿 LeafLens"); title.setObjectName("SidebarTitle")
        
        self.nav_btns = []
        btn_data = [("🔍 Analysis", 0), ("📜 History", 1), ("⚙️ Settings", 2), ("ℹ️ About", 3)]
        for text, idx in btn_data:
            btn = QPushButton(f"  {text}")
            btn.setObjectName("NavBtn"); btn.setCheckable(True)
            btn.clicked.connect(self.make_switch_fn(idx))
            side_ly.addWidget(btn)
            self.nav_btns.append(btn)

        self.nav_btns[0].setChecked(True)
        side_ly.insertWidget(0, title); side_ly.addStretch()

        # 2. Stacked Pages
        self.stack = QStackedWidget()
        self.stack.addWidget(self.create_analysis_page())
        self.stack.addWidget(self.create_history_page())
        self.stack.addWidget(self.create_settings_page())
        self.stack.addWidget(self.create_about_page())

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stack, 1)

    def make_switch_fn(self, idx):
        return lambda: self.switch_page(idx)

    def switch_page(self, index):
        for i, btn in enumerate(self.nav_btns):
            btn.setChecked(i == index)
        self.stack.setCurrentIndex(index)

    # 📄 PAGES
    def create_analysis_page(self):
        page = QWidget(); ly = QVBoxLayout(page)
        ly.setContentsMargins(40, 40, 40, 40); ly.setSpacing(20)
        
        ly.addWidget(QLabel("Plant Health Scanner", objectName="Header"))
        ly.addWidget(QLabel("Upload a leaf image to detect diseases using AI.", objectName="SubHeader"))
        
        work = QHBoxLayout(); work.setSpacing(30)
        
        # Left: Image Drop Zone
        img_card = QFrame(objectName="Card"); img_ly = QVBoxLayout(img_card)
        self.image_label = QLabel("Drag & Drop Leaf Image Here", objectName="DropZone")
        self.image_label.setAlignment(Qt.AlignCenter); self.image_label.setMinimumSize(500, 450)
        
        row = QHBoxLayout()
        b_btn = QPushButton("📁 Browse", objectName="SecondaryBtn"); b_btn.clicked.connect(self.browse_image)
        a_btn = QPushButton("🚀 Analyze Image", objectName="PrimaryBtn"); a_btn.clicked.connect(self.analyze_image)
        row.addWidget(b_btn); row.addWidget(a_btn)
        img_ly.addWidget(self.image_label); img_ly.addLayout(row)

        # Right: Results
        res_card = QFrame(objectName="Card"); res_card.setFixedWidth(350); res_ly = QVBoxLayout(res_card)
        res_ly.setContentsMargins(20, 20, 20, 20)
        
        self.res_title = QLabel("Ready to Scan", objectName="Header")
        self.res_conf = QLabel("--%", objectName="ResultText"); self.res_conf.setAlignment(Qt.AlignCenter)
        self.res_details = QLabel("Results will appear here after analysis."); self.res_details.setWordWrap(True)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        
        res_ly.addWidget(self.res_title)
        res_ly.addWidget(self.res_conf)
        res_ly.addSpacing(15)
        res_ly.addWidget(self.res_details)
        res_ly.addStretch()
        res_ly.addWidget(self.progress)
        
        work.addWidget(img_card, 2); work.addWidget(res_card, 1)
        ly.addLayout(work); return page

    def create_history_page(self):
        page = QWidget(); ly = QVBoxLayout(page); ly.setContentsMargins(40, 40, 40, 40)
        ly.addWidget(QLabel("Analysis History", objectName="Header"))
        self.history_list = QListWidget()
        ly.addWidget(self.history_list)
        btn = QPushButton("Clear History", objectName="SecondaryBtn"); btn.clicked.connect(self.clear_history)
        ly.addWidget(btn); return page

    def create_settings_page(self):
        page = QWidget(); ly = QVBoxLayout(page); ly.setContentsMargins(100, 50, 100, 50); ly.setSpacing(30)
        ly.addWidget(QLabel("Application Settings", objectName="Header"))
        card = QFrame(objectName="Card"); c_ly = QVBoxLayout(card); c_ly.setContentsMargins(30,30,30,30)

        self.dark_check = QCheckBox("Enable Dark Mode")
        self.dark_check.setChecked(self.settings.get("dark_mode", False))
        self.dark_check.toggled.connect(self.toggle_dark_mode)

        self.voice_check = QCheckBox("Enable Voice Feedback")
        self.voice_check.setChecked(self.settings.get("voice_enabled", True))
        self.voice_check.toggled.connect(self.toggle_voice)

        c_ly.addWidget(self.dark_check); c_ly.addSpacing(20); c_ly.addWidget(self.voice_check)
        ly.addWidget(card); ly.addStretch(); return page

    def create_about_page(self):
        page = QWidget(); ly = QVBoxLayout(page); ly.setAlignment(Qt.AlignCenter)
        info = QLabel("<h2 style='text-align:center; color:#059669;'>LeafLens AI Pro</h2><p style='color:#64748b;'>Background Removal + EfficientNet Analysis.<br>Version 2.0</p>")
        info.setAlignment(Qt.AlignCenter); ly.addWidget(info); return page

    # ⚙️ LOGIC
    def toggle_dark_mode(self, enabled):
        self.settings["dark_mode"] = enabled
        self.save_settings(); self.apply_current_theme()

    def toggle_voice(self, enabled):
        self.settings["voice_enabled"] = enabled; self.save_settings()

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png *.jpeg)")
        if path: self.load_image(path)

    def load_image(self, path):
        self.image_path = path
        pix = QPixmap(path).scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix); self.image_label.setText("")

    def dragEnterEvent(self, e): e.accept() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e):
        path = e.mimeData().urls()[0].toLocalFile()
        if path.lower().endswith(('.png', '.jpg', '.jpeg')): self.load_image(path)

    def analyze_image(self):
        if not self.image_path: return
        self.progress.setVisible(True); self.progress.setRange(0, 0) # Indeterminate loading
        self.thread = AnalysisThread(self.image_path, self.model, self.class_names)
        self.thread.finished.connect(self.handle_result); self.thread.start()

    def handle_result(self, res):
        self.progress.setVisible(False)
        if isinstance(res, Exception):
            QMessageBox.critical(self, "Error", f"Analysis failed: {res}"); return
        
        # 1. Show Segmented Image
        res['processed_pil'].save("debug_view.png")
        self.image_label.setPixmap(QPixmap("debug_view.png").scaled(500, 500, Qt.KeepAspectRatio))
        
        # 2. Display Result
        clean_name = res['disease'].replace("___", " - ").replace("_", " ")
        self.res_title.setText(clean_name)
        self.res_conf.setText(f"{res['confidence']:.1f}% Match")
        
        details = "<br><b>Top Alternatives:</b><br>"
        for n, p in res['top_3']: details += f"• {n.replace('_',' ')}: {p:.1f}%<br>"
        self.res_details.setText(details)
        
        if self.settings.get("voice_enabled", True):
            self.tts.say(f"Result: {clean_name}")
            
        # 3. History
        entry = {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"), 'disease': clean_name, 'conf': f"{res['confidence']:.1f}%"}
        self.prediction_history.insert(0, entry)
        self.save_history(); self.update_history_list()

    def update_history_list(self):
        self.history_list.clear()
        for h in self.prediction_history:
            conf_val = h.get('conf') or h.get('confidence') or "--"
            self.history_list.addItem(f"{h['timestamp']} - {h['disease']} ({conf_val})")

    def clear_history(self):
        self.prediction_history = []; self.update_history_list()
        if os.path.exists(HISTORY_PATH): os.remove(HISTORY_PATH)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LeafLens(); win.show(); sys.exit(app.exec())
import sys
import os
import json
import pickle
import io
import re
from datetime import datetime

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from rembg import remove

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame,
    QListWidget, QProgressBar, QStackedWidget, QCheckBox,
    QSizePolicy, QScrollArea
)
from PySide6.QtGui import QPixmap, QColor, QIcon, QDragEnterEvent, QDropEvent, QFont
from PySide6.QtCore import Qt, Signal, QThread, QSize, QEvent
from PySide6.QtTextToSpeech import QTextToSpeech

# ───────────────────────────────────────────────
# ⚙️ CONFIG & PATHS
# ───────────────────────────────────────────────

NUM_CLASSES = 13
MODEL_PATH = "final_trained_model.pth"
CLASS_NAMES_PATH = "class_names.pkl"
HISTORY_PATH = "prediction_history.json"
SETTINGS_PATH = "app_settings.json"

# NEW: Active Learning Config
FAILED_SAMPLES_DIR = "uncertain_samples"
UNCERTAINTY_THRESHOLD = 75.0  # Images with confidence < 75% will be saved for review

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────
# 📖 RECOMMENDATION DATABASE
# ───────────────────────────────────────────────

DISEASE_KNOWLEDGE_BASE = {
    "potatoearlyblight": {
        "Remedy": "Fungicides with Chlorothalonil or Mancozeb.",
        "Organic": "Copper-based sprays / Bacillus subtilis.",
        "Action": "Rotate crops and keep leaves dry."
    },
    "potatolateblight": {
        "Remedy": "Curzate, Revus Top, or Copper fungicides.",
        "Organic": "Strict sanitation; destroy infected plants.",
        "Action": "High-risk! Alert nearby farmers immediately."
    },
    "potatohealthy": {
        "Remedy": "None required.",
        "Organic": "Compost tea for immunity.",
        "Action": "Maintain soil pH 4.8-5.5."
    },
    "tomatobacterialspot": {
        "Remedy": "Copper bactericides + Mancozeb.",
        "Organic": "Potassium bicarbonate / Organic copper.",
        "Action": "Do not handle plants when wet."
    },
    "tomatoearlyblight": {
        "Remedy": "Daconil / Chlorothalonil.",
        "Organic": "Neem oil; remove lower leaves.",
        "Action": "Mulch base to stop soil splash."
    },
    "tomatolateblight": {
        "Remedy": "Ranman / Copper sprays (7-10 days).",
        "Organic": "Aggressive pruning.",
        "Action": "Act immediately; highly destructive."
    },
    "tomatoleafmold": {
        "Remedy": "Difenoconazole.",
        "Organic": "Increase ventilation.",
        "Action": "Prune lower suckers for airflow."
    },
    "tomatoseptorialeafspot": {
        "Remedy": "Chlorothalonil / Azoxystrobin.",
        "Organic": "Remove infected leaves early.",
        "Action": "Rotate crops every 3 years."
    },
    "tomatoyellowleafcurlvirus": {
        "Remedy": "None (Viral). Control Whiteflies.",
        "Organic": "Yellow sticky traps / Insecticidal soap.",
        "Action": "Remove and bury infected plants."
    },
    "tomatohealthy": {
        "Remedy": "None needed.",
        "Organic": "Balanced nutrition.",
        "Action": "Monitor for aphids/hornworms."
    }
}

# ───────────────────────────────────────────────
# 🎨 STYLESHEETS
# ───────────────────────────────────────────────

def get_stylesheet(dark_mode=False):
    primary = "#34d399" if dark_mode else "#059669"
    bg = "#0f172a" if dark_mode else "#f8fafc"
    card_bg = "#1e293b" if dark_mode else "#ffffff"
    text_main = "#f1f5f9" if dark_mode else "#1e293b"
    text_sub = "#94a3b8" if dark_mode else "#64748b"
    border = "#334155" if dark_mode else "#cbd5e1"

    return f"""
    QWidget {{ font-family: 'Segoe UI', sans-serif; font-size: 14px; color: {text_sub}; }}
    QWidget#RootWidget {{ background-color: {bg}; }}
    
    QFrame#Sidebar {{ background-color: {card_bg}; border-right: 1px solid {border}; min-width: 240px; }}
    QLabel#SidebarTitle {{ color: {primary}; font-size: 26px; font-weight: 800; padding: 30px 20px; }}
    QPushButton#NavBtn {{ 
        background-color: transparent; color: {text_sub}; text-align: left; 
        padding: 12px 20px; border: none; border-radius: 8px; margin: 4px 15px; font-weight: 600; font-size: 15px;
    }}
    QPushButton#NavBtn:checked {{ background-color: {primary}; color: {bg}; }}
    QPushButton#NavBtn:hover {{ background-color: {border}; }}

    QFrame#Card {{ background-color: {card_bg}; border-radius: 16px; border: 1px solid {border}; }}
    QLabel#Header {{ font-size: 24px; font-weight: 700; color: {text_main}; margin-bottom: 10px; }}
    
    QLabel#BigPercent {{ font-size: 48px; font-weight: 900; color: {primary}; margin: 5px 0; }}
    QLabel#ResultTitle {{ font-size: 20px; font-weight: 700; color: {text_main}; }}
    QLabel#SectionTitle {{ font-size: 14px; font-weight: 700; color: {text_main}; text-transform: uppercase; letter-spacing: 1px; }}

    QLabel#DropZone {{ 
        background-color: {bg}; border: 3px dashed {border}; border-radius: 16px; 
        color: {text_sub}; font-size: 16px; font-weight: 600;
    }}

    QPushButton#PrimaryBtn {{ background-color: {primary}; color: {bg}; font-weight: 700; border-radius: 10px; padding: 12px; font-size: 15px; }}
    QPushButton#PrimaryBtn:hover {{ background-color: #10b981; }}
    QPushButton#SecondaryBtn {{ background-color: {card_bg}; color: {text_main}; border: 1px solid {border}; border-radius: 10px; padding: 12px; font-weight: 600; }}
    QPushButton#SecondaryBtn:hover {{ background-color: {border}; }}

    QCheckBox {{ color: {text_main}; font-size: 15px; spacing: 10px; }}
    QCheckBox::indicator {{ width: 20px; height: 20px; border-radius: 4px; border: 2px solid {border}; }}
    QCheckBox::indicator:checked {{ background-color: {primary}; border-color: {primary}; }}
    
    QListWidget {{ background-color: {card_bg}; border: 1px solid {border}; border-radius: 10px; padding: 10px; font-size: 14px; color: {text_main}; }}
    QProgressBar {{ border: 1px solid {border}; border-radius: 5px; text-align: center; }}
    QProgressBar::chunk {{ background-color: {primary}; }}
    """

# ───────────────────────────────────────────────
# 🧵 WORKER THREAD
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
            raw_image = Image.open(self.image_path).convert("RGB")
            img_byte_arr = io.BytesIO()
            raw_image.save(img_byte_arr, format='PNG')
            output_bytes = remove(img_byte_arr.getvalue())
            
            segmented_rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
            white_bg = Image.new("RGBA", segmented_rgba.size, "WHITE")
            white_bg.paste(segmented_rgba, (0, 0), segmented_rgba)
            processed_rgb = white_bg.convert("RGB")

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
        self.setObjectName("RootWidget") 
        self.setWindowTitle("LeafLens AI")
        self.resize(1280, 800)
        self.setMinimumSize(1000, 700)
        self.setAcceptDrops(True)
        
        # Ensure active learning folder exists
        if not os.path.exists(FAILED_SAMPLES_DIR):
            os.makedirs(FAILED_SAMPLES_DIR)

        self.settings = self.load_settings()
        self.prediction_history = self.load_history()
        self.current_pixmap = None 
        self.image_path = None
        self.tts = QTextToSpeech()
        
        self.init_model()
        self.init_ui()
        self.apply_theme()
        self.update_history_ui()

    def load_settings(self):
        default = {"dark_mode": True, "voice_enabled": True, "read_recommendations": True}
        if os.path.exists(SETTINGS_PATH):
            try: 
                with open(SETTINGS_PATH, 'r') as f: return json.load(f)
            except: return default
        return default

    def save_settings(self):
        with open(SETTINGS_PATH, 'w') as f: json.dump(self.settings, f)
        self.apply_theme()

    def load_history(self):
        if os.path.exists(HISTORY_PATH):
            try: 
                with open(HISTORY_PATH, 'r') as f: return json.load(f)
            except: return []
        return []

    def save_history(self):
        with open(HISTORY_PATH, 'w') as f: json.dump(self.prediction_history[:50], f)

    def init_model(self):
        self.model = models.efficientnet_b0()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, NUM_CLASSES)
        if os.path.exists(MODEL_PATH):
            try: self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            except: pass
        self.model.to(device).eval()
        
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, "rb") as f: self.class_names = pickle.load(f)
        else: self.class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

    def apply_theme(self):
        self.setStyleSheet(get_stylesheet(self.settings.get("dark_mode", True)))

    def clean_text_for_speech(self, text):
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = re.sub(r'\s+', ' ', clean)
        clean = clean.encode('ascii', 'ignore').decode('ascii')
        return clean.strip()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame(); sidebar.setObjectName("Sidebar")
        side_ly = QVBoxLayout(sidebar)
        side_ly.setContentsMargins(0, 0, 0, 20)
        
        title = QLabel("🌿 LeafLens"); title.setObjectName("SidebarTitle")
        side_ly.addWidget(title)

        self.nav_btns = []
        pages = [("🔍 Analysis", 0), ("📜 History", 1), ("⚙️ Settings", 2), ("ℹ️ About", 3)]
        
        for label, idx in pages:
            btn = QPushButton(f"  {label}")
            btn.setObjectName("NavBtn"); btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self.switch_page(i))
            side_ly.addWidget(btn)
            self.nav_btns.append(btn)
        
        self.nav_btns[0].setChecked(True)
        side_ly.addStretch()
        
        ver = QLabel("v1.1.0 - Active Learning")
        ver.setStyleSheet("color: #64748b; padding-left: 35px; font-size: 11px;")
        side_ly.addWidget(ver)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.create_analysis_page())
        self.stack.addWidget(self.create_history_page())
        self.stack.addWidget(self.create_settings_page())
        self.stack.addWidget(self.create_about_page())

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stack, 1)

    def switch_page(self, idx):
        for i, b in enumerate(self.nav_btns): b.setChecked(i == idx)
        self.stack.setCurrentIndex(idx)

    def create_analysis_page(self):
        page = QWidget(); ly = QVBoxLayout(page)
        ly.setContentsMargins(30, 30, 30, 30); ly.setSpacing(20)
        
        header = QLabel("AI Crop Diagnosis", objectName="Header")
        ly.addWidget(header)

        work_area = QHBoxLayout(); work_area.setSpacing(20)

        img_card = QFrame(objectName="Card")
        img_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_ly = QVBoxLayout(img_card)
        
        self.image_label = QLabel("Drag & Drop Leaf Image Here\n\n(Supports JPG, PNG)", objectName="DropZone")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        btn_row = QHBoxLayout()
        b_btn = QPushButton("📁 Browse Image", objectName="SecondaryBtn"); b_btn.clicked.connect(self.browse_image)
        a_btn = QPushButton("🚀 Run Analysis", objectName="PrimaryBtn"); a_btn.clicked.connect(self.analyze_image)
        btn_row.addWidget(b_btn); btn_row.addWidget(a_btn)
        
        img_ly.addWidget(self.image_label, 1)
        img_ly.addLayout(btn_row)

        res_card = QFrame(objectName="Card"); res_card.setFixedWidth(420)
        res_ly = QVBoxLayout(res_card)
        
        self.res_title = QLabel("System Ready", objectName="ResultTitle")
        self.res_percent = QLabel("--%", objectName="BigPercent")
        self.res_percent.setAlignment(Qt.AlignLeft)

        top3_lbl = QLabel("PROBABILITY BREAKDOWN", objectName="SectionTitle")
        self.top3_box = QLabel("1. --\n2. --\n3. --")
        self.top3_box.setStyleSheet("color: #94a3b8; line-height: 140%; margin-bottom: 10px;")
        
        rec_lbl = QLabel("RECOMMENDATIONS", objectName="SectionTitle")
        self.rec_text = QLabel("Upload an image to generate a treatment plan.")
        self.rec_text.setWordWrap(True)
        self.rec_text.setStyleSheet("color: #cbd5e1; line-height: 130%;")
        
        self.progress = QProgressBar(); self.progress.setVisible(False)

        res_ly.addWidget(self.res_title)
        res_ly.addWidget(self.res_percent)
        res_ly.addSpacing(10)
        res_ly.addWidget(top3_lbl)
        res_ly.addWidget(self.top3_box)
        res_ly.addSpacing(10)
        res_ly.addWidget(rec_lbl)
        res_ly.addWidget(self.rec_text, 1)
        res_ly.addWidget(self.progress)

        work_area.addWidget(img_card, 3)
        work_area.addWidget(res_card, 0) 
        
        ly.addLayout(work_area)
        return page

    def create_history_page(self):
        page = QWidget(); ly = QVBoxLayout(page); ly.setContentsMargins(40,40,40,40)
        ly.addWidget(QLabel("Recent Diagnoses", objectName="Header"))
        self.history_list = QListWidget()
        ly.addWidget(self.history_list, 1)
        ly.addSpacing(20)
        ly.addWidget(QLabel("DISEASE PREVALENCE TRENDS", objectName="SectionTitle"))
        self.stats_container = QFrame(objectName="Card")
        self.stats_ly = QVBoxLayout(self.stats_container)
        self.stats_ly.setContentsMargins(20, 20, 20, 20)
        ly.addWidget(self.stats_container)
        btn = QPushButton("Clear History", objectName="SecondaryBtn"); btn.clicked.connect(self.clear_history)
        ly.addWidget(btn, 0, Qt.AlignRight)
        return page

    def create_settings_page(self):
        page = QWidget(); ly = QVBoxLayout(page); ly.setContentsMargins(50,50,50,50)
        ly.addWidget(QLabel("Preferences", objectName="Header"))
        card = QFrame(objectName="Card"); card_ly = QVBoxLayout(card)
        
        self.voice_check = QCheckBox("Enable Voice: Auto-read Results")
        self.voice_check.setChecked(self.settings.get("voice_enabled", True))
        self.voice_check.toggled.connect(lambda v: self.update_setting("voice_enabled", v))
        
        self.rec_voice_check = QCheckBox("Voice: Include Recommendations in Auto-read")
        self.rec_voice_check.setChecked(self.settings.get("read_recommendations", True))
        self.rec_voice_check.toggled.connect(lambda v: self.update_setting("read_recommendations", v))

        self.dark_check = QCheckBox("Enable Dark Mode")
        self.dark_check.setChecked(self.settings.get("dark_mode", True))
        self.dark_check.toggled.connect(self.toggle_dark)

        card_ly.addWidget(self.voice_check); card_ly.addSpacing(5)
        card_ly.addWidget(self.rec_voice_check); card_ly.addSpacing(10)
        card_ly.addWidget(self.dark_check)
        ly.addWidget(card); ly.addStretch()
        return page

    def create_about_page(self):
        page = QWidget(); ly = QVBoxLayout(page)
        ly.setAlignment(Qt.AlignCenter) 
        title = QLabel("LeafLens AI")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #34d399;")
        title.setAlignment(Qt.AlignCenter) 
        desc = QLabel("AI-Powered Crop Disease Detection with Active Learning\nBuilt with EfficientNet-B0 and Human-in-the-Loop Feedback")
        desc.setAlignment(Qt.AlignCenter) 
        desc.setStyleSheet("line-height: 150%; font-size: 16px; margin-top: 10px;")
        copy = QLabel("© 2026 LeafLens Inc.\nDesigned for Agricultural Efficiency")
        copy.setStyleSheet("color: #64748b; margin-top: 25px; font-size: 13px;")
        copy.setAlignment(Qt.AlignCenter) 
        ly.addWidget(title); ly.addWidget(desc); ly.addWidget(copy)
        return page

    def update_setting(self, key, value):
        self.settings[key] = value; self.save_settings()

    def toggle_dark(self, checked):
        self.settings["dark_mode"] = checked; self.save_settings()

    def resizeEvent(self, event):
        if self.current_pixmap:
            self.image_label.setPixmap(self.current_pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        super().resizeEvent(event)

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png *.jpeg)")
        if path: self.load_image(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files: self.load_image(files[0])

    def load_image(self, path):
        self.image_path = path
        self.current_pixmap = QPixmap(path)
        self.image_label.setPixmap(self.current_pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.image_label.setText("")

    def analyze_image(self):
        if not self.image_path: return
        self.progress.setVisible(True); self.progress.setRange(0, 0)
        self.thread = AnalysisThread(self.image_path, self.model, self.class_names)
        self.thread.finished.connect(self.handle_result); self.thread.start()

    def handle_result(self, res):
        self.progress.setVisible(False)
        if isinstance(res, Exception): 
            QMessageBox.warning(self, "Error", str(res)); return

        res['processed_pil'].save("processed_temp.png")
        self.current_pixmap = QPixmap("processed_temp.png")
        self.image_label.setPixmap(self.current_pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        raw_name = res['disease']
        conf = res['confidence']
        top_3 = res['top_3']
        
        # ───────────────────────────────────────────────
        # 🧪 NEW: ACTIVE LEARNING LOGIC
        # ───────────────────────────────────────────────
        if conf < UNCERTAINTY_THRESHOLD:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Format: 20260212_143005_guess_potatoearlyblight_62pct.png
            save_name = f"{ts}_guess_{raw_name.lower().replace('_','')}_{int(conf)}pct.png"
            save_path = os.path.join(FAILED_SAMPLES_DIR, save_name)
            res['processed_pil'].save(save_path)
            print(f"DEBUG: Low confidence sample saved to {save_path}")

        self.res_percent.setText(f"{conf:.1f}%")
        clean_name = raw_name.replace("___", " - ").replace("_", " ")
        self.res_title.setText(clean_name)
        
        if "healthy" in raw_name.lower(): self.res_title.setStyleSheet("color: #34d399; font-weight: bold; font-size: 20px;")
        else: self.res_title.setStyleSheet("color: #f87171; font-weight: bold; font-size: 20px;")

        top_str = ""
        for i, (n, p) in enumerate(top_3):
            n_clean = n.replace("___", " ").replace("_", " ")
            top_str += f"{i+1}. {n_clean} ({p:.1f}%)\n"
        self.top3_box.setText(top_str.strip())

        search_key = raw_name.lower().replace("_", "")
        info = DISEASE_KNOWLEDGE_BASE.get(search_key, {
            "Remedy": "Consult an expert.", "Organic": "Monitor closely.", "Action": "Isolate plant."
        })
        
        rec_html = f"""
        <p style='color:#38bdf8'><b>💊 Chemical:</b><br>{info['Remedy']}</p>
        <p style='color:#4ade80'><b>🌿 Organic:</b><br>{info['Organic']}</p>
        <p style='color:#fbbf24'><b>🚜 Action:</b><br>{info['Action']}</p>
        """
        self.rec_text.setText(rec_html)

        full_date_time = datetime.now().strftime("%d/%m/%Y %H:%M") 
        entry = {'time': full_date_time, 'name': clean_name, 'conf': f"{conf:.1f}%"}
        self.prediction_history.insert(0, entry)
        self.save_history()
        self.update_history_ui()

        if self.settings.get("voice_enabled", True):
            speech = f"Diagnosis complete. Detected {clean_name} with {int(conf)} percent confidence."
            if self.settings.get("read_recommendations", True):
                speech += f" Recommendations: Chemical remedy is {info['Remedy']}. Organic remedy is {info['Organic']}."
            self.tts.stop()
            self.tts.say(self.clean_text_for_speech(speech))

    def update_history_ui(self):
        self.history_list.clear()
        for h in self.prediction_history:
            self.history_list.addItem(f"[{h.get('time')}] {h.get('name')} - {h.get('conf')}")

        for i in reversed(range(self.stats_ly.count())): 
            item = self.stats_ly.itemAt(i)
            if item.widget(): item.widget().setParent(None)

        if not self.prediction_history:
            self.stats_ly.addWidget(QLabel("No data yet."))
            return

        counts = {}
        for h in self.prediction_history:
            name = h['name']
            counts[name] = counts.get(name, 0) + 1
        
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        total = len(self.prediction_history)

        for name, count in sorted_counts:
            percent = (count / total) * 100
            lbl_row = QHBoxLayout()
            lbl_row.addWidget(QLabel(name)); lbl_row.addStretch(); lbl_row.addWidget(QLabel(f"{int(percent)}%"))
            bar = QProgressBar(); bar.setFixedHeight(8); bar.setRange(0, 100); bar.setValue(int(percent)); bar.setTextVisible(False)
            self.stats_ly.addLayout(lbl_row); self.stats_ly.addWidget(bar); self.stats_ly.addSpacing(10)

    def clear_history(self):
        self.prediction_history = []
        self.update_history_ui()
        if os.path.exists(HISTORY_PATH): os.remove(HISTORY_PATH)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LeafLens(); win.show(); sys.exit(app.exec())
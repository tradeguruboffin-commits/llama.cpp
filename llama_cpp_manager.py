#!/usr/bin/env python3
import sys
import subprocess
import threading
import os
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QFileDialog,
    QMessageBox, QProgressBar, QInputDialog, QFormLayout, QDialog, QDialogButtonBox
)

# llama.cpp bin পাথ — তোমার নতুন স্ট্রাকচার অনুযায়ী (llama_cpp_gui ফোল্ডারে থাকলে)
LLAMA_BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "llama.cpp", "build", "bin"))
CONVERT_SCRIPT = os.path.join(LLAMA_BIN_DIR, "convert-hf-to-gguf.py")

class ArgsDialog(QDialog):
    def __init__(self, title, fields, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        layout = QFormLayout(self)
        self.inputs = {}
        for label, default in fields:
            edit = QLineEdit(default)
            layout.addRow(label + ":", edit)
            self.inputs[label] = edit
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_args(self):
        return [f"--{k.replace(' ', '-').lower()} {v.text().strip()}" for k, v in self.inputs.items() if v.text().strip()]

class LlamaCppManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦙 llama.cpp Tools Manager")
        self.resize(1400, 800)
        self.server_process = None
        self.cli_process = None
        self.init_ui()
        self.log(f"llama.cpp bin directory: {LLAMA_BIN_DIR}")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(40)
        layout.setContentsMargins(40, 40, 40, 40)

        left = QVBoxLayout()
        left.setSpacing(35)

        title = QLabel("🦙 llama.cpp Local Tools")
        title.setStyleSheet("font-size: 38px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        left.addWidget(title)

        # Server
        server_layout = QHBoxLayout()
        self.server_btn = QPushButton("🖥️ Start llama-server")
        self.server_btn.clicked.connect(self.toggle_server)
        self.server_btn.setStyleSheet("padding: 22px; font-size: 32px;")
        self.server_btn.setMinimumHeight(90)
        server_layout.addWidget(self.server_btn)
        self.server_args_btn = QPushButton("⚙️ Args")
        self.server_args_btn.clicked.connect(self.server_advanced_args)
        self.server_args_btn.setStyleSheet("padding: 22px; font-size: 28px;")
        self.server_args_btn.setMinimumHeight(90)
        server_layout.addWidget(self.server_args_btn)
        left.addLayout(server_layout)

        # Interactive Chat
        chat_layout = QHBoxLayout()
        self.chat_btn = QPushButton("💬 Interactive Chat (llama-cli)")
        self.chat_btn.clicked.connect(self.run_interactive_chat)
        self.chat_btn.setStyleSheet("padding: 22px; font-size: 32px;")
        self.chat_btn.setMinimumHeight(90)
        chat_layout.addWidget(self.chat_btn)
        self.chat_args_btn = QPushButton("⚙️ Args")
        self.chat_args_btn.clicked.connect(self.chat_advanced_args)
        self.chat_args_btn.setStyleSheet("padding: 22px; font-size: 28px;")
        self.chat_args_btn.setMinimumHeight(90)
        chat_layout.addWidget(self.chat_args_btn)
        left.addLayout(chat_layout)

        # Quantize
        quant_layout = QHBoxLayout()
        self.quant_input = QLineEdit()
        self.quant_input.setPlaceholderText("Select GGUF file for quantization...")
        self.quant_input.setStyleSheet("font-size: 30px; padding: 18px;")
        quant_layout.addWidget(self.quant_input, 3)
        browse_quant = QPushButton("📂")
        browse_quant.clicked.connect(self.browse_quantize)
        browse_quant.setMinimumWidth(100)
        quant_layout.addWidget(browse_quant, 1)
        left.addLayout(quant_layout)

        quant_btn_layout = QHBoxLayout()
        self.quant_btn = QPushButton("⚡ Quantize GGUF")
        self.quant_btn.clicked.connect(self.quantize)
        self.quant_btn.setStyleSheet("padding: 22px; font-size: 32px;")
        self.quant_btn.setMinimumHeight(90)
        quant_btn_layout.addWidget(self.quant_btn)
        self.quant_args_btn = QPushButton("⚙️ Args")
        self.quant_args_btn.clicked.connect(self.quant_advanced_args)
        self.quant_args_btn.setStyleSheet("padding: 22px; font-size: 28px;")
        self.quant_args_btn.setMinimumHeight(90)
        quant_btn_layout.addWidget(self.quant_args_btn)
        left.addLayout(quant_btn_layout)

        # Convert HF to GGUF
        convert_layout = QHBoxLayout()
        self.convert_input = QLineEdit()
        self.convert_input.setPlaceholderText("Select HuggingFace model folder...")
        self.convert_input.setStyleSheet("font-size: 30px; padding: 18px;")
        convert_layout.addWidget(self.convert_input, 3)
        browse_convert = QPushButton("📂")
        browse_convert.clicked.connect(self.browse_convert)
        browse_convert.setMinimumWidth(100)
        convert_layout.addWidget(browse_convert, 1)
        left.addLayout(convert_layout)

        convert_btn_layout = QHBoxLayout()
        self.convert_btn = QPushButton("🔄 Convert HF → GGUF")
        self.convert_btn.clicked.connect(self.convert_hf_to_gguf)
        self.convert_btn.setStyleSheet("padding: 22px; font-size: 32px;")
        self.convert_btn.setMinimumHeight(90)
        convert_btn_layout.addWidget(self.convert_btn)
        self.convert_args_btn = QPushButton("⚙️ Args")
        self.convert_args_btn.clicked.connect(self.convert_advanced_args)
        self.convert_args_btn.setStyleSheet("padding: 22px; font-size: 28px;")
        self.convert_args_btn.setMinimumHeight(90)
        convert_btn_layout.addWidget(self.convert_args_btn)
        left.addLayout(convert_btn_layout)

        left.addStretch()

        # Right - Log
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>📜 Log & Output</b>"), alignment=Qt.AlignCenter)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        right.addWidget(self.log_area, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMinimumHeight(70)
        right.addWidget(self.progress)

        # Assemble
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(650)
        layout.addWidget(left_widget)
        layout.addLayout(right, 1)

        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #0d1117; color: #c9d1d9; }
            QLabel { color: #c9d1d9; font-size: 32px; font-weight: bold; }
            QPushButton {
                background: #21262d; color: #c9d1d9; border: 2px solid #30363d;
                padding: 22px; border-radius: 18px; font-size: 32px;
            }
            QPushButton:hover { background: #30363d; }
            QPushButton:pressed { background: #444c56; }
            QLineEdit {
                background: #161b22; color: #c9d1d9; border: 2px solid #30363d;
                padding: 20px; border-radius: 15px; font-size: 30px;
            }
            QTextEdit {
                background: #0d1117; color: #58a6ff;
                font-family: Consolas, monospace; font-size: 28px; padding: 25px;
            }
            QProgressBar {
                border: 2px solid #30363d; border-radius: 15px; background: #161b22;
                text-align: center; font-size: 28px; min-height: 70px;
            }
            QProgressBar::chunk { background: #238636; }
        """)

    def log(self, text):
        self.log_area.append(text)
        self.log_area.ensureCursorVisible()

    # ================= Server =================
    def toggle_server(self):
        if self.server_process and self.server_process.state() == QProcess.Running:
            self.server_process.terminate()
            self.server_process = None
            self.server_btn.setText("🖥️ Start llama-server")
            self.log("🛑 llama-server stopped.")
        else:
            model_path, ok = QFileDialog.getOpenFileName(self, "Select GGUF Model", "", "GGUF Files (*.gguf)")
            if not ok or not model_path:
                return

            bin_path = os.path.join(LLAMA_BIN_DIR, "llama-server")
            if not os.path.exists(bin_path):
                QMessageBox.critical(self, "Error", f"llama-server not found!\n{bin_path}")
                return

            args = self.server_extra_args if hasattr(self, 'server_extra_args') else []
            cmd = [bin_path, "-m", model_path] + args

            self.server_process = QProcess()
            self.server_process.readyReadStandardOutput.connect(self.handle_server_output)
            self.server_process.readyReadStandardError.connect(self.handle_server_output)
            self.server_process.finished.connect(lambda: self.server_btn.setText("🖥️ Start llama-server"))

            self.server_btn.setText("🛑 Stop llama-server")
            self.log(f"Starting server: {' '.join(cmd)}")
            self.server_process.start(cmd[0], cmd[1:])

    def handle_server_output(self):
        data = self.server_process.readAllStandardOutput().data().decode() + \
               self.server_process.readAllStandardError().data().decode()
        self.log(data.strip())

    def server_advanced_args(self):
        fields = [
            ("Port", "8080"),
            ("Host", "127.0.0.1"),
            ("GPU Layers", "33"),
            ("Context Size", "8192"),
            ("Threads", "8")
        ]
        dialog = ArgsDialog("llama-server Advanced Args", fields, self)
        if dialog.exec_():
            self.server_extra_args = dialog.get_args()
            self.log(f"Server args set: {' '.join(self.server_extra_args)}")

    # ================= CLI Chat =================
    def run_interactive_chat(self):
        model_path, ok = QFileDialog.getOpenFileName(self, "Select GGUF Model", "", "GGUF Files (*.gguf)")
        if not ok or not model_path:
            return

        bin_path = os.path.join(LLAMA_BIN_DIR, "llama-cli")
        if not os.path.exists(bin_path):
            QMessageBox.critical(self, "Error", f"llama-cli not found!\n{bin_path}")
            return

        args = self.chat_extra_args if hasattr(self, 'chat_extra_args') else []
        cmd = [bin_path, "-m", model_path, "--interactive", "--color"] + args

        self.log(f"Launching CLI chat: {' '.join(cmd)}")
        subprocess.Popen(cmd)

    def chat_advanced_args(self):
        fields = [
            ("Temperature", "0.8"),
            ("Top P", "0.9"),
            ("Context Size", "8192"),
            ("GPU Layers", "33"),
            ("Repeat Penalty", "1.1")
        ]
        dialog = ArgsDialog("llama-cli Advanced Args", fields, self)
        if dialog.exec_():
            self.chat_extra_args = dialog.get_args()
            self.log(f"CLI args set: {' '.join(self.chat_extra_args)}")

    # ================= Quantize =================
    def browse_quantize(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF Files (*.gguf)")
        if path:
            self.quant_input.setText(path)

    def quantize(self):
        infile = self.quant_input.text().strip()
        if not infile or not os.path.exists(infile):
            QMessageBox.warning(self, "Error", "Select valid GGUF!")
            return

        outfile, _ = QFileDialog.getSaveFileName(self, "Save Quantized", "", "GGUF Files (*.gguf)")
        if not outfile:
            return

        types = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16"]
        qtype, ok = QInputDialog.getItem(self, "Quant Type", "Select:", types, 0, False)
        if not ok:
            return

        bin_path = os.path.join(LLAMA_BIN_DIR, "llama-quantize")
        if not os.path.exists(bin_path):
            QMessageBox.critical(self, "Error", f"llama-quantize not found!")
            return

        args = self.quant_extra_args if hasattr(self, 'quant_extra_args') else []
        cmd = [bin_path, infile, outfile, qtype] + args

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log(f"Quantizing: {' '.join(cmd)}")

        def run():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.log(line.strip())
                proc.wait()
                self.log(f"✅ Quantized: {os.path.basename(outfile)}")
            except Exception as e:
                self.log(f"❌ {e}")
            finally:
                self.progress.setVisible(False)

        threading.Thread(target=run, daemon=True).start()

    def quant_advanced_args(self):
        fields = [("Threads", "8"), ("Allow Requantize", "")]
        dialog = ArgsDialog("Quantize Extra Args", fields, self)
        if dialog.exec_():
            self.quant_extra_args = dialog.get_args()
            self.log(f"Quant args set: {' '.join(self.quant_extra_args)}")

    # ================= Convert HF → GGUF =================
    def browse_convert(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select HF Model Folder")
        if dir_path:
            self.convert_input.setText(dir_path)

    def convert_hf_to_gguf(self):
        hf_dir = self.convert_input.text().strip()
        if not hf_dir or not os.path.isdir(hf_dir):
            QMessageBox.warning(self, "Error", "Select valid folder!")
            return

        outfile, _ = QFileDialog.getSaveFileName(self, "Save GGUF", "", "GGUF Files (*.gguf)")
        if not outfile:
            return

        if not os.path.exists(CONVERT_SCRIPT):
            QMessageBox.critical(self, "Error", f"convert-hf-to-gguf.py not found!")
            return

        outtypes = ["f16", "q8_0", "f32"]
        outtype, ok = QInputDialog.getItem(self, "Output Type", "Select:", outtypes, 0, False)
        if not ok:
            return

        args = self.convert_extra_args if hasattr(self, 'convert_extra_args') else []
        cmd = [sys.executable, CONVERT_SCRIPT, hf_dir, "--outfile", outfile, "--outtype", outtype] + args

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log(f"Converting: {' '.join(cmd)}")

        def run():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=LLAMA_BIN_DIR)
                for line in proc.stdout:
                    self.log(line.strip())
                proc.wait()
                self.log(f"✅ Converted: {os.path.basename(outfile)}")
            except Exception as e:
                self.log(f"❌ {e}")
            finally:
                self.progress.setVisible(False)

        threading.Thread(target=run, daemon=True).start()

    def convert_advanced_args(self):
        fields = [("Vocab Type", ""), ("Concurrency", "8")]
        dialog = ArgsDialog("Convert Extra Args", fields, self)
        if dialog.exec_():
            self.convert_extra_args = dialog.get_args()
            self.log(f"Convert args set: {' '.join(self.convert_extra_args)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LlamaCppManager()
    win.show()
    sys.exit(app.exec_())

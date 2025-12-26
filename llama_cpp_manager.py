#!/usr/bin/env python3
import os
import subprocess
import shutil
import tkinter as tk
from tkinter import font, ttk, filedialog, messagebox, scrolledtext
import threading

# ---------------- Converter Manager (Pro Version) ---------------- #

class ConverterManager(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("llama.cpp Converter Manager (Pro)")
        self.geometry("1500x950")
        self.minsize(1200, 800)

        style = ttk.Style(self)
        style.theme_use('clam')

        default_font = tk.font.nametofont("TkDefaultFont")
        default_font.configure(size=30, family="DejaVu Sans")
        self.option_add("*Font", default_font)

        self.output_font = tk.font.Font(family="DejaVu Sans Mono", size=26)

        style.configure("TButton", padding=20)
        style.configure("TFrame", padding=20)

        self.llama_dir = tk.StringVar()
        self.script = tk.StringVar()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.extra_args = tk.StringVar()
        self.dry_run = tk.BooleanVar(value=False)

        # LoRA-র জন্য বেস মডেল
        self.base_model_path = tk.StringVar()

        self.create_ui()

    def create_ui(self):
        frm = ttk.Frame(self, padding=20)
        frm.pack(fill="both", expand=True)

        # llama.cpp dir
        ttk.Label(frm, text="llama.cpp Directory").grid(row=0, column=0, sticky="w", pady=10)
        ttk.Entry(frm, textvariable=self.llama_dir, width=60).grid(row=0, column=1, padx=10, pady=10)
        ttk.Button(frm, text="Browse", command=self.browse_llama).grid(row=0, column=2, padx=10, pady=10)

        # script selection
        ttk.Label(frm, text="Convert Script").grid(row=1, column=0, sticky="w", pady=10)
        scripts = [
            "convert_hf_to_gguf.py",
            "convert_llama_ggml_to_gguf.py",
            "convert_lora_to_gguf.py"
        ]
        self.script_cb = ttk.Combobox(frm, values=scripts, textvariable=self.script, width=57, state="readonly")
        self.script_cb.grid(row=1, column=1, padx=10, pady=10)
        self.script_cb.current(0)
        self.script_cb.bind("<<ComboboxSelected>>", self.on_script_change)

        # input
        ttk.Label(frm, text="Input Model / Path").grid(row=2, column=0, sticky="w", pady=10)
        ttk.Entry(frm, textvariable=self.input_path, width=60).grid(row=2, column=1, padx=10, pady=10)
        ttk.Button(frm, text="Browse", command=self.browse_input).grid(row=2, column=2, padx=10, pady=10)

        # LoRA base model (শুধু LoRA সিলেক্ট করলে দেখাবে)
        self.base_row = 3
        self.base_label = ttk.Label(frm, text="Base Model (for LoRA)", foreground="red")
        self.base_entry = ttk.Entry(frm, textvariable=self.base_model_path, width=60)
        self.base_btn = ttk.Button(frm, text="Browse", command=self.browse_base)

        # output
        ttk.Label(frm, text="Output File / Dir").grid(row=4, column=0, sticky="w", pady=10)
        ttk.Entry(frm, textvariable=self.output_path, width=60).grid(row=4, column=1, padx=10, pady=10)
        ttk.Button(frm, text="Browse", command=self.browse_output).grid(row=4, column=2, padx=10, pady=10)

        # extra args
        ttk.Label(frm, text="Extra Arguments").grid(row=5, column=0, sticky="w", pady=10)
        ttk.Entry(frm, textvariable=self.extra_args, width=60).grid(row=5, column=1, columnspan=2, padx=10, pady=10, sticky="ew")

        # Dry run checkbox
        ttk.Checkbutton(frm, text="Show command only (Dry Run)", variable=self.dry_run, padding=10).grid(row=6, column=0, columnspan=3, pady=15)

        # run button
        ttk.Button(frm, text="Run Conversion 🚀", command=self.run).grid(row=7, column=0, columnspan=3, pady=30)

        # output box
        output_frame = ttk.Frame(frm)
        output_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=10)
        self.output = tk.Text(
            output_frame,
            font=self.output_font,
            bg="#1e1e1e",
            fg="#00ff88",
            insertbackground="#00ff88",
            relief="flat",
            padx=15,
            pady=15
        )
        self.output.pack(fill="both", expand=True)

        frm.rowconfigure(8, weight=1)
        frm.columnconfigure(1, weight=1)

        # প্রথমবার স্ক্রিপ্ট চেঞ্জ ট্রিগার
        self.on_script_change(None)

    def on_script_change(self, event):
        script = self.script.get()
        self.extra_args.set("")  # রিসেট

        # LoRA-র জন্য বেস মডেল ফিল্ড
        if script == "convert_lora_to_gguf.py":
            self.base_label.grid(row=self.base_row, column=0, sticky="w", pady=10)
            self.base_entry.grid(row=self.base_row, column=1, padx=10, pady=10)
            self.base_btn.grid(row=self.base_row, column=2, padx=10, pady=10)
            self.log("⚠️ LoRA conversion requires base model GGUF!\n")
        else:
            self.base_label.grid_forget()
            self.base_entry.grid_forget()
            self.base_btn.grid_forget()

        # অটো প্রিসেট
        if script == "convert_hf_to_gguf.py":
            self.extra_args.set("--outtype f16")
            self.log("ℹ️ Auto-filled: --outtype f16 (recommended for HF)\n")
        elif script == "convert_llama_ggml_to_gguf.py":
            self.extra_args.set("--outtype f16")
            self.log("⚠️ LEGACY GGML → GGUF (use f16)\n")
        elif script == "convert_lora_to_gguf.py":
            self.extra_args.set("--outtype q8_0")

    def browse_llama(self):
        d = filedialog.askdirectory(title="Select llama.cpp directory")
        if d: self.llama_dir.set(d)

    def browse_input(self):
        d = filedialog.askdirectory(title="Select input model/path")
        if d: self.input_path.set(d)

    def browse_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d: self.output_path.set(d)

    def browse_base(self):
        p = filedialog.askopenfilename(title="Select base model GGUF", filetypes=[("GGUF", "*.gguf")])
        if p: self.base_model_path.set(p)

    def log(self, text):
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.update_idletasks()

    def run(self):
        if not self.llama_dir.get():
            messagebox.showerror("Error", "llama.cpp directory missing", parent=self)
            return

        script_path = os.path.join(self.llama_dir.get(), self.script.get())
        if not os.path.isfile(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}", parent=self)
            return

        cmd = ["python3", script_path, self.input_path.get()]

        if self.output_path.get():
            cmd += ["--outfile", self.output_path.get()]

        # LoRA-র জন্য বেস মডেল
        if self.script.get() == "convert_lora_to_gguf.py":
            if not self.base_model_path.get():
                messagebox.showerror("Error", "Base model required for LoRA conversion!", parent=self)
                return
            cmd += ["--base", self.base_model_path.get()]

        if self.extra_args.get().strip():
            cmd += self.extra_args.get().split()

        full_cmd = " ".join(cmd)
        self.log(f"\n▶ Command:\n{full_cmd}\n\n")

        if self.dry_run.get():
            self.log("🛠️ Dry Run mode – command shown only (not executed)\n")
            return

        threading.Thread(target=self.execute, args=(cmd,), daemon=True).start()

    def execute(self, cmd):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in process.stdout:
                self.log(line)
            process.wait()
            if process.returncode == 0:
                self.log("\n✅ Conversion completed successfully!\n")
            else:
                self.log(f"\n⚠️ Exited with code {process.returncode}\n")
        except Exception as e:
            self.log(f"\n❌ Error: {e}\n")

# ---------------- মেইন অ্যাপ (উন্নত Quantize ড্রপডাউন সহ) ---------------- #

APP = "llama.cpp SAFE Tools (Termux / proot)"

def which(cmd):
    return shutil.which(cmd)

def detect_terminal():
    for t in ["xfce4-terminal", "gnome-terminal", "xterm"]:
        if which(t):
            return t
    return None

TERMINAL = detect_terminal()

def run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

def supports(flag, exe):
    try:
        return flag in run([exe, "--help"])
    except:
        return False

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP)
        self.geometry("1700x900")
        self.minsize(1280, 720)

        style = ttk.Style()
        style.theme_use('clam')

        default_font = tk.font.nametofont("TkDefaultFont")
        default_font.configure(size=30, family="DejaVu Sans")
        self.option_add("*Font", default_font)

        self.log_font = tk.font.Font(family="DejaVu Sans Mono", size=26)

        style.configure("TButton", padding=20)
        style.configure("TNotebook", padding=10)
        style.configure("TNotebook.Tab", padding=(20, 10))
        style.configure("TFrame", padding=20)

        self.bin = ""
        self.model = ""
        self.gguf = ""

        self.build_ui()

        # Convert Tools বাটন
        convert_btn = ttk.Button(self, text="Convert Tools", command=self.open_converter)
        convert_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

    def open_converter(self):
        ConverterManager(self)

    def log(self, msg):
        self.logbox.insert(tk.END, msg + "\n")
        self.logbox.see(tk.END)

    def build_ui(self):
        tabs = ttk.Notebook(self)
        tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chat_tab = ttk.Frame(tabs)
        self.quant_tab = ttk.Frame(tabs)

        tabs.add(self.chat_tab, text="Chat (SAFE)")
        tabs.add(self.quant_tab, text="Quantize")

        self.build_chat()
        self.build_quant()

    def build_chat(self):
        f = ttk.Frame(self.chat_tab)
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Button(f, text="Select llama.cpp build/bin", command=self.pick_bin).pack(fill=tk.X, pady=10)
        ttk.Button(f, text="Select GGUF model", command=self.pick_model).pack(fill=tk.X, pady=10)

        args = ttk.Frame(f)
        args.pack(fill=tk.X, pady=20)

        self.template = tk.StringVar(value="chatml")
        self.ctx = tk.StringVar(value="2048")
        self.threads = tk.StringVar(value="2")

        ttk.Label(args, text="Chat template").grid(row=0, column=0, padx=15, sticky="e")
        ttk.Entry(args, textvariable=self.template, width=12).grid(row=0, column=1, padx=10)

        ttk.Label(args, text="CTX").grid(row=0, column=2, padx=15, sticky="e")
        ttk.Entry(args, textvariable=self.ctx, width=8).grid(row=0, column=3, padx=10)

        ttk.Label(args, text="Threads").grid(row=0, column=4, padx=15, sticky="e")
        ttk.Entry(args, textvariable=self.threads, width=8).grid(row=0, column=5, padx=10)

        ttk.Button(f, text="▶ Interactive Chat (FIXED)", command=self.run_chat).pack(fill=tk.X, pady=20)

        self.logbox = scrolledtext.ScrolledText(f, font=self.log_font, bg="#1e1e1e", fg="#00ff88")
        self.logbox.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        if TERMINAL:
            self.log(f"✔ terminal: {TERMINAL}")
        else:
            self.log("❌ no terminal detected")

    def build_quant(self):
        f = ttk.Frame(self.quant_tab)
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Button(f, text="Select GGUF file", command=self.pick_gguf).pack(fill=tk.X, pady=15)

        qframe = ttk.Frame(f)
        qframe.pack(fill=tk.X, pady=15)
        ttk.Label(qframe, text="Quant type:").pack(side=tk.LEFT, padx=10)

        quant_types = [
            "q2_K", "q3_K_S", "q3_K_M", "q4_0", "q4_K_S", "q4_K_M",
            "q5_K_S", "q5_K_M", "q6_K", "q8_0", "f16"
        ]
        self.qtype_cb = ttk.Combobox(qframe, values=quant_types, width=12, state="readonly")
        self.qtype_cb.pack(side=tk.LEFT, padx=10)
        self.qtype_cb.current(quant_types.index("q4_K_M"))  # ডিফল্ট ভালো কোয়ালিটি

        ttk.Button(f, text="▶ Quantize", command=self.quantize).pack(fill=tk.X, pady=30)

    def pick_bin(self):
        d = filedialog.askdirectory()
        if d and os.path.exists(os.path.join(d, "llama-cli")):
            self.bin = d
            self.log(f"✔ bin set: {d}")
        else:
            messagebox.showerror("Error", "llama-cli not found")

    def pick_model(self):
        p = filedialog.askopenfilename(filetypes=[("GGUF", "*.gguf")])
        if p:
            self.model = p
            self.log(f"✔ model: {p}")

    def pick_gguf(self):
        p = filedialog.askopenfilename(filetypes=[("GGUF", "*.gguf")])
        if p:
            self.gguf = p
            self.log(f"✔ gguf: {p}")

    def run_chat(self):
        if not self.bin or not self.model:
            messagebox.showerror("Error", "select bin + model")
            return

        cli = os.path.join(self.bin, "llama-cli")
        flags = [
            f"-m \"{self.model}\"",
            f"--ctx-size {self.ctx.get()}",
            f"--threads {self.threads.get()}",
        ]

        if supports("--chat-template", cli):
            flags.append(f"--chat-template {self.template.get()}")
            self.log("✔ chat-template enabled")

        if supports("--interactive-first", cli):
            flags.append("--interactive-first")
            self.log("✔ interactive-first enabled")

        cmd = f"{cli} " + " ".join(flags)
        self.log("▶ " + cmd)

        subprocess.Popen([
            TERMINAL,
            "--command",
            f"bash -c \"{cmd}; echo; echo '--- exited ---'; read -n1\""
        ])

    def quantize(self):
        if not self.bin or not self.gguf:
            messagebox.showerror("Error", "select bin + gguf file")
            return

        qtype = self.qtype_cb.get()
        if not qtype:
            messagebox.showerror("Error", "Select quant type!")
            return

        exe = os.path.join(self.bin, "llama-quantize")
        out = self.gguf.replace(".gguf", f"-{qtype}.gguf")
        cmd = f"{exe} \"{self.gguf}\" \"{out}\" {qtype}"
        self.log("▶ " + cmd)

        subprocess.Popen([TERMINAL, "--command", f"bash -c \"{cmd}; read -n1\""])

if __name__ == "__main__":
    App().mainloop()

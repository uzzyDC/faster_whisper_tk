# -*- coding: utf-8 -*-
r"""
Trascrizione batch con faster-whisper + Tkinter.
Avvio consigliato con il Python del proprio virtualenv:
    C:\git\_whisperEnv\Scripts\python.exe faster_whisper_tk.py
"""

import os
import threading
import queue
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import os
import sys
from pathlib import Path

_NV = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
_dll_handles = []

if _NV.is_dir():
    _dll_dirs = [p for p in _NV.glob("*/bin") if p.is_dir()]
    for _dll_dir in _dll_dirs:
        _dll_handles.append(os.add_dll_directory(str(_dll_dir)))

    os.environ["PATH"] = (
        os.pathsep.join(str(p) for p in _dll_dirs)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

print("Python in uso:", sys.executable)

# Estensioni audio/video decodificabili da PyAV
MEDIA_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".m4b", ".mp3", ".ogg",
    ".opus", ".wav", ".wma", ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".mpeg", ".mpg", ".m4v", ".wmv", ".3gp",
}

MODEL_PRESETS = [
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "medium",
    "small",
    "base",
    "tiny",
    "distil-large-v3",
]


def srt_timestamp(seconds):
    """Formatta secondi come HH:MM:SS,mmm."""
    millis_total = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(millis_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def safe_stem(path):
    # Evita collisioni con nomi file uguali in sottocartelle: il chiamante
    # gestisce la collisione aggiungendo un suffisso se necessario.
    return Path(path).stem


class FasterWhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Faster-Whisper — trascrizione batch")
        self.root.geometry("820x690")
        self.root.minsize(720, 580)

        self.messages = queue.Queue()
        self.worker = None
        self.cancel_event = threading.Event()
        self.stop_current_event = threading.Event()
        self.skip_current_event = threading.Event()

        self.input_var = tk.StringVar()
        self.input_mode = tk.StringVar(value="cartella")
        self.output_var = tk.StringVar()
        self.model_var = tk.StringVar(value="large-v2")
        self.custom_model_var = tk.StringVar()
        self.device_var = tk.StringVar(value="cuda")
        self.compute_var = tk.StringVar(value="float16")
        self.language_var = tk.StringVar(value="it")
        self.beam_var = tk.StringVar(value="5")
        self.vad_var = tk.BooleanVar(value=True)
        self.prompt_var = tk.StringVar(
            value="Ciao, questo è un incontro di Dong Sheng. Benvenuto!"
        )
        self.recursive_var = tk.BooleanVar(value=False)
        self.txt_var = tk.BooleanVar(value=True)
        self.srt_var = tk.BooleanVar(value=True)
        self.word_ts_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.after(100, self._poll_messages)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(11, weight=1)
        main.rowconfigure(9, weight=1)

        ttk.Label(main, text="Input audio/video").grid(
            row=0, column=0, sticky="w", pady=4
        )

        input_frame = ttk.Frame(main)
        input_frame.grid(row=0, column=1, columnspan=2, sticky="ew", padx=8)
        input_frame.columnconfigure(0, weight=1)
        ttk.Entry(input_frame, textvariable=self.input_var).grid( row=0, column=0, sticky="ew" )
        ttk.Button( input_frame, text="Cartella…", command=self._choose_input ).grid(row=0, column=1, padx=(5, 0))
        ttk.Button( input_frame, text="File…", command=self._choose_file ).grid(row=0, column=2, padx=(5, 0))

        # Cartella output
        ttk.Label(main, text="Cartella output").grid( row=1, column=0, sticky="w", pady=4 )
        ttk.Entry( main, textvariable=self.output_var ).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button( main, text="Sfoglia…", command=self._choose_output ).grid(row=1, column=2)

        ttk.Label(main, text="Modello").grid(row=2, column=0, sticky="w", pady=4)
        model_frame = ttk.Frame(main)
        model_frame.grid(row=2, column=1, columnspan=2, sticky="ew", padx=8)
        model_frame.columnconfigure(1, weight=1)
        ttk.Combobox( model_frame, textvariable=self.model_var, values=MODEL_PRESETS, state="readonly", width=22 ).grid(row=0, column=0, sticky="w")
        ttk.Label(model_frame, text="ID Hugging Face o percorso locale (facoltativo):").grid( row=0, column=1, sticky="e", padx=(12, 5) )
        ttk.Entry(model_frame, textvariable=self.custom_model_var).grid( row=0, column=2, sticky="ew" )
        model_frame.columnconfigure(2, weight=1)

        ttk.Label(main, text="Dispositivo").grid(row=3, column=0, sticky="w", pady=4)
        device_frame = ttk.Frame(main)
        device_frame.grid(row=3, column=1, columnspan=2, sticky="w", padx=8)
        ttk.Combobox( device_frame, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=10 ).pack(side="left")
        ttk.Label(device_frame, text="Precisione").pack(side="left", padx=(18, 6))
        ttk.Combobox( device_frame, textvariable=self.compute_var, values=["float16", "int8_float16", "int8", "float32"], state="readonly", width=15 ).pack(side="left")
        ttk.Label(device_frame, text="Lingua (it, en, fr…; vuoto = rilevamento)").pack( side="left", padx=(18, 5) )
        ttk.Entry(device_frame, textvariable=self.language_var, width=8).pack(side="left")

        ttk.Label(main, text="Beam size").grid(row=4, column=0, sticky="w", pady=4)
        opts = ttk.Frame(main)
        opts.grid(row=4, column=1, columnspan=2, sticky="w", padx=8)
        ttk.Entry(opts, textvariable=self.beam_var, width=6).pack(side="left")
        ttk.Checkbutton(opts, text="Filtro VAD (riduce i silenzi)", variable=self.vad_var).pack( side="left", padx=(14, 0) )
        ttk.Checkbutton(opts, text="Cerca anche nelle sottocartelle", variable=self.recursive_var).pack( side="left", padx=(14, 0) )
        ttk.Label(main, text="Prompt iniziale (facoltativo)").grid( row=5, column=0, sticky="w", pady=4 )
        ttk.Entry(main, textvariable=self.prompt_var).grid( row=5, column=1, columnspan=2, sticky="ew", padx=8 )

        output_opts = ttk.Frame(main)
        output_opts.grid(row=6, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Label(output_opts, text="Crea:").pack(side="left")
        ttk.Checkbutton(output_opts, text=".srt", variable=self.srt_var).pack(side="left", padx=5)
        ttk.Checkbutton(output_opts, text=".txt", variable=self.txt_var).pack(side="left", padx=5)
        ttk.Checkbutton( output_opts, text="Timestamp per parola (più lento)", variable=self.word_ts_var ).pack(side="left", padx=14)

        buttons = ttk.Frame(main)
        buttons.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(6, 4))

        self.start_btn = ttk.Button( buttons, text="Avvia trascrizione", command=self._start )
        self.start_btn.pack(side="left")

        self.stop_btn = ttk.Button( buttons, text="Transcribe & Stop", command=self._request_stop_current, state="disabled" )
        self.stop_btn.pack(side="left", padx=(8, 0))

        self.skip_btn = ttk.Button( buttons, text="Transcribe & Skip", command=self._request_skip_current, state="disabled" )
        self.skip_btn.pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(buttons, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(12, 0))

        # Avanzamento del file corrente
        self.current_progress = ttk.Progressbar( main, mode="determinate", maximum=100 )
        self.current_progress.grid( row=8, column=0, columnspan=3, sticky="ew", pady=(4, 4) )

        self.current_progress_var = tk.StringVar(value="File corrente: 0%")
        ttk.Label(main, textvariable=self.current_progress_var).grid( row=9, column=0, columnspan=3, sticky="w" )

        self.status_var = tk.StringVar(value="Pronto.")
        ttk.Label(main, textvariable=self.status_var).grid( row=10, column=0, columnspan=3, sticky="w", pady=(2, 6) )

        self.log = ScrolledText(main, height=16, wrap="word", state="disabled")
        self.log.grid(row=11, column=0, columnspan=3, sticky="nsew")

        ttk.Label( main, text=("Nota: gli ID Hugging Face personalizzati devono indicare un modello " "già convertito in formato CTranslate2, non il normale checkpoint Transformers."), wraplength=780
            ).grid(row=12, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _choose_input(self):
        path = filedialog.askdirectory(title="Scegli la cartella con i file audio/video")
        if path:
            self.input_var.set(path)
            if not self.output_var.get():
                self.output_var.set(str(Path(path) / "subs"))

    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="Scegli un file audio/video",
            filetypes=[
                ("Audio e video", "*.aac *.aif *.aiff *.flac *.m4a *.m4b "
                "*.mp3 *.ogg *.opus *.wav *.wma *.mp4 *.mkv *.mov "
                "*.avi *.webm *.mpeg *.mpg *.m4v *.wmv *.3gp"),
                ("Tutti i file", "*.*")
            ]
        )
        if path:
            self.input_var.set(path)

    def _choose_output(self):
        path = filedialog.askdirectory(title="Scegli la cartella dei sottotitoli")
        if path:
            self.output_var.set(path)

    def _log(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self):
        input_path = Path(self.input_var.get().strip().strip('"'))
        output_dir = Path(self.output_var.get().strip().strip('"'))

        if input_path.is_file():
            if input_path.suffix.lower() not in MEDIA_EXTENSIONS:
                messagebox.showerror("Input non valido", "Formato audio/video non supportato.")
                return
            files = [input_path]
        elif input_path.is_dir():
            files_iter = input_path.rglob("*") if self.recursive_var.get() else input_path.iterdir()
            files = sorted( p for p in files_iter if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS )
        else:
            messagebox.showerror("Input non valido", "Scegli un file o una cartella esistente.")
            return
        if not str(output_dir):
            messagebox.showerror("Output non valido", "Scegli una cartella output.")
            return
        if not (self.srt_var.get() or self.txt_var.get()):
            messagebox.showerror("Output non selezionato", "Seleziona almeno .srt o .txt.")
            return
        try:
            beam_size = int(self.beam_var.get())
            if beam_size < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Beam size non valido", "Inserisci un intero maggiore di zero.")
            return

        """         files_iter = input_dir.rglob("*") if self.recursive_var.get() else input_dir.iterdir()
        files = sorted(
            p for p in files_iter
            if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
        ) """
        if not files:
            messagebox.showinfo("Nessun file", "Non ho trovato file audio/video nella cartella.")
            return

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Output non valido", str(exc))
            return

        settings = {
            "files": files,
            "output_dir": output_dir,
            "model": self.custom_model_var.get().strip() or self.model_var.get().strip(),
            "device": self.device_var.get(),
            "compute_type": self.compute_var.get(),
            "language": self.language_var.get().strip() or None,
            "beam_size": beam_size,
            "vad_filter": self.vad_var.get(),
            "prompt": self.prompt_var.get().strip() or None,
            "write_srt": self.srt_var.get(),
            "write_txt": self.txt_var.get(),
            "word_timestamps": self.word_ts_var.get(),
        }

        self.cancel_event.clear()
        self.stop_current_event.clear()
        self.skip_current_event.clear()
        self.progress["maximum"] = len(files)
        self.progress["value"] = 0
        self.current_progress["value"] = 0
        self.current_progress_var.set("File corrente: 0%")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.skip_btn.configure(state="normal")
        self.status_var.set(f"Caricamento modello: {settings['model']}…")
        self._log(f"\n=== Avvio: {len(files)} file | modello: {settings['model']} | device: {settings['device']} ===")
        self.worker = threading.Thread(target=self._work, args=(settings,), daemon=True)
        self.worker.start()

    def _request_stop(self):
        self.cancel_event.set()
        self.status_var.set("Stop richiesto: terminerà il file in corso.")

    def _request_stop_current(self):
        self.stop_current_event.set()
        self.status_var.set( "Stop richiesto: salvo l'output parziale del file corrente..." )

    def _request_skip_current(self):
        self.skip_current_event.set()
        self.status_var.set( "Skip richiesto: interrompo il file corrente e passo al successivo..." )

    def _work(self, settings):
        try:
            from faster_whisper import WhisperModel

            model_id = settings["model"]
            device = settings["device"]
            compute_type = settings["compute_type"]

            if device == "cpu" and compute_type in ("float16", "int8_float16"):
                compute_type = "int8" if compute_type == "int8_float16" else "float32"
                self.messages.put(("log", f"CPU: precisione impostata automaticamente a {compute_type}."))

            self.messages.put(("log", f"Caricamento modello '{model_id}' ({device}, {compute_type})."))
            model = WhisperModel(model_id, device=device, compute_type=compute_type)
            self.messages.put(("log", "Modello caricato."))

            total = len(settings["files"])
            used_names = set()

            for index, media_path in enumerate(settings["files"], start=1):
                if self.cancel_event.is_set():
                    break
                self.stop_current_event.clear()
                self.skip_current_event.clear()
                self.messages.put(("status", f"File {index}/{total}: {media_path.name}"))
                self.messages.put(("log", f"\n[{index}/{total}] {media_path}"))

                try:
                    kwargs = {
                        "beam_size": settings["beam_size"],
                        "vad_filter": settings["vad_filter"],
                        "word_timestamps": settings["word_timestamps"],
                        "condition_on_previous_text": False,
                    }
                    if settings["language"]:
                        kwargs["language"] = settings["language"]
                    if settings["prompt"]:
                        kwargs["initial_prompt"] = settings["prompt"]

                    print("DEVICE:", device)
                    print("COMPUTE TYPE:", compute_type)
                    segments, info = model.transcribe(str(media_path), **kwargs)
                    lines = []
                    srt_blocks = []

                    duration = getattr(info, "duration", 0) or 0
                    self.messages.put(("current_progress", 0))
                    self.messages.put(("current_label", f"File {index}/{total}: 0%"))
                    # IMPORTANTE: transcribe restituisce un generatore; iterarlo esegue davvero l'inferenza.
                    for segment in segments:
                        if duration > 0:
                            percent = min(100, int(segment.end / duration * 100))
                            self.messages.put(("current_progress", percent))
                            self.messages.put(( "current_label", f"File {index}/{total}: {percent}%" ))
                        print(segment.text, flush=True)
                        self.messages.put(("log", segment.text))
                        # Stop o Skip richiesti dall'interfaccia
                        if ( self.stop_current_event.is_set() or self.skip_current_event.is_set() ):
                            break

                        text = segment.text.strip()
                        if not text:
                            continue
                        lines.append(text)
                        srt_blocks.append(
                            f"{len(srt_blocks) + 1}\n"
                            f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}\n"
                            f"{text}\n"
                        )

                    stem = safe_stem(media_path)
                    output_dir = settings["output_dir"]
                    base = stem
                    suffix = 2
                    while base.lower() in used_names:
                        base = f"{stem}_{suffix}"
                        suffix += 1
                    used_names.add(base.lower())

                    if settings["write_txt"]: (output_dir / f"{base}.txt").write_text( "\n".join(lines).strip() + "\n" if lines else "", encoding="utf-8-sig" )
                    if settings["write_srt"]: (output_dir / f"{base}.srt").write_text( "\n".join(srt_blocks), encoding="utf-8-sig" )

                    self.messages.put(( "log", f"Lingua: {info.language} (prob. {info.language_probability:.3f}) | " f"segmenti: {len(srt_blocks)} | completato." ))

                except Exception as exc:
                    self.messages.put(("log", f"ERRORE su {media_path.name}: {exc}\n{traceback.format_exc()}"))

                if self.skip_current_event.is_set():
                    self.messages.put((
                        "log",
                        f"SKIP: {media_path.name} — output parziale salvato; passo al file successivo."
                    ))
                    self.messages.put(("progress", index))
                    continue

                if self.stop_current_event.is_set():
                    self.messages.put(( "log", f"STOP: {media_path.name} — output parziale salvato." ))
                    self.messages.put(("progress", index))
                    break

                self.messages.put(("progress", index))
                if self.stop_current_event.is_set():
                    final_status = "Interrotto su richiesta: output parziale salvato."
                elif self.cancel_event.is_set():
                    final_status = "Interrotto su richiesta."
                else:
                    final_status = "Elaborazione terminata."
                self.messages.put(("done", final_status))

        except Exception as exc:
            self.messages.put(("fatal", f"{exc}\n\n{traceback.format_exc()}"))

    def _poll_messages(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "current_progress":
                    self.current_progress["value"] = payload
                elif kind == "current_label":
                    self.current_progress_var.set(payload)
                elif kind == "done":
                    self.status_var.set(payload)
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self._log(payload)
                    self.stop_btn.configure(state="disabled")
                    self.skip_btn.configure(state="disabled")
                elif kind == "fatal":
                    self.status_var.set("Errore durante l'avvio o il caricamento del modello.")
                    self._log("ERRORE FATALE:\n" + payload)
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.stop_btn.configure(state="disabled")
                    self.skip_btn.configure(state="disabled")
                    messagebox.showerror( "Errore faster-whisper", "Il caricamento o l'avvio è fallito. Leggi il log nella finestra." )
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)


def main():
    root = tk.Tk()
    FasterWhisperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

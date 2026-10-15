import os
import sys
import glob
import re
import time
import datetime
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from PIL import Image, ImageTk, ImageGrab
import pyautogui

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pdf2image
import easyocr

# Configure CustomTkinter Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Initialize Deep Learning EasyOCR Engine once at startup
print("Initializing Deep Learning OCR Engine (EasyOCR)...")
EASYOCR_READER = easyocr.Reader(['en'], gpu=False)


# =========================================================
# NON-DESTRUCTIVE DEEP LEARNING PRE-PROCESSING ENGINE
# =========================================================
class AdvancedImagePreprocessor:
    """
    Clean pre-processor that enhances contrast and sharpens camera/screenshot text
    without eroding character stems or creating pixel artifacts.
    """

    @staticmethod
    def assess_and_enhance(cv_img: np.ndarray, attempt_stage: int = 1) -> np.ndarray:
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img.copy()

        # 1. Moderate Upscaling to Preserve Character Geometry
        h, w = gray.shape[:2]
        if h < 1200 or w < 1200:
            scale = 2.0 if attempt_stage == 1 else 2.5
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. CLAHE Contrast Normalization
        clip = 2.0 if attempt_stage == 1 else 3.5
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. Non-Destructive Sharpening Filter for Camera Focus Blur
        if attempt_stage >= 2:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced


# =========================================================
# NEURAL NETWORK SPATIAL TABLE EXTRACTOR (EASYOCR)
# =========================================================
class SpatialTableExtractor:
    """
    Uses EasyOCR neural recognition to detect bounding boxes and text,
    then spatially clusters words into horizontal rows and vertical columns.
    """

    @staticmethod
    def extract_grid(cv_img: np.ndarray, page_num: int = 1, attempt_stage: int = 1) -> pd.DataFrame:
        results = EASYOCR_READER.readtext(cv_img, detail=1, paragraph=False)

        if not results:
            df = pd.DataFrame([["[No Data Extracted]", ""]])
            df.insert(0, "Page_Number", page_num)
            return df

        words_data = []
        for bbox, text, confidence in results:
            text_str = text.strip()
            if not text_str or confidence < 0.15:
                continue

            min_x = min(pt[0] for pt in bbox)
            min_y = min(pt[1] for pt in bbox)
            max_y = max(pt[1] for pt in bbox)
            center_y = (min_y + max_y) / 2.0

            words_data.append({
                "x": min_x,
                "y": center_y,
                "text": text_str
            })

        if not words_data:
            df = pd.DataFrame([["[No High-Confidence Text Found]", ""]])
            df.insert(0, "Page_Number", page_num)
            return df

        words_data.sort(key=lambda item: item["y"])

        rows = []
        current_row = []
        last_y = None
        y_tolerance = 18.0 if attempt_stage == 1 else 14.0

        for word in words_data:
            if last_y is None or abs(word["y"] - last_y) <= y_tolerance:
                current_row.append(word)
            else:
                rows.append(current_row)
                current_row = [word]
            last_y = word["y"]

        if current_row:
            rows.append(current_row)

        table_matrix = []
        for row in rows:
            row_sorted = sorted(row, key=lambda item: item["x"])
            row_text = " ".join([w["text"] for w in row_sorted])

            if "|" in row_text:
                parts = [p.strip() for p in row_text.split("|")]
                table_matrix.append(parts)
            else:
                split_parts = re.split(r'\s{3,}', row_text)
                table_matrix.append(split_parts)

        max_cols = max(len(r) for r in table_matrix) if table_matrix else 1
        padded_matrix = [r + [""] * (max_cols - len(r)) for r in table_matrix]

        cols = [f"Extracted_Col_{i+1}" for i in range(max_cols)]
        df = pd.DataFrame(padded_matrix, columns=cols)
        
        # Insert Page Number header as first column
        df.insert(0, "Page_Number", page_num)
        return df


# ==========================================
# FILE SELECTION POP-UP MODAL
# ==========================================
class FileSelectionModal(ctk.CTkToplevel):
    def __init__(self, parent, file_list, on_confirm):
        super().__init__(parent)
        self.title("Select Files for Data Extraction Analysis")
        self.geometry("600x450")
        self.grab_set()

        self.file_list = file_list
        self.on_confirm = on_confirm
        self.checkbox_vars = {}

        self.label = ctk.CTkLabel(
            self, text="Select files to include in processing:", font=ctk.CTkFont(size=14, weight="bold")
        )
        self.label.pack(anchor="w", padx=20, pady=(15, 5))

        self.ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ctrl_frame.pack(fill="x", padx=20, pady=5)

        self.btn_all = ctk.CTkButton(self.ctrl_frame, text="Select All", width=100, command=self.select_all)
        self.btn_all.pack(side="left", padx=(0, 10))

        self.btn_none = ctk.CTkButton(self.ctrl_frame, text="Deselect All", width=100, command=self.deselect_all)
        self.btn_none.pack(side="left")

        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

        for fpath in self.file_list:
            fname = os.path.basename(fpath)
            var = ctk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.scroll_frame, text=fname, variable=var)
            cb.pack(anchor="w", pady=4, padx=5)
            self.checkbox_vars[fpath] = var

        self.btn_confirm = ctk.CTkButton(
            self, text="Proceed with Selected Files", fg_color="green", command=self.confirm_selection
        )
        self.btn_confirm.pack(pady=15)

    def select_all(self):
        for v in self.checkbox_vars.values():
            v.set(True)

    def deselect_all(self):
        for v in self.checkbox_vars.values():
            v.set(False)

    def confirm_selection(self):
        selected = [fpath for fpath, v in self.checkbox_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("Warning", "Please select at least one file to analyze.")
            return
        self.destroy()
        self.on_confirm(selected)


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
class MasterDataExtractorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Master Data Extraction — Deep Learning EasyOCR Engine")
        self.geometry("1400x850")

        self.input_dir = ""
        self.output_dir = ""
        self.selected_files = []
        self.extracted_data_store = {}
        self.current_file = None
        self.current_page = 1
        self.total_pages = 1
        self.manifest_path = ""
        self.clipboard_pil_img = None

        self._build_top_bar()
        self._build_split_dashboard()
        self._build_bottom_status_bar()

        self.after(200, self._prompt_initial_directories)

    def _prompt_initial_directories(self):
        messagebox.showinfo("Directory Setup", "Select your SOURCE INPUT folder.")
        inp = filedialog.askdirectory(title="Select Source Input Directory")
        if not inp:
            sys.exit(0)
        self.input_dir = inp

        messagebox.showinfo("Directory Setup", "Select your OUTPUT DESTINATION folder.")
        outp = filedialog.askdirectory(title="Select Output Destination Directory")
        if not outp:
            sys.exit(0)
        self.output_dir = outp

        self.lbl_input_path.configure(text=f"Input: {self.input_dir}")
        self.lbl_output_path.configure(text=f"Output: {self.output_dir}")

        self.manifest_path = os.path.join(self.output_dir, "Master_Extraction_Log.csv")
        self._init_manifest_log()
        self._scan_input_directory()

    def _init_manifest_log(self):
        if not os.path.exists(self.manifest_path):
            df_manifest = pd.DataFrame(
                columns=["Timestamp", "FileName", "FileType", "PageCount", "Status", "AttemptStage", "ErrorDiagnostics"]
            )
            df_manifest.to_csv(self.manifest_path, index=False)

    def _log_manifest(self, fname, ftype, pages, status, stage, err=""):
        try:
            entry = pd.DataFrame([{
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "FileName": fname,
                "FileType": ftype,
                "PageCount": pages,
                "Status": status,
                "AttemptStage": stage,
                "ErrorDiagnostics": err
            }])
            entry.to_csv(self.manifest_path, mode="a", header=False, index=False)
        except Exception as e:
            print(f"Manifest Log Error: {e}")

    # ------------------------------------------
    # GUI LAYOUT
    # ------------------------------------------
    def _build_top_bar(self):
        self.top_frame = ctk.CTkFrame(self, height=60)
        self.top_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.lbl_input_path = ctk.CTkLabel(self.top_frame, text="Input: Not Set", font=ctk.CTkFont(size=12), anchor="w")
        self.lbl_input_path.pack(side="left", padx=10)

        self.lbl_output_path = ctk.CTkLabel(self.top_frame, text="Output: Not Set", font=ctk.CTkFont(size=12), anchor="w")
        self.lbl_output_path.pack(side="left", padx=10)

        self.btn_change_dirs = ctk.CTkButton(self.top_frame, text="Change Paths", width=100, command=self._prompt_initial_directories)
        self.btn_change_dirs.pack(side="right", padx=5)

        self.btn_auto_snip = ctk.CTkButton(
            self.top_frame, text="Auto Snip & Extract", fg_color="#9333ea", width=160, command=self.trigger_windows_snipping_tool
        )
        self.btn_auto_snip.pack(side="right", padx=5)

        self.btn_analyze = ctk.CTkButton(self.top_frame, text="Start Analysis", fg_color="green", width=120, command=self.start_analysis_process)
        self.btn_analyze.pack(side="right", padx=5)

    def _build_split_dashboard(self):
        self.paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#2b2b2b", sashwidth=6)
        self.paned_window.pack(fill="both", expand=True, padx=10, pady=5)

        # LEFT PANEL
        self.left_panel = ctk.CTkFrame(self.paned_window)
        self.paned_window.add(self.left_panel, minsize=500)

        self.left_nav_frame = ctk.CTkFrame(self.left_panel, height=40)
        self.left_nav_frame.pack(fill="x", padx=5, pady=5)

        self.doc_dropdown = ctk.CTkOptionMenu(self.left_nav_frame, values=["No Files Loaded"], command=self._on_doc_selected)
        self.doc_dropdown.pack(side="left", padx=5)

        self.btn_prev_page = ctk.CTkButton(self.left_nav_frame, text="< Prev", width=60, command=self.prev_page)
        self.btn_prev_page.pack(side="left", padx=5)

        self.lbl_page_num = ctk.CTkLabel(self.left_nav_frame, text="Page: 0 / 0")
        self.lbl_page_num.pack(side="left", padx=5)

        self.btn_next_page = ctk.CTkButton(self.left_nav_frame, text="Next >", width=60, command=self.next_page)
        self.btn_next_page.pack(side="left", padx=5)

        self.media_canvas_frame = ctk.CTkScrollableFrame(self.left_panel)
        self.media_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.lbl_image_display = ctk.CTkLabel(self.media_canvas_frame, text="Select Document or Click 'Auto Snip & Extract'")
        self.lbl_image_display.pack(expand=True, pady=20)

        # RIGHT PANEL
        self.right_panel = ctk.CTkFrame(self.paned_window)
        self.paned_window.add(self.right_panel, minsize=600)

        self.right_ctrl_frame = ctk.CTkFrame(self.right_panel, height=45)
        self.right_ctrl_frame.pack(fill="x", padx=5, pady=5)

        self.lbl_data_status = ctk.CTkLabel(self.right_ctrl_frame, text="Extracted Data Preview", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_data_status.pack(side="left", padx=10)

        self.btn_more_analysis = ctk.CTkButton(
            self.right_ctrl_frame, text="More Analysis (Retry Auto-Tune)", fg_color="#d97706", command=self.trigger_more_analysis
        )
        self.btn_more_analysis.pack(side="right", padx=5)

        self.btn_approve = ctk.CTkButton(
            self.right_ctrl_frame, text="Approve Page", fg_color="green", width=110, command=self.approve_current_page
        )
        self.btn_approve.pack(side="right", padx=5)

        self.table_container = ctk.CTkFrame(self.right_panel)
        self.table_container.pack(fill="both", expand=True, padx=5, pady=5)

        self.tree_scroll_y = ttk.Scrollbar(self.table_container, orient="vertical")
        self.tree_scroll_x = ttk.Scrollbar(self.table_container, orient="horizontal")

        self.tree = ttk.Treeview(
            self.table_container,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set,
            selectmode="extended"
        )
        self.tree_scroll_y.config(command=self.tree.yview)
        self.tree_scroll_x.config(command=self.tree.xview)

        self.tree_scroll_y.pack(side="right", fill="y")
        self.tree_scroll_x.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

    def _build_bottom_status_bar(self):
        self.bottom_frame = ctk.CTkFrame(self, height=40)
        self.bottom_frame.pack(fill="x", padx=10, pady=(5, 10))

        self.progress_bar = ctk.CTkProgressBar(self.bottom_frame, width=300)
        self.progress_bar.pack(side="left", padx=15)
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(self.bottom_frame, text="Ready", anchor="w")
        self.lbl_status.pack(side="left", padx=10, fill="x", expand=True)

        self.btn_export_master = ctk.CTkButton(
            self.bottom_frame, text="Export Master Excel", fg_color="#2563eb", command=self.export_master_excel
        )
        self.btn_export_master.pack(side="right", padx=15)

    # ------------------------------------------
    # AUTOMATED WINDOWS SNIPPING TOOL CONNECTOR
    # ------------------------------------------
    def trigger_windows_snipping_tool(self):
        self.iconify()
        time.sleep(0.5)

        pyautogui.hotkey('win', 'shift', 's')
        threading.Thread(target=self._poll_clipboard_for_snip, daemon=True).start()

    def _poll_clipboard_for_snip(self):
        self._update_status("Waiting for Snipping Tool capture...", 0.2)

        max_wait = 20
        start_time = time.time()
        captured_img = None

        while time.time() - start_time < max_wait:
            time.sleep(0.5)
            current_clipboard = ImageGrab.grabclipboard()
            if isinstance(current_clipboard, Image.Image):
                captured_img = current_clipboard
                break

        self.after(0, self.deiconify)

        if captured_img is None:
            self._update_status("Snipping Tool cancelled or timed out.", 0.0)
            return

        self._update_status("Processing Snipping Tool Capture with EasyOCR...", 0.7)

        cv_img = cv2.cvtColor(np.array(captured_img.convert('RGB')), cv2.COLOR_RGB2BGR)

        enhanced_img = AdvancedImagePreprocessor.assess_and_enhance(cv_img, attempt_stage=1)
        df = SpatialTableExtractor.extract_grid(enhanced_img, page_num=1, attempt_stage=1)

        snip_filename = f"Snip_{int(time.time())}.png"
        snip_key = (snip_filename, 1)
        self.extracted_data_store[snip_key] = {
            "df": df, "attempt": 1, "approved": False, "track": "EasyOCR (Deep Learning)"
        }

        self.current_file = snip_filename
        self.current_page = 1
        self.total_pages = 1
        self.clipboard_pil_img = captured_img

        preview_img = captured_img.copy()
        preview_img.thumbnail((550, 750))
        ctk_img = ctk.CTkImage(light_image=preview_img, dark_image=preview_img, size=preview_img.size)

        self.after(0, lambda: self.lbl_image_display.configure(image=ctk_img, text=""))
        self.after(0, self._render_right_table)
        self._update_status("Snipping Tool Data Extracted Successfully!", 1.0)
        self._log_manifest(snip_filename, ".png", 1, "Success", 1)

    # ------------------------------------------
    # FILE DISCOVERY
    # ------------------------------------------
    def _scan_input_directory(self):
        valid_exts = ("*.pdf", "*.jpg", "*.jpeg", "*.png", "*.PDF", "*.JPG", "*.JPEG", "*.PNG")
        files_found = []
        for ext in valid_exts:
            files_found.extend(glob.glob(os.path.join(self.input_dir, ext)))

        files_found = [f for f in files_found if not f.endswith((".xlsx", ".xls", ".csv", ".txt"))]

        if not files_found:
            messagebox.showwarning("No Files", "No eligible PDF or Image files found.")
            return

        FileSelectionModal(self, files_found, self._on_files_confirmed)

    def _on_files_confirmed(self, selected_files):
        self.selected_files = selected_files
        doc_names = [os.path.basename(f) for f in self.selected_files]
        self.doc_dropdown.configure(values=doc_names)
        self.doc_dropdown.set(doc_names[0])
        self.current_file = self.selected_files[0]
        self.current_page = 1
        self._load_document_view()

    # ------------------------------------------
    # PROCESSING WORKER
    # ------------------------------------------
    def start_analysis_process(self):
        if not self.selected_files:
            messagebox.showwarning("Warning", "No files selected.")
            return

        self.btn_analyze.configure(state="disabled")
        self.progress_bar.set(0)
        threading.Thread(target=self._async_analysis_worker, daemon=True).start()

    def _async_analysis_worker(self):
        total = len(self.selected_files)
        for idx, fpath in enumerate(self.selected_files):
            fname = os.path.basename(fpath)
            self._update_status(f"Analyzing {fname} ({idx+1}/{total})...", idx / total)

            ext = os.path.splitext(fpath)[1].lower()
            try:
                if ext == ".pdf":
                    self._process_pdf_file(fpath)
                else:
                    self._process_image_file(fpath)

                self._log_manifest(fname, ext, 1, "Success", 1)
            except Exception as e:
                self._log_manifest(fname, ext, 1, "Failed", 1, str(e))

            self.progress_bar.set((idx + 1) / total)

        self._update_status("Analysis Completed Successfully!", 1.0)
        self.after(0, lambda: self.btn_analyze.configure(state="normal"))
        self.after(0, self._load_document_view)

    def _process_pdf_file(self, fpath):
        try:
            with pdfplumber.open(fpath) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    tables = page.extract_tables()
                    if tables and len(tables[0]) > 0:
                        df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
                        df.insert(0, "Page_Number", page_num)
                        self.extracted_data_store[(fpath, page_num)] = {
                            "df": df, "attempt": 1, "approved": False, "track": "Digital PDF"
                        }
                    else:
                        self._process_scanned_pdf_page(fpath, page_num)
        except Exception:
            self._process_scanned_pdf_page(fpath, 1)

    def _process_scanned_pdf_page(self, fpath, page_num):
        images = pdf2image.convert_from_path(fpath, first_page=page_num, last_page=page_num, dpi=300)
        if images:
            cv_img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            enhanced = AdvancedImagePreprocessor.assess_and_enhance(cv_img, attempt_stage=1)
            df = SpatialTableExtractor.extract_grid(enhanced, page_num=page_num, attempt_stage=1)
            self.extracted_data_store[(fpath, page_num)] = {
                "df": df, "attempt": 1, "approved": False, "track": "EasyOCR (Scanned PDF)"
            }

    def _process_image_file(self, fpath):
        cv_img = cv2.imread(fpath)
        enhanced = AdvancedImagePreprocessor.assess_and_enhance(cv_img, attempt_stage=1)
        df = SpatialTableExtractor.extract_grid(enhanced, page_num=1, attempt_stage=1)
        self.extracted_data_store[(fpath, 1)] = {
            "df": df, "attempt": 1, "approved": False, "track": "EasyOCR (Image)"
        }

    # ------------------------------------------
    # MORE ANALYSIS (RETRY AUTO-TUNE)
    # ------------------------------------------
    def trigger_more_analysis(self):
        if not self.current_file:
            return

        key = (self.current_file, self.current_page)
        current_attempt = self.extracted_data_store.get(key, {}).get("attempt", 1)
        next_attempt = current_attempt + 1

        self.btn_more_analysis.configure(state="disabled")
        self._update_status(f"Running More Analysis (Attempt Stage {next_attempt})...", 0.5)

        def _re_analysis_worker():
            fpath = self.current_file

            if fpath.startswith("Snip_") and self.clipboard_pil_img:
                cv_img = cv2.cvtColor(np.array(self.clipboard_pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
                ext = ".png"
            else:
                ext = os.path.splitext(fpath)[1].lower()
                if ext == ".pdf":
                    images = pdf2image.convert_from_path(fpath, first_page=self.current_page, last_page=self.current_page, dpi=300)
                    cv_img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
                else:
                    cv_img = cv2.imread(fpath)

            enhanced = AdvancedImagePreprocessor.assess_and_enhance(cv_img, attempt_stage=next_attempt)
            new_df = SpatialTableExtractor.extract_grid(enhanced, page_num=self.current_page, attempt_stage=next_attempt)

            self.extracted_data_store[key] = {
                "df": new_df, "attempt": next_attempt, "approved": False, "track": f"EasyOCR Stage {next_attempt}"
            }

            self._log_manifest(os.path.basename(fpath), ext, self.current_page, "Re-Analyzed", next_attempt)
            self._update_status(f"Stage {next_attempt} Auto-Tuning Complete!", 1.0)

            self.after(0, lambda: self.btn_more_analysis.configure(state="normal"))
            self.after(0, self._render_right_table)

        threading.Thread(target=_re_analysis_worker, daemon=True).start()

    def approve_current_page(self):
        key = (self.current_file, self.current_page)
        if key in self.extracted_data_store:
            self.extracted_data_store[key]["approved"] = True
            messagebox.showinfo("Approved", f"Page {self.current_page} approved for Master Export!")
            self._render_right_table()

    # ------------------------------------------
    # RENDERING ENGINE
    # ------------------------------------------
    def _on_doc_selected(self, choice_name):
        for fpath in self.selected_files:
            if os.path.basename(fpath) == choice_name:
                self.current_file = fpath
                self.current_page = 1
                self._load_document_view()
                break

    def _load_document_view(self):
        if not self.current_file:
            return

        ext = os.path.splitext(self.current_file)[1].lower()
        if ext == ".pdf":
            try:
                with pdfplumber.open(self.current_file) as pdf:
                    self.total_pages = len(pdf.pages)
            except Exception:
                self.total_pages = 1
        else:
            self.total_pages = 1

        self.lbl_page_num.configure(text=f"Page: {self.current_page} / {self.total_pages}")
        self._render_left_media()
        self._render_right_table()

    def _render_left_media(self):
        if self.current_file and self.current_file.startswith("Snip_") and self.clipboard_pil_img:
            pil_img = self.clipboard_pil_img.copy()
            pil_img.thumbnail((550, 750))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.lbl_image_display.configure(image=ctk_img, text="")
            return

        ext = os.path.splitext(self.current_file)[1].lower()
        try:
            if ext == ".pdf":
                images = pdf2image.convert_from_path(
                    self.current_file, first_page=self.current_page, last_page=self.current_page, dpi=150
                )
                pil_img = images[0]
            else:
                pil_img = Image.open(self.current_file)

            pil_img.thumbnail((550, 750))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.lbl_image_display.configure(image=ctk_img, text="")
        except Exception as e:
            self.lbl_image_display.configure(image="", text=f"Error Loading Image: {e}")

    def _render_right_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        key = (self.current_file, self.current_page)
        stored_entry = self.extracted_data_store.get(key, None)

        if not stored_entry:
            self.lbl_data_status.configure(text="No Extracted Data Yet — Click 'Start Analysis' or 'Auto Snip & Extract'")
            return

        df = stored_entry["df"]
        approved = stored_entry["approved"]
        track = stored_entry["track"]
        attempt = stored_entry["attempt"]

        status_txt = f"[{track}] | Stage: {attempt} | Approved: {'YES' if approved else 'NO'}"
        self.lbl_data_status.configure(text=status_txt)

        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"

        for col in df.columns:
            self.tree.heading(col, text=str(col))
            self.tree.column(col, width=150 if col == "Page_Number" else 220, anchor="w")

        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._load_document_view()

    def next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self._load_document_view()

    # ------------------------------------------
    # EXPORT MASTER EXCEL (CRASH-PROOF SINGLE SHEET PER DOC)
    # ------------------------------------------
    def export_master_excel(self):
        if not self.extracted_data_store:
            messagebox.showwarning("Warning", "No extracted data available to export.")
            return

        master_excel_path = os.path.join(self.output_dir, "Master_Extracted_Data.xlsx")
        try:
            # Group extracted pages by document file path
            grouped_docs = {}
            for (fpath, page_num), entry in self.extracted_data_store.items():
                if fpath not in grouped_docs:
                    grouped_docs[fpath] = []
                grouped_docs[fpath].append((page_num, entry["df"]))

            with pd.ExcelWriter(master_excel_path, engine="openpyxl") as writer:
                for fpath, pages_data in grouped_docs.items():
                    # Sort pages sequentially by page number
                    pages_data.sort(key=lambda x: x[0])
                    
                    cleaned_dfs = []
                    for page_num, df in pages_data:
                        df_copy = df.copy()
                        
                        # Fix Duplicate Columns: Ensure all column headers are unique strings
                        cols = []
                        col_count = {}
                        for col in df_copy.columns:
                            col_str = str(col)
                            if col_str in col_count:
                                col_count[col_str] += 1
                                cols.append(f"{col_str}_{col_count[col_str]}")
                            else:
                                col_count[col_str] = 0
                                cols.append(col_str)
                        df_copy.columns = cols
                        cleaned_dfs.append(df_copy)
                    
                    # Combine all page DataFrames safely
                    combined_df = pd.concat(cleaned_dfs, axis=0, ignore_index=True, sort=False)
                    
                    # Fill any empty missing cells from mismatched column counts with empty string
                    combined_df = combined_df.fillna("")
                    
                    # Generate a clean sheet title from the document filename (max 31 chars)
                    fname = os.path.basename(fpath)
                    sheet_title = re.sub(r'[:\*\\/1\?\[\]]', '_', fname)[:31]
                    
                    combined_df.to_excel(writer, sheet_name=sheet_title, index=False)

            messagebox.showinfo("Export Success", f"Master Excel generated cleanly at:\n{master_excel_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save Master Excel: {e}")

    def _update_status(self, msg, progress_val=None):
        self.lbl_status.configure(text=msg)
        if progress_val is not None:
            self.progress_bar.set(progress_val)


if __name__ == "__main__":
    app = MasterDataExtractorApp()
    app.mainloop()
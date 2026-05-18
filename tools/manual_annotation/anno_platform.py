# -*- coding: utf-8 -*-
import tkinter as tk
import re
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
import os
import glob
import traceback
import itertools
import sys # Import sys for command line arguments
import argparse # Import argparse for better command line argument handling


# --- Constants ---
DEFAULT_BOX_COLOR_SUBJECT = "red"  # Subject box color
DEFAULT_BOX_COLOR_OBJECT = "blue"   # Object box color

# New relationship types, grouped by category
RELATIONSHIP_CATEGORIES = {
    "静态位置关系": ["above","behind","beneath","left","right","front", "next_to","inside"],
    "动态位置关系": ["away","toward","past","follow","chase"],
    "静态动作关系": ["bite","carry","drive","feed","hold","pull","push","ride","touch","watch","clean",
                     "close","open","cut","hit","hug","kiss","knock","eat","use",
                     "shake_hand_with","smell","speak_to","sit_on","kick"],
    "动态动作关系": ["play","get_off","get_on","lift","throw","wave_hand_to","release"]
}

# Flattened list for easy lookup
ALL_RELATIONSHIP_TYPES = list(itertools.chain(*RELATIONSHIP_CATEGORIES.values()))

class VideoAnnotator:
    def __init__(self, master, data_folder=None, track_json_path=None, annotation_folder=None):
        self.master = master
        self.master.title("视频关系标注工具 (自动匹配)")
        # Try to set an initial size more suitable for most screens
        try:
            screen_width = master.winfo_screenwidth()
            screen_height = master.winfo_screenheight()
            # Set to 80% width and 90% height of the screen, but not exceeding a certain maximum
            width = min(int(screen_width * 0.8), 1200)
            height = min(int(screen_height * 0.9), 900)
            self.master.geometry(f"{width}x{height}")
        except tk.TclError: # Fallback if screen info not available
             self.master.geometry("1000x800")

        # Try to load a font that supports Chinese characters
        self.font = None
        try:
            # Try common Chinese font paths
            for font_path in ["C:/Windows/Fonts/simsun.ttc", "/System/Library/Fonts/STHeiti Light.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]: # Added another common path
                 if os.path.exists(font_path):
                     # Adjust font size if needed
                     self.font = ImageFont.truetype(font_path, 16) # Increased font size slightly
                     break
        except Exception as e:
             print(f"Warning: Could not load font, text display may be incorrect: {e}")
             self.font = ImageFont.load_default() # Fallback to default font


        # --- Data Storage ---
        self.data_folder = data_folder # Folder containing video frame subfolders
        self.track_json_path = track_json_path # Path to the track JSON file
        self.annotation_folder = annotation_folder # Folder to save/load annotations

        # Structure: { "video_folder_name": { "track_id": { "frame_index_str": [x1, y1, x2, y2], ... } }, ... }
        self.tracks_data = {}
        # New dictionary to store track metadata {video_id: {track_id: {category: ..., traj_name: ...}, ...}, ...}
        self.track_metadata = {}

        self.current_video_folder = None # Current video folder name (i.e., video ID)
        self.frame_files = [] # Paths of all frame files for the current video
        self.all_frame_indices = [] # List of original indices of all frames (0, 1, 2, ...)

        # Stores automatically identified track pairs (Subject ID, Object ID) and their common frame information
        # Structure: [{ "pair": (subject_id, object_id), "common_frames": [f1, f2, ...]}, ...]
        self.potential_pairs = []
        self.current_pair_index = -1 # Index of the currently displayed track pair
        # Stores the list of original indices of frames common to the current track pair, in order
        self.current_pair_common_frames = []
        self.current_frame_in_common_index = 0 # Index of the currently displayed frame in the common_frames list

        self.current_subject_track_id = None # Subject ID of the current track pair
        self.current_object_track_id = None # Object ID of the current track pair

        self.annotations = [] # List to store annotation dicts for the current video
        self.tk_image = None # Reference to prevent garbage collection

        # --- UI Layout ---
        # Main frame using grid layout
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=0) # Left control column (fixed width)
        main_frame.columnconfigure(1, weight=1) # Right display/annotation area (expands horizontally)
        main_frame.rowconfigure(0, weight=3) # Top area (control + display) takes more vertical space
        main_frame.rowconfigure(1, weight=1) # Bottom area (annotation) takes less vertical space


        # Left control panel (fixed width) - placed in top-left grid cell
        control_panel = ttk.Frame(main_frame, width=320)
        control_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10)) # Only in row 0, column 0
        control_panel.pack_propagate(False) # Prevent internal widgets from expanding the panel


        # Right top area (Image display) - placed in top-right grid cell
        display_panel = ttk.Frame(main_frame)
        display_panel.grid(row=0, column=1, sticky="nsew")
        display_panel.columnconfigure(0, weight=1) # Allow canvas to expand within display_panel
        display_panel.rowconfigure(0, weight=1) # Allow canvas to expand within display_panel


        # Bottom area frame (Annotations + Save) - placed in bottom-right grid cell, spans across the original right panel width
        bottom_annotation_area = ttk.Frame(main_frame)
        bottom_annotation_area.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0)) # Spans both columns in row 1
        bottom_annotation_area.columnconfigure(0, weight=3) # Annotate frame takes more horizontal space
        bottom_annotation_area.columnconfigure(1, weight=1) # Save frame takes less horizontal space
        bottom_annotation_area.rowconfigure(0, weight=1) # Allow content to expand vertically


        # --- Control Panel Widgets (Sections 1, 2, 3) ---
        # 1. Load Data
        load_frame = ttk.LabelFrame(control_panel, text="1. 加载数据")
        load_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(load_frame, text="选择数据文件夹", command=self.select_data_folder).pack(pady=2, fill=tk.X, padx=5)
        ttk.Button(load_frame, text="选择轨迹 JSON", command=self.select_track_json).pack(pady=2, fill=tk.X, padx=5)
        ttk.Button(load_frame, text="选择标注文件夹", command=self.select_annotation_folder).pack(pady=2, fill=tk.X, padx=5) # New button for annotation folder
        self.load_status_label = ttk.Label(load_frame, text="未加载数据", wraplength=280) # Automatic wrapping
        self.load_status_label.pack(pady=2, padx=5)

        # 2. Select Video
        video_frame = ttk.LabelFrame(control_panel, text="2. 选择视频")
        video_frame.pack(fill=tk.X, pady=5, padx=5)
        video_list_frame = ttk.Frame(video_frame) # Frame to contain Listbox and Scrollbar
        video_list_frame.pack(fill=tk.X, pady=2, padx=5)
        video_scrollbar = ttk.Scrollbar(video_list_frame, orient=tk.VERTICAL)
        self.video_listbox = tk.Listbox(video_list_frame, height=6, exportselection=False,
                                        yscrollcommand=video_scrollbar.set)
        video_scrollbar.config(command=self.video_listbox.yview)
        video_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)
        self.current_video_label = ttk.Label(video_frame, text="当前视频: 无", wraplength=280)
        self.current_video_label.pack(pady=2, padx=5)

        # 3. Automatic Track Pair Matching and Navigation
        pair_frame = ttk.LabelFrame(control_panel, text="3. 轨迹对 (自动匹配)")
        pair_frame.pack(fill=tk.X, pady=5, padx=5)

        # Modified label to show subject and object
        self.current_pair_label = ttk.Label(pair_frame, text="当前轨迹对: 无", wraplength=280)
        self.current_pair_label.pack(pady=2, padx=5)

        pair_nav_frame = ttk.Frame(pair_frame)
        pair_nav_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(pair_nav_frame, text="上一对", command=lambda: self.change_pair(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(pair_nav_frame, text="下一对", command=lambda: self.change_pair(1)).pack(side=tk.LEFT, expand=True, fill=tk.X)


        # --- Annotate Relationships (Section 4) - placed in bottom_annotation_area column 0 ---
        annotate_frame = ttk.LabelFrame(bottom_annotation_area, text="4. 标注关系")
        annotate_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        annotate_frame.columnconfigure(0, weight=4) # Relationship categories grid takes more space
        annotate_frame.columnconfigure(1, weight=1) # Frame selection/add button frame takes less space
        annotate_frame.rowconfigure(0, weight=1) # Allow content to expand vertically


        # Frame to hold Relationship type selection (grouped by category) - placed in annotate_frame column 0
        rel_types_grid_container = ttk.Frame(annotate_frame)
        rel_types_grid_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        rel_types_grid_container.columnconfigure((0, 1, 2, 3), weight=1) # Four columns for categories share space equally
        rel_types_grid_container.rowconfigure(0, weight=1) # Allow content to expand vertically


        self.relationship_listboxes = {} # Store references to the four Listboxes

        col_idx = 0
        for category, types in RELATIONSHIP_CATEGORIES.items():
            cat_frame = ttk.LabelFrame(rel_types_grid_container, text=category)
            cat_frame.grid(row=0, column=col_idx, sticky="nsew", padx=2, pady=2)
            cat_frame.columnconfigure(0, weight=1) # Make Listbox fill horizontal space
            cat_frame.rowconfigure(0, weight=1) # Make Listbox fill vertical space


            rel_scrollbar = ttk.Scrollbar(cat_frame, orient=tk.VERTICAL)
            # Set a fixed height that can display most items, or allow expansion
            rel_listbox = tk.Listbox(cat_frame, height=min(len(types), 15), selectmode=tk.MULTIPLE, exportselection=False,
                                     yscrollcommand=rel_scrollbar.set)
            rel_scrollbar.config(command=rel_listbox.yview)
            # Only pack scrollbar if the fixed height is less than the number of types
            # This ensures scrollbar only appears when necessary
            if len(types) > 15:
                 rel_scrollbar.grid(row=0, column=1, sticky="ns")
            rel_listbox.grid(row=0, column=0, sticky="nsew")


            # Populate relationship type list
            for rel_type in types:
                rel_listbox.insert(tk.END, rel_type)

            self.relationship_listboxes[category] = rel_listbox # Store reference
            col_idx += 1


        # Frame to hold Frame range selection and Add button - placed in annotate_frame column 1
        frame_select_add_frame = ttk.Frame(annotate_frame)
        frame_select_add_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        frame_select_add_frame.columnconfigure(0, weight=1) # Allow content to expand horizontally
        # frame_select_add_frame.rowconfigure((0, 1, 2), weight=1) # Allow rows to expand


        start_frame_frame = ttk.Frame(frame_select_add_frame) # Frame for start frame
        start_frame_frame.pack(fill=tk.X, pady=2)
        ttk.Label(start_frame_frame, text="起始帧索引:").pack(side=tk.LEFT)
        self.start_frame_combobox = ttk.Combobox(start_frame_frame, state="readonly", width=8)
        self.start_frame_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X) # Allow combobox to expand

        end_frame_frame = ttk.Frame(frame_select_add_frame) # Frame for end frame
        end_frame_frame.pack(fill=tk.X, pady=2)
        ttk.Label(end_frame_frame, text="结束帧索引:").pack(side=tk.LEFT)
        self.end_frame_combobox = ttk.Combobox(end_frame_frame, state="readonly", width=8) # Width can be adjusted
        self.end_frame_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X) # Allow combobox to expand


        ttk.Button(frame_select_add_frame, text="添加标注", command=self.add_annotation).pack(pady=5, fill=tk.X, padx=5)


        # --- Annotation View Area Widgets (Section 5) - placed in bottom_annotation_area column 1 ---
        save_frame = ttk.LabelFrame(bottom_annotation_area, text="5. 当前标注")
        save_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        save_frame.columnconfigure(0, weight=1) # Allow content to expand horizontally
        save_frame.rowconfigure(0, weight=1) # Allow text area to expand vertically


        ann_text_frame = ttk.Frame(save_frame) # Frame to contain Text and Scrollbar
        # Use grid to place ann_text_frame within save_frame
        ann_text_frame.grid(row=0, column=0, sticky="nsew", pady=2, padx=5)
        ann_text_frame.columnconfigure(0, weight=1) # Make Text fill horizontal space
        ann_text_frame.rowconfigure(0, weight=1) # Make Text fill vertical space


        ann_scrollbar = ttk.Scrollbar(ann_text_frame, orient=tk.VERTICAL)
        self.annotation_text = tk.Text(ann_text_frame, height=8, wrap=tk.WORD, state=tk.DISABLED,
                                       yscrollcommand=ann_scrollbar.set)
        ann_scrollbar.config(command=self.annotation_text.yview)
        self.annotation_text.grid(row=0, column=0, sticky="nsew")
        ann_scrollbar.grid(row=0, column=1, sticky="ns")

        # Save button - placed below the text area in the grid
        ttk.Button(save_frame, text="保存所有标注", command=self.save_annotations).grid(row=1, column=0, pady=5, padx=5, sticky="ew")


        # --- Display Panel Widgets (Image display) ---
        # Image display canvas
        self.canvas = tk.Canvas(display_panel, bg="lightgrey", bd=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew") # Use grid and sticky to make canvas expand


        # Frame control bar
        frame_control_frame = ttk.Frame(display_panel, padding=(0, 5))
        frame_control_frame.grid(row=1, column=0, sticky="ew") # Place below canvas and make it expand horizontally

        ttk.Button(frame_control_frame, text="<<", width=4, command=lambda: self.change_frame(-1)).pack(side=tk.LEFT, padx=(5,0))
        # Modified frame label to show current frame index in common frames list and total common frames
        self.frame_label_var = tk.StringVar(value="帧: - / -")
        self.frame_label = ttk.Label(frame_control_frame, textvariable=self.frame_label_var, anchor='center', width=25) # Wider label
        self.frame_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_control_frame, text=">>", width=4, command=lambda: self.change_frame(1)).pack(side=tk.LEFT, padx=(0,5))

        self.frame_slider_var = tk.DoubleVar()
        self.frame_slider = ttk.Scale(frame_control_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                      variable=self.frame_slider_var, command=self.on_slider_move)
        # Make slider fill remaining space
        self.frame_slider.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)

        # --- Event Bindings ---
        # Bind left/right arrow keys to change frame
        self.master.bind('<Left>', lambda event: self.change_frame(-1))
        self.master.bind('<Right>', lambda event: self.change_frame(1))
        # Bind window size change event to redraw the image to fit the canvas
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        # --- Initial Load from Arguments ---
        # If initial paths were provided via command line, load data automatically
        if self.data_folder and self.track_json_path:
             self._update_load_status()
             self._load_track_data()
             # _populate_video_list is called at the end of _load_track_data if successful
             # If annotation folder was also provided, populate video list after setting it
             if self.annotation_folder:
                 self._populate_video_list()
        elif self.data_folder or self.track_json_path or self.annotation_folder:
             # If only some paths were provided, update status but don't auto-load fully
             self._update_load_status()
             if self.data_folder: # If data folder is provided, populate video list even without tracks
                 self._populate_video_list()


    # --- Callback Functions and Methods ---

    def on_canvas_resize(self, event):
        """Redraw the current frame when the canvas size changes"""
        # Add a small delay to avoid redrawing too frequently when resizing quickly
        # If there was a previous delayed task, cancel it
        if hasattr(self, '_after_id_resize'):
             self.master.after_cancel(self._after_id_resize)
        # Set a new delayed task
        self._after_id_resize = self.master.after(100, self.display_current_frame)


    def validate_spinbox_input(self, P):
        """Validate Spinbox input is a number (not used for Combobox)"""
        if P == "" or P.isdigit():
            return True
        else:
            return False

    def select_data_folder(self):
        """Select the main directory containing video frame folders"""
        folder = filedialog.askdirectory(title="选择数据文件夹")
        if folder:
            self.data_folder = folder
            self._update_load_status()
            self._populate_video_list()
        else:
             messagebox.showwarning("选择文件夹", "未选择文件夹")

    def select_track_json(self):
        """Select the track JSON file"""
        filepath = filedialog.askopenfilename(title="选择轨迹 JSON 文件", filetypes=[("JSON files", "*.json")])
        if filepath:
            self.track_json_path = filepath
            self._update_load_status()
            self._load_track_data()
        else:
            messagebox.showwarning("选择文件", "未选择 JSON 文件")

    def select_annotation_folder(self):
        """Select the folder to save/load annotations"""
        folder = filedialog.askdirectory(title="选择标注文件夹")
        if folder:
            self.annotation_folder = folder
            self._update_load_status()
            # Re-populate video list to show annotated status
            self._populate_video_list()
        else:
             messagebox.showwarning("选择文件夹", "未选择标注文件夹")


    def _update_load_status(self):
        """Update the load status label"""
        folder_status = f"数据文件夹: {os.path.basename(self.data_folder)}" if self.data_folder else "数据文件夹: 未选"
        json_status = f"轨迹 JSON: {os.path.basename(self.track_json_path)}" if self.track_json_path else "轨迹 JSON: 未选"
        ann_folder_status = f"标注文件夹: {os.path.basename(self.annotation_folder)}" if self.annotation_folder else "标注文件夹: 未选"
        self.load_status_label.config(text=f"{folder_status}\n{json_status}\n{ann_folder_status}")

    def _populate_video_list(self):
        """Populate the video list and indicate annotated status"""
        self.video_listbox.delete(0, tk.END)
        if self.data_folder:
            try:
                subfolders = [f.name for f in os.scandir(self.data_folder) if f.is_dir()]
                video_folders = sorted(subfolders)

                for folder_name in video_folders:
                    display_name = folder_name
                    if self.annotation_folder:
                        # Check if annotation file exists for this video
                        annotation_file = os.path.join(self.annotation_folder, f"anno_{folder_name}.json")
                        if os.path.exists(annotation_file):
                            display_name += " (已标注)"
                        else:
                            display_name += " (未标注)"
                    self.video_listbox.insert(tk.END, display_name)

            except Exception as e:
                messagebox.showerror("错误", f"无法列出视频文件夹:\n{e}")

    def _load_track_data(self):
        """Load and convert the track JSON file to match the internal data structure, including metadata"""
        if not self.track_json_path:
            return
        try:
            with open(self.track_json_path, 'r', encoding='utf-8') as f:
                raw_tracks_data = json.load(f)

            self.tracks_data = {}
            self.track_metadata = {} # Initialize metadata storage

            for video_id, video_content in raw_tracks_data.items():
                video_id_str = str(video_id) # Ensure it's a string
                self.tracks_data[video_id_str] = {} # Create a dictionary for this video
                self.track_metadata[video_id_str] = {} # Initialize metadata for this video

                if "anno" in video_content and isinstance(video_content["anno"], list):
                    for track_info in video_content["anno"]:
                        if "tid" in track_info and "trajectory" in track_info:
                            track_id = str(track_info["tid"]) # Ensure track ID is a string
                            trajectory_data = track_info["trajectory"]

                            # Store metadata for this track
                            metadata = {}
                            if "category" in track_info:
                                metadata["category"] = track_info["category"]
                            if "traj_name" in track_info:
                                metadata["traj_name"] = track_info["traj_name"]
                            self.track_metadata[video_id_str][track_id] = metadata

                            if isinstance(trajectory_data, dict):
                                # Convert BBox coordinates to floats and check validity
                                valid_trajectory = {}
                                for frame_idx_str, bbox in trajectory_data.items():
                                    # Check if bbox is a list containing 4 numbers
                                    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                                        try:
                                            # Ensure frame index is in string format, consistent with JSON
                                            valid_trajectory[str(frame_idx_str)] = [float(coord) for coord in bbox]
                                        except (ValueError, TypeError):
                                             print(f"Warning: Video {video_id_str} Track {track_id} Frame {frame_idx_str} has invalid bbox coordinates: {bbox}")
                                    else:
                                        print(f"Warning: Video {video_id_str} Track {track_id} Frame {frame_idx_str} has invalid bbox format or contains non-numeric values: {bbox}")
                                # Only store tracks with valid trajectory data
                                if valid_trajectory:
                                     self.tracks_data[video_id_str][track_id] = valid_trajectory
                                else:
                                     print(f"Info: Track {track_id} in video {video_id_str} has no valid trajectory data.")
                            else:
                                print(f"Warning: 'trajectory' for track {track_id} in video {video_id_str} is not a dictionary.")
                        else:
                            print(f"Warning: A track in video {video_id_str} is missing 'tid' or 'trajectory'.")
                else:
                     print(f"Warning: Data for video {video_id_str} is missing the 'anno' list.")

            messagebox.showinfo("加载成功", f"轨迹数据已加载并处理 {len(self.tracks_data)} 个视频。")
            # If data folder is already set, populate video list after loading tracks
            if self.data_folder:
                 self._populate_video_list()

        except json.JSONDecodeError as e:
             messagebox.showerror("JSON 解析错误", f"无法解析轨迹 JSON 文件:\n{e}")
             self.tracks_data = {}
             self.track_metadata = {}
        except FileNotFoundError:
             messagebox.showerror("文件未找到", f"无法找到轨迹文件:\n{self.track_json_path}")
             self.tracks_data = {}
             self.track_metadata = {}
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("加载错误", f"加载或转换轨迹 JSON 时发生未知错误:\n{e}\n(详情请查看终端)")
            self.tracks_data = {}
            self.track_metadata = {}

    def on_video_select(self, event):
        """Called when a video is selected in the video listbox"""
        selection = self.video_listbox.curselection()
        if not selection:
            return

        # Get the actual video folder name from the listbox item (remove status text)
        listbox_text = self.video_listbox.get(selection[0])
        selected_video_folder_name = listbox_text.split(" (")[0]


        if not self.data_folder or not self.track_json_path:
             messagebox.showwarning("数据未加载", "请先选择数据文件夹和轨迹 JSON 文件。")
             self.video_listbox.selection_clear(selection[0]) # Deselect
             return

        # Ensure JSON data is loaded and converted
        if not self.tracks_data:
            messagebox.showwarning("数据未处理", "轨迹 JSON 未加载或处理失败，请重新选择 JSON 文件。")
            self.video_listbox.selection_clear(selection[0])
            return # Prevent further processing

        if selected_video_folder_name not in self.tracks_data:
             messagebox.showwarning("数据不匹配", f"在加载的轨迹 JSON 数据中未找到视频 '{selected_video_folder_name}' 的轨迹信息。请检查文件夹名称是否与 JSON 中的顶层键完全一致。")
             self.video_listbox.selection_clear(selection[0])
             return # Prevent further processing


        if selected_video_folder_name == self.current_video_folder:
            return # Prevent duplicate loading

        # --- Reset State ---
        self.current_video_folder = selected_video_folder_name
        self.current_video_label.config(text=f"当前视频: {self.current_video_folder}")
        self.frame_files = []
        self.all_frame_indices = []
        self.potential_pairs = []
        self.current_pair_index = -1
        self.current_pair_common_frames = []
        self.current_frame_in_common_index = 0
        self.current_subject_track_id = None
        self.current_object_track_id = None
        self.canvas.delete("all") # Clear canvas
        self.frame_label_var.set("帧: - / -")
        self.tk_image = None # Clear old image reference
        self._reset_frame_controls() # Reset frame controls and Comboboxes
        self._update_pair_display() # Clear track pair display
        self.annotations = [] # Clear annotations for the previous video
        self._update_annotation_display() # Clear annotation display

        # --- Load Frame Files ---
        video_path = os.path.join(self.data_folder, self.current_video_folder)
        try:
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
            all_files = []
            for ext in extensions:
                 all_files.extend(glob.glob(os.path.join(video_path, ext)))

            # *** Very Important: Ensure frames are sorted in the expected order ***
            # Try to sort by numbers in the filename
            try:
                 self.frame_files = sorted(all_files, key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f))) or 0))
            except ValueError:
                 print("Warning: Could not sort frame filenames numerically, using alphabetical order.")
                 self.frame_files = sorted(all_files) # Fallback to alphabetical sort

            self.all_frame_indices = list(range(len(self.frame_files))) # Store original frame indices

            if not self.frame_files:
                 messagebox.showwarning("无帧文件", f"在文件夹 {video_path} 中未找到支持的图片帧文件（jpg, png等）。")
                 self._reset_video_selection() # Reset video related state
                 return

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("错误", f"加载帧文件时出错:\n{e}\n(详情请查看终端)")
            self._reset_video_selection() # Reset video related state
            return

        # --- Load existing annotations if annotation folder is set ---
        if self.annotation_folder and self.current_video_folder:
            annotation_file = os.path.join(self.annotation_folder, f"anno_{self.current_video_folder}.json")
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                        # Assuming the loaded data is a dictionary where the key is the video folder name
                        if self.current_video_folder in loaded_data:
                            self.annotations = loaded_data[self.current_video_folder]
                            self._update_annotation_display()
                            print(f"Loaded {len(self.annotations)} existing annotations for video {self.current_video_folder}.")
                        else:
                            print(f"No annotations found for video {self.current_video_folder} in {annotation_file}.")

                except json.JSONDecodeError:
                     messagebox.showerror("加载错误", f"无法解析现有标注文件: {annotation_file}")
                except Exception as e:
                     traceback.print_exc()
                     messagebox.showerror("加载错误", f"加载现有标注时出错: {e}\n(详情请查看终端)")


        # --- Identify Track Pairs and Select the First One ---
        self._identify_potential_pairs()
        if self.potential_pairs:
            self._select_pair(0) # Automatically select the first track pair
        else:
            messagebox.showinfo("无轨迹对", f"在视频 '{self.current_video_folder}' 中未找到共同存在的轨迹对。")
            self.display_current_frame() # Display the first frame, but don't draw boxes


    def _reset_video_selection(self):
         """Reset video selection related state"""
         self.current_video_folder = None
         self.current_video_label.config(text="当前视频: 无")
         self.frame_files = []
         self.all_frame_indices = []
         self.potential_pairs = []
         self.current_pair_index = -1
         self.current_pair_common_frames = []
         self.current_frame_in_common_index = 0
         self.current_subject_track_id = None
         self.current_object_track_id = None
         self.canvas.delete("all")
         self.frame_label_var.set("帧: - / -")
         self.tk_image = None
         self._reset_frame_controls()
         self._update_pair_display()
         self.annotations = []
         self._update_annotation_display()


    def _reset_frame_controls(self):
        """Reset frame controls and Spinboxes to initial state (now Comboboxes)"""
        self.frame_slider.config(from_=0, to=0)
        self.frame_slider_var.set(0)
        # Clear Combobox options
        self.start_frame_combobox['values'] = []
        self.end_frame_combobox['values'] = []
        self.start_frame_combobox.set("")
        self.end_frame_combobox.set("")


    def _update_pair_display(self):
        """Update the display label for the current track pair"""
        if self.current_subject_track_id is not None and self.current_object_track_id is not None:
            pair_info = f"当前轨迹对: 主语:{self.current_subject_track_id}, 宾语:{self.current_object_track_id}"
            if self.potential_pairs:
                pair_info += f" ({self.current_pair_index + 1}/{len(self.potential_pairs)})"
            self.current_pair_label.config(text=pair_info)
        else:
            self.current_pair_label.config(text="当前轨迹对: 无")


    def _identify_potential_pairs(self):
        """Identify all potential track pairs (Subject, Object) in the current video"""
        self.potential_pairs = []
        self.current_pair_index = -1
        self.current_pair_common_frames = []
        self.current_frame_in_common_index = 0
        self.current_subject_track_id = None
        self.current_object_track_id = None

        if self.current_video_folder and self.current_video_folder in self.tracks_data:
            video_tracks = self.tracks_data[self.current_video_folder]
            track_ids = list(video_tracks.keys())

            # Generate all ordered track pairs (Subject, Object)
            for subject_id, object_id in itertools.permutations(track_ids, 2):
                # Get frame lists for both tracks (convert to integers)
                # Ensure only tracks with valid trajectory data are considered
                if subject_id not in video_tracks or object_id not in video_tracks:
                     continue

                # Ensure frame indices are valid numeric strings
                frames_subject = set(int(f) for f in video_tracks[subject_id].keys() if isinstance(f, str) and f.isdigit())
                frames_object = set(int(f) for f in video_tracks[object_id].keys() if isinstance(f, str) and f.isdigit())


                # Find common frames
                common_frames = sorted(list(frames_subject.intersection(frames_object)))

                if common_frames and len(common_frames) >= 30:
                    # Store the track pair (Subject ID, Object ID) and their list of common frames
                    self.potential_pairs.append({
                        "pair": (subject_id, object_id),
                        "common_frames": common_frames
                    })

        print(f"Found {len(self.potential_pairs)} track pairs for video {self.current_video_folder}.")


    def _select_pair(self, index):
        """Select the track pair at the specified index and update the display"""
        if not self.potential_pairs:
            self._reset_pair_info()
            return

        # Ensure index is valid
        self.current_pair_index = index % len(self.potential_pairs)
        selected_pair_info = self.potential_pairs[self.current_pair_index]

        self.current_subject_track_id, self.current_object_track_id = selected_pair_info["pair"]
        self.current_pair_common_frames = selected_pair_info["common_frames"]
        self.current_frame_in_common_index = 0 # Go back to the first common frame when switching pairs

        self._update_pair_display()
        self._update_frame_controls_for_pair()
        self.display_current_frame() # Display the first frame of the current track pair


    def change_pair(self, delta):
        """Switch to the previous or next track pair"""
        if not self.potential_pairs:
            messagebox.showinfo("无轨迹对", "当前视频没有可切换的轨迹对。")
            return

        new_index = self.current_pair_index + delta
        self._select_pair(new_index)


    def _update_frame_controls_for_pair(self):
        """Update frame controls and Comboboxes based on the current track pair's common frames"""
        num_common_frames = len(self.current_pair_common_frames)
        max_common_index = max(0, num_common_frames - 1)

        # Update slider, range is the index of the common frames list
        self.frame_slider.config(from_=0, to=max_common_index)
        self.frame_slider_var.set(self.current_frame_in_common_index)

        # Update Comboboxes, options are multiples of 15 within the common frame range
        frame_options = []
        if self.current_pair_common_frames:
            min_original_frame = self.current_pair_common_frames[0]
            max_original_frame = self.current_pair_common_frames[-1]

            # Generate multiples of 15 as options
            # Start from min_original_frame, find the first multiple of 15 >= min_original_frame
            start_option = (min_original_frame // 15) * 15
            if start_option < min_original_frame:
                 start_option += 15

            for frame_idx in range(start_option, max_original_frame + 1, 15):
                 if frame_idx >= min_original_frame: # Ensure options are within the common frame range
                     frame_options.append(str(frame_idx)) # Combobox values should be strings

            # If the common frame range is small, may not have multiples of 15, include at least the start and end frames
            if not frame_options and self.current_pair_common_frames:
                 frame_options = [str(self.current_pair_common_frames[0]), str(self.current_pair_common_frames[-1])]
            elif frame_options and str(max_original_frame) not in frame_options:
                 # Ensure the end frame (if not a multiple of 15) is also included as an option
                 frame_options.append(str(max_original_frame))
                 frame_options = sorted(list(set(frame_options)), key=int) # Remove duplicates and sort by integer value


        self.start_frame_combobox['values'] = frame_options
        self.end_frame_combobox['values'] = frame_options

        # Default select the original index of the first and last common frame (if options exist)
        if frame_options:
            self.start_frame_combobox.set(frame_options[0])
            self.end_frame_combobox.set(frame_options[-1])
        else:
            self.start_frame_combobox.set("")
            self.end_frame_combobox.set("")


    def on_slider_move(self, value):
        """Called when the slider moves (based on common frame list index)"""
        if not self.current_pair_common_frames:
            return
        new_index_in_common = int(float(value))
        if new_index_in_common != self.current_frame_in_common_index:
            self.current_frame_in_common_index = new_index_in_common
            self.display_current_frame()

    def change_frame(self, delta):
        """Change frame via button or keyboard (based on common frame list index)"""
        if not self.current_pair_common_frames:
            return
        num_common = len(self.current_pair_common_frames)
        if num_common == 0:
             return

        new_index_in_common = self.current_frame_in_common_index + delta

        # No looping
        new_index_in_common = max(0, min(new_index_in_common, num_common - 1))

        if new_index_in_common != self.current_frame_in_common_index:
            self.current_frame_in_common_index = new_index_in_common
            self.frame_slider_var.set(new_index_in_common) # Update slider
            self.display_current_frame()


    def display_current_frame(self):
        """Display the frame at the current index and draw bounding boxes on the canvas"""
        # Check if there are valid common frames to display
        if not self.current_pair_common_frames or not (0 <= self.current_frame_in_common_index < len(self.current_pair_common_frames)):
            self.canvas.delete("all") # Clear canvas
            # Get current canvas background color for filling
            bg_color = self.canvas.cget('bg')
            if bg_color:
                self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height(), fill=bg_color, outline="")
            self.frame_label_var.set("帧: - / -")
            self.tk_image = None # Clear reference
            return

        # Get the original frame index of the current common frame
        # try:
        
        original_frame_index = self.current_pair_common_frames[self.current_frame_in_common_index]
        first_frame_path = self.frame_files[0]
        frame_folder_path = os.path.dirname(first_frame_path)
        first_filename = os.path.basename(first_frame_path)
        match = re.match(r"(\d+)(\.\w+)", first_filename)
        if match:
            number_width = len(match.group(1))  # e.g. 6
            file_ext = match.group(2)           # e.g. ".jpg"
        else:
            number_width = 6  # 默认宽度
            file_ext = ".jpg"
        frame_filename_format = f"{{:0{number_width}d}}{file_ext}"
        # frame_path = self.frame_files[original_frame_index]
        frame_key = str(original_frame_index) # Frame index in JSON is a string
        frame_key_ = frame_filename_format.format(int(frame_key))
        print(frame_key_)
        frame_path = os.path.join(frame_folder_path, frame_key_)
        # except:
        #     list_tem = self.current_pair_common_frames
        #     list_new = []
        #     for value in list_tem:
        #         list_new.append(value - int(self.start_frame_combobox.get()))
        #     # self.current_pair_common_frames = list_new
        #     original_frame_index = list_new[self.current_frame_in_common_index]
        #     # original_frame_index = self.current_frame_in_common_index - int(self.start_frame_combobox.get())
        #     print(self.current_pair_common_frames)
        #     print(original_frame_index)
        #     frame_path = self.frame_files[original_frame_index]
        #     frame_key = str(original_frame_index + int(self.start_frame_combobox.get())) # Frame index in JSON is a string
        #     print(frame_key)
            # frame_key = str(original_frame_index) # Frame index in JSON is a string


        # Update frame label, show index in common frames list and total count, and original frame index
        self.frame_label_var.set(f"共同帧: {self.current_frame_in_common_index + 1} / {len(self.current_pair_common_frames)} (原始索引: {frame_key})")
        try:
            # --- Load Image ---
            img_pil = Image.open(frame_path).convert("RGB") # Ensure it's RGB
            draw = ImageDraw.Draw(img_pil)
            print(frame_path)

            # --- Draw Bounding Boxes ---
            if self.current_video_folder in self.tracks_data:
                video_tracks = self.tracks_data[self.current_video_folder]
                # print(video_tracks)
                # print(self.current_subject_track_id, self.current_subject_track_id)
                # Draw Subject track (if exists and has data in the current frame)
                if self.current_subject_track_id and self.current_subject_track_id in video_tracks:
                    subject_track_data = video_tracks[self.current_subject_track_id]
                    if frame_key in subject_track_data:
                        bbox_subject = subject_track_data[frame_key] # Already a list of floats
                        # print(bbox_subject)
                        draw.rectangle(bbox_subject, outline=DEFAULT_BOX_COLOR_SUBJECT, width=3) # Thicker line
                        # Get subject category from metadata
                        subject_category = self.track_metadata.get(self.current_video_folder, {}).get(self.current_subject_track_id, {}).get("category", "未知类别")
                        text = f"主语:{self.current_subject_track_id} ({subject_category})"
                        text_pos = (bbox_subject[0], bbox_subject[1] - 18) # Adjust position, leave space
                        # Draw text using the loaded font
                        draw.text(text_pos, text, fill=DEFAULT_BOX_COLOR_SUBJECT, font=self.font)

                # Draw Object track (if exists and has data in the current frame)
                if self.current_object_track_id and self.current_object_track_id in video_tracks:
                    object_track_data = video_tracks[self.current_object_track_id]
                    if frame_key in object_track_data:
                        bbox_object = object_track_data[frame_key] # Already a list of floats
                        draw.rectangle(bbox_object, outline=DEFAULT_BOX_COLOR_OBJECT, width=3) # Thicker line
                        # Get object category from metadata
                        object_category = self.track_metadata.get(self.current_video_folder, {}).get(self.current_object_track_id, {}).get("category", "未知类别")
                        text = f"宾语:{self.current_object_track_id} ({object_category})"
                        text_pos = (bbox_object[0], bbox_object[1] - 18) # Adjust position
                        # Draw text using the loaded font
                        draw.text(text_pos, text, fill=DEFAULT_BOX_COLOR_OBJECT, font=self.font)

            # --- Resize Image to Fit Canvas ---
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1: # Canvas not fully initialized or too small
                 self.tk_image = ImageTk.PhotoImage(img_pil) # Use original image
            else:
                img_ratio = img_pil.width / img_pil.height
                canvas_ratio = canvas_width / canvas_height
                if img_ratio > canvas_ratio: # Image is wider than canvas -> scale by canvas width
                    new_width = canvas_width
                    new_height = int(new_width / img_ratio)
                else: # Image is taller than canvas or same ratio -> scale by canvas height
                    new_height = canvas_height
                    new_width = int(new_height * img_ratio)

                # Use LANCZOS (high quality) or BILINEAR (faster)
                img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.tk_image = ImageTk.PhotoImage(img_resized)

            # --- Display Image on Tkinter Canvas ---
            self.canvas.delete("all") # Clear old content
            # Draw image centered on the canvas
            x_offset = max(0, (canvas_width - self.tk_image.width()) // 2)
            y_offset = max(0, (canvas_height - self.tk_image.height()) // 2)
            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image)

        except FileNotFoundError:
             messagebox.showerror("错误", f"无法找到帧文件:\n{frame_path}")
             self.canvas.delete("all")
             bg_color = self.canvas.cget('bg')
             if bg_color: self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height(), fill=bg_color, outline="")
             self.tk_image = None
        except Exception as e:
             traceback.print_exc()
             messagebox.showerror("显示错误", f"显示帧 {frame_key} 时出错:\n{e}\n(详情请查看终端)")
             self.canvas.delete("all")
             bg_color = self.canvas.cget('bg')
             if bg_color: self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height(), fill=bg_color, outline="")
             self.tk_image = None


    def validate_spinbox_value(self):
        """Method name retained, but now validates Combobox values"""
        # Combobox state="readonly" ensures values are valid options, mainly checks if a value is selected here
        pass


    def add_annotation(self):
        """Add relationship annotation to the list"""
        if self.current_subject_track_id is None or self.current_object_track_id is None:
            messagebox.showwarning("缺少选择", "请先选择一个视频并等待轨迹对自动匹配。")
            return

        if not self.current_video_folder:
             messagebox.showwarning("缺少选择", "请先选择一个视频。")
             return

        # Get selected relationship types
        selected_relationships = []
        for category, listbox in self.relationship_listboxes.items():
             selected_indices = listbox.curselection()
             for i in selected_indices:
                 selected_relationships.append(listbox.get(i))

        if not selected_relationships:
             messagebox.showwarning("缺少选择", "请至少选择一个关系类型。")
             return

        # Get selected frame indices from Comboboxes
        start_frame_str = self.start_frame_combobox.get()
        end_frame_str = self.end_frame_combobox.get()

        if not start_frame_str or not end_frame_str:
             messagebox.showwarning("缺少选择", "请选择起始帧和结束帧。")
             return

        try:
            start_frame = int(start_frame_str)
            end_frame = int(end_frame_str)
        except ValueError:
            messagebox.showerror("输入错误", "起始帧和结束帧必须是有效的数字。")
            return

        # Re-check range and order (although Combobox restricts options, just in case)
        if not self.current_pair_common_frames:
             messagebox.showerror("错误", "当前轨迹对没有共同帧范围。")
             return

        min_original_frame = self.current_pair_common_frames[0]
        max_original_frame = self.current_pair_common_frames[-1]

        if not (min_original_frame <= start_frame <= max_original_frame and min_original_frame <= end_frame <= max_original_frame):
             messagebox.showerror("输入错误", f"帧索引必须在当前轨迹对的共同帧范围 [{min_original_frame} - {max_original_frame}] 之间。")
             return
        if start_frame > end_frame:
            messagebox.showerror("输入错误", "起始帧不能大于结束帧。")
            return

        # Add an annotation for each selected relationship type
        added_count = 0
        for rel_type in selected_relationships:
            annotation = {
                "video_folder": self.current_video_folder,
                "subject_track_id": self.current_subject_track_id,
                "object_track_id": self.current_object_track_id,
                "relationship_type": rel_type, # Store the specific relationship type
                "start_frame": start_frame,
                "end_frame": end_frame
            }
            self.annotations.append(annotation)
            added_count += 1

        self._update_annotation_display()
        messagebox.showinfo("标注添加成功", f"为轨迹对 ({self.current_subject_track_id}, {self.current_object_track_id}) 添加了 {added_count} 条标注 [帧 {start_frame}-{end_frame}]。")

        # --- Clear selected relationships after adding annotation ---
        self._clear_selected_relationships()


    def _clear_selected_relationships(self):
        """Clears the selection in all relationship listboxes."""
        for category, listbox in self.relationship_listboxes.items():
            listbox.selection_clear(0, tk.END)


    def _update_annotation_display(self):
        """Update the list of annotations displayed in the text box"""
        self.annotation_text.config(state=tk.NORMAL) # Allow editing temporarily
        self.annotation_text.delete('1.0', tk.END) # Clear content
        if not self.annotations:
            self.annotation_text.insert(tk.END, "尚未添加任何标注。")
        else:
            # Simplified display for the current video's annotations
            self.annotation_text.insert(tk.END, f"--- 当前视频: {self.current_video_folder or '无'} ---\n")
            if self.annotations:
                 for i, ann in enumerate(self.annotations):
                     text = (f"  {i+1}. 主语: {ann['subject_track_id']}, 宾语: {ann['object_track_id']}, "
                             f"关系: {ann['relationship_type']}, "
                             f"帧: [{ann['start_frame']} - {ann['end_frame']}]\n")
                     self.annotation_text.insert(tk.END, text)
            else:
                 self.annotation_text.insert(tk.END, "  当前视频没有标注。\n")


        self.annotation_text.config(state=tk.DISABLED) # Disable editing

    def save_annotations(self):
        """Save all annotations for the current video to a JSON file in the annotation folder"""
        if not self.annotations:
            messagebox.showinfo("无标注", "没有可保存的标注。")
            return

        if not self.current_video_folder:
             messagebox.showwarning("保存失败", "当前未选择视频，无法确定文件名。")
             return

        if not self.annotation_folder:
             messagebox.showwarning("标注文件夹未选", "请先选择一个标注文件夹。")
             return

        # Construct the save path automatically
        save_path = os.path.join(self.annotation_folder, f"anno_{self.current_video_folder}.json")

        try:
            # Data structure organized by video for saving
            output_data = {}
            # We only save the annotations for the current video in its specific file
            output_data[self.current_video_folder] = []
            for ann in self.annotations:
                # Remove the video_folder key as it's the top-level key in this file structure
                ann_data_to_save = {k: v for k, v in ann.items() if k != 'video_folder'}
                output_data[self.current_video_folder].append(ann_data_to_save)


            with open(save_path, 'w', encoding='utf-8') as f:
                # Use indent for readability
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("保存成功", f"标注已成功保存到:\n{save_path}")

            # After successful save, update the video list display to show \"(已标注)\"
            self._populate_video_list()

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("保存失败", f"无法保存标注文件:\\n{e}\\n(Details in terminal)")


# --- Main Program Entry Point ---
if __name__ == "__main__":
    # Use argparse to handle command line arguments
    parser = argparse.ArgumentParser(description="Video Relationship Annotation Tool")
    parser.add_argument("--data_folder", help="Path to the folder containing video frame subfolders")
    parser.add_argument("--track_json", help="Path to the track JSON file")
    parser.add_argument("--annotation_folder", help="Path to the folder for saving/loading annotations")

    args = parser.parse_args()

    root = tk.Tk()
    # Pass initial paths from arguments to the VideoAnnotator constructor
    app = VideoAnnotator(root, data_folder=args.data_folder, track_json_path=args.track_json, annotation_folder=args.annotation_folder)
    root.mainloop()

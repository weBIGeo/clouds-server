import sys
import os
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QGridLayout,
        QWidget,
        QSlider,
        QLabel,
        QComboBox,
        QVBoxLayout,
        QGroupBox,
        QFrame,
        QStatusBar,
        QCheckBox,
        QHBoxLayout,
        QPushButton,
        QSizePolicy,
        QSpacerItem,
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent
except ImportError:
    print("Error: PyQt6 is missing. Run: pip install PyQt6")
    sys.exit(1)

try:
    import matplotlib
except ImportError:
    print("Error: matplotlib is missing. Run: pip install matplotlib==3.10.8")
    sys.exit(1)

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from io_ktx import Ktx2
except ImportError as e:
    print(f"Error: io_ktx.py not found.")
    sys.exit(1)

try:
    from bc4 import BC4Compressor
except ImportError as e:
    print(f"Error: bc4.py not found.")
    sys.exit(1)


class KtxViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KTX2 Viewer")
        self.resize(1300, 950)
        self.setAcceptDrops(True)

        # --- State ---
        self.vol = None
        self.indices = [0, 0, 0]  # Z, Y, X
        self.max_indices = [0, 0, 0]
        self.mode = "Slice"
        self.cache_projections = {}
        self.maximized_idx = None

        # Pan/Zoom
        self.press_xy = None
        self.pan_active = False

        # Contrast
        self.fmt_min = 0.0
        self.fmt_max = 1.0
        self.slider_steps = 1000

        self.setup_ui()

        if len(sys.argv) > 1:
            self.load_file(sys.argv[1])

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # 1. Grid Area
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(4)
        self.main_layout.addWidget(self.grid_container, stretch=1)

        # --- Viewports ---
        self.canvases = []
        self.axes = []
        self.images = []
        self.frames = []
        self.grid_pos = [(0, 0), (1, 0), (0, 1)]

        view_titles = ["Top (XY)", "Front (XZ)", "Side (YZ)"]

        for i in range(3):
            fig = Figure(facecolor="#f0f0f0")
            canvas = FigureCanvas(fig)

            # Full viewport axes
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            # Events
            canvas.mpl_connect(
                "button_press_event", lambda e, idx=i: self.on_click(e, idx)
            )
            canvas.mpl_connect("button_release_event", self.on_release)
            canvas.mpl_connect(
                "motion_notify_event", lambda e, idx=i: self.on_drag(e, idx)
            )
            canvas.mpl_connect("scroll_event", lambda e, idx=i: self.on_scroll(e, idx))

            self.canvases.append(canvas)
            self.axes.append(ax)
            self.images.append(None)

            # Frame
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setStyleSheet("background: #333; border: 1px solid #555;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(0, 0, 0, 0)

            # Title
            lbl = QLabel(view_titles[i], canvas)
            lbl.setStyleSheet(
                "background: rgba(0,0,0,0.6); color: white; padding: 4px; font-weight: bold;"
            )
            lbl.move(5, 5)

            fl.addWidget(canvas)
            self.frames.append(frame)
            r, c = self.grid_pos[i]
            self.grid_layout.addWidget(frame, r, c)

        # --- 3D View ---
        self.fig3d = Figure(facecolor="#e0e0e0")
        self.canvas3d = FigureCanvas(self.fig3d)
        self.ax3d = self.fig3d.add_subplot(111, projection="3d")
        self.ax3d.view_init(elev=25, azim=-45)
        self.ax3d.mouse_init(rotate_btn=1, zoom_btn=[])
        self.ax3d._pan_btn = []

        self.frame3d = QFrame()
        self.frame3d.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame3d.setStyleSheet("border: 1px solid #555;")
        fl3 = QVBoxLayout(self.frame3d)
        fl3.addWidget(self.canvas3d)

        self.grid_layout.addWidget(self.frame3d, 1, 1)
        self.frames.append(self.frame3d)
        self.grid_pos.append((1, 1))

        # --- Controls Panel ---
        self.ctrl_box = QGroupBox("Controls")
        self.ctrl_box.setFixedHeight(180)
        cl = QGridLayout(self.ctrl_box)

        # [Left] Navigation
        nav_group = QWidget()
        nav_layout = QGridLayout(nav_group)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(5)

        self.sliders = []
        self.lbl_indices = []
        labels = ["Z (Height)", "Y (Depth)", "X (Width)"]

        for i in range(3):
            lbl = QLabel(f"{labels[i]}: 0")
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 100)
            sl.setEnabled(False)
            sl.valueChanged.connect(lambda v, axis=i: self.update_axis(axis, v))

            self.lbl_indices.append(lbl)
            self.sliders.append(sl)
            nav_layout.addWidget(lbl, i, 0)
            nav_layout.addWidget(sl, i, 1)

        self.lbl_mip = QLabel("MIP: 0")
        self.sl_mip = QSlider(Qt.Orientation.Horizontal)
        self.sl_mip.setRange(0, 0)
        self.sl_mip.setEnabled(False)
        self.sl_mip.valueChanged.connect(self.change_mip_level)
        nav_layout.addWidget(self.lbl_mip, 3, 0)
        nav_layout.addWidget(self.sl_mip, 3, 1)

        nav_layout.addWidget(QLabel("Mode:"), 4, 0)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Slice Mode", "MIP (Max)", "AIP (Avg)"])
        self.combo_mode.currentTextChanged.connect(self.change_mode)
        nav_layout.addWidget(self.combo_mode, 4, 1)

        cl.addWidget(nav_group, 0, 0)

        # [Right] Contrast & View
        contrast_group = QGroupBox("Windowing & Display")
        c_layout = QHBoxLayout(contrast_group)
        c_layout.setContentsMargins(10, 10, 10, 10)
        c_layout.setSpacing(10)

        # Sliders Container
        sliders_cont = QWidget()
        sl_grid = QGridLayout(sliders_cont)
        sl_grid.setContentsMargins(0, 0, 0, 0)

        # Labels for Format Range
        self.lbl_range_min = QLabel("-")
        self.lbl_range_max = QLabel("-")
        self.lbl_range_min.setStyleSheet("color: #777; font-size: 10px;")
        self.lbl_range_max.setStyleSheet("color: #777; font-size: 10px;")

        # Min Slider Row
        sl_grid.addWidget(QLabel("Min:"), 0, 0)
        self.sl_cmin = QSlider(Qt.Orientation.Horizontal)
        self.sl_cmin.setRange(0, self.slider_steps)
        self.sl_cmin.valueChanged.connect(self.update_contrast_from_sliders)
        sl_grid.addWidget(self.sl_cmin, 0, 1)
        self.lbl_cur_min = QLabel("0.00")
        self.lbl_cur_min.setFixedWidth(45)
        sl_grid.addWidget(self.lbl_cur_min, 0, 2)
        sl_grid.addWidget(self.lbl_range_min, 1, 1)  # Label below slider

        # Max Slider Row
        sl_grid.addWidget(QLabel("Max:"), 2, 0)
        self.sl_cmax = QSlider(Qt.Orientation.Horizontal)
        self.sl_cmax.setRange(0, self.slider_steps)
        self.sl_cmax.setValue(self.slider_steps)
        self.sl_cmax.valueChanged.connect(self.update_contrast_from_sliders)
        sl_grid.addWidget(self.sl_cmax, 2, 1)
        self.lbl_cur_max = QLabel("1.00")
        self.lbl_cur_max.setFixedWidth(45)
        sl_grid.addWidget(self.lbl_cur_max, 2, 2)
        sl_grid.addWidget(self.lbl_range_max, 3, 1)  # Label below slider

        c_layout.addWidget(sliders_cont, stretch=1)

        # Buttons Container
        btn_cont = QWidget()
        btn_layout = QVBoxLayout(btn_cont)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_auto = QPushButton("Auto")
        self.btn_auto.clicked.connect(self.auto_contrast)
        self.btn_auto.setFixedHeight(25)

        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.clicked.connect(self.reset_all_views)
        self.btn_reset_view.setFixedHeight(25)

        self.chk_cmap = QCheckBox("Turbo")
        self.chk_cmap.toggled.connect(self.refresh_views)

        btn_layout.addWidget(self.btn_auto)
        btn_layout.addWidget(self.btn_reset_view)
        btn_layout.addSpacing(5)
        btn_layout.addWidget(self.chk_cmap)
        btn_layout.addStretch()

        c_layout.addWidget(btn_cont)

        cl.addWidget(contrast_group, 0, 1)
        cl.setColumnStretch(0, 1)
        cl.setColumnStretch(1, 1)

        self.main_layout.addWidget(self.ctrl_box)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Double-click view to maximize.")

    # -------------------------------------------------------------------------
    # Logic: Interaction
    # -------------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.load_file(files[0])

    def on_click(self, event, view_idx):
        if event.inaxes != self.axes[view_idx]:
            return
        if event.dblclick and event.button == 1:
            self.toggle_maximize(view_idx)
            return
        if event.button == 3:  # Right click
            self.pan_active = True
            self.press_xy = (event.xdata, event.ydata)

    def on_release(self, event):
        self.pan_active = False
        self.press_xy = None

    def on_drag(self, event, view_idx):
        if not self.pan_active or event.inaxes != self.axes[view_idx]:
            return
        if self.press_xy is None or event.xdata is None:
            return

        ax = self.axes[view_idx]
        dx = event.xdata - self.press_xy[0]
        dy = event.ydata - self.press_xy[1]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self.canvases[view_idx].draw_idle()

    def on_scroll(self, event, view_idx):
        if event.inaxes != self.axes[view_idx]:
            return
        ax = self.axes[view_idx]
        base = 1.2
        factor = 1 / base if event.button == "up" else base

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        x, y = event.xdata, event.ydata

        w = (xlim[1] - xlim[0]) * factor
        h = (ylim[1] - ylim[0]) * factor

        # Zoom towards mouse pointer
        new_x0 = x - (x - xlim[0]) * factor
        new_y0 = y - (y - ylim[0]) * factor

        ax.set_xlim(new_x0, new_x0 + w)
        ax.set_ylim(new_y0, new_y0 + h)
        self.canvases[view_idx].draw_idle()

    def toggle_maximize(self, idx):
        if self.maximized_idx is not None:
            self.grid_layout.removeWidget(self.frames[self.maximized_idx])
            for i, f in enumerate(self.frames):
                f.show()
                r, c = self.grid_pos[i]
                self.grid_layout.addWidget(f, r, c, 1, 1)
            self.maximized_idx = None
            self.ctrl_box.show()
        else:
            for i, f in enumerate(self.frames):
                if i != idx:
                    self.grid_layout.removeWidget(f)
                    f.hide()
            self.ctrl_box.hide()
            self.grid_layout.addWidget(self.frames[idx], 0, 0, 2, 2)
            self.frames[idx].show()
            self.maximized_idx = idx

    def reset_all_views(self):
        if self.vol is None:
            return

        for ax in self.axes:
            ax.autoscale(enable=True, axis="both", tight=True)

        self.ax3d.view_init(elev=25, azim=-45)
        self.canvas3d.draw_idle()

        for c in self.canvases:
            c.draw_idle()

    # -------------------------------------------------------------------------
    # Logic: Data
    # -------------------------------------------------------------------------

    def load_file(self, filepath):
        self.status.showMessage(f"Loading {os.path.basename(filepath)}...")
        QApplication.processEvents()

        try:
            raw_mips, meta = Ktx2.load(filepath)

            # --- NEW CODE START ---
            # Pre-process all MIP levels
            self.all_mips = []

            # Base dimensions for calculating MIP sizes
            # Note: Assuming meta["_ktx_block_dimensions"] exists for BC4 or we derive from mips[0] shape
            base_w, base_h, base_d = 0, 0, 0
            is_bc4 = meta.get("_ktx_format", "").startswith("BC4")

            if is_bc4:
                base_w, base_h, base_d = meta["_ktx_block_dimensions"]

            for i, vol in enumerate(raw_mips):
                # Calculate dimensions for this MIP level
                if is_bc4:
                    mw = max(1, base_w >> i)
                    mh = max(1, base_h >> i)
                    md = max(1, base_d >> i)
                    vol = BC4Compressor.decompress(vol, mw, mh, md)

                if vol.ndim == 3:
                    vol = vol[..., np.newaxis]
                if vol.dtype == np.float16:
                    vol = vol.astype(np.float32)

                D, H, W, C = vol.shape
                if C == 2:
                    new_vol = np.zeros((D, H, W, 3), dtype=vol.dtype)
                    new_vol[..., 0:2] = vol
                    vol = new_vol

                self.all_mips.append(vol)

            # Set initial volume to MIP 0
            self.curr_mip_idx = 0
            vol = self.all_mips[0]
            D, H, W, C = vol.shape

            # Reset State
            self.vol = None
            self.D, self.H, self.W, self.C = D, H, W, C
            self.is_gray = C == 1
            self.max_indices = [D - 1, H - 1, W - 1]
            self.indices = [D // 2, H // 2, W // 2]

            # Clear Axes
            for ax in self.axes:
                ax.cla()
                ax.axis("off")
            self.images = [None, None, None]

            # Format Range Logic
            if np.issubdtype(vol.dtype, np.integer):
                # Signed Int (SNORM usually)
                if np.issubdtype(vol.dtype, np.signedinteger):
                    info = np.iinfo(vol.dtype)
                    self.fmt_min, self.fmt_max = info.min, info.max
                else:
                    # Unsigned
                    info = np.iinfo(vol.dtype)
                    self.fmt_min, self.fmt_max = 0, info.max
            else:
                self.fmt_min, self.fmt_max = 0.0, 1.0

            # Update Labels
            self.lbl_range_min.setText(str(self.fmt_min))
            self.lbl_range_max.setText(str(self.fmt_max))

            # Auto-Scan Data Range
            step = max(1, int((vol.size / 500000) ** (1 / 3)))
            subset = vol[::step, ::step, ::step]
            self.data_min = float(np.min(subset))
            self.data_max = float(np.max(subset))
            if self.data_min == self.data_max:
                self.data_max += 0.001

            # Sliders reset
            for i in range(3):
                self.sliders[i].blockSignals(True)
                self.sliders[i].setRange(0, self.max_indices[i])
                self.sliders[i].setValue(self.indices[i])
                self.sliders[i].setEnabled(True)
                self.sliders[i].blockSignals(False)

            # Reset MIP Slider
            self.sl_mip.blockSignals(True)
            self.sl_mip.setRange(0, len(self.all_mips) - 1)
            self.sl_mip.setValue(0)
            self.sl_mip.setEnabled(True)
            self.lbl_mip.setText(f"MIP: 0")
            self.sl_mip.blockSignals(False)

            self.cache_projections = {}
            self.vol = vol

            # Reset view limits
            self.reset_all_views()

            # Reset Contrast
            self.set_sliders_to_range(self.fmt_min, self.fmt_max)

            self.status.showMessage(
                f"Loaded {os.path.basename(filepath)} ({W}x{H}x{D})"
            )
            self.update_all_views(redraw_3d=True)

        except Exception as e:
            self.status.showMessage(f"Error: {e}")
            import traceback

            traceback.print_exc()

    def change_mip_level(self, level):
        if not hasattr(self, "all_mips") or level >= len(self.all_mips):
            return

        self.curr_mip_idx = level
        self.lbl_mip.setText(f"MIP: {level}")

        # 1. Update Volume
        new_vol = self.all_mips[level]
        self.vol = new_vol
        self.cache_projections = {}  # Clear projection cache for new size

        # 2. Update Dimensions
        old_D, old_H, old_W = self.D, self.H, self.W
        self.D, self.H, self.W, self.C = self.vol.shape
        self.max_indices = [self.D - 1, self.H - 1, self.W - 1]

        # 3. Update Indices (Scale position relative to new size)
        # Z
        self.indices[0] = int((self.indices[0] / max(1, old_D)) * self.D)
        # Y
        self.indices[1] = int((self.indices[1] / max(1, old_H)) * self.H)
        # X
        self.indices[2] = int((self.indices[2] / max(1, old_W)) * self.W)

        # 4. Update Spatial Sliders Range and Value
        for i in range(3):
            self.sliders[i].blockSignals(True)
            self.sliders[i].setRange(0, self.max_indices[i])
            self.sliders[i].setValue(self.indices[i])
            self.lbl_indices[i].setText(f"{['Z','Y','X'][i]}: {self.indices[i]}")
            self.sliders[i].blockSignals(False)

        # 5. Refresh
        self.reset_all_views()  # Reset zoom/pan as aspect/scale changed
        self.update_all_views(redraw_3d=True)

    # --- Contrast ---

    def set_sliders_to_range(self, vmin, vmax):
        span = self.fmt_max - self.fmt_min
        if span == 0:
            span = 1
        n_min = (vmin - self.fmt_min) / span
        n_max = (vmax - self.fmt_min) / span
        v_min = int(np.clip(n_min, 0, 1) * self.slider_steps)
        v_max = int(np.clip(n_max, 0, 1) * self.slider_steps)

        self.sl_cmin.blockSignals(True)
        self.sl_cmax.blockSignals(True)
        self.sl_cmin.setValue(v_min)
        self.sl_cmax.setValue(v_max)
        self.sl_cmin.blockSignals(False)
        self.sl_cmax.blockSignals(False)
        self.update_contrast_from_sliders()

    def update_contrast_from_sliders(self):
        s_min = self.sl_cmin.value()
        s_max = self.sl_cmax.value()
        if s_min >= s_max:
            if self.sender() == self.sl_cmin:
                s_min = s_max - 10
                self.sl_cmin.setValue(s_min)
            else:
                s_max = s_min + 10
                self.sl_cmax.setValue(s_max)

        span = self.fmt_max - self.fmt_min
        self.cur_min = self.fmt_min + (s_min / self.slider_steps) * span
        self.cur_max = self.fmt_min + (s_max / self.slider_steps) * span

        self.lbl_cur_min.setText(f"{self.cur_min:.2f}")
        self.lbl_cur_max.setText(f"{self.cur_max:.2f}")
        self.refresh_views()

    def auto_contrast(self):
        self.set_sliders_to_range(self.data_min, self.data_max)

    # --- Rendering ---

    def change_mode(self, text):
        self.mode = "Slice" if "Slice" in text else ("MIP" if "MIP" in text else "AIP")
        enabled = self.mode == "Slice"
        for sl in self.sliders:
            sl.setEnabled(enabled)
        self.update_all_views(redraw_3d=True)

    def refresh_views(self):
        self.update_all_views(redraw_3d=False)

    def update_axis(self, axis, val):
        if self.vol is None:
            return
        self.indices[axis] = val
        self.lbl_indices[axis].setText(f"{['Z','Y','X'][axis]}: {val}")
        if self.mode == "Slice":
            self.update_all_views(redraw_3d=True)

    def get_projection(self, axis, mode):
        key = (axis, mode)
        if key in self.cache_projections:
            return self.cache_projections[key]
        if mode == "MIP":
            proj = np.max(self.vol, axis=axis)
        else:
            proj = np.mean(self.vol, axis=axis).astype(self.vol.dtype)
        self.cache_projections[key] = proj
        return proj

    def update_all_views(self, redraw_3d=True):
        if self.vol is None:
            return

        # Always Physical Aspect (Equal)
        aspect = "equal"

        cmap = None
        if self.is_gray:
            cmap = "turbo" if self.chk_cmap.isChecked() else "gray"

        z, y, x = self.indices

        views = []
        if self.mode == "Slice":
            views.append(self.vol[z])
            views.append(self.vol[:, y, :])
            views.append(self.vol[:, :, x])
        else:
            views.append(self.get_projection(0, self.mode))
            views.append(self.get_projection(1, self.mode))
            views.append(self.get_projection(2, self.mode))

        for i in range(3):
            data = views[i]
            if i > 0:
                data = np.flipud(data)
            if self.is_gray:
                data = data.squeeze()

            if self.images[i] is None:
                # FIX: Use adjustable='datalim' to allow panning into void space
                self.axes[i].set_aspect(aspect, adjustable="datalim")
                self.images[i] = self.axes[i].imshow(
                    data,
                    cmap=cmap,
                    vmin=self.cur_min,
                    vmax=self.cur_max,
                    interpolation="nearest",
                )
            else:
                self.images[i].set_data(data)
                self.images[i].set_clim(self.cur_min, self.cur_max)
                self.images[i].set_cmap(cmap)
                self.axes[i].set_aspect(aspect, adjustable="datalim")

            self.canvases[i].draw_idle()

        if redraw_3d and self.mode == "Slice":
            self.draw_3d_cube()

    def draw_3d_cube(self):
        self.ax3d.clear()
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_xlim(0, self.W)
        self.ax3d.set_ylim(0, self.H)
        self.ax3d.set_zlim(0, self.D)

        max_dim = max(self.W, self.H, self.D)
        self.ax3d.set_box_aspect((self.W / max_dim, self.H / max_dim, self.D / max_dim))

        z, y, x = self.indices

        p1 = [[0, 0, z], [self.W, 0, z], [self.W, self.H, z], [0, self.H, z]]
        self.ax3d.add_collection3d(
            Poly3DCollection([p1], color="cyan", alpha=0.2, edgecolors="blue")
        )

        p2 = [[0, y, 0], [self.W, y, 0], [self.W, y, self.D], [0, y, self.D]]
        self.ax3d.add_collection3d(
            Poly3DCollection([p2], color="magenta", alpha=0.2, edgecolors="magenta")
        )

        p3 = [[x, 0, 0], [x, self.H, 0], [x, self.H, self.D], [x, 0, self.D]]
        self.ax3d.add_collection3d(
            Poly3DCollection([p3], color="yellow", alpha=0.2, edgecolors="orange")
        )

        self.canvas3d.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KtxViewer()
    window.show()
    sys.exit(app.exec())

#############################################################################
# weBIGeo Clouds
# Copyright (C) 2026 Wendelin Muth
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#############################################################################

import sys
import os
import re
import math
import io
import requests
import numpy as np
import xarray as xr
try:
    from PIL import Image
except ImportError:
    print("Error: PIL is missing. Run: pip install Pillow==12.1.1")
    sys.exit(1)
from concurrent.futures import ThreadPoolExecutor

from numba import jit, prange

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QGridLayout, QPushButton, QLabel, QSlider, QComboBox, QFileDialog, 
        QGroupBox, QCheckBox, QStatusBar, QListWidget, QProgressBar, QMessageBox,
        QLineEdit
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
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
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# --- Configuration ---
REGION_CFG = {"lat_min": 46.2, "lat_max": 49.2, "lon_min": 9.4, "lon_max": 17.4}

@jit(nopython=True, parallel=True)
def slice_at_altitude(data_cube, hhl_cube, target_alt, out_slice):
    nz_data, ny, nx = data_cube.shape
    nz_hhl = hhl_cube.shape[0]
    is_staggered = (nz_data == nz_hhl)
    for y in prange(ny):
        for x in range(nx):
            sfc_h = hhl_cube[nz_hhl - 1, y, x]
            top_h = hhl_cube[0, y, x]
            if target_alt < sfc_h or target_alt > top_h:
                out_slice[y, x] = np.nan
                continue
            found = False
            if not is_staggered:
                for k in range(nz_data):
                    h_upper = hhl_cube[k, y, x]
                    h_lower = hhl_cube[k + 1, y, x]
                    if h_lower <= target_alt <= h_upper:
                        out_slice[y, x] = data_cube[k, y, x]
                        found = True
                        break
            else:
                dist_min = np.inf
                k_closest = -1
                for k in range(nz_data):
                    h = hhl_cube[k, y, x]
                    dist = abs(h - target_alt)
                    if dist < dist_min:
                        dist_min = dist
                        k_closest = k
                if k_closest >= 0:
                    out_slice[y, x] = data_cube[k_closest, y, x]
                    found = True
            if not found:
                out_slice[y, x] = np.nan

# --- Advanced Map Fetcher with Reprojection ---
class MapFetcher:
    @staticmethod
    def latlon_to_merc(lat, lon):
        """Converts Lat/Lon to Web Mercator (0..1 range)"""
        r_major = 6378137.000
        x = r_major * math.radians(lon)
        scale = x / lon
        y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 + lat * (math.pi / 180.0) / 2.0)) * scale
        
        # Normalize to 0..1 for tiles (where 0 is top-left in standard XYZ tiling?)
        # Standard Tile Coordinates:
        # x = (lon + 180) / 360
        # y = (1 - log(tan(lat) + sec(lat)) / pi) / 2
        
        n_x = (lon + 180.0) / 360.0
        n_y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0
        return n_x, n_y

    @staticmethod
    def deg2num(lat_deg, lon_deg, zoom):
        n = 2.0 ** zoom
        nx, ny = MapFetcher.latlon_to_merc(lat_deg, lon_deg)
        return nx * n, ny * n

    @staticmethod
    def get_reprojected_map(lat_min, lat_max, lon_min, lon_max, zoom=8):
        """
        Fetches Web Mercator tiles and re-samples them to a Linear Latitude Grid.
        This fixes the South-Shift issue.
        """
        # 1. Determine Tile Bounds
        x_min_f, y_max_f = MapFetcher.deg2num(lat_min, lon_min, zoom) # lat_min is bottom (high Y)
        x_max_f, y_min_f = MapFetcher.deg2num(lat_max, lon_max, zoom) # lat_max is top (low Y)
        
        x_min, x_max = int(math.floor(x_min_f)), int(math.ceil(x_max_f))
        y_min, y_max = int(math.floor(y_min_f)), int(math.ceil(y_max_f))
        
        w_tiles = x_max - x_min
        h_tiles = y_max - y_min
        
        # 2. Fetch Raw Stitch
        raw_w = w_tiles * 256
        raw_h = h_tiles * 256
        raw_img = Image.new('RGB', (raw_w, raw_h))
        
        url_template = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
        session = requests.Session()
        session.headers.update({'User-Agent': 'GribViewer/1.0'})

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                try:
                    resp = session.get(url_template.format(z=zoom, x=x, y=y), timeout=2)
                    if resp.status_code == 200:
                        tile = Image.open(io.BytesIO(resp.content))
                        raw_img.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))
                except: pass
        
        # 3. Reprojection: Web Mercator -> Equirectangular (Linear Lat/Lon)
        # We assume X axis (Lon) is linear in both, just scaling.
        # Y axis needs row re-sampling.
        
        src_arr = np.array(raw_img) # (H, W, 3)
        target_h = int(raw_h) # Keep resolution approx same
        target_w = int(raw_w)
        
        reproj_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Iterate over TARGET rows (Linear Latitude)
        # y_dst=0 -> lat_max (Top), y_dst=H -> lat_min (Bottom)
        
        # Pre-compute source Y indices for every target Y
        target_ys = np.arange(target_h)
        # Map target pixel Y to Latitude
        lats = lat_max - (target_ys / target_h) * (lat_max - lat_min)
        
        # Map Latitude to Source Mercator Y (relative to tile grid)
        # n_y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0
        # pixel_y = (n_y * 2^zoom - y_min) * 256
        
        n = 2.0 ** zoom
        # Vectorized calculation
        lat_rads = np.radians(lats)
        # Clip to avoid poles if necessary (austria is safe)
        merc_ys_norm = (1.0 - np.arcsinh(np.tan(lat_rads)) / np.pi) / 2.0
        src_ys_float = (merc_ys_norm * n - y_min) * 256.0
        
        # Clip to bounds
        src_ys = np.clip(src_ys_float, 0, raw_h - 1).astype(int)
        
        # Resample rows
        # Since X is linear in both, we just copy the whole row 
        # (Technically X scale might differ slightly due to tile alignment, 
        #  but crop usually handles it. We assume X linearity is sufficient).
        
        # However, the aspect ratio of the raw stitch might not match the target perfectly.
        # We should also resample X if we want perfection. 
        # For simplicity: We fetch a tile block. We will just return the image 
        # AND the exact bounds of the tile block. 
        # BUT, the tile block boundaries are Mercator aligned, not LatLon aligned.
        # Correct approach:
        # We reproject the vertical axis of the *entire* stitched block.
        # The result covers the LatLon box defined by the tile corners.
        
        # Calculate bounds of the raw tile grid in Lat/Lon
        top_lat_grid, left_lon_grid = MapFetcher.num2deg(x_min, y_min, zoom) # Mercator -> LatLon
        bot_lat_grid, right_lon_grid = MapFetcher.num2deg(x_max, y_max, zoom)
        
        # Now we re-run the row mapping using THESE bounds
        lats = top_lat_grid - (target_ys / target_h) * (top_lat_grid - bot_lat_grid)
        lat_rads = np.radians(lats)
        merc_ys_norm = (1.0 - np.arcsinh(np.tan(lat_rads)) / np.pi) / 2.0
        src_ys_float = (merc_ys_norm * n - y_min) * 256.0
        src_ys = np.clip(src_ys_float, 0, raw_h - 1).astype(int)
        
        reproj_img = src_arr[src_ys, :, :]
        
        final_pil = Image.fromarray(reproj_img)
        
        # Extent is now the Lat/Lon bounds of the tile grid
        extent = [left_lon_grid, right_lon_grid, bot_lat_grid, top_lat_grid]
        
        return final_pil, extent
        
    @staticmethod
    def num2deg(xtile, ytile, zoom):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)


# --- Workers ---
class GribLoaderWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, str, tuple, dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_list, var_name, slices=None):
        super().__init__()
        self.file_list = file_list; self.var_name = var_name; self.slices = slices; self._run = True
    
    def run(self):
        if not self.file_list: return
        try:
            with xr.open_dataset(self.file_list[0], engine='cfgrib', backend_kwargs={'indexpath':''}) as ds0:
                k = list(ds0.data_vars)[0]
                d0 = ds0[k].isel(latitude=self.slices[0], longitude=self.slices[1]).values.squeeze() if self.slices else ds0[k].values.squeeze()
                h, w = d0.shape
            
            vol = np.zeros((len(self.file_list), h, w), dtype=np.float32)
            
            def read(i, p):
                if not self._run: return i, None
                try:
                    with xr.open_dataset(p, engine='cfgrib', backend_kwargs={'indexpath':''}) as ds:
                        k = list(ds.data_vars)[0]
                        v = ds[k].isel(latitude=self.slices[0], longitude=self.slices[1]).values.squeeze() if self.slices else ds[k].values.squeeze()
                        return i, v
                except: return i, None

            with ThreadPoolExecutor(max_workers=4) as ex:
                for f in [ex.submit(read, i, f) for i, f in enumerate(self.file_list)]:
                    i, v = f.result()
                    if v is not None: vol[i] = v
                    self.progress.emit(int((i+1)/len(self.file_list)*100))
            
            self.finished.emit(vol, self.var_name, vol.shape, REGION_CFG.copy())
        except Exception as e: self.error.emit(str(e))

class MapLoaderWorker(QThread):
    finished = pyqtSignal(object, list)
    def run(self):
        # Fetch slightly larger area
        img, ext = MapFetcher.get_reprojected_map(
            REGION_CFG['lat_min']-0.2, REGION_CFG['lat_max']+0.2,
            REGION_CFG['lon_min']-0.2, REGION_CFG['lon_max']+0.2,
            zoom=9 # Higher zoom for better precision
        )
        if img: self.finished.emit(img, ext)

# --- Main Window ---
class GribViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ICON-D2 Advanced Viewer")
        self.resize(1300, 950)
        
        self.hhl_cube = None; self.data_cube = None; self.map_img = None
        self.roi_slices = None; self.coords_lat = None; self.coords_lon = None
        self.marker_pos = None
        self.altitude_m = 2000
        self.slice_indices = [0,0,0]
        self.available_vars = {}
        self.view_mode = "Top (XY)"
        self.vmin, self.vmax = 0.0, 1.0
        self.is_staggered = (self.data_cube.shape[0] == self.hhl_cube.shape[0]) if self.hhl_cube is not None else False

        self.init_ui()
        self.load_map_bg()

    def init_ui(self):
        main = QWidget(); self.setCentralWidget(main); layout = QHBoxLayout(main)
        lp = QWidget(); lp.setFixedWidth(320); lpl = QVBoxLayout(lp)
        
        gb = QGroupBox("Data Source"); gbl = QVBoxLayout(gb)
        self.btn_hhl = QPushButton("Select HHL Dir"); self.btn_hhl.clicked.connect(self.sel_hhl)
        self.btn_dat = QPushButton("Select Data Dir"); self.btn_dat.clicked.connect(self.sel_dat)
        gbl.addWidget(self.btn_hhl); gbl.addWidget(self.btn_dat); lpl.addWidget(gb)
        
        self.lst = QListWidget(); self.lst.itemClicked.connect(self.sel_var)
        lpl.addWidget(QLabel("Variables:")); lpl.addWidget(self.lst)
        self.prog = QProgressBar(); self.prog.setVisible(False); lpl.addWidget(self.prog)
        
        gbv = QGroupBox("View"); gvl = QVBoxLayout(gbv)
        self.cmb = QComboBox(); self.cmb.addItems(["Top (XY)", "Side (XZ)", "Side (YZ)", "Projection (Sum)"])
        self.cmb.currentTextChanged.connect(self.set_mode)
        self.chk_met = QCheckBox("Vertical: Meters"); self.chk_met.setEnabled(False); self.chk_met.toggled.connect(self.toggle_met)
        self.chk_map = QCheckBox("Show Map"); self.chk_map.setChecked(True); self.chk_map.toggled.connect(self.redraw)
        gvl.addWidget(self.cmb); gvl.addWidget(self.chk_met); gvl.addWidget(self.chk_map); lpl.addWidget(gbv)

        gbm = QGroupBox("Marker"); gml = QHBoxLayout(gbm)
        self.inp_mk = QLineEdit(); self.inp_mk.setPlaceholderText("Lat, Lon"); gml.addWidget(self.inp_mk)
        b_mk = QPushButton("Set"); b_mk.clicked.connect(self.set_mark); gml.addWidget(b_mk)
        lpl.addWidget(gbm)
        
        gbs = QGroupBox("Slicing"); self.gls = QGridLayout(gbs)
        self.sls = []; self.lbls = []
        names = ["Level", "Lat", "Lon"]
        for i in range(3):
            l = QLabel(f"{names[i]}: 0"); s = QSlider(Qt.Orientation.Horizontal); s.setEnabled(False)
            s.valueChanged.connect(lambda v, x=i: self.on_slide(x, v))
            self.sls.append(s); self.lbls.append(l)
            self.gls.addWidget(l, i, 0); self.gls.addWidget(s, i, 1)
        lpl.addWidget(gbs)
        
        gbc = QGroupBox("Contrast"); cl = QVBoxLayout(gbc)
        self.smin = QSlider(Qt.Orientation.Horizontal); self.smax = QSlider(Qt.Orientation.Horizontal)
        self.smin.setRange(0,1000); self.smax.setRange(0,1000); self.smax.setValue(1000)
        self.smin.valueChanged.connect(self.cont); self.smax.valueChanged.connect(self.cont)
        cl.addWidget(QLabel("Min")); cl.addWidget(self.smin); cl.addWidget(QLabel("Max")); cl.addWidget(self.smax)
        cl.addWidget(QPushButton("Auto", clicked=self.auto_cont)); lpl.addWidget(gbc)
        layout.addWidget(lp)

        self.fig = Figure(facecolor="#F0F0F0"); self.can = FigureCanvas(self.fig); self.ax = self.fig.add_subplot(111)
        rp = QWidget(); rpl = QVBoxLayout(rp); rpl.addWidget(NavigationToolbar(self.can, self)); rpl.addWidget(self.can)
        layout.addWidget(rp); self.stat = QStatusBar(); self.setStatusBar(self.stat)
        # Track whether the user has panned/zoomed and save limits
        self._user_set_lims = False
        self._saved_xlim = None
        self._saved_ylim = None
        # Connect canvas events to detect user interaction (mouse release and scroll)
        try:
            self.can.mpl_connect('button_release_event', self._on_mouse_release)
            self.can.mpl_connect('scroll_event', self._on_mouse_release)
        except Exception:
            pass

    # --- Logic ---
    def load_map_bg(self):
        self.mw = MapLoaderWorker()
        self.mw.finished.connect(lambda i, e: (setattr(self, 'map_img', i), setattr(self, 'map_extent', e), self.redraw()))
        self.mw.start()

    def calc_roi(self, path):
        try:
            with xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath':''}) as ds:
                lat = ds['latitude'].values if 'latitude' in ds else ds['lat'].values
                lon = ds['longitude'].values if 'longitude' in ds else ds['lon'].values
                if lat.ndim==1:
                    ym = (lat>=REGION_CFG['lat_min'])&(lat<=REGION_CFG['lat_max'])
                    xm = (lon>=REGION_CFG['lon_min'])&(lon<=REGION_CFG['lon_max'])
                    yi, xi = np.where(ym)[0], np.where(xm)[0]
                    self.roi_slices = (slice(yi.min(), yi.max()+1), slice(xi.min(), xi.max()+1))
                    self.coords_lat = lat[self.roi_slices[0]]
                    self.coords_lon = lon[self.roi_slices[1]]
        except: pass

    def scan(self, d):
        m = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".grib2"):
                ma = re.search(r"([a-zA-Z0-9_]+)_(\d+)", f)
                if ma:
                    v, l = ma.groups()
                    if v.endswith('_'): v=v[:-1]
                    if v not in m: m[v]={}
                    m[v][int(l)] = os.path.join(d, f)
        return m

    def sel_hhl(self):
        d = QFileDialog.getExistingDirectory(self, "HHL Directory")
        if not d: return
        self.load_hhl_dir(d)
        
        # --- Auto-Detect Data Directory ---
        # Look for sibling folders: ../000, ../001, etc.
        parent = os.path.dirname(d)
        candidates = []
        try:
            for item in sorted(os.listdir(parent)):
                # Check for 3-digit folder names (e.g. 000, 005)
                if os.path.isdir(os.path.join(parent, item)) and re.match(r"^\d{3}$", item):
                    candidates.append(os.path.join(parent, item))
            
            if candidates:
                # Pick the first one (lowest number, e.g. 000)
                auto_data = candidates[0]
                self.stat.showMessage(f"Auto-detected data dir: {os.path.basename(auto_data)}")
                self.load_data_dir(auto_data)
            else:
                self.stat.showMessage("No timestamped data folders (e.g. '000') found in parent.")
        except Exception as e:
            print(f"Auto-detect failed: {e}")

    def load_hhl_dir(self, d):
        m = self.scan(d)
        k = next((x for x in m if 'hhl' in x.lower()), None)
        if not k: return
        fs = [m[k][l] for l in sorted(m[k])]
        if not self.roi_slices: self.calc_roi(fs[0])
        self.gw = GribLoaderWorker(fs, "hhl", self.roi_slices)
        self.gw.finished.connect(self.done_hhl)
        self.gw.start()

    def done_hhl(self, v, n, s, m):
        self.hhl_cube = v; self.h_centers = 0.5*(v[:-1]+v[1:])
        self.chk_met.setEnabled(True)
        self.btn_hhl.setStyleSheet("background:#cfc")
        if self.data_cube is not None: self.redraw()

    def sel_dat(self):
        d = QFileDialog.getExistingDirectory(self, "Data Directory")
        if d: self.load_data_dir(d)

    def load_data_dir(self, d):
        self.available_vars = {}
        self.lst.clear()
        m = self.scan(d)
        for v, ls in m.items():
            self.available_vars[v] = [ls[l] for l in sorted(ls)]
            self.lst.addItem(f"{v} ({len(ls)})")
        self.btn_dat.setStyleSheet("background:#cfc")
        if not self.roi_slices and m: self.calc_roi(m[list(m.keys())[0]][0])

    def sel_var(self, item):
        v = item.text().split(" (")[0]
        self.lst.setEnabled(False); self.prog.setVisible(True)
        self.gw = GribLoaderWorker(self.available_vars[v], v, self.roi_slices)
        self.gw.finished.connect(self.done_dat)
        self.gw.start()

    def done_dat(self, v, n, s, m):
        self.data_cube = v; self.current_var = n; self.dims = s
        self.lst.setEnabled(True); self.prog.setVisible(False)
        self.data_min, self.data_max = np.nanmin(v), np.nanmax(v)
        print(f"Data loaded: {n} with shape {s}, min={self.data_min:.6f}, max={self.data_max:.6f}", flush=True)
        # Reset sliders
        for i in range(3):
            self.sls[i].blockSignals(True); self.sls[i].setEnabled(True)
            mx = s[i]-1
            if i==0 and self.chk_met.isChecked(): mx=14000
            self.sls[i].setRange(0, mx)
            self.slice_indices[i] = s[i]//2
            if i==0 and self.chk_met.isChecked(): self.slice_indices[i]=2000; self.sls[i].setValue(2000)
            else: self.sls[i].setValue(s[i]//2)
            self.sls[i].blockSignals(False)
            self.on_slide(i, self.sls[i].value())
        self.auto_cont()
        self.redraw(keep_lims=True)

    def toggle_met(self, chk):
        self.sls[0].blockSignals(True)
        if chk:
            self.sls[0].setRange(0, 14000); self.sls[0].setValue(self.altitude_m)
            self.lbls[0].setText(f"Height: {self.altitude_m}m")
        else:
            if self.dims:
                self.sls[0].setRange(0, self.dims[0]-1); self.sls[0].setValue(0)
                self.slice_indices[0]=0
                self.lbls[0].setText("Level: 0")
        self.sls[0].blockSignals(False)
        self.redraw(keep_lims=True)

    def on_slide(self, i, v):
        if i==0 and self.chk_met.isChecked(): self.altitude_m = v; self.lbls[0].setText(f"Height: {v}m")
        else: self.slice_indices[i] = v; self.lbls[i].setText(f"{['Lvl','Lat','Lon'][i]}: {v}")
        if i==1 and self.coords_lat is not None: self.lbls[1].setText(f"Lat: {self.coords_lat[v]:.2f}")
        if i==2 and self.coords_lon is not None: self.lbls[2].setText(f"Lon: {self.coords_lon[v]:.2f}")
        self.redraw(keep_lims=True)

    def set_mode(self, t):
        self.view_mode = t.split(" -")[0]
        self.sls[0].setEnabled("Projection" not in self.view_mode)
        self.redraw(keep_lims=True)

    def set_mark(self):
        try: self.marker_pos = list(map(float, self.inp_mk.text().split(','))); self.redraw(keep_lims=True)
        except: pass

    def _on_mouse_release(self, event):
        # Save current x/y limits as user-set limits if they differ from the data extent
        try:
            if self.coords_lon is None or self.coords_lat is None:
                return
            ext = [self.coords_lon[0], self.coords_lon[-1], self.coords_lat[0], self.coords_lat[-1]]
            cur_xlim = self.ax.get_xlim(); cur_ylim = self.ax.get_ylim()
            # If limits are meaningfully different from the full extent, treat as user-set
            dx = abs(cur_xlim[0] - ext[0]) + abs(cur_xlim[1] - ext[1])
            dy = abs(cur_ylim[0] - ext[2]) + abs(cur_ylim[1] - ext[3])
            if dx > 1e-6 or dy > 1e-6:
                self._user_set_lims = True
                self._saved_xlim = cur_xlim
                self._saved_ylim = cur_ylim
        except Exception:
            pass

    def auto_cont(self):
        if self.data_cube is None: return
        self.smin.setValue(0); self.smax.setValue(1000); self.cont()

    def cont(self):
        if self.data_cube is None: return
        # Adjust range for Projection mode
        mx = self.data_max
        if "Projection" in self.view_mode: mx *= self.data_cube.shape[0]*0.5
        
        self.vmin = self.data_min + (self.smin.value()/1000)*(mx-self.data_min)
        self.vmax = self.data_min + (self.smax.value()/1000)*(mx-self.data_min)
        self.redraw(keep_lims=True)

    def redraw(self, keep_lims=False):
        if self.data_cube is None:
            return

        # Compute extent if coordinates available
        if self.coords_lon is None or self.coords_lat is None:
            ext = None
        else:
            ext = [self.coords_lon[0], self.coords_lon[-1], self.coords_lat[0], self.coords_lat[-1]]

        use_m = self.chk_met.isChecked() and self.hhl_cube is not None

        # Determine whether to restore saved limits: only when user previously panned/zoomed
        restore_limits = False
        if keep_lims and self._user_set_lims and self._saved_xlim is not None and ("Top" in self.view_mode or "Projection" in self.view_mode):
            restore_limits = True

        # Clear axes before drawing
        self.ax.clear()

        def draw_ov():
            if self.chk_map.isChecked() and self.map_img:
                self.ax.imshow(self.map_img, extent=self.map_extent, aspect='auto', zorder=0)
            if self.marker_pos:
                self.ax.plot(self.marker_pos[1], self.marker_pos[0], 'rx', ms=12, mew=2, zorder=10)

        if "Top" in self.view_mode:
            draw_ov()
            if use_m:
                out = np.zeros((self.data_cube.shape[1], self.data_cube.shape[2]), dtype=np.float32)
                slice_at_altitude(self.data_cube, self.hhl_cube, float(self.altitude_m), out)
                d = out; t = f"{self.current_var} @ {self.altitude_m}m"
            else:
                d = self.data_cube[self.slice_indices[0]]; t = f"{self.current_var} @ Lvl {self.slice_indices[0]}"

            self.ax.imshow(d, cmap="turbo", vmin=self.vmin, vmax=self.vmax, origin='lower', extent=ext, zorder=1, alpha=0.8)
            self.ax.set_title(t)
            self.ax.set_aspect(1.0/np.cos(np.deg2rad(48.0)))

            # Crosshairs
            self.ax.axhline(self.coords_lat[self.slice_indices[1]], c='w', alpha=0.3)
            self.ax.axvline(self.coords_lon[self.slice_indices[2]], c='w', alpha=0.3)

        elif "Projection" in self.view_mode:
            draw_ov()
            tot = np.nansum(self.data_cube, axis=0)
            self.ax.imshow(tot, cmap="turbo", vmin=self.vmin, vmax=self.vmax, origin='lower', extent=ext, zorder=1, alpha=0.8)
            self.ax.set_aspect(1.0/np.cos(np.deg2rad(48.0)))

        elif "Side (XZ)" in self.view_mode:
            y=self.slice_indices[1]; d=self.data_cube[:, y, :]
            if use_m:
                h = self.hhl_cube[:, y, :] if self.is_staggered else self.h_centers[:, y, :]
                X, _ = np.meshgrid(self.coords_lon, np.arange(d.shape[0]))
                self.ax.pcolormesh(X, h, d, cmap="turbo", vmin=self.vmin, vmax=self.vmax, shading='auto')
                if self.marker_pos: self.ax.axvline(self.marker_pos[1], c='r', ls='--')
                self.ax.set_ylabel("Altitude (m)")
            else:
                self.ax.imshow(d, cmap="turbo", vmin=self.vmin, vmax=self.vmax, aspect='auto')
                self.ax.set_ylabel("Level")
            self.ax.set_title(f"Lat: {self.coords_lat[y]:.2f}")

        elif "Side (YZ)" in self.view_mode:
            x=self.slice_indices[2]; d=self.data_cube[:, :, x]
            if use_m:
                h = self.hhl_cube[:, :, x] if self.is_staggered else self.h_centers[:, :, x]
                X, _ = np.meshgrid(self.coords_lat, np.arange(d.shape[0]))
                self.ax.pcolormesh(X, h, d, cmap="turbo", vmin=self.vmin, vmax=self.vmax, shading='auto')
                if self.marker_pos: self.ax.axvline(self.marker_pos[0], c='r', ls='--')
                self.ax.set_ylabel("Altitude (m)")
            else:
                self.ax.imshow(d, cmap="turbo", vmin=self.vmin, vmax=self.vmax, aspect='auto')
                self.ax.set_ylabel("Level")
            self.ax.set_title(f"Lon: {self.coords_lon[x]:.2f}")

        # Restore saved user limits when appropriate
        if restore_limits:
            try:
                self.ax.set_xlim(self._saved_xlim)
                self.ax.set_ylim(self._saved_ylim)
            except Exception:
                pass

        self.can.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GribViewer()
    window.show()
    sys.exit(app.exec())
import sys
import os
import shutil
import subprocess
import glob
import copy
import math
import numpy as np
from stl import mesh

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QScrollArea, QGroupBox, 
                             QFormLayout, QLineEdit, QPushButton, QComboBox,QAbstractSpinBox,
                             QDoubleSpinBox, QTabWidget, QTextEdit, QLabel, QSpinBox,
                             QDockWidget, QToolBar, QStatusBar, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QFont, QAction
import re

try:
    import matplotlib
    matplotlib.use('qtagg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: Matplotlib is not installed. Plotting will be disabled.")

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: PyVista or pyvistaqt not installed. 3D View will be disabled.")

import datetime
import webbrowser
from PyQt6.QtWidgets import QDialog
from PyQt6.QtCore import QTimer

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GeminiWorker(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_key, prompt, history=[]):
        super().__init__()
        self.api_key = api_key
        self.prompt = prompt
        self.history = history

    def run(self):
        if not HAS_GENAI:
            self.error_occurred.emit("Error: 'google-genai' library missing.")
            return
        try:
            client = genai.Client(api_key=self.api_key)
            chat = client.chats.create(model="gemini-2.5-flash", history=self.history)
            response = chat.send_message(self.prompt)
            self.result_ready.emit(response.text)
        except Exception as e:
            self.error_occurred.emit(f"AI Error: {str(e)}")


class WSLWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, wsl_dir, command_sequence):
        super().__init__()
        self.wsl_dir = wsl_dir
        self.command_sequence = command_sequence
        self.process = None
        self._is_stopped = False

    def run(self):
        wsl_cmd = f'wsl -e bash -ic "cd {self.wsl_dir} && {self.command_sequence}"'
        self.log_signal.emit(f"\n> Executing: {wsl_cmd}\n")
        
        try:
            self.process = subprocess.Popen(
                wsl_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            for line in iter(self.process.stdout.readline, ''):
                if self._is_stopped:
                    break
                self.log_signal.emit(line.strip())
            
            self.process.stdout.close()
            return_code = self.process.wait()
            
            if not self._is_stopped:
                self.finished_signal.emit(return_code)

        except Exception as e:
            if not self._is_stopped:
                self.log_signal.emit(f"ERROR: {str(e)}")
                self.finished_signal.emit(-1)

    def stop(self):
        self._is_stopped = True
        if self.process:
            self.log_signal.emit("\n> Stopping process... Please wait.")
            self.process.terminate()
            self.finished_signal.emit(130)


class SalomeWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, salome_bat_path, script_path, work_dir):
        super().__init__()
        self.salome_bat_path = salome_bat_path
        self.script_path = script_path
        self.work_dir = work_dir
        self.process = None
        self._is_stopped = False

    def run(self):
        safe_script_path = os.path.normpath(self.script_path)
        cmd = [self.salome_bat_path, "-t", safe_script_path]
        
        self.log_signal.emit(f"\n> Executing SALOME: {' '.join(cmd)}\n")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            for line in iter(self.process.stdout.readline, ''):
                if self._is_stopped:
                    break
                self.log_signal.emit(line.strip())
            
            self.process.stdout.close()
            return_code = self.process.wait()
            
            if not self._is_stopped:
                self.finished_signal.emit(return_code)

        except Exception as e:
            if not self._is_stopped:
                self.log_signal.emit(f"SALOME ERROR: {str(e)}")
                self.finished_signal.emit(-1)

    def stop(self):
        self._is_stopped = True
        if self.process:
            self.log_signal.emit("\n> Stopping SALOME process... Please wait.")
            self.process.terminate()
            self.finished_signal.emit(130)


class SafeSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, event): event.ignore()


class SafeDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, event): event.ignore()


class YPlusDialog(QDialog):
    def __init__(self, velocity, length, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Y+ Calculator (Water)")
        self.setFixedWidth(300)
        layout = QFormLayout(self)
        self.spin_u = SafeDoubleSpinBox(); self.spin_u.setRange(0.1, 100); self.spin_u.setValue(velocity)
        self.spin_l = SafeDoubleSpinBox(); self.spin_l.setRange(0.1, 500); self.spin_l.setValue(length)
        self.spin_target_y = SafeDoubleSpinBox(); self.spin_target_y.setRange(0.1, 300); self.spin_target_y.setValue(30)
        self.lbl_result = QLabel("Result: ..."); self.lbl_result.setStyleSheet("font-weight: bold; color: blue;")
        btn_calc = QPushButton("Calculate")
        btn_calc.clicked.connect(self.calculate)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.accept)
        layout.addRow("Velocity (m/s):", self.spin_u)
        layout.addRow("Ref. Length (m):", self.spin_l)
        layout.addRow("Target Y+:", self.spin_target_y)
        layout.addRow(btn_calc)
        layout.addRow(self.lbl_result)
        layout.addRow(self.btn_apply)
        self.calculated_thickness = 0.0

    def calculate(self):
        U, L, y_plus = self.spin_u.value(), self.spin_l.value(), self.spin_target_y.value()
        rho, nu = 998.8, 1.09e-6
        Re = (U * L) / nu
        Cf = (2 * math.log10(Re) - 0.65)**(-2.3) if Re < 1e9 else 0.058 * (Re**-0.2)
        tau_w = 0.5 * rho * (U**2) * Cf
        u_star = math.sqrt(tau_w / rho)
        ds = (y_plus * nu) / u_star
        self.calculated_thickness = ds
        self.lbl_result.setText(f"First Layer: {ds:.6f} m")
        self.btn_apply.setEnabled(True)


class TowingTankGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Towing Tank Ultimate")
        self.setGeometry(50, 50, 1200, 800)
        
        self.create_menu_bar()
        self.create_tool_bar()
        self.create_status_bar()
         
        self.init_central_layout()

        self.chat_history = []

        self.working_dir = None

        self.current_process_type = "" 
        self.residuals_data = {}
        self.residuals_time = {}
        self.current_sim_time = 0.0
        self.seen_fields_this_step = set()
        
        self.sys_timer = QTimer()
        self.sys_timer.timeout.connect(self.update_system_monitor)
        self.sys_timer.start(2000)
        
        self.refresh_wsl()

    def create_menu_bar(self):
        menubar = self.menuBar()

        menu_file = menubar.addMenu("File")
        
        act_new = QAction("New Project", self)
        act_new.triggered.connect(self.new_project)
        menu_file.addAction(act_new)

        act_open = QAction("Open Project", self)
        act_open.triggered.connect(self.open_project)
        menu_file.addAction(act_open)
        
        menu_file.addSeparator()
        
        act_import_stl = QAction("Import STL...", self)
        act_import_stl.triggered.connect(self.import_stl)
        menu_file.addAction(act_import_stl)

        act_export_csv = QAction("Export CSV...", self)
        act_export_csv.triggered.connect(self.export_csv_action)
        menu_file.addAction(act_export_csv)

        menu_sim = menubar.addMenu("Simulation")
        
        act_setup = QAction("Setup Case", self)
        act_setup.triggered.connect(self.setup_case)
        menu_sim.addAction(act_setup)

        act_mesh = QAction("Generate Mesh", self)
        act_mesh.triggered.connect(self.generate_mesh)
        menu_sim.addAction(act_mesh)

        menu_sim.addSeparator()

        act_run = QAction("RUN", self)
        act_run.triggered.connect(self.run_simulation)
        menu_sim.addAction(act_run)

        act_stop = QAction("STOP", self)
        act_stop.triggered.connect(self.stop_simulation)
        menu_sim.addAction(act_stop)

        menu_sim.addSeparator()

        act_clean = QAction("Clean Case", self)
        act_clean.triggered.connect(self.clean_case_action)
        menu_sim.addAction(act_clean)

        menu_view = menubar.addMenu("View")
        
        self.act_show_model = QAction("Show Model", self, checkable=True)
        self.act_show_model.setChecked(True)
        self.act_show_model.triggered.connect(self.toggle_view_model)
        menu_view.addAction(self.act_show_model)

        self.act_show_edges = QAction("Show Edges", self, checkable=True)
        self.act_show_edges.triggered.connect(self.toggle_view_edges)
        menu_view.addAction(self.act_show_edges)

        self.act_show_mesh = QAction("Show Mesh", self, checkable=True)
        self.act_show_mesh.triggered.connect(self.toggle_view_mesh)
        menu_view.addAction(self.act_show_mesh)

        menu_tools = menubar.addMenu("Tools")
        
        act_console = QAction("Show/Hide Console", self)
        act_console.triggered.connect(self.toggle_console_action)
        menu_tools.addAction(act_console)

        act_ai = QAction("Show/Hide AI Assistant", self)
        act_ai.triggered.connect(self.toggle_ai_action)
        menu_tools.addAction(act_ai)

        act_plot = QAction("Plot Results", self)
        act_plot.triggered.connect(self.plot_results_action)
        menu_tools.addAction(act_plot)

        act_paraview = QAction("ParaView", self)
        act_paraview.triggered.connect(self.open_paraview)
        menu_tools.addAction(act_paraview)

        menu_settings = menubar.addMenu("Settings")

        act_set_paraview = QAction("ParaView Path...", self)
        act_set_paraview.triggered.connect(self.set_paraview_path_action)
        menu_settings.addAction(act_set_paraview)

        act_set_unity = QAction("Unity VR Path...", self)
        act_set_unity.triggered.connect(self.set_unity_vr_path_action)
        menu_settings.addAction(act_set_unity)

        act_set_salome = QAction("SALOME Path...", self)
        act_set_salome.triggered.connect(self.set_salome_path_action)
        menu_settings.addAction(act_set_salome)

        menu_help = menubar.addMenu("Help")
        
        act_guide = QAction("User Guide", self)
        act_guide.triggered.connect(self.show_user_guide)
        menu_help.addAction(act_guide)

        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about)
        menu_help.addAction(act_about)

    def toggle_ai_action(self):
        if hasattr(self, 'right_panel'):
            current_state = self.right_panel.isVisible()
            self.right_panel.setVisible(not current_state)

    def export_csv_action(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Warning", "No project opened.")
            return
            
        search = os.path.join(self.working_dir, "postProcessing", "forces", "*", "forces.dat")
        files = sorted(glob.glob(search))
        if not files:
            QMessageBox.information(self, "Info", "No force data found to export. Please run the simulation first.")
            return
            
        csv_path, _ = QFileDialog.getSaveFileName(self, "Export CSV", os.path.join(self.working_dir, "forces_export.csv"), "CSV Files (*.csv)")
        if not csv_path:
            return
            
        t_list, drag_list = [], []
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line: 
                        continue
                    clean_line = line.replace('(', '').replace(')', '')
                    parts = clean_line.split()
                    if len(parts) >= 10:
                        try:
                            time_val = float(parts[0])
                            px = float(parts[1])
                            vx = float(parts[4])
                            pox = float(parts[7])
                            t_list.append(str(time_val))
                            drag_list.append(str(px + vx + pox))
                        except ValueError:
                            pass
                            
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Time(s),Total_Drag_X(N)\n")
            for t_val, d_val in zip(t_list, drag_list):
                f.write(f"{t_val},{d_val}\n")
                
        self.log(f"Force data successfully exported to {csv_path}")
        QMessageBox.information(self, "Success", "CSV export completed!")

    def stop_simulation(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()

    def clean_case_action(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Warning", "No project opened.")
            return
            
        reply = QMessageBox.question(self, "Confirm Clean", "Are you sure you want to clean the case? This will delete all mesh and simulation results.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
            
        if hasattr(self, 'plotter') and self.plotter:
            self.plotter.clear()
            self.plotter.add_axes()
            
        if HAS_MATPLOTLIB and hasattr(self, 'ax_resid'):
            self.ax_resid.clear()
            self.canvas_resid.draw()
            self.ax.clear()
            self.canvas.draw()
            
        self.log("Cleaning case data...")
        
        folders_to_remove = ["processor", "postProcessing"]
        
        for item in os.listdir(self.working_dir):
            item_path = os.path.join(self.working_dir, item)
            
            if os.path.isdir(item_path):
                try:
                    float(item)
                    import shutil
                    shutil.rmtree(item_path, ignore_errors=True)
                    continue
                except ValueError:
                    pass
                    
                if any(item.startswith(f) for f in folders_to_remove):
                    import shutil
                    shutil.rmtree(item_path, ignore_errors=True)
                    
            elif os.path.isfile(item_path) and item.endswith(".foam"):
                try:
                    os.remove(item_path)
                except OSError:
                    pass
                    
        polymesh_dir = os.path.join(self.working_dir, "constant", "polyMesh")
        if os.path.exists(polymesh_dir):
            import shutil
            shutil.rmtree(polymesh_dir, ignore_errors=True)
            
        self.residuals_data.clear()
        if hasattr(self, 'residuals_time'):
            self.residuals_time.clear()
        self.seen_fields_this_step.clear()
        self.current_sim_time = 0.0
        
        self.log("Case cleaned successfully. You can now Setup Case and Generate Mesh again.")

    def toggle_view_model(self, checked):
        if hasattr(self, 'plotter') and self.plotter:
            if "hull_actor" in self.plotter.actors:
                self.plotter.actors["hull_actor"].SetVisibility(checked)
                self.plotter.render()

    def toggle_view_edges(self, checked):
        if hasattr(self, 'plotter') and self.plotter:
            for key, actor in self.plotter.actors.items():
                if hasattr(actor, 'prop'):
                    actor.prop.show_edges = checked
            self.plotter.render()

    def toggle_view_mesh(self, checked):
        if hasattr(self, 'plotter') and self.plotter:
            if "slice_actor" in self.plotter.actors:
                self.plotter.actors["slice_actor"].SetVisibility(checked)
                self.plotter.render()

    def plot_results_action(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Warning", "No project opened.")
            return
            
        self.log("Plotting force results from postProcessing data...")
        self.plot_forces_on_tab()
        QMessageBox.information(self, "Success", "Plot updated! Please check the 'Results' tab in the center panel.")

    def show_user_guide(self):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
            
        guide_path = os.path.join(base_path, "user_guide.html")
        
        if os.path.exists(guide_path):
            webbrowser.open(f"file:///{guide_path}")
        else:
            QMessageBox.warning(self, "Error", "Cannot find user_guide.html in the application folder.")

    def show_about(self):
        about_text = (
            "<h3>CFD Ultimate</h3>"
            "<p><b>Version:</b> 1.0.0</p>"
            "<p>Automation software for Aerodynamics and Hydrodynamics simulations using the OpenFOAM solver.</p>"
            "<p>Developed on Python, PyQt6, and PyVista platforms.</p>"
        )
        QMessageBox.about(self, "About", about_text)

    def toggle_console_action(self):
        if hasattr(self, 'console_widget'):
            current_state = self.console_widget.isVisible()
            self.console_widget.setVisible(not current_state)

    def init_central_layout(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.main_vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_horizontal_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = self.create_left_panel()
        center_panel = self.create_center_panel()
        self.right_panel = self.create_right_panel()

        self.main_horizontal_splitter.addWidget(left_panel)
        self.main_horizontal_splitter.addWidget(center_panel)
        self.main_horizontal_splitter.addWidget(self.right_panel)
        self.main_horizontal_splitter.setSizes([350, 900, 350])
        self.main_horizontal_splitter.setCollapsible(0, False)

        self.main_vertical_splitter.addWidget(self.main_horizontal_splitter)

        self.create_bottom_console_widget()
        self.main_vertical_splitter.addWidget(self.console_widget)
        self.main_vertical_splitter.setSizes([800, 200])

        self.main_layout.addWidget(self.main_vertical_splitter)

    def create_bottom_console_widget(self):
        self.console_widget = QWidget()
        layout = QVBoxLayout(self.console_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel(" Console Output")
        lbl.setStyleSheet("background: #333; color: #00cc00; font-weight: bold;")
        layout.addWidget(lbl)
        
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("background: #1e1e1e; color: #00ff00; font-family: Consolas, Courier New;")
        layout.addWidget(self.console_text)

    def create_tool_bar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        self.act_new = QAction("New Project", self)
        self.act_new.triggered.connect(self.new_project)
        toolbar.addAction(self.act_new)

        self.act_open = QAction("Open Project", self)
        self.act_open.triggered.connect(self.open_project)
        toolbar.addAction(self.act_open)

        toolbar.addSeparator()

        self.act_setup = QAction("Setup Case", self)
        self.act_setup.triggered.connect(self.setup_case)
        toolbar.addAction(self.act_setup)

        self.act_mesh = QAction("Generate Mesh", self)
        self.act_mesh.triggered.connect(self.generate_mesh)
        toolbar.addAction(self.act_mesh)

        self.act_run = QAction("RUN", self)
        self.act_run.triggered.connect(self.run_simulation)
        toolbar.addAction(self.act_run)

        self.act_stop = QAction("STOP", self)
        self.act_stop.triggered.connect(self.stop_simulation)
        self.act_stop.setEnabled(False)
        toolbar.addAction(self.act_stop)
        toolbar.addSeparator()

        self.act_plot = QAction("Plot Results", self)
        self.act_plot.triggered.connect(self.plot_results_action)
        toolbar.addAction(self.act_plot)

        self.act_paraview = QAction("ParaView", self)
        self.act_paraview.triggered.connect(self.open_paraview)
        toolbar.addAction(self.act_paraview)

        self.act_unity = QAction("Unity VR", self)
        self.act_unity.triggered.connect(self.launch_unity_vr)
        toolbar.addAction(self.act_unity)

    def create_left_panel(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        content_widget.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { padding: 3px; }
        """)

        group_wsl = QGroupBox("Project Info")
        form_wsl = QFormLayout()
        
        wsl_layout = QHBoxLayout()
        self.combo_wsl = QComboBox()
        self.combo_wsl.addItems(["Ubuntu-22.04", "Ubuntu-20.04", "Debian"])
        
        self.btn_refresh_wsl = QPushButton("Refresh")
        self.btn_refresh_wsl.clicked.connect(self.refresh_wsl)
        
        wsl_layout.addWidget(self.combo_wsl)
        wsl_layout.addWidget(self.btn_refresh_wsl)
        
        form_wsl.addRow("WSL Linux:", wsl_layout)
        group_wsl.setLayout(form_wsl)
        layout.addWidget(group_wsl)

        group_geom = QGroupBox("Geometry")
        form_geom = QFormLayout()
        
        stl_layout = QHBoxLayout()
        self.stl_input = QLineEdit()
        self.stl_input.setPlaceholderText("Select .stl file...")
        self.stl_input.setReadOnly(True)
        
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.import_stl)
        
        stl_layout.addWidget(self.stl_input)
        stl_layout.addWidget(btn_browse)
        form_geom.addRow("STL File:", stl_layout)
        
        self.combo_units = QComboBox()
        self.combo_units.addItems(["Meters", "Centimeters", "Millimeters"])
        self.combo_units.currentIndexChanged.connect(self.process_geometry)
        form_geom.addRow("Units:", self.combo_units)
        
        self.spin_rot = SafeDoubleSpinBox()
        self.spin_rot.setRange(-360.0, 360.0)
        self.spin_rot.setValue(0.0)
        self.spin_rot.valueChanged.connect(self.process_geometry)
        form_geom.addRow("Rotation (Z deg):", self.spin_rot)
        
        group_geom.setLayout(form_geom)
        layout.addWidget(group_geom)

        group_domain = QGroupBox("BlockMesh Domain")
        form_domain = QFormLayout()
        
        self.input_L = QLineEdit("20.0")
        self.input_W = QLineEdit("10.0")
        self.input_H = QLineEdit("10.0")
        self.input_wl = QLineEdit("5.0")
        
        form_domain.addRow("Tank Length (m):", self.input_L)
        form_domain.addRow("Tank Width (m):", self.input_W)
        form_domain.addRow("Tank Height (m):", self.input_H)
        form_domain.addRow("Water Level (m):", self.input_wl)
        
        group_domain.setLayout(form_domain)
        layout.addWidget(group_domain)

        group_mesh = QGroupBox("Mesh Settings")
        form_mesh = QFormLayout()
        
        self.spin_refine = SafeSpinBox()
        self.spin_refine.setRange(0, 5)
        self.spin_refine.setValue(2)

        self.combo_mesher = QComboBox()
        self.combo_mesher.addItems(["OpenFOAM (snappyHexMesh)", "SALOME"])

        self.combo_layer_mode = QComboBox()
        self.combo_layer_mode.addItems(["Auto (Y+ Calculator)", "Custom Input"])
        self.combo_layer_mode.currentIndexChanged.connect(self.toggle_layer_mode)

        self.btn_yplus = QPushButton("Y+ Calculator")
        self.btn_yplus.clicked.connect(self.open_yplus_tool)
        
        self.spin_layer_thick = SafeDoubleSpinBox()
        self.spin_layer_thick.setDecimals(6)
        self.spin_layer_thick.setRange(0.000001, 1.0)
        self.spin_layer_thick.setSingleStep(0.001)
        self.spin_layer_thick.setValue(0.005)
        self.spin_layer_thick.setEnabled(False)
        self.spin_layer_thick.setStyleSheet("color: blue; font-weight: bold;")

        form_mesh.addRow("Mesher:", self.combo_mesher)
        form_mesh.addRow("Refinement Level:", self.spin_refine)
        form_mesh.addRow("Layer Mode:", self.combo_layer_mode)
        form_mesh.addRow(self.btn_yplus, self.spin_layer_thick)
        
        group_mesh.setLayout(form_mesh)
        layout.addWidget(group_mesh)

        group_phys = QGroupBox("Physics")
        form_phys = QFormLayout()
        
        self.input_vel = QLineEdit("2.0")
        self.input_deltaT = QLineEdit("0.001")
        self.input_endTime = QLineEdit("10.0")

        self.spin_cores = SafeSpinBox()
        self.spin_cores.setRange(1, 64)
        self.spin_cores.setValue(8)
        
        self.spin_write_interval = SafeSpinBox()
        self.spin_write_interval.setRange(1, 100000)
        self.spin_write_interval.setValue(1)
    
        form_phys.addRow("Velocity (m/s):", self.input_vel)
        form_phys.addRow("CPU Cores:", self.spin_cores)
        form_phys.addRow("End Time:", self.input_endTime)
        form_phys.addRow("Delta T:", self.input_deltaT)
        form_phys.addRow("Write Interval:", self.spin_write_interval)
        
        group_phys.setLayout(form_phys)
        layout.addWidget(group_phys)

        layout.addStretch()
        scroll_area.setWidget(content_widget)
        return scroll_area
    
    def toggle_layer_mode(self):
        mode = self.combo_layer_mode.currentText()
        if mode == "Custom Input":
            self.btn_yplus.setEnabled(False)
            self.spin_layer_thick.setEnabled(True)
        else:
            self.btn_yplus.setEnabled(True)
            self.spin_layer_thick.setEnabled(False)

    def create_center_panel(self):
        tabs = QTabWidget()
        
        tab_3d = QWidget()
        layout_3d = QVBoxLayout(tab_3d)
        
        view_btn_layout = QHBoxLayout()
        btn_iso = QPushButton("Isometric")
        btn_top = QPushButton("Top (XY)")
        btn_front = QPushButton("Front (XZ)")
        btn_side = QPushButton("Side (YZ)")
        
        btn_iso.clicked.connect(self.set_view_iso)
        btn_top.clicked.connect(self.set_view_top)
        btn_front.clicked.connect(self.set_view_front)
        btn_side.clicked.connect(self.set_view_side)
        
        view_btn_layout.addWidget(btn_iso)
        view_btn_layout.addWidget(btn_top)
        view_btn_layout.addWidget(btn_front)
        view_btn_layout.addWidget(btn_side)
        layout_3d.addLayout(view_btn_layout)
        
        if HAS_PYVISTA:
            self.plotter = QtInteractor(tab_3d)
            layout_3d.addWidget(self.plotter.interactor)
            self.plotter.set_background("#263238")
            self.plotter.add_axes()
        else:
            placeholder_3d = QLabel("PyVista is not installed.\nPlease run: pip install pyvista pyvistaqt")
            placeholder_3d.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder_3d.setStyleSheet("background-color: #2b2b2b; color: white; border: 1px solid #555;")
            layout_3d.addWidget(placeholder_3d)
            self.plotter = None
        
        tabs.addTab(tab_3d, "3D Scene")

        self.tab_residuals = QWidget()
        self.resid_layout = QVBoxLayout(self.tab_residuals)
        if HAS_MATPLOTLIB:
            self.fig_resid = Figure()
            self.canvas_resid = FigureCanvas(self.fig_resid)
            self.ax_resid = self.fig_resid.add_subplot(111)
            self.resid_layout.addWidget(self.canvas_resid)
        else:
            self.resid_layout.addWidget(QLabel("Matplotlib not installed."))
        tabs.addTab(self.tab_residuals, "Residuals (Live)")

        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_widget)
        if HAS_MATPLOTLIB:
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.graph_layout.addWidget(self.canvas)
            btn_refresh = QPushButton("Refresh Drag Plot")
            btn_refresh.clicked.connect(self.plot_forces_on_tab)
            self.graph_layout.addWidget(btn_refresh)
        else:
            self.graph_layout.addWidget(QLabel("Matplotlib not installed."))
        tabs.addTab(self.graph_widget, "Results")
        
        self.residuals_data = {}
        
        return tabs

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        title = QLabel("<b>AI Assistant</b>")
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(title)
        
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("Key:"))
        self.txt_apikey = QLineEdit()
        self.txt_apikey.setPlaceholderText("Paste Gemini API Key...")
        self.txt_apikey.setEchoMode(QLineEdit.EchoMode.Password)
        api_layout.addWidget(self.txt_apikey)
        layout.addLayout(api_layout)
        
        self.ai_chat_area = QTextEdit()
        self.ai_chat_area.setReadOnly(True)
        self.ai_chat_area.setPlaceholderText("Ready to assist...")
        layout.addWidget(self.ai_chat_area)
        
        action_layout = QHBoxLayout()
        btn_analyze = QPushButton("Analyze Error")
        btn_analyze.clicked.connect(self.ai_analyze_log)
        btn_suggest = QPushButton("Suggest Settings")
        btn_suggest.clicked.connect(self.ai_suggest_settings)
        action_layout.addWidget(btn_analyze)
        action_layout.addWidget(btn_suggest)
        layout.addLayout(action_layout)
        
        input_layout = QHBoxLayout()
        self.input_ai = QLineEdit()
        self.input_ai.setPlaceholderText("Ask Gemini about CFD...")
        self.input_ai.returnPressed.connect(self.ai_send_manual)
        btn_send = QPushButton("Send")
        btn_send.clicked.connect(self.ai_send_manual)
        input_layout.addWidget(self.input_ai)
        input_layout.addWidget(btn_send)
        layout.addLayout(input_layout)
        
        return panel

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight: bold; padding-left: 5px;")
        self.status_bar.addWidget(self.lbl_status)
        
        lbl_cpu = QLabel("<b>CPU:</b> Monitoring...")
        lbl_cpu.setStyleSheet("color: #d32f2f;")
        lbl_ram = QLabel("<b>RAM:</b> Monitoring...")
        lbl_ram.setStyleSheet("color: #1976d2;")
        
        self.status_bar.addPermanentWidget(lbl_cpu)
        self.status_bar.addPermanentWidget(lbl_ram)

    def set_view_iso(self): 
        if self.plotter: self.plotter.view_isometric(); self.plotter.reset_camera()
    def set_view_top(self): 
        if self.plotter: self.plotter.view_xy(); self.plotter.reset_camera()
    def set_view_front(self): 
        if self.plotter: self.plotter.view_xz(); self.plotter.reset_camera()
    def set_view_side(self): 
        if self.plotter: self.plotter.view_yz(); self.plotter.reset_camera()

    def open_yplus_tool(self):
        try:
            v = float(self.input_vel.text())
            l_ref = float(self.input_L.text()) / 5.0
        except ValueError:
            v, l_ref = 2.0, 1.0
            
        dlg = YPlusDialog(v, l_ref, self)
        if dlg.exec():
            self.spin_layer_thick.setValue(dlg.calculated_thickness)
            self.log(f"New mesh layer thickness applied from calculator: {dlg.calculated_thickness:.6f} m")

    def update_3d_view(self, stl_path):
        if not HAS_PYVISTA or not self.plotter: return
        try:
            self.plotter.clear()
            self.plotter.add_axes()
            
            mesh = pv.read(stl_path)
            
            self.plotter.add_mesh(mesh, color="#f0f0f0", show_edges=True, edge_color="#555555", opacity=1.0, name="hull_actor")
            
            b = list(mesh.bounds)
            margin = max(b[1]-b[0], b[3]-b[2]) * 0.5
            self.plotter.show_grid(bounds=[b[0]-margin, b[1]+margin, b[2]-margin, b[3]+margin, b[4]-margin, b[5]+margin])
            
            arrow_start = [b[0] - margin*1.5, (b[2]+b[3])/2, (b[4]+b[5])/2]
            arrow = pv.Arrow(start=arrow_start, direction=(1, 0, 0), scale=margin)
            self.plotter.add_mesh(arrow, color="cyan", label="Water Flow")
            
            self.plotter.reset_camera()
        except Exception as e:
            self.log(f"Error 3D: {e}")

    def refresh_wsl(self):
        self.log("Refreshing WSL distribution list...")

    def win_to_wsl_path(self, win_path):
        if not win_path:
            return ""
        drive, tail = os.path.splitdrive(win_path)
        if drive:
            drive_letter = drive[0].lower()
            tail = tail.replace('\\', '/')
            return f"/mnt/{drive_letter}{tail}"
        return win_path.replace('\\', '/')

    def log(self, message):
        self.console_text.append(message)
        scrollbar = self.console_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        if hasattr(self, 'lbl_status'):
            clean_msg = message.replace("\n", " ").strip()
            self.lbl_status.setText(clean_msg[:80])

    def toggle_buttons(self, state):
        self.act_setup.setEnabled(state)
        self.act_mesh.setEnabled(state)
        self.act_run.setEnabled(state)
        self.act_stop.setEnabled(not state)

    def write_foam_dict(self, filepath, content):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())

    def new_project(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Folder for New Project")
        if directory:
            if os.listdir(directory):
                msg = "Directory is not empty. Do you want to PERMANENTLY DELETE all files inside to start fresh?"
                reply = QMessageBox.warning(self, "Warning", msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                
                if reply == QMessageBox.StandardButton.Yes:
                    try:
                        for item in os.listdir(directory):
                            item_path = os.path.join(directory, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                        self.log(f"Project directory cleared successfully: {directory}")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Could not wipe directory: {str(e)}")
                        return
                else:
                    self.log("Project creation aborted. Please choose an empty directory.")
                    return

            self.working_dir = directory
            self.lbl_status.setText(f"Project Dir: {self.working_dir}")
            self.log(f"New Project set at: {self.working_dir}")
            if hasattr(self, 'plotter') and self.plotter: 
                self.plotter.clear()
                self.plotter.add_axes()

    def open_project(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Existing Project")
        if directory:
            self.working_dir = directory
            self.lbl_status.setText(f"Project Dir: {self.working_dir}")
            self.log(f"Opened Existing Project: {self.working_dir}")
            
            stl_path = os.path.join(directory, "constant", "triSurface", "hull.stl")
            if os.path.exists(stl_path):
                self.stl_input.setText(stl_path)
                self.stl_filename = "hull.stl"
                self.stl_basename = "hull"
                self.log("Found existing STL model. Loading into 3D view...")
                self.update_3d_view(stl_path)
                
            self.load_project_settings()

    def import_stl(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Error", "Please select/open a project directory first.")
            return

        stl_path, _ = QFileDialog.getOpenFileName(self, "Select STL File", "", "STL Files (*.stl)")
        if stl_path:
            self.stl_input.setText(stl_path)
            self.log(f"Loading immutable original STL into memory: {os.path.basename(stl_path)}...")
            try:
                self.original_mesh = mesh.Mesh.from_file(stl_path)
                self.stl_filename = "hull.stl"
                self.stl_basename = "hull"
                self.process_geometry()
            except Exception as e:
                self.log(f"ERROR: Failed to read STL file. {str(e)}")

    def export_unity_package(self):
        if not self.working_dir: return None
        
        gltf_path = os.path.join(self.working_dir, "unity_scene.gltf").replace("\\", "/")
        physics_path = os.path.join(self.working_dir, "physics_data.csv").replace("\\", "/")
        foam_path = os.path.join(self.working_dir, "result.foam").replace("\\", "/")
        
        script_content = f"""
from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

foam = OpenFOAMReader(registrationName='result.foam', FileName='{foam_path}')
foam.UpdatePipeline()

c2p = CellDatatoPointData(Input=foam)
contour = Contour(Input=c2p)
contour.ContourBy = ['POINTS', 'alpha.water']
contour.Isosurfaces = [0.5]
contour.UpdatePipeline()

view = CreateRenderView()
display = Show(contour, view)
display.Representation = 'Surface'
ColorBy(display, ('POINTS', 'U', 'Magnitude'))
ExportView('{gltf_path}', view=view)

surf = ExtractSurface(Input=foam)

centers = CellCenters(Input=surf)

SaveData('{physics_path}', proxy=centers, Precision=5)
print("Data exported successfully.")
"""
        py_script = os.path.join(self.working_dir, "export_for_unity.py")
        with open(py_script, 'w', encoding='utf-8') as f:
            f.write(script_content.strip())
            
        pv_exe_path = self.find_paraview_exe()
        if pv_exe_path:
            pvpython_exe = os.path.join(os.path.dirname(pv_exe_path), "pvpython.exe")
            if os.path.exists(pvpython_exe):
                self.log("Exporting 3D scene (Free Surface) and physics cloud silently via pvpython...")
                subprocess.run([pvpython_exe, py_script], cwd=self.working_dir)
                return gltf_path, physics_path
            else:
                self.log("ERROR: pvpython.exe not found. Cannot export data silently.")
                return None
        return None
    
    def is_vr_headset_connected(self):
        if not HAS_PSUTIL:
            QMessageBox.critical(self, "Missing Library", "Library 'psutil' is missing. Please run 'pip install psutil' in terminal.")
            self.log("ERROR: Cannot verify VR connection without psutil.")
            return False

        vr_processes = ['OVRServer_x64.exe', 'vrmonitor.exe', 'vrserver.exe']

        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] in vr_processes:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

        return False

    def launch_unity_vr(self):
        if not self.working_dir:
            QMessageBox.warning(self, "Error", "Please open a project first.")
            return

        if not self.is_vr_headset_connected():
            msg = "No VR Headset detected!\n\nPlease ensure your Oculus/Meta Quest is plugged in, powered on, and the Quest Link software (or SteamVR) is running."
            QMessageBox.critical(self, "VR Not Found", msg)
            self.log("VR launch aborted: Headset not connected. No files were exported.")
            return

        self.log("Packaging Towing Tank data for Unity VR...")
        paths = self.export_unity_package()

        if paths:
            gltf_file, csv_file = paths
            config = {
                "model_path": gltf_file,
                "data_path": csv_file,
                "type": "TowingTank",
                "project_dir": self.working_dir.replace("\\", "/")
            }
            with open(os.path.join(self.working_dir, "vr_config.json"), 'w', encoding='utf-8') as f:
                import json
                json.dump(config, f)

            if not hasattr(self, 'unity_exe_path') or not os.path.exists(self.unity_exe_path):
                exe_path, _ = QFileDialog.getOpenFileName(self, "Locate Unity VR Executable", "C:\\", "Executable (*.exe)")
                if exe_path:
                    self.unity_exe_path = os.path.normpath(exe_path)
                else:
                    QMessageBox.information(self, "Export Complete", "VR Data is ready in project folder.")
                    return

            if hasattr(self, 'unity_exe_path') and os.path.exists(self.unity_exe_path):
                subprocess.Popen([self.unity_exe_path, "--config", os.path.join(self.working_dir, "vr_config.json")])
                self.log(f"Unity VR App launched: {self.unity_exe_path}")

    def process_geometry(self):
        if not hasattr(self, 'original_mesh') or self.original_mesh is None:
            return
        if not self.working_dir:
            return

        current_mesh = copy.deepcopy(self.original_mesh)

        yaw_deg = self.spin_rot.value()
        if yaw_deg != 0.0:
            yaw_rad = math.radians(yaw_deg)
            cos_val = math.cos(yaw_rad)
            sin_val = math.sin(yaw_rad)
            for i in range(3): 
                x = current_mesh.vectors[:, i, 0]
                y = current_mesh.vectors[:, i, 1]
                new_x = x * cos_val - y * sin_val
                new_y = x * sin_val + y * cos_val
                current_mesh.vectors[:, i, 0] = new_x
                current_mesh.vectors[:, i, 1] = new_y

        unit = self.combo_units.currentText()
        if unit == "Centimeters":
            scale = 0.01
        elif unit == "Millimeters":
            scale = 0.001
        else:
            scale = 1.0
        
        current_mesh.vectors *= scale

        minx, maxx = current_mesh.x.min(), current_mesh.x.max()
        miny, maxy = current_mesh.y.min(), current_mesh.y.max()
        minz, maxz = current_mesh.z.min(), current_mesh.z.max()

        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        cz = minz 

        current_mesh.x -= cx
        current_mesh.y -= cy
        current_mesh.z -= cz

        if hasattr(current_mesh, 'attr'):
            current_mesh.attr[:] = 0

        l_ship = maxx - minx
        w_ship = maxy - miny
        h_ship = maxz - minz

        self.log(f"\n--- Processed Geometry -> Yaw: {yaw_deg}°, Scale: {scale}x ---")
        self.log(f"Updated Ship Dimensions: L={l_ship:.3f}m, W={w_ship:.3f}m, H={h_ship:.3f}m")

        tank_L = 5.0 * l_ship
        tank_W = 3.0 * l_ship
        tank_H = 3.0 * h_ship
        water_level = 0.5 * h_ship

        self.input_L.setText(f"{tank_L:.2f}")
        self.input_W.setText(f"{tank_W:.2f}")
        self.input_H.setText(f"{tank_H:.2f}")
        self.input_wl.setText(f"{water_level:.2f}")

        target_dir = os.path.join(self.working_dir, "constant", "triSurface")
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, self.stl_filename)
        
        current_mesh.save(target_file)
        self.log(f"Final processed STL successfully overwritten at constant/triSurface/{self.stl_filename}")

        self.update_3d_view(target_file)

    def setup_case(self):
        if not self.working_dir:
            self.log("ERROR: No working directory selected.")
            return
            
        if not hasattr(self, 'stl_filename'):
            self.log("ERROR: No STL file imported. Please import a hull model first.")
            return

        self.log("Setting up OpenFOAM case directories and dictionary files...")
        for folder in ['0', 'constant/triSurface', 'system']:
            os.makedirs(os.path.join(self.working_dir, folder), exist_ok=True)
            
        try:
            L = float(self.input_L.text())
            W = float(self.input_W.text())
            H = float(self.input_H.text())
            wl = float(self.input_wl.text())
            U_in = float(self.input_vel.text())
            dT = float(self.input_deltaT.text())
            eTime = float(self.input_endTime.text())
            w_interval = self.spin_write_interval.value()
            refine_lvl = self.spin_refine.value()
            cores = self.spin_cores.value()
        except ValueError:
            self.log("ERROR: Invalid input parameters. Please enter numerical values.")
            return

        xMin, xMax = -L/3, 2*L/3
        yMin, yMax = -W/2, W/2
        zMin, zMax = -H/2, H/2
        
        self.inside_point_x = -L/4 + 0.123
        self.inside_point_y = 0.123
        self.inside_point_z = zMin + (H * 0.1) + 0.123
        self.log(f"Calculated insidePoint: ({self.inside_point_x}, {self.inside_point_y}, {self.inside_point_z})")
        
        sys_dir = os.path.join(self.working_dir, "system")
        zero_dir = os.path.join(self.working_dir, "0")
        const_dir = os.path.join(self.working_dir, "constant")

        self.write_foam_dict(os.path.join(zero_dir, "alpha.water"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      alpha.water;
}}

dimensions      [0 0 0 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           $internalField;
    }}
    outlet
    {{
        type            variableHeightFlowRate;
        lowerBound      0;
        upperBound      1;
        value           $internalField;
    }}
    atmosphere
    {{
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }}
    hull
    {{
        type            zeroGradient;
    }}
    "(bottom|sides|sym)"
    {{
        type            symmetry;
    }}
}}
        """)

        self.write_foam_dict(os.path.join(zero_dir, "k"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      k;
}}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0.00015;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           $internalField;
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }}
    atmosphere
    {{
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }}
    hull
    {{
        type            kqRWallFunction;
        value           $internalField;
    }}
    "(bottom|sides|sym)"
    {{
        type            symmetry;
    }}
}}
        """)

        self.write_foam_dict(os.path.join(zero_dir, "nut"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      nut;
}}

dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 5e-07;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           $internalField;
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    atmosphere
    {{
        type            zeroGradient;
    }}
    hull
    {{
        type            nutkRoughWallFunction;
        Ks              uniform 100e-6;
        Cs              uniform 0.5;
        value           $internalField;
    }}
    "(bottom|sides|sym)"
    {{
        type            symmetry;
    }}
}}
        """)

        self.write_foam_dict(os.path.join(zero_dir, "omega"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      omega;
}}

dimensions      [0 0 -1 0 0 0 0];
internalField   uniform 2;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           $internalField;
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }}
    atmosphere
    {{
        type            inletOutlet;
        inletValue      $internalField;
        value           $internalField;
    }}
    hull
    {{
        type            omegaWallFunction;
        value           $internalField;
    }}
    "(bottom|sides|sym)"
    {{
        type            symmetry;
    }}
}}
        """)

        self.write_foam_dict(os.path.join(zero_dir, "p_rgh"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p_rgh;
}}

dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            fixedFluxPressure;
        value           $internalField;
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    atmosphere
    {{
        type            prghTotalPressure;
        p0              uniform 0;
    }}
    hull
    {{
        type            fixedFluxPressure;
        value           $internalField;
    }}
    "(bottom|sides|sym)"
    {{
        type            symmetry;
    }}
}}
        """)

        self.write_foam_dict(os.path.join(zero_dir, "pointDisplacement"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       pointVectorField;
    location    "0";
    object      pointDisplacement;
}}

dimensions      [0 1 0 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    atmosphere
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    hull
    {{
        type            calculated;
    }}
    "(bottom|sides|sym)"
    {{
        type            symmetry;
    }}
}}
        """)

        self.write_foam_dict(os.path.join(zero_dir, "U"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({U_in} 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({U_in} 0 0);
    }}
    outlet
    {{
        type            outletPhaseMeanVelocity;
        alpha           alpha.water;
        UnMean          {U_in};
        value           uniform ({U_in} 0 0);
    }}
    atmosphere
    {{
        type            pressureInletOutletVelocity;
        tangentialVelocity uniform ({U_in} 0 0);
        value           uniform (0 0 0);
    }}
    hull
    {{
        type            movingWallVelocity;
        value           uniform (0 0 0);
    }}
    "(bottom|sides|sym)"
    {{
        type            symmetry;
    }}
}}
        """)

        self.write_foam_dict(os.path.join(const_dir, "g"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    location    "constant";
    object      g;
}}

dimensions      [0 1 -2 0 0 0 0];
value           (0 0 -9.81);
        """)

        self.write_foam_dict(os.path.join(const_dir, "hRef"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       uniformDimensionedScalarField;
    location    "constant";
    object      hRef;
}}

dimensions      [0 1 0 0 0 0 0];
value           {wl};
        """)

        self.write_foam_dict(os.path.join(const_dir, "momentumTransport"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      momentumTransport;
}}

simulationType  RAS;

RAS
{{
    model           kOmegaSST;
    turbulence      on;
}}
        """)

        self.write_foam_dict(os.path.join(const_dir, "phaseProperties"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      phaseProperties;
}}

phases          (water air);
sigma           0;
        """)

        self.write_foam_dict(os.path.join(const_dir, "physicalProperties.air"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      physicalProperties.air;
}}

viscosityModel  constant;
nu              1.48e-05;
rho             1;
        """)

        self.write_foam_dict(os.path.join(const_dir, "physicalProperties.water"), f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      physicalProperties.water;
}}

viscosityModel  constant;
nu              1.09e-06;
rho             998.8;
        """)

        base_divs = [20, 30, 40, 50, 60]
        idx = max(0, min(refine_lvl - 1, 4))
        target_size = L / base_divs[idx]

        nx = int(max(1, L / target_size))
        ny = int(max(1, W / target_size))
        nz = int(max(1, H / target_size))

        feat_lvl = max(1, refine_lvl - 1)
        surf_min = refine_lvl
        surf_max = refine_lvl + 3
        
        if refine_lvl >= 4:
            surf_max = refine_lvl

        self.log(f"Calculated background mesh: nx={nx}, ny={ny}, nz={nz}")

        self.write_foam_dict(os.path.join(sys_dir, "blockMeshDict"), f"""
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    ({xMin} {yMin} {zMin})
    ({xMax} {yMin} {zMin})
    ({xMax} {yMax} {zMin})
    ({xMin} {yMax} {zMin})

    ({xMin} {yMin} {zMax})
    ({xMax} {yMin} {zMax})
    ({xMax} {yMax} {zMax})
    ({xMin} {yMax} {zMax})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    atmosphere
    {{
        type patch;
        faces ((4 5 6 7));
    }}
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    bottom
    {{
        type symmetry;
        faces ((0 1 2 3));
    }}
    sides
    {{
        type symmetry;
        faces (
            (0 1 5 4)
            (3 2 6 7)
        );
    }}
);
        """)

        self.write_foam_dict(os.path.join(sys_dir, "controlDict"), f"""
FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}

solver          incompressibleVoF;
startFrom       latestTime; 
startTime       0;
stopAt          endTime;
endTime         {eTime};
deltaT          {dT};
adjustTimeStep  yes;
maxCo           0.5;
maxAlphaCo      0.5;
maxDeltaT       1;
writeControl    adjustableRunTime;
writeInterval   {w_interval};
purgeWrite      0;
writeFormat     binary;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;

functions
{{
    #include "functions"
}}
        """)

        self.write_foam_dict(os.path.join(sys_dir, "setFieldsDict"), f"""
FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      setFieldsDict;
}}

defaultValues
{{
    alpha.water 0;
}}

zones
{{
    cells
    {{
        type        box;
        box         (-999 -999 -999) (999 999 {wl});
        values
        {{
            alpha.water 1;
        }}
    }}
}}

extrapolatePatches
{{
    "inlet|outlet"   (alpha.water);
}}
        """)

        self.write_foam_dict(os.path.join(sys_dir, "refineMeshDict"), f"""
FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      refineMeshDict;
}}

coordinates
{{
    type        global;
    e1          (1 0 0);
    e2          (0 1 0);
    directions  (e1 e2);
}}

zones
{{
    level1 {{ type box; box (-10 -6 -3) (10 6 3); }}
    level2 {{ type box; box (-5 -3 -2.5) (9 3 2); }}
    level3 {{ type box; box (-3 -1.5 -1) (8 1.5 1.5); }}
    level4 {{ type box; box (-2 -1 -0.6) (7 1 1); }}
    level5 {{ type box; box (-1 -0.6 -0.3) (6.5 0.6 0.8); }}
    level6 {{ type box; box (-0.5 -0.55 -0.15) (6.25 0.55 0.65); }}
}}
        """)

        self.write_foam_dict(os.path.join(sys_dir, "functions"), r"""
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      functions;
}

forces
{
    type            forces;
    libs            ("libforces.so");
    patches         (hull);
    log             on;
    writeControl    timeStep;
    writeInterval   1;
    CofR            (2.929541 0 0.2);
}
        """)

        self.write_foam_dict(os.path.join(sys_dir, "decomposeParDict"), f"""
FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}}

numberOfSubdomains {cores};
method          scotch;
        """)

        self.write_foam_dict(os.path.join(sys_dir, "fvSchemes"), r"""
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes { default localEuler; }
gradSchemes { default Gauss linear; limitedGrad cellLimited Gauss linear 1; }
divSchemes
{
    div(rhoPhi,U)   Gauss linearUpwind grad(U);
    div(phi,alpha)  Gauss interfaceCompression vanLeer 1;
    div(phi,k)      Gauss linearUpwind limitedGrad;
    div(phi,omega)  Gauss linearUpwind limitedGrad;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
        """)

        self.write_foam_dict(os.path.join(sys_dir, "fvSolution"), r"""
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

solvers
{
    "alpha.water.*"
    {
        nCorrectors     2;
        nSubCycles      1;
        MULESCorr       yes;
        alphaApplyPrevCorr  yes;
        MULES { nIter 1; }
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
        minIter         1;
    }
    "pcorr.*"
    {
        solver          PCG;
        preconditioner  { preconditioner GAMG; smoother GaussSeidel; tolerance 1e-5; relTol 0; };
        tolerance       1e-5;
        relTol          0;
    }
    p_rgh
    {
        solver          GAMG;
        smoother        DIC;
        tolerance       1e-8;
        relTol          0.01;
    }
    p_rghFinal { $p_rgh; relTol 0; }
    "(U|k|omega).*"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         1;
        tolerance       1e-7;
        relTol          0.1;
        minIter         1;
    }
}

PIMPLE
{
    momentumPredictor   no;
    nOuterCorrectors    1;
    nCorrectors         2;
    nNonOrthogonalCorrectors 0;
    maxCo               10;
    maxAlphaCo          1;
    rDeltaTSmoothingCoeff 0.05;
    rDeltaTDampingCoeff 0.5;
    nAlphaSpreadIter    0;
    nAlphaSweepIter     0;
    maxDeltaT           1;
}

relaxationFactors { equations { ".*" 1; } }
cache { grad(U); }
        """)

        self.write_foam_dict(os.path.join(sys_dir, "meshQualityDict"), r"""
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      meshQualityDict;
}

maxNonOrtho 75;
maxBoundarySkewness 20;
maxInternalSkewness 8;
maxConcave 80;
minVol -1e30;
minTetQuality -1e30;
minTwist -1e30;
minDeterminant -1;
minFaceWeight 0.02;
minVolRatio 0.01;
nSmoothScale 4;
errorReduction 0.75;

relaxed
{
    maxNonOrtho 75;
}
        """)

        self.write_foam_dict(os.path.join(sys_dir, "snappyHexMeshDict"), f"""
FoamFile {{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }}

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    hull {{ type triSurface; file "{self.stl_filename}"; patchInfo {{ type wall; }} }}
}}

castellatedMeshControls
{{
    maxLocalCells 2000000;
    maxGlobalCells 10000000;
    minRefinementCells 10;
    nCellsBetweenLevels 5;
    features ( {{ file "{self.stl_basename}.eMesh"; level {refine_lvl+1}; }} );
    refinementSurfaces {{ hull {{ level ({refine_lvl} {refine_lvl + 3}); }} }}
    resolveFeatureAngle 30;
    insidePoint ({self.inside_point_x} {self.inside_point_y} {self.inside_point_z});
    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    nSmoothPatch 9;
    tolerance 1.0;
    nSolveIter 300;
    nRelaxIter 20;
    nFeatureSnapIter 15;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes false;
    layers {{ "hull.*" {{ nSurfaceLayers 3; }} }}
    expansionRatio 1;
    minThickness 1e-6;
    firstLayerThickness {self.spin_layer_thick.value():.6f};
    nGrow 0;
    featureAngle 180;
    slipFeatureAngle 10;
    nRelaxIter 5;
    nSmoothSurfaceNormals 20;
    nSmoothNormals 30;
    nLayerIter 50;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nSmoothThickness 10;
}}

meshQualityControls {{ #include "meshQualityDict" }}
mergeTolerance 1e-6;
""")

        self.write_foam_dict(os.path.join(sys_dir, "surfaceFeaturesDict"), f"""
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}

surfaces ("{self.stl_filename}");

includedAngle   150;

subsetFeatures
{{
    nonManifoldEdges       yes;
    openEdges              yes;
}}

        """)
        self.write_foam_dict(os.path.join(sys_dir, "changeDictionaryDict"), f"""
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      changeDictionaryDict;
}}

dictionaryReplacement
{{
    boundary
    {{
        hull
        {{
            type            wall;
        }}
        sym
        {{
            type            symmetry;
        }}
    }}
}}
        """)

        self.log("Case setup complete. All dictionary files successfully written.")

    def execute_wsl_command(self, commands):
        if not self.working_dir:
            self.log("ERROR: Working directory not set.")
            return

        self.toggle_buttons(False)
        wsl_path = self.win_to_wsl_path(self.working_dir)
        
        self.worker = WSLWorker(wsl_path, commands)
        self.worker.log_signal.connect(self.handle_log_and_parse)
        self.worker.finished_signal.connect(self.on_process_finished)
        self.worker.start()

    def generate_mesh(self):
        if hasattr(self, 'plotter') and self.plotter:
            self.plotter.clear()
            self.plotter.add_axes()
            
        if self.working_dir:
            self.log("Cleaning old time directories...")
            for item in os.listdir(self.working_dir):
                item_path = os.path.join(self.working_dir, item)
                if os.path.isdir(item_path):
                    try:
                        val = float(item)
                        if val > 0:
                            import shutil
                            shutil.rmtree(item_path, ignore_errors=True)
                    except ValueError:
                        pass
                        
        mesher_choice = self.combo_mesher.currentText()
        
        if "OpenFOAM" in mesher_choice:
            self.current_process_type = "openfoam_mesh"
            commands = "surfaceFeatures && blockMesh && snappyHexMesh -overwrite && checkMesh -allTopology -allGeometry > checkMesh.log"
            self.log("Starting Meshing Process via OpenFOAM...")
            self.execute_wsl_command(commands)
            
        elif "SALOME" in mesher_choice:
            self.current_process_type = "salome_mesh"
            if not hasattr(self, 'salome_bat_path') or not os.path.exists(self.salome_bat_path):
                QMessageBox.warning(self, "Error", "Please go to Settings -> SALOME Path... and select run_salome.bat first.")
                return

            salome_script_path = os.path.join(self.working_dir, "run_salome_mesh.py")
            polyMesh_dir = os.path.join(self.working_dir, "constant", "polyMesh").replace("\\", "/")
            stl_path = os.path.join(self.working_dir, "constant", "triSurface", "hull.stl").replace("\\", "/")
            error_log = os.path.join(self.working_dir, "salome_error.log").replace("\\", "/")
            
            try:
                L = float(self.input_L.text())
                W = float(self.input_W.text())
                H = float(self.input_H.text())
            except:
                L, W, H = 20.0, 10.0, 10.0

            xMin, xMax = -L/3, 2*L/3
            yMin, yMax = -W/2, W/2
            zMin, zMax = -H/2, H/2
            dx = xMax - xMin
            dy = yMax - yMin
            dz = zMax - zMin

            v_thick = self.spin_layer_thick.value()
            fineness = self.spin_refine.value()

            vars_header = f"""
import sys, os, traceback, tempfile, time
import salome
import SMESH, SALOMEDS
import GEOM
from salome.geom import geomBuilder
from salome.smesh import smeshBuilder
from salome.shaper import model

STL_PATH = r"{stl_path}"
POLY_DIR = r"{polyMesh_dir}"
ERROR_LOG = r"{error_log}"

XMIN, XMAX = {xMin}, {xMax}
YMIN, YMAX = {yMin}, {yMax}
ZMIN, ZMAX = {zMin}, {zMax}
DX, DY, DZ = {dx}, {dy}, {dz}

MAX_SIZE = 1.0
MIN_SIZE = 0.06
LOCAL_SIZE_HULL = 0.06
VISCOUS_THICKNESS = {v_thick}
VISCOUS_LAYERS = 3
FINENESS = {fineness}
"""

            logic_script = r"""
debug = 1
verify = True

class MeshBuffer(object):
    def __init__(self, mesh, v):
        i = 0
        faces = list()
        keys = list()
        fnodes = mesh.GetElemFaceNodes(v, i)
        while fnodes:
            faces.append(fnodes)
            keys.append(tuple(sorted(fnodes)))
            i += 1
            fnodes=mesh.GetElemFaceNodes(v, i)
        self.v = v
        self.faces = faces
        self.keys = keys
        self.fL = i
    @staticmethod
    def Key(fnodes): return tuple(sorted(fnodes))
    @staticmethod
    def ReverseKey(fnodes):
        if type(fnodes) is tuple: return tuple(reversed(fnodes))
        else: return tuple(sorted(fnodes, reverse=True))

def __writeHeader__(file,fileType,nrPoints=0,nrCells=0,nrFaces=0,nrIntFaces=0):
    file.write("/*" + "-"*68 + "*\\\n" )
    file.write("|" + " "*70 + "|\n")
    file.write("|" + " "*4 + "File exported from Salome Platform using SalomeToFoamExporter" +" "*5 +"|\n")
    file.write("|" + " "*70 + "|\n")
    file.write("\*" + "-"*68 + "*/\n")
    file.write("FoamFile\n{\n\tversion\t\t2.0;\n\tformat\t\tascii;\n\tclass\t\t")
    if(fileType =="points"): file.write("vectorField;\n")
    elif(fileType =="faces"): file.write("faceList;\n")
    elif(fileType =="owner" or fileType=="neighbour"):
        file.write("labelList;\n\tnote\t\t\"nPoints: %d nCells: %d nFaces: %d nInternalFaces: %d\";\n" %(nrPoints,nrCells,nrFaces,nrIntFaces))
    elif(fileType == "boundary"): file.write("polyBoundaryMesh;\n")
    elif(fileType=="cellZones"): file.write("regIOobject;\n")
    file.write("\tlocation\t\"constant/polyMesh\";\n\tobject\t\t" + fileType +";\n}\n\n")

def __debugPrint__(msg,level=1):
    if(debug >= level): print(msg)

def __cog__(mesh,nodes):
    c=[0.0,0.0,0.0]
    for n in nodes:
        pos=mesh.GetNodeXYZ(n)
        c[0]+=pos[0]; c[1]+=pos[1]; c[2]+=pos[2]
    c[0]/=len(nodes); c[1]/=len(nodes); c[2]/=len(nodes)
    return c

def __diff__(u,v): return [u[0]-v[0], u[1]-v[1], u[2]-v[2]]
def __dotprod__(u,v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
def __crossprod__(u,v): return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]

def __calcNormal__(mesh,nodes):
    p0=mesh.GetNodeXYZ(nodes[0]); p1=mesh.GetNodeXYZ(nodes[1]); pn=mesh.GetNodeXYZ(nodes[-1])
    return __crossprod__(__diff__(p1,p0), __diff__(pn,p0))

def __verifyFaceOrder__(mesh,vnodes,fnodes):
    vc=__cog__(mesh,vnodes); fc=__cog__(mesh,fnodes)
    fcTovc=__diff__(vc,fc); fn=__calcNormal__(mesh,fnodes)
    return False if __dotprod__(fn,fcTovc)>0.0 else True

def __isGroupBaffle__(mesh,group,extFaces):
    for sid in group.GetIDs():
        if not sid in extFaces: return True
    return False

def exportToFoam(mesh, dirname):
    starttime=time.time()
    if not os.path.exists(dirname): os.makedirs(dirname)
    
    filePoints = open(dirname + '/points', 'w')
    fileFaces = open(dirname + '/faces', 'w')
    fileOwner = open(dirname + '/owner', 'w')
    fileNeighbour = open(dirname + '/neighbour', 'w')
    fileBoundary = open(dirname + '/boundary', 'w')

    smesh_builder = smeshBuilder.New()
    volumes=mesh.GetElementsByType(SMESH.VOLUME)
    
    filter_faces = smesh_builder.GetFilter(SMESH.EDGE, SMESH.FT_FreeFaces)
    extFaces = set(mesh.GetIdsFromFilter(filter_faces))
    nrBCfaces = len(extFaces)
    nrExtFaces = len(extFaces)
    buffers=list()
    nrFaces = 0

    for v in volumes:
        b = MeshBuffer(mesh, v) 
        nrFaces += b.fL
        buffers.append(b)

    nrFaces = int((nrFaces + nrExtFaces) / 2)
    nrIntFaces = int(nrFaces - nrBCfaces)

    faces = []; facesSorted = dict(); bcFaces = []; bcFacesSorted = dict()
    owner = []; neighbour = []
    grpStartFace = []; grpNrFaces = []; grpNames = []
    ofbcfid = 0; nrExtFacesInGroups = 0

    for gr in mesh.GetGroups():
        if gr.GetType() == SMESH.FACE:
            grpNames.append(gr.GetName())
            grIds = gr.GetIDs()
            nr = len(grIds)
            if nr > 0:
                grpStartFace.append(nrIntFaces+ofbcfid)
                grpNrFaces.append(nr)
            for sfid in grIds:
                fnodes = mesh.GetElemNodes(sfid)
                key = MeshBuffer.Key(fnodes)
                if not key in bcFacesSorted:
                    bcFaces.append(fnodes)
                    bcFacesSorted[key] = ofbcfid
                    ofbcfid += 1
            if __isGroupBaffle__(mesh, gr, extFaces):
                nrBCfaces += nr; nrFaces += nr; nrIntFaces -= nr
                grpStartFace = [x - nr for x in grpStartFace]
                grpNrFaces[-1] = nr*2
                for sfid in gr.GetIDs():
                    fnodes = mesh.GetElemNodes(sfid)
                    key = MeshBuffer.ReverseKey(fnodes)
                    bcFaces.append(fnodes)
                    bcFacesSorted[key] = ofbcfid
                    ofbcfid += 1
            else:
                nrExtFacesInGroups += nr

    owner = [-1] * nrFaces
    neighbour = [-1] * nrIntFaces

    offid = 0; ofvid = 0
    for b in buffers:
        nodes = mesh.GetElemNodes(b.v)
        fi = 0
        while fi < b.fL:
            fnodes = b.faces[fi]; key = b.keys[fi]
            try:
                fidinof = facesSorted[key]
                neighbour[fidinof] = ofvid
            except KeyError:
                try:
                    bcind = bcFacesSorted[key]
                    if owner[nrIntFaces + bcind] == -1:
                        owner[nrIntFaces + bcind] = ofvid
                        bcFaces[bcind] = fnodes
                    else:
                        key = MeshBuffer.ReverseKey(fnodes)
                        bcind = bcFacesSorted[key]
                        bcFaces[bcind] = fnodes
                        owner[nrIntFaces + bcind] = ofvid
                except KeyError:
                    if verify:
                        if not __verifyFaceOrder__(mesh, nodes, fnodes): fnodes.reverse()
                    faces.append(fnodes)
                    key = b.keys[fi]
                    facesSorted[key] = offid
                    owner[offid] = ofvid
                    offid += 1
            fi += 1
        ofvid += 1

    nrCells = ofvid
    ownedfaces = 1
    quickrange = range if sys.version_info.major > 2 else xrange

    for faceId in quickrange(0, nrIntFaces):
        cellId = owner[faceId]; nextCellId = owner[faceId + 1]
        if cellId == nextCellId:
            ownedfaces += 1
            continue
        if ownedfaces > 1:
            sId = faceId - ownedfaces + 1; eId = faceId
            inds = range(sId, eId + 1)
            if sys.version_info.major > 2: sorted(inds, key=neighbour.__getitem__)
            else: inds.sort(key = neighbour.__getitem__)
            neighbour[sId:eId + 1] = map(neighbour.__getitem__, inds)
            faces[sId:eId + 1] = map(faces.__getitem__, inds)
        ownedfaces = 1

    __writeHeader__(filePoints, 'points')
    points = mesh.GetElementsByType(SMESH.NODE)
    nrPoints = len(points)
    filePoints.write('\n%d\n(\n' % nrPoints)
    for n, ni in enumerate(points):
        pos = mesh.GetNodeXYZ(ni)
        filePoints.write('\t(%.10g %.10g %.10g)\n' % (pos[0], pos[1], pos[2]))
    filePoints.write(')\n'); filePoints.close()

    __writeHeader__(fileFaces, 'faces')
    fileFaces.write('\n%d\n(\n' % nrFaces)
    for node in faces:
        fileFaces.write('\t%d(' % len(node))
        for p in node: fileFaces.write('%d ' % (p - 1))
        fileFaces.write(')\n')
    for node in bcFaces:
        fileFaces.write('\t%d(' % len(node))
        for p in node: fileFaces.write('%d ' % (p - 1))
        fileFaces.write(')\n')
    fileFaces.write(')\n'); fileFaces.close()

    __writeHeader__(fileOwner, 'owner', nrPoints, nrCells, nrFaces, nrIntFaces)
    fileOwner.write('\n%d\n(\n' % len(owner))
    for cell in owner: fileOwner.write(' %d \n' % cell)
    fileOwner.write(')\n'); fileOwner.close()

    __writeHeader__(fileNeighbour, 'neighbour', nrPoints, nrCells, nrFaces, nrIntFaces)
    fileNeighbour.write('\n%d\n(\n' %(len(neighbour)))
    for cell in neighbour: fileNeighbour.write(' %d\n' %(cell))
    fileNeighbour.write(')\n'); fileNeighbour.close()

    __writeHeader__(fileBoundary, 'boundary')
    fileBoundary.write('%d\n(\n' %len(grpStartFace))
    
    patch_types = {'inlet': 'patch', 'outlet': 'patch', 'sym': 'symmetry', 'hull': 'wall'}
    
    for ind, gname in enumerate(grpNames):
        fileBoundary.write('\t%s\n\t{\n' %gname)
        fileBoundary.write('\ttype\t\t')
        
        b_type = patch_types.get(gname, 'patch')
        fileBoundary.write(b_type + ";\n")
        
        fileBoundary.write('\tnFaces\t\t%d;\n' %grpNrFaces[ind])
        fileBoundary.write('\tstartFace\t%d;\n' %grpStartFace[ind])
        fileBoundary.write('\t}\n')
    fileBoundary.write(')\n'); fileBoundary.close()

def run_meshing():
    salome.salome_init()
    geompy = geomBuilder.New()
    smesh = smeshBuilder.New()

    model.begin()
    partSet = model.moduleDocument()
    Part_1 = model.addPart(partSet)
    Part_1_doc = Part_1.document()

    Import_1 = model.addImport(Part_1_doc, STL_PATH)
    model.do()
    Import_1.result().setName("Hull_STL")

    Box_1 = model.addBox(Part_1_doc, XMIN, YMIN, ZMIN, DX, DY, DZ)
    Cut_1 = model.addCut(Part_1_doc, [model.selection("COMPOUND", "all-in-Box_1")], [model.selection("SOLID", "Hull_STL")], keepSubResults=True)
    model.do()

    temp_dir = tempfile.gettempdir()
    xao_path = os.path.join(temp_dir, 'shaper_fluid_domain_auto.xao').replace('\\', '/')
    model.exportToXAO(Part_1_doc, xao_path, model.selection("SOLID", "Cut_1_1"), 'XAO')
    model.end()

    imported_xao = geompy.ImportXAO(xao_path)
    Cut_Model = imported_xao[1] 
    geompy.addToStudy(Cut_Model, 'Fluid_Domain')

    faces = geompy.ExtractShapes(Cut_Model, geompy.ShapeType["FACE"], True)
    inlet_faces, outlet_faces, sym_faces, hull_faces = [], [], [], []
    D_xmin, D_xmax, D_ymin, D_ymax, D_zmin, D_zmax = geompy.BoundingBox(Cut_Model)
    tol = 1e-2 

    for f in faces:
        bbf = geompy.BoundingBox(f)
        f_xmin, f_xmax, f_ymin, f_ymax, f_zmin, f_zmax = bbf
        if (f_xmax - f_xmin) < tol and abs(f_xmin - D_xmin) < tol: outlet_faces.append(f)
        elif (f_xmax - f_xmin) < tol and abs(f_xmax - D_xmax) < tol: inlet_faces.append(f)
        elif ((f_ymax - f_ymin) < tol and (abs(f_ymin - D_ymin) < tol or abs(f_ymax - D_ymax) < tol)) or \
             ((f_zmax - f_zmin) < tol and (abs(f_zmin - D_zmin) < tol or abs(f_zmax - D_zmax) < tol)):
            sym_faces.append(f)
        else: hull_faces.append(f)

    inlet = geompy.CreateGroup(Cut_Model, geompy.ShapeType["FACE"])
    geompy.UnionList(inlet, inlet_faces)
    geompy.addToStudyInFather(Cut_Model, inlet, 'inlet')
    
    outlet = geompy.CreateGroup(Cut_Model, geompy.ShapeType["FACE"])
    geompy.UnionList(outlet, outlet_faces)
    geompy.addToStudyInFather(Cut_Model, outlet, 'outlet')
    
    sym = geompy.CreateGroup(Cut_Model, geompy.ShapeType["FACE"])
    geompy.UnionList(sym, sym_faces)
    geompy.addToStudyInFather(Cut_Model, sym, 'sym')
    
    hull = geompy.CreateGroup(Cut_Model, geompy.ShapeType["FACE"])
    geompy.UnionList(hull, hull_faces)
    geompy.addToStudyInFather(Cut_Model, hull, 'hull')

    outer_face_ids = geompy.GetObjectIDs(inlet) + geompy.GetObjectIDs(outlet) + geompy.GetObjectIDs(sym)

    max_length = max(DX, DY, DZ)
    ref_length = 7.0 
    scale_factor = max_length / ref_length

    Mesh_Final = smesh.Mesh(Cut_Model, 'Mesh_Final')
    NETGEN_1D2D3D = Mesh_Final.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)
    NETGEN_Parameters = NETGEN_1D2D3D.Parameters()
    NETGEN_Parameters.SetMaxSize(1.0 * scale_factor)
    NETGEN_Parameters.SetMinSize(MIN_SIZE * scale_factor)
    NETGEN_Parameters.SetLocalSizeOnShape(hull, LOCAL_SIZE_HULL * scale_factor)
    NETGEN_Parameters.SetSecondOrder(0)
    NETGEN_Parameters.SetOptimize(1)
    NETGEN_Parameters.SetFineness(FINENESS)

    Viscous_Layers = NETGEN_1D2D3D.ViscousLayers(VISCOUS_THICKNESS, VISCOUS_LAYERS, 1.25, outer_face_ids, 1, smeshBuilder.SURF_OFFSET_SMOOTH)

    inlet_mesh = Mesh_Final.GroupOnGeom(inlet, 'inlet', SMESH.FACE)
    outlet_mesh = Mesh_Final.GroupOnGeom(outlet, 'outlet', SMESH.FACE)
    sym_mesh = Mesh_Final.GroupOnGeom(sym, 'sym', SMESH.FACE)
    hull_mesh = Mesh_Final.GroupOnGeom(hull, 'hull', SMESH.FACE)

    Mesh_Final.Compute()
    
    exportToFoam(Mesh_Final, POLY_DIR)

try:
    run_meshing()
    sys.exit(0)
except Exception as e:
    with open(ERROR_LOG, "w") as f:
        f.write(traceback.format_exc())
    sys.exit(1)
"""
            with open(salome_script_path, 'w', encoding='utf-8') as f:
                f.write(vars_header)
                f.write(logic_script)
                
            self.toggle_buttons(False)
            self.worker = SalomeWorker(self.salome_bat_path, salome_script_path, self.working_dir)
            self.worker.log_signal.connect(self.handle_log_and_parse)
            self.worker.finished_signal.connect(self.on_process_finished)
            self.worker.start()

    def run_simulation(self):
        self.current_process_type = "run"   
        if not self.working_dir:
            self.log("ERROR: Working directory not set.")
            return

        cores = self.spin_cores.value()

        is_resume = False
        try:
            for item in os.listdir(self.working_dir):
                item_path = os.path.join(self.working_dir, item)
                if os.path.isdir(item_path):
                    try:
                        time_val = float(item)
                        if time_val > 0:
                            is_resume = True
                            break
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass

        if cores > 1:
            if is_resume:
                self.log(f"Resuming simulation in PARALLEL ({cores} Cores)...")
                commands = f"mpirun --oversubscribe -np {cores} interFoam -parallel && reconstructPar"
            else:
                self.log(f"Starting fresh simulation in PARALLEL ({cores} Cores)...")
                commands = f"setFields && decomposePar -force && mpirun --oversubscribe -np {cores} interFoam -parallel && reconstructPar"
        else:
            if is_resume:
                self.log("Resuming simulation in SERIAL (1 Core)...")
                commands = "interFoam"
            else:
                self.log("Starting fresh simulation in SERIAL (1 Core)...")
                commands = "setFields && interFoam"

        self.execute_wsl_command(commands)

    @pyqtSlot(int)
    def on_process_finished(self, return_code):
        self.toggle_buttons(True)
        
        if return_code == 130:
            self.log("Process was manually stopped.")
            self.current_process_type = ""
            return
            
        if self.current_process_type in ["openfoam_mesh", "check_mesh"]:
            log_path = os.path.join(self.working_dir, "checkMesh.log")
            if os.path.exists(log_path):
                self.parse_check_mesh_log()
            elif return_code != 0:
                self.log(f"Meshing failed with return code: {return_code}. checkMesh did not run.")
                
            self.preview_mesh_slice()
            self.current_process_type = ""
            return

        if return_code == 0:
            self.log("Process finished successfully.")
            
            if self.current_process_type == "salome_mesh":
                error_log = os.path.join(self.working_dir, "salome_error.log")
                polyMesh_dir = os.path.join(self.working_dir, "constant", "polyMesh")
                
                if not os.path.exists(polyMesh_dir) or not os.path.exists(os.path.join(polyMesh_dir, "points")):
                    self.log("\n[!!!] ERROR: SALOME failed to generate polyMesh!")
                    if os.path.exists(error_log):
                        with open(error_log, "r") as f:
                            self.log("--- SALOME PYTHON TRACEBACK ---")
                            self.log(f.read())
                    return
                
                self.log("SALOME meshing complete. Running checkMesh to verify quality...")
                self.current_process_type = "check_mesh"
                self.execute_wsl_command("checkMesh -allTopology -allGeometry > checkMesh.log")
                
        else:
            self.log(f"Process failed with return code: {return_code}")

    def parse_check_mesh_log(self):
        log_path = os.path.join(self.working_dir, "checkMesh.log")
        if not os.path.exists(log_path):
            return

        self.log("\n" + "="*50)
        self.log("          MESH QUALITY REPORT")
        self.log("="*50)

        mesh_ok = False
        failed_lines = []
        cells = "Unknown"

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    clean_line = line.strip()
                    if clean_line.startswith("cells:"):
                        cells = clean_line.split(":")[1].strip()
                    elif "Mesh OK." in clean_line:
                        mesh_ok = True
                    elif clean_line.startswith("***") or "Failed" in clean_line or "failed" in clean_line:
                        if "checkMesh" not in clean_line:
                            failed_lines.append(clean_line)

            self.log(f"Total Cells: {cells}")
            
            if mesh_ok:
                self.log("[+] MESH OK. All geometry and topology checks passed.")
                self.lbl_status.setText("Mesh Generation Complete - MESH OK")
            else:
                self.log("[-] MESH FAILED QUALITY CHECKS!")
                self.lbl_status.setText("Mesh Generation Complete - MESH FAILED")
                for err in failed_lines:
                    self.log(f"    {err}")
                    
                QMessageBox.warning(self, "Mesh Quality Warning", 
                                    f"checkMesh found issues in the mesh.\n\n"
                                    f"Total Cells: {cells}\n\n"
                                    f"Please check the Console Output for details on the failed faces.")
            
            self.log("="*50 + "\n")
            
        except Exception as e:
            self.log(f"Error reading checkMesh.log: {e}")

    def preview_mesh_slice(self):
        if not HAS_PYVISTA or not self.plotter: return
        try:
            self.log("Generating 3D Mesh Preview (Slice at Y=0)...")
            foam_file = os.path.join(self.working_dir, "preview.foam")
            with open(foam_file, 'w', encoding='utf-8') as f: f.write("")
            
            self.plotter.clear()
            self.plotter.add_axes()
            
            mesh_data = pv.read(foam_file)
            
            internal_mesh = None
            if isinstance(mesh_data, pv.MultiBlock):
                if len(mesh_data) > 0: internal_mesh = mesh_data[0]
            else:
                internal_mesh = mesh_data

            if internal_mesh and internal_mesh.n_points > 0:
                slice_mesh = internal_mesh.slice(normal='y', origin=(0, 0, 0))
                self.plotter.add_mesh(slice_mesh, show_edges=True, color="lightgreen", line_width=1, opacity=1.0, name="slice_actor")
            
            stl_path = os.path.join(self.working_dir, "constant", "triSurface", "hull.stl")
            if os.path.exists(stl_path):
                hull = pv.read(stl_path)
                self.plotter.add_mesh(hull, color="white", opacity=0.1, name="hull_actor")

            self.plotter.view_xz()
            self.plotter.reset_camera()
            self.log("Mesh preview loaded on 3D Scene.")
        except Exception as e:
            self.log(f"Error rendering mesh slice: {e}")

    def handle_log_and_parse(self, line):
        self.log(line) 
        
        if line.startswith("Time = "):
            self.seen_fields_this_step.clear() 
        
        m = re.search(r"Solving for ([\w\.]+),.*Initial residual = ([\d\.eE\-\+]+)", line)
        if m:
            field, val = m.group(1), float(m.group(2))
            if field not in self.seen_fields_this_step:
                self.seen_fields_this_step.add(field)
                
                if field not in self.residuals_data:
                    self.residuals_data[field] = []
                self.residuals_data[field].append(val)

    def update_residual_plot(self):
        if not HAS_MATPLOTLIB: return
        self.ax_resid.clear()
        self.ax_resid.set_yscale('log')
        self.ax_resid.set_xlabel("Time Steps")
        self.ax_resid.grid(True, alpha=0.3)
        
        has_data = False
        
        for field, data in self.residuals_data.items():
            if len(data) > 0: 
                self.ax_resid.plot(data, label=field, linewidth=1.0)
                has_data = True
                
        if has_data:
            self.ax_resid.legend(loc='upper right', fontsize='small')
            
        self.canvas_resid.draw()

    def plot_forces_on_tab(self):
        if not HAS_MATPLOTLIB: return
        wd = self.working_dir 
        if not wd or not os.path.exists(wd): return
        
        search = os.path.join(wd, "postProcessing", "forces", "*", "forces.dat")
        files = sorted(glob.glob(search))
        if not files: return

        t, drag_total = [], []
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line: continue
                    clean_line = line.replace('(', '').replace(')', '')
                    parts = clean_line.split()
                    
                    if len(parts) >= 10: 
                        try: 
                            time_val = float(parts[0])
                            px, vx, pox = float(parts[1]), float(parts[4]), float(parts[7])
                            total_drag_x = px + vx + pox
                            t.append(time_val)
                            drag_total.append(total_drag_x)
                        except: pass
                        
        if not t: return
        self.ax.clear()
        self.ax.plot(t, drag_total, 'r-', label='Total Drag Force (X) [N]', linewidth=1.5)
        self.ax.set_title(f"Ship Resistance (Current Drag: {drag_total[-1]:.2f} N)")
        self.ax.set_xlabel("Time (s)", color='black')
        self.ax.set_ylabel("Force (Newtons)", color='black')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend()
        self.canvas.draw()

    def refresh_wsl(self):
        try:
            res = subprocess.run(["wsl", "-l", "-q"], capture_output=True)
            out = res.stdout.decode('utf-16-le', errors='ignore').strip()
            distros = [l.strip() for l in out.splitlines() if l.strip()]
            
            self.combo_wsl.clear()
            if distros:
                self.combo_wsl.addItems(distros)
                sel_idx = next((i for i, d in enumerate(distros) if "Ubuntu" in d), 0)
                self.combo_wsl.setCurrentIndex(sel_idx)
                self.log(f"[INFO] WSL Detected: {distros[sel_idx]}")
            else:
                self.combo_wsl.addItem("Ubuntu")
        except: 
            self.combo_wsl.clear()
            self.combo_wsl.addItem("Ubuntu")
            self.log("[WARNING] Could not auto-detect WSL. Defaulting to 'Ubuntu'")

    def find_paraview_exe(self):
        if hasattr(self, 'paraview_path') and os.path.exists(self.paraview_path):
            return self.paraview_path

        search_paths = [
            r"C:\Program Files\ParaView*\bin\paraview.exe",
            r"C:\Program Files (x86)\ParaView*\bin\paraview.exe"
        ]
        
        for path_pattern in search_paths:
            matches = glob.glob(path_pattern)
            if matches:
                self.paraview_path = matches[0] 
                self.log(f"Auto-detected ParaView at: {self.paraview_path}")
                return self.paraview_path

        exe_path, _ = QFileDialog.getOpenFileName(self, "Select paraview.exe", r"C:\Program Files", "Executables (*.exe)")
        if exe_path:
            self.paraview_path = exe_path
            self.log(f"User manually mapped ParaView to: {self.paraview_path}")
            return self.paraview_path
            
        return None

    def open_paraview(self):
        if not self.working_dir:
            self.log("ERROR: Working directory not set.")
            return

        foam_file = "result.foam"
        foam_path = os.path.join(self.working_dir, foam_file)
        if not os.path.exists(foam_path):
            with open(foam_path, 'w', encoding='utf-8') as f: f.write("")

        pv_script_path = os.path.join(self.working_dir, "render_setup.py")
        
        pv_script_content = r"""
from paraview.simple import *

foam_file = "result.foam"
renderView = GetActiveViewOrCreate('RenderView')

reader = OpenFOAMReader(registrationName='SimulationData', FileName=foam_file)
reader.MeshRegions = ['internalMesh']
UpdatePipeline(proxy=reader)
Hide(reader, renderView)

GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
GetAnimationScene().GoToLast()
UpdatePipeline(proxy=reader)

c2p = CellDatatoPointData(registrationName='C2P_Converter', Input=reader)
c2p.ProcessAllArrays = 1
UpdatePipeline(proxy=c2p)
Hide(c2p, renderView)

free_surf = Contour(registrationName='FreeSurface', Input=c2p)
free_surf.ContourBy = ['POINTS', 'alpha.water']
free_surf.Isosurfaces = [0.5]
UpdatePipeline(proxy=free_surf)

surf_display = Show(free_surf, renderView)
surf_display.Representation = 'Surface'
ColorBy(surf_display, ('POINTS', 'U', 'Magnitude'))
surf_display.SetScalarBarVisibility(renderView, True)

hull_reader = OpenFOAMReader(registrationName='ShipHull', FileName=foam_file)
hull_reader.MeshRegions = ['patch/hull']
UpdatePipeline(proxy=hull_reader)

hull_display = Show(hull_reader, renderView)
hull_display.Representation = 'Surface'
ColorBy(hull_display, None)
hull_display.DiffuseColor = [0.6, 0.6, 0.6]

renderView.ResetCamera()
Render()
"""
        with open(pv_script_path, 'w', encoding='utf-8') as f:
            f.write(pv_script_content.strip())

        pv_exe = self.find_paraview_exe()
        if not pv_exe:
            self.log("ERROR: ParaView execution cancelled (executable not found).")
            return

        self.log("Launching ParaView and rendering final simulation state...")
        try:
            subprocess.Popen([pv_exe, '--script=render_setup.py'], cwd=self.working_dir)
        except Exception as e:
            self.log(f"ERROR: Failed to launch ParaView. {str(e)}")
    
    def update_system_monitor(self):
        if HAS_PSUTIL and hasattr(self, 'status_bar'):
            cpu = psutil.cpu_percent()
            ram_info = psutil.virtual_memory()
            ram_mb = ram_info.used / (1024 * 1024)
            
            try:
                widgets = self.status_bar.findChildren(QLabel)
                for w in widgets:
                    if "CPU:" in w.text(): w.setText(f"<b>CPU:</b> {cpu}%")
                    elif "RAM:" in w.text(): w.setText(f"<b>RAM:</b> {int(ram_mb)} MB")
            except: pass

        if hasattr(self, 'current_process_type') and self.current_process_type == "run":
            self.update_residual_plot()

    def start_ai(self, prompt):
        k = self.txt_apikey.text().strip()
        if not k: 
            QMessageBox.warning(self, "AI", "Missing API Key")
            return
      
        self.ai_chat_area.append(f"<b>You:</b> {prompt}")
        self.ai_chat_area.append("<i>Gemini is processing...</i>")
        
        sys_instruct = "\n\n[SYSTEM: Format your response using HTML tags ONLY (<b>, <br>, <ul>, etc). DO NOT use Markdown.]"
        
        self.ai_thread = GeminiWorker(k, prompt + sys_instruct, self.chat_history)
        self.ai_thread.result_ready.connect(self.on_ai_res)
        self.ai_thread.error_occurred.connect(lambda e: self.ai_chat_area.append(f"<b style='color:red'>{e}</b>"))
        self.ai_thread.start()

    def on_ai_res(self, txt):
        self.ai_chat_area.append(f"<b>Gemini:</b> {txt}<br><hr>")
        self.chat_history.append({"role": "user", "parts": [{"text": self.ai_thread.prompt}]})
        self.chat_history.append({"role": "model", "parts": [{"text": txt}]})

    def ai_send_manual(self):
        t = self.input_ai.text()
        self.input_ai.clear()
        if t: self.start_ai(t)

    def ai_analyze_log(self):
        log_txt = self.console_text.toPlainText()[-1500:]
        self.start_ai(f"Analyze this OpenFOAM terminal log and tell me what is wrong or if it's running fine:\n{log_txt}")

    def ai_suggest_settings(self):
        vel = self.input_vel.text()
        L = self.input_L.text()
        prompt = f"Act as a Marine Hydrodynamics expert. I am simulating a ship in a towing tank at velocity {vel} m/s, tank length {L}m. Suggest a good Delta T to keep Courant number < 1, and what physical phenomena I should look out for (like Kelvin wake pattern)."
        self.start_ai(prompt)

    def set_paraview_path_action(self):
        exe_path, _ = QFileDialog.getOpenFileName(self, "Select paraview.exe", "C:\\", "Executables (*.exe)")
        if exe_path:
            self.paraview_path = os.path.normpath(exe_path)
            self.log(f"ParaView path manually updated to: {self.paraview_path}")
            QMessageBox.information(self, "Success", "ParaView path updated successfully!")

    def set_unity_vr_path_action(self):
        exe_path, _ = QFileDialog.getOpenFileName(self, "Locate Unity VR Executable", "C:\\", "Executables (*.exe)")
        if exe_path:
            self.unity_exe_path = os.path.normpath(exe_path)
            self.log(f"Unity VR path manually updated to: {self.unity_exe_path}")
            QMessageBox.information(self, "Success", "Unity VR path updated successfully!")

    def set_salome_path_action(self):
        bat_path, _ = QFileDialog.getOpenFileName(self, "Locate run_salome.bat", "C:\\", "Batch Files (*.bat);;All Files (*)")
        if bat_path:
            self.salome_bat_path = os.path.normpath(bat_path)
            self.log(f"SALOME path manually updated to: {self.salome_bat_path}")
            QMessageBox.information(self, "Success", "SALOME path updated successfully!")

    def load_project_settings(self):
        if not self.working_dir:
            return

        sys_dir = os.path.join(self.working_dir, "system")
        zero_dir = os.path.join(self.working_dir, "0")
        const_dir = os.path.join(self.working_dir, "constant")

        u_file = os.path.join(zero_dir, "U")
        if os.path.exists(u_file):
            with open(u_file, 'r', encoding='utf-8') as f:
                content = f.read()
                m = re.search(r"internalField\s+uniform\s+\(([\d\.\-]+)\s+0\s+0\);", content)
                if m:
                    self.input_vel.setText(m.group(1))

        ctrl_file = os.path.join(sys_dir, "controlDict")
        if os.path.exists(ctrl_file):
            with open(ctrl_file, 'r', encoding='utf-8') as f:
                content = f.read()
                m_dt = re.search(r"deltaT\s+([\d\.\-]+);", content)
                if m_dt:
                    self.input_deltaT.setText(m_dt.group(1))
                m_et = re.search(r"endTime\s+([\d\.\-]+);", content)
                if m_et:
                    self.input_endTime.setText(m_et.group(1))
                m_wi = re.search(r"writeInterval\s+(\d+);", content)
                if m_wi:
                    self.spin_write_interval.setValue(int(m_wi.group(1)))

        dec_file = os.path.join(sys_dir, "decomposeParDict")
        if os.path.exists(dec_file):
            with open(dec_file, 'r', encoding='utf-8') as f:
                content = f.read()
                m_cores = re.search(r"numberOfSubdomains\s+(\d+);", content)
                if m_cores:
                    self.spin_cores.setValue(int(m_cores.group(1)))

        href_file = os.path.join(const_dir, "hRef")
        if os.path.exists(href_file):
            with open(href_file, 'r', encoding='utf-8') as f:
                content = f.read()
                m_href = re.search(r"value\s+([\d\.\-]+);", content)
                if m_href:
                    self.input_wl.setText(m_href.group(1))

        block_file = os.path.join(sys_dir, "blockMeshDict")
        if os.path.exists(block_file):
            with open(block_file, 'r', encoding='utf-8') as f:
                content = f.read()
                v_match = re.search(r"vertices\s*\(\s*(.*?)\s*\);", content, re.DOTALL)
                if v_match:
                    v_text = v_match.group(1)
                    coords = re.findall(r"\(([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\)", v_text)
                    if coords:
                        xs = [float(c[0]) for c in coords]
                        ys = [float(c[1]) for c in coords]
                        zs = [float(c[2]) for c in coords]
                        self.input_L.setText(f"{max(xs) - min(xs):.2f}")
                        self.input_W.setText(f"{max(ys) - min(ys):.2f}")
                        self.input_H.setText(f"{max(zs) - min(zs):.2f}")

        snappy_file = os.path.join(sys_dir, "snappyHexMeshDict")
        if os.path.exists(snappy_file):
            with open(snappy_file, 'r', encoding='utf-8') as f:
                content = f.read()
                m_ref = re.search(r"level\s*\(\s*(\d+)\s+\d+\s*\)", content)
                if m_ref:
                    self.spin_refine.setValue(int(m_ref.group(1)))
                m_layer = re.search(r"firstLayerThickness\s+([\d\.\-]+);", content)
                if m_layer:
                    loaded_thickness = float(m_layer.group(1))
                    self.combo_layer_mode.setCurrentText("Custom Input")
                    self.spin_layer_thick.setValue(loaded_thickness)

        self.log("Project parameters successfully loaded to GUI.")

    def closeEvent(self, event):
        if hasattr(self, 'plotter') and self.plotter:
            self.plotter.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    window = TowingTankGUI()
    window.show()
    sys.exit(app.exec())
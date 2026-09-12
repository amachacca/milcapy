import matplotlib.pyplot as plt
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QShortcut, QIcon, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QComboBox, QLineEdit, QColorDialog, QWidget, QWidgetAction
)
from milcapy.model.model import SystemMilcaModel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import copy
import numpy as np

from milcapy.plotter.widgets import InternalForceDiagramWidget

import os
import tkinter as tk
from tkinter import filedialog


# Clase para mostrar un gráfico de Matplotlib en un widget de Qt
class MatplotlibCanvas(QWidget):
    def __init__(self, parent, model: 'SystemMilcaModel', main_window: 'MainWindow'):
        """
        Clase para mostrar un gráfico de Matplotlib en un widget de Qt.

        :param parent: El widget padre.
        :type parent: QWidget
        :param model: El modelo del sistema.
        :type model: SystemMilcaModel
        :param main_window: La ventana principal.
        :type main_window: MainWindow
        """

        super().__init__(parent)
        ############################################################
        self.model = model
        self.figure = model.plotter.figure
        self.plotter_options = model.plotter_options
        self.main_window = main_window
        self.axes = self.figure.axes  # Obtener todos los ejes
        ############################################################

        # Almacenar límites originales de cada eje
        self.original_limits = {
            ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.axes}

        # Crear el lienzo de Matplotlib
        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Variables de paneo
        self._pan_start = None
        self.pan_active = False
        self.active_ax = None  # Eje activo

        # ==================== NODO SELECCIONABLE ====================
        # Datos de nodos (coordenadas X, Y)
        self.ax = self.model.plotter.axes
        # self.nodes = self.model.plotter.deformed_nodes[self.current_load_pattern] # {id: ax.scater(x, y)}
        self.selected_node = None  # Nodo actualmente seleccionado

        # Anotación para mostrar información del nodo seleccionado
        self.annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,               # Tamaño de letra
            fontweight="normal",      # Negrita ("bold", "light", "normal")
            color="blue",             # Color del texto
            style="italic",           # Cursiva ("italic", "normal")
            zorder=100,
            # bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="blue"), # Estilo del fondo
            arrowprops=dict(arrowstyle="->", color="black")  # Flecha
        )

        self.annotation.set_visible(False)
        # ===============================================================

        # Conectar eventos de ratón
        self._connect_events()

    @property
    def nodes(self):
        """
        Devuelve los nodos deformados para el patrón de carga actual.

        :return: Un diccionario con los nodos deformados.
        :rtype: dict
        """
        # {id: ax.scater(x, y)}
        return self.model.plotter.deformed_nodes[self.current_load_pattern]

    @property
    def members(self):
        return self.model.plotter.members  # {id: [line2D], ...}

    @property
    def trusses(self):
        return self.model.plotter.trusses   # {id: [line2D], ...}

    @property
    def csts(self):
        return self.model.plotter.csts  # {id: [line2D, polygon2D], ...}

    @property
    def current_load_pattern(self) -> str | None:
        return self.model.current_load_pattern

    def _connect_events(self):
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('button_press_event', self._on_right_click)

    def _on_click(self, event):
        """ Detecta el nodo más cercano en píxeles y activa el snap """
        if event.inaxes is None and 1:  # Clic fuera del gráfico
            return

        # Convertir coordenadas de nodos a píxeles de pantalla
        node_pixels = {node_id: self.ax.transData.transform(s.get_offsets()[0])
                       for node_id, s in self.nodes.items()}  # {id: (px_x, px_y)}

        # Coordenadas del clic en píxeles
        click_pixel = np.array([event.x, event.y])

        # Calcular distancia en píxeles entre el clic y cada nodo
        distances = {node_id: np.linalg.norm(
            pos - click_pixel) for node_id, pos in node_pixels.items()}

        # Definir umbral de snap (10 píxeles)
        snap_threshold = 10
        closest_node = min(distances, key=distances.get)  # Nodo más cercano
        closest_distance = distances[closest_node]

        if closest_distance <= snap_threshold and event.button == 1:
            # Restaurar color del nodo previamente seleccionado
            if self.selected_node is not None:
                self.nodes[self.selected_node].set_color('blue')

            # Seleccionar nuevo nodo
            self.selected_node = closest_node
            self.nodes[self.selected_node].set_color('red')  # Cambiar color

            # Actualizar anotación
            node_x, node_y = self.nodes[self.selected_node].get_offsets()[0]
            decimals = self.plotter_options.disp_nodes_decimals
            if self.model.nodes[self.selected_node].local_axis is not None:
                displacements = self.model.results[self.current_load_pattern].get_node_displacements(
                self.selected_node)
                T = self.model.nodes[self.selected_node].local_axis.get_transformation_matrix()
                displacements = T.T @ displacements
                self.annotation.set_text(
                    f"Nodo {self.selected_node}\nu1 = {displacements[0]:.{decimals}f}\nu2 = {displacements[1]:.{decimals}f}\nr3 = {displacements[2]:.{decimals}f}")
            else:
                displacements = self.model.results[self.current_load_pattern].get_node_displacements(
                self.selected_node)
                self.annotation.set_text(
                    f"Nodo {self.selected_node}\nux = {displacements[0]:.{decimals}f}\nuy = {displacements[1]:.{decimals}f}\nrz = {displacements[2]:.{decimals}f}")

            self.annotation.xy = (node_x, node_y)
            self.annotation.set_visible(True)
        elif closest_distance > snap_threshold and event.button == 1:
            # Si no hay nodos cercanos, ocultar anotación y restaurar color
            if self.selected_node is not None:
                self.nodes[self.selected_node].set_color('blue')
            self.selected_node = None
            self.annotation.set_visible(False)

        # Actualizar ploteo
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """Zoom con la rueda del ratón."""
        ax = event.inaxes
        if ax is None:
            return

        base_scale = 1.2
        scale_factor = 1 / base_scale if event.step > 0 else base_scale  # Qt usa 'step'

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_center = event.xdata
        y_center = event.ydata

        x_min = x_center - (x_center - x_min) * scale_factor
        x_max = x_center + (x_max - x_center) * scale_factor
        y_min = y_center - (y_center - y_min) * scale_factor
        y_max = y_center + (y_max - y_center) * scale_factor

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        self.canvas.draw_idle()

    def _on_button_press(self, event):
        """Activar el paneo con el botón del medio."""
        ax = event.inaxes
        if ax is not None and event.button == 2:  # Botón medio
            self._pan_start = (event.xdata, event.ydata)
            self.pan_active = True
            self.active_ax = ax
            self.setCursor(Qt.ClosedHandCursor)

    def _on_button_release(self, event):
        """Desactivar el paneo."""
        if event.button == 2:
            self.pan_active = False
            self._pan_start = None
            self.active_ax = None
            self.unsetCursor()

    def _on_mouse_move(self, event):
        """Mover el gráfico al hacer paneo."""
        if self.pan_active and self.active_ax and event.inaxes == self.active_ax and self._pan_start:
            dx = self._pan_start[0] - event.xdata
            dy = self._pan_start[1] - event.ydata

            x_min, x_max = self.active_ax.get_xlim()
            y_min, y_max = self.active_ax.get_ylim()

            self.active_ax.set_xlim(x_min + dx, x_max + dx)
            self.active_ax.set_ylim(y_min + dy, y_max + dy)
            self.canvas.draw_idle()

    def _on_right_click(self, event):
        """
        abrir ventana emergente para click derecho, una venta con las respuestas del miembro
        """
        if event.inaxes is None:  # Clic fuera del gráfico
            return
        global_contains = False

        if event.button == 3:  # Clic derecho
            for member_id, line in list(self.members.items()) + list(self.trusses.items()):
                contains, _ = line.contains(event)
                if contains:
                    global_contains = True
                    self.model.plotter.selected_member = member_id
                    # Guardar el color original y cambiar a rojo
                    if member_id in self.trusses:
                        original_color = self.model.plotter.trusses[member_id].get_color()
                        original_linewidth = self.model.plotter.trusses[member_id].get_linewidth()
                        self.model.plotter.trusses[member_id].set_color("red")
                        self.model.plotter.trusses[member_id].set_linewidth(2)
                        self.model.plotter.figure.canvas.draw_idle()
                        element = self.model.trusses[member_id]
                    else:
                        original_color = self.model.plotter.members[member_id].get_color()
                        original_linewidth = self.model.plotter.members[member_id].get_linewidth()
                        self.model.plotter.members[member_id].set_color("red")
                        self.model.plotter.members[member_id].set_linewidth(2)
                        self.model.plotter.figure.canvas.draw_idle()
                        element = self.model.members[member_id]
                    try:
                        x = self.model.results[self.current_load_pattern].get_member_x_val(member_id)
                    except:
                        x = np.linspace(0, element.length(), self.model.postprocessing_options.n)

                    # Crear ventana emergente de Tkinter y esperar a que se cierre
                    InternalForceDiagramWidget(element, self.model.plotter.diagrams, x)
                    break

            if global_contains:
                # Restaurar el color original
                plt.close("all")
                if member_id in self.trusses:
                    self.model.plotter.trusses[member_id].set_color(original_color)
                    self.model.plotter.trusses[member_id].set_linewidth(original_linewidth)
                    self.model.plotter.figure.canvas.draw_idle()
                else:
                    self.model.plotter.members[member_id].set_color(original_color)
                    self.model.plotter.members[member_id].set_linewidth(original_linewidth)
                    self.model.plotter.figure.canvas.draw_idle()
                return

            # Membranas: clic derecho muestra tensiones nodales del elemento
            for bucket, kind in (
                (self.model.plotter.csts, "cst"),
                (self.model.plotter.membrane_q3dof, "q3"),
                (self.model.plotter.membrane_q2dof, "q2"),
            ):
                for ele_id, artists in bucket.items():
                    line = artists[0] if artists else None
                    if line is None:
                        continue
                    contains, _ = line.contains(event)
                    if contains:
                        self.annotation.set_text(
                            self._membrane_stress_text(kind, ele_id))
                        self.annotation.xy = (event.xdata, event.ydata)
                        self.annotation.set_visible(True)
                        self.canvas.draw_idle()
                        return

    def _membrane_stress_text(self, kind: str, ele_id: int) -> str:
        """Texto de anotación con tensiones nodales promedio del elemento."""
        decimals = self.plotter_options.disp_nodes_decimals
        results = self.model.results[self.current_load_pattern]
        try:
            if kind == "cst":
                s = np.asarray(results.get_cst_stresses(ele_id)).ravel()[:3]
                e = np.asarray(results.get_cst_strains(ele_id)).ravel()[:3]
                label = f"CST {ele_id}"
            elif kind == "q3":
                s = np.asarray(results.get_membrane_q3dof_stresses(ele_id)).reshape(-1, 3).mean(axis=0)
                e = np.asarray(results.get_membrane_q3dof_strains(ele_id)).reshape(-1, 3).mean(axis=0)
                label = f"Membrana Q3DOF {ele_id}"
            else:
                s = np.asarray(results.get_membrane_q2dof_stresses(ele_id)).reshape(-1, 3).mean(axis=0)
                e = np.asarray(results.get_membrane_q2dof_strains(ele_id)).reshape(-1, 3).mean(axis=0)
                label = f"Membrana Q2DOF {ele_id}"
        except Exception:
            return f"Elemento {ele_id}\n(sin resultados de esfuerzos)"
        return (
            f"{label}\n"
            f"SX = {s[0]:.{decimals}f}\nSY = {s[1]:.{decimals}f}\nSXY = {s[2]:.{decimals}f}\n"
            f"EX = {e[0]:.{decimals}f}\nEY = {e[1]:.{decimals}f}\nEXY = {e[2]:.{decimals}f}"
        )


class GraphicOptionsDialog(QDialog):
    """Ventana emergente para opciones de gráfico."""

    def __init__(self, parent=None, model: 'SystemMilcaModel' = None, options=None):
        super().__init__(parent)
        """
        Inicializa la ventana de opciones de gráfico.

        Args:
            parent (QWidget, optional): El widget padre. Defaults to None.
            model (SystemMilcaModel, optional): El modelo del sistema. Defaults to None.
            options (dict, optional): Opciones de gráfico. Defaults to None.
        """
        self.setWindowTitle("Opciones de Gráfico")
        self.setMinimumSize(400, 500)
        self.model = model
        self.plotter_options = model.plotter.plotter_options
        self.options = options  # Diccionario para almacenar opciones seleccionadas

        main_layout = QVBoxLayout()

        # Secciones de opciones
        main_layout.addWidget(self.create_general_options())
        main_layout.addWidget(self.create_node_options())
        main_layout.addWidget(self.create_member_options())
        main_layout.addWidget(self.create_assignations_options())
        main_layout.addWidget(self.create_stress_options())
        main_layout.addWidget(self.create_scale_options())

        # Botones Aceptar / Restaurar / Aplicar / Cancelar
        button_layout = QHBoxLayout()
        accept_button = QPushButton("Aceptar")
        restore_button = QPushButton("Restaurar Valores")
        apply_button = QPushButton("Aplicar")
        cancel_button = QPushButton("Cancelar")

        accept_button.clicked.connect(self.accept_changes)
        restore_button.clicked.connect(self.restore_defaults)
        apply_button.clicked.connect(self.apply_changes)
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(accept_button)
        button_layout.addWidget(restore_button)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    @property
    def current_load_pattern(self) -> str | None:
        """
        Devuelve el Nombre del patrón de carga actual.

        Returns:
            str | None: El nombre del patrón de carga actual.
        """
        return self.model.current_load_pattern

    def create_general_options(self):
        """
        Crea las opciones generales como color de fondo.

        Returns:
            QGroupBox: El grupo de opciones generales.
        """
        group = QGroupBox("Generales")
        layout = QVBoxLayout()

        self.background_color_button = QPushButton(
            "Seleccionar Color de Fondo")
        self.background_color_button.clicked.connect(
            self._select_background_color)
        layout.addWidget(self.background_color_button)

        group.setLayout(layout)
        return group

    def create_node_options(self):
        """
        Crea las opciones para nodos.

        Returns:
            QGroupBox: El grupo de opciones para nodos.
        """
        group = QGroupBox("Nodos")
        layout = QVBoxLayout()

        self.show_nodes_checkbox = QCheckBox("Mostrar Nodos")
        self.node_labels_checkbox = QCheckBox("Mostrar Etiquetas de Nodos")

        layout.addWidget(self.show_nodes_checkbox)
        layout.addWidget(self.node_labels_checkbox)
        group.setLayout(layout)
        return group

    def create_member_options(self):
        """
        Crea las opciones para miembros (elementos).

        Returns:
            QGroupBox: El grupo de opciones para miembros.
        """
        group = QGroupBox("Miembros")
        layout = QVBoxLayout()

        self.show_members_checkbox = QCheckBox("Mostrar Miembros")
        self.member_labels_checkbox = QCheckBox(
            "Mostrar Etiquetas de Miembros")

        layout.addWidget(self.show_members_checkbox)
        layout.addWidget(self.member_labels_checkbox)
        group.setLayout(layout)
        return group

    def create_assignations_options(self):
        """
        Crea las opciones para visualización de asignaciones.

        Returns:
            QGroupBox: El grupo de opciones para visualización de asignaciones.
        """
        group = QGroupBox("Asignaciones")
        layout = QVBoxLayout()

        self.show_loads_checkbox = QCheckBox("Mostrar Cargas")
        self.show_supports_checkbox = QCheckBox("Mostrar Soportes")

        layout.addWidget(self.show_loads_checkbox)
        layout.addWidget(self.show_supports_checkbox)
        group.setLayout(layout)
        return group

    def create_stress_options(self):
        """
        Crea las opciones del mapa de esfuerzos en membranas.

        Returns:
            QGroupBox: El grupo de opciones de esfuerzos.
        """
        from milcapy.utils.types import FieldType as _FieldType

        group = QGroupBox("Esfuerzos en membranas")
        layout = QVBoxLayout()

        self.show_stress_checkbox = QCheckBox("Mostrar esfuerzos")
        layout.addWidget(self.show_stress_checkbox)

        layout.addWidget(QLabel("Campo"))
        self.stress_field_combo = QComboBox()
        self.stress_field_combo.addItems([f.value for f in _FieldType])
        layout.addWidget(self.stress_field_combo)

        layout.addWidget(QLabel("Colormap"))
        self.stress_colormap_combo = QComboBox()
        self.stress_colormap_combo.addItems(
            ["jet", "viridis", "plasma", "coolwarm", "rainbow", "turbo"])
        layout.addWidget(self.stress_colormap_combo)

        group.setLayout(layout)
        return group

    def create_scale_options(self):
        """
        Crea las opciones para visualización de la forma deformada.

        Returns:
            QGroupBox: El grupo de opciones para visualización de la forma deformada.
        """
        group = QGroupBox("Escalas")
        layout = QVBoxLayout()

        self.deformation_scale_input = QLineEdit()
        self.deformation_scale_input.setPlaceholderText(
            "Escala de deformación")
        self.deformation_scale_input.setValidator(
            QDoubleValidator(1e-9, 1e9, 20))

        layout.addWidget(QLabel("Escala de Deformación"))
        layout.addWidget(self.deformation_scale_input)

        # self.internal_forces_scale_input = QLineEdit()
        # self.internal_forces_scale_input.setPlaceholderText(
        #     "Escala de fuerzas internas")
        # self.internal_forces_scale_input.setValidator(
        #     QDoubleValidator(1e-9, 1e9, 20))

        # layout.addWidget(QLabel("Escala de Fuerzas Internas"))
        # layout.addWidget(self.internal_forces_scale_input)

        group.setLayout(layout)
        return group

    def _select_background_color(self):
        """
        Abre el selector de color para el fondo.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.options["UI_background_color"] = color.name()

    def restore_defaults(self):
        """
        Restablece las opciones a los valores por defecto.
        """
        self.show_nodes_checkbox.setChecked(False)
        self.node_labels_checkbox.setChecked(False)
        self.show_members_checkbox.setChecked(True)
        self.member_labels_checkbox.setChecked(False)
        self.show_loads_checkbox.setChecked(True)
        self.show_supports_checkbox.setChecked(True)
        self.show_stress_checkbox.setChecked(False)
        self.stress_field_combo.setCurrentText("SX")
        self.stress_colormap_combo.setCurrentText("jet")
        scale = self.plotter_options.UI_deformation_scale.get(self.current_load_pattern, 40)
        self.deformation_scale_input.setText(str(round(scale, 2)))
        # scale = self.plotter_options.UI_internal_forces_scale.get(self.current_load_pattern, 40)
        # self.internal_forces_scale_input.setText(str(round(scale, 2)))
        self.__recover_values()
        # ! RESTABLECE EL COLOR DE FONDO INICIAL
        self.options["UI_background_color"] = 'white'
        self.__transfer_changes()
        self.update_changer(reset=True)
        print("Restaurado a valores por defecto.")

    def accept_changes(self):
        """
        Guarda las opciones seleccionadas y las imprime.
        """
        self.__recover_values()
        self.__transfer_changes()
        self.update_changer()
        self.accept()

    def apply_changes(self):
        """
        Aplica las opciones seleccionadas pero no cierra la ventana.
        """
        self.__recover_values()
        self.__transfer_changes()
        self.update_changer()

    def __recover_values(self):
        """
        Recupera los valores de los widgets.
        """
        self.options["UI_show_nodes"] = self.show_nodes_checkbox.isChecked()
        self.options["UI_node_labels"] = self.node_labels_checkbox.isChecked()
        self.options["UI_show_members"] = self.show_members_checkbox.isChecked()
        self.options["UI_member_labels"] = self.member_labels_checkbox.isChecked()
        self.options["UI_load"] = self.show_loads_checkbox.isChecked()
        self.options["UI_support"] = self.show_supports_checkbox.isChecked()
        self.options["UI_stress"] = self.show_stress_checkbox.isChecked()
        self.options["UI_stress_field"] = self.stress_field_combo.currentText()
        self.options["UI_stress_colormap"] = self.stress_colormap_combo.currentText()
        deformation_scale_text = self.deformation_scale_input.text()
        self.options["UI_deformation_scale"][self.current_load_pattern] = float(
            deformation_scale_text) if deformation_scale_text else 40

    def __transfer_changes(self):
        """
        Transfiere las opciones seleccionadas al objeto PlotterOptions.
        """
        op = self.plotter_options
        op.UI_background_color = self.options.get("UI_background_color", None)
        op.UI_show_nodes = self.options.get("UI_show_nodes", False)
        op.UI_show_members = self.options.get("UI_show_members", True)
        op.UI_node_labels = self.options.get("UI_node_labels", False)
        op.UI_member_labels = self.options.get("UI_member_labels", False)
        op.UI_load = self.options.get("UI_load", True)
        op.UI_support = self.options.get("UI_support", True)
        op.UI_stress = self.options.get("UI_stress", False)
        from milcapy.utils.types import FieldType as _FieldType, to_enum as _to_enum
        try:
            op.stress_field = _to_enum(
                self.options.get("UI_stress_field", "SX"), _FieldType)
        except ValueError:
            op.stress_field = _FieldType.SX
        op.stress_colormap = self.options.get("UI_stress_colormap", "jet")

    def _keep_data(self):
        """
        Asigna los valores de las opciones anteriores (mantiene) a los widgets.
        """
        self.show_nodes_checkbox.setChecked(
            self.options.get("UI_show_nodes", False))
        self.node_labels_checkbox.setChecked(
            self.options.get("UI_node_labels", False))
        self.show_members_checkbox.setChecked(
            self.options.get("UI_show_members", True))
        self.member_labels_checkbox.setChecked(
            self.options.get("UI_member_labels", False))
        self.show_loads_checkbox.setChecked(self.options.get("UI_load", True))
        self.show_supports_checkbox.setChecked(self.options.get("UI_support", True))
        self.show_stress_checkbox.setChecked(self.options.get("UI_stress", False))
        self.stress_field_combo.setCurrentText(self.options.get("UI_stress_field", "SX"))
        self.stress_colormap_combo.setCurrentText(self.options.get("UI_stress_colormap", "jet"))
        self.deformation_scale_input.setText(str(self.options.get(
            "UI_deformation_scale", {}).get(self.current_load_pattern, 40)))

    def update_changer(self, reset: bool = False):
        """
        Actualiza los cambios a la figura principal y su vista inmediata.
        """
        if self.options.get("UI_background_color", None) is not None:
            self.model.plotter.change_background_color()    # cambia el color de fondo
        self.model.plotter.update_nodes()                   # actualiza los nodos
        self.model.plotter.update_members()                 # actualiza los miembros
        self.model.plotter.update_trusses()                 # actualiza los trusses
        self.model.plotter.update_cst()                     # actualiza los CSTs
        self.model.plotter.update_membrane_q3dof()          # actualiza los Membranas de 3 dof / nodo
        self.model.plotter.update_membrane_q2dof()          # actualiza los Membranas de 2 dof / nodo
        self.model.plotter.update_node_labels()             # actualiza los labels de los nodos
        self.model.plotter.update_member_labels()           # actualiza los labels de los miembros
        self.model.plotter.update_trusses_labels()          # actualiza los labels de los trusses
        self.model.plotter.update_cst_labels()              # actualiza los labels de los CSTs
        self.model.plotter.update_membrane_q3dof_labels()   # actualiza los labels de los Membranas de 3 dof / nodo
        self.model.plotter.update_membrane_q2dof_labels()   # actualiza los labels de los Membranas de 2 dof / nodo
        self.model.plotter.update_end_length_offset()       # actualiza los labels de los miembros
        self.model.plotter.update_frame_release()           # actualiza los labels de los miembros
        self.model.plotter.update_point_load()              # actualiza las cargas
        self.model.plotter.update_point_load_labels()       # actualiza los labels de las cargas
        self.model.plotter.update_distributed_loads()       # actualiza las cargas distribuidas
        self.model.plotter.update_distributed_load_labels() # actualiza los labels de las cargas distribuidas
        self.model.plotter.update_prescribed_dofs()         # actualiza los grados de libertad prescindidos
        self.model.plotter.update_prescribed_dofs_labels()  # actualiza los labels de los grados de libertad prescindidos
        self.model.plotter.update_supports()                # actualiza los soportes
        self.model.plotter.update_elastic_supports()        # actualiza los soportes elásticos
        # mapa de esfuerzos (campo/colormap desde el diálogo)
        self.model.plotter.update_stress_field(visibility=self.options.get("UI_stress", False))
        # verificar si cambio la escala de deformación
        escala = self.options.get("UI_deformation_scale", {}).get(
            self.current_load_pattern, 40)
        if escala != self.plotter_options.UI_deformation_scale.get(self.current_load_pattern, 40):
            self.model.plotter.update_deformed(escala=escala)
            self.model.plotter.update_rigid_deformed(escala=escala)
            self.model.plotter.update_displaced_nodes(scale=escala)
            if self.model.plotter_options.UI_rigid_deformed == True or self.model.plotter_options.UI_deformed == True:
                self.model.plotter.update_supports(scale=escala)                # actualiza los soportes
                self.model.plotter.update_elastic_supports(scale=escala)        # actualiza los soportes elásticos

        if reset:
            escala = self.plotter_options.UI_deformation_scale.get(self.current_load_pattern, 40)
            self.model.plotter.update_deformed(escala=escala)
            self.model.plotter.update_rigid_deformed(escala=escala)
            self.model.plotter.update_displaced_nodes(scale=escala)
            if self.model.plotter_options.UI_rigid_deformed == True or self.model.plotter_options.UI_deformed == True:
                self.model.plotter.update_supports(scale=escala)                # actualiza los soportes
                self.model.plotter.update_elastic_supports(scale=escala)        # actualiza los soportes elásticos


# Clase para la ventana principal
class MainWindow(QMainWindow):
    def __init__(self, model: 'SystemMilcaModel'):
        """
        Inicializa la ventana principal del Visualizador del objeto "SystemMilcaModel".
        """
        super().__init__()
        self.setWindowTitle("MILCApy")
        self.setGeometry(800, 80, 800, 800)
        try:
            self.setWindowIcon(QIcon("milcapy/plotter/assets/milca.ico"))
        except:
            pass

        self.model = model

        # Crear el widget de Matplotlib
        self.plot_widget = MatplotlibCanvas(self, self.model, self)
        self.setCentralWidget(self.plot_widget)

        # Crear la barra de herramientas
        self.toolbar = self.addToolBar("Herramientas")

        # Acción para guardar IMAGENES
        action_save = QAction(QIcon("otros/logo.png"), "Guardar", self)
        action_save.triggered.connect(self.on_save_figure)
        self.toolbar.addAction(action_save)

        # SEPARADOR
        self.toolbar.addSeparator()

        # Acción para OPCIONES DE GRAFICO
        action_save = QAction(
            QIcon("otros/opciones_de_grafico.png"), "Opciones de gráfico", self)
        action_save.triggered.connect(self.abrir_opciones_grafico)
        self.toolbar.addAction(action_save)

        # SEPARADOR
        self.toolbar.addSeparator()

        # SELECCIONAR LOAD PATERNS
        self.combo = QComboBox()
        lista_patterns = list(self.model.results.keys())
        self.combo.addItems(lista_patterns)
        self.combo.currentTextChanged.connect(self.on_pattern_selected)

        # Envolver el ComboBox en un QWidgetAction
        combo_action = QWidgetAction(self)
        combo_action.setDefaultWidget(self.combo)

        # Agregar a la barra de herramientas
        self.toolbar.addAction(combo_action)
        self.toolbar.addSeparator()

        # CASILLAS DE VERIFICACION:
        self.DFA = QCheckBox("DFA")
        self.DFA.stateChanged.connect(self.mostrar_fuerzas_axiales)
        self.toolbar.addWidget(self.DFA)
        self.DFA.setChecked(self.model.plotter_options.UI_axial)
        self.toolbar.addSeparator()
        self.DFC = QCheckBox("DFC")
        self.DFC.stateChanged.connect(self.mostrar_fuerzas_cortantes)
        self.toolbar.addWidget(self.DFC)
        self.DFC.setChecked(self.model.plotter_options.UI_shear)
        self.toolbar.addSeparator()
        self.DMF = QCheckBox("DMF")
        self.DMF.stateChanged.connect(self.mostrar_momentos)
        self.toolbar.addWidget(self.DMF)
        self.DMF.setChecked(self.model.plotter_options.UI_moment)
        self.toolbar.addSeparator()
        self.REACIONES = QCheckBox("Reacciones")
        self.REACIONES.stateChanged.connect(self.mostrar_reacciones)
        self.toolbar.addWidget(self.REACIONES)
        self.REACIONES.setChecked(self.model.plotter_options.UI_reactions)
        self.toolbar.addSeparator()
        self.DEFORMADA = QCheckBox("Deformada")
        self.DEFORMADA.stateChanged.connect(self.mostrar_deformada)
        self.toolbar.addWidget(self.DEFORMADA)
        self.DEFORMADA.setChecked(self.model.plotter_options.UI_deformed)
        self.toolbar.addSeparator()
        self.DEFORMADA_RIGIDA = QCheckBox("Deformada rígida")
        self.DEFORMADA_RIGIDA.stateChanged.connect(
            self.mostrar_deformada_rigida)
        self.toolbar.addWidget(self.DEFORMADA_RIGIDA)
        self.DEFORMADA_RIGIDA.setChecked(
            self.model.plotter_options.UI_rigid_deformed)
        self.toolbar.addSeparator()
        self.ESFUERZOS = QCheckBox("Esfuerzos")
        self.ESFUERZOS.stateChanged.connect(self.mostrar_esfuerzos)
        self.toolbar.addWidget(self.ESFUERZOS)
        self.ESFUERZOS.setChecked(self.model.plotter_options.UI_stress)
        self.toolbar.addSeparator()

        # SELECTOR DE CAMPO DE ESFUERZOS
        from milcapy.utils.types import FieldType as _FieldType
        self.combo_field = QComboBox()
        self.combo_field.addItems([f.value for f in _FieldType])
        current_field = self.model.plotter_options.stress_field
        self.combo_field.setCurrentText(
            current_field.value if isinstance(current_field, _FieldType) else str(current_field))
        self.combo_field.currentTextChanged.connect(self.on_field_selected)
        combo_field_action = QWidgetAction(self)
        combo_field_action.setDefaultWidget(self.combo_field)
        self.toolbar.addAction(combo_field_action)
        self.toolbar.addSeparator()

        # Atajo de teclado (Ctrl + H) para mostrar/ocultar la barra de herramientas
        self.toggle_toolbar_shortcut = QShortcut(Qt.CTRL | Qt.Key_H, self)
        self.toggle_toolbar_shortcut.activated.connect(self.toggle_toolbar)

        # Inicialmente visible la barra de herramientas
        self.toolbar.setVisible(True)

        #####################################################
        self.options_values = {
            "UI_background_color": None,
            "UI_show_nodes": False,
            "UI_node_labels": False,
            "UI_show_members": True,
            "UI_member_labels": False,
            "UI_load": True,
            "UI_deformation_scale": copy.deepcopy(self.model.plotter_options.UI_deformation_scale),
            "UI_internal_forces_scale": 0.03,
            "UI_filling_type": "Colormap",
            "UI_colormap": "rainbow",
            "UI_support": True,
            "UI_stress": False,
            "UI_stress_field": "SX",
            "UI_stress_colormap": "jet",
        }
        #####################################################

    def toggle_toolbar(self):
        """Activa/Desactiva la barra de herramientas"""
        self.toolbar.setVisible(not self.toolbar.isVisible())


    def on_save_figure(self):
        # Inicializar tkinter en modo oculto
        root = tk.Tk()
        root.withdraw()

        # Abrir cuadro de diálogo para elegir ruta
        path = filedialog.asksaveasfilename(
            title="Guardar figura como...",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos los archivos", "*.*")]
        )

        # Si el usuario canceló
        if not path:
            return

        # Guardar figura con matplotlib (sobrescribe si ya existe)
        self.model.plotter.figure.savefig(
            path,
            dpi=self.model.plotter_options.save_fig_dpi,
            bbox_inches="tight"
        )

        print(f"Figura guardada en: {path}")


    def abrir_opciones_grafico(self):
        """Abre la ventana emergente de opciones de gráfico"""
        dialog = GraphicOptionsDialog(self, self.model, self.options_values)
        dialog._keep_data()
        dialog.show()

    def on_pattern_selected(self, value):
        # ! NOTIFICATION
        self.model.current_load_pattern = value
        # VISUALIZACIONES AL INICIO
        self.DFA.setChecked(self.model.plotter_options.UI_axial)
        self.DFC.setChecked(self.model.plotter_options.UI_shear)
        self.DMF.setChecked(self.model.plotter_options.UI_moment)
        self.REACIONES.setChecked(self.model.plotter_options.UI_reactions)
        self.DEFORMADA.setChecked(self.model.plotter_options.UI_deformed)
        self.DEFORMADA_RIGIDA.setChecked(
            self.model.plotter_options.UI_rigid_deformed)
        self.ESFUERZOS.setChecked(self.model.plotter_options.UI_stress)
        current_field = self.model.plotter_options.stress_field
        self.combo_field.setCurrentText(
            current_field.value if hasattr(current_field, "value") else str(current_field))

        # ! ACTUALIZACIONES DE VISIBILIDAD AL CAMBIAR EL PATRON
        if self.model.plotter_options.UI_deformed:
            self.mostrar_deformada(2)
        elif not self.model.plotter_options.UI_deformed:
            self.mostrar_deformada(0)
        if self.model.plotter_options.UI_rigid_deformed:
            self.mostrar_deformada_rigida(2)
        elif not self.model.plotter_options.UI_rigid_deformed:
            self.mostrar_deformada_rigida(0)
        self.model.plotter.update_change()
        if self.model.plotter_options.UI_axial:
            self.mostrar_fuerzas_axiales(2)
        elif not self.model.plotter_options.UI_axial:
            self.mostrar_fuerzas_axiales(0)
        if self.model.plotter_options.UI_shear:
            self.mostrar_fuerzas_cortantes(2)
        elif not self.model.plotter_options.UI_shear:
            self.mostrar_fuerzas_cortantes(0)
        if self.model.plotter_options.UI_moment:
            self.mostrar_momentos(2)
        elif not self.model.plotter_options.UI_moment:
            self.mostrar_momentos(0)
        if self.model.plotter_options.UI_reactions:
            self.mostrar_reacciones(2)
        elif not self.model.plotter_options.UI_reactions:
            self.mostrar_reacciones(0)
        if self.model.plotter_options.UI_support:
            self.mostrar_soportes(2)
        elif not self.model.plotter_options.UI_support:
            self.mostrar_soportes(0)
        if self.model.plotter_options.UI_stress:
            self.mostrar_esfuerzos(2)
        elif not self.model.plotter_options.UI_stress:
            self.mostrar_esfuerzos(0)

    def mostrar_fuerzas_axiales(self, state):
        """Muestra las fuerzas axiales"""
        if state == 2:
            self.model.plotter_options.UI_axial = True
            self.model.plotter.update_axial_force(visibility=True)
        elif state == 0:
            self.model.plotter_options.UI_axial = False
            self.model.plotter.update_axial_force(visibility=False)
        self.diagrams_and_deformed("F")

    def mostrar_fuerzas_cortantes(self, state):
        """Muestra las fuerzas cortantes"""
        if state == 2:
            self.model.plotter_options.UI_shear = True
            self.model.plotter.update_shear_force(visibility=True)
        elif state == 0:
            self.model.plotter_options.UI_shear = False
            self.model.plotter.update_shear_force(visibility=False)
        self.diagrams_and_deformed("F")

    def mostrar_momentos(self, state):
        """Muestra las fuerzas momentos"""
        if state == 2:
            self.model.plotter_options.UI_moment = True
            self.model.plotter.update_bending_moment(visibility=True)
        elif state == 0:
            self.model.plotter_options.UI_moment = False
            self.model.plotter.update_bending_moment(visibility=False)
        self.diagrams_and_deformed("F")

    def mostrar_reacciones(self, state):
        """Muestra las reacciones"""
        if state == 2:
            self.model.plotter_options.UI_reactions = True
            self.model.plotter.update_reactions(visibility=True)
        elif state == 0:
            self.model.plotter_options.UI_reactions = False
            self.model.plotter.update_reactions(visibility=False)

    def mostrar_deformada(self, state):
        """Muestra la deformada"""
        if state == 2: # estado activo
            scale = self.options_values.get("UI_deformation_scale", {}).get(self.model.current_load_pattern, 40)
            self.model.plotter_options.UI_deformed = True
            self.model.plotter.update_deformed()
            self.model.plotter.update_displaced_nodes()
            self.model.plotter.update_supports(scale=scale)
            self.model.plotter.update_elastic_supports(scale=scale)
        elif state == 0: # estado desactivado
            self.model.plotter_options.UI_deformed = False
            self.model.plotter.update_deformed()
            self.model.plotter.update_displaced_nodes()
            if self.model.plotter_options.UI_rigid_deformed == False and self.model.plotter_options.UI_deformed == False:
                self.model.plotter.reset_supports()
                self.model.plotter.reset_elastic_supports()
        self.diagrams_and_deformed("D")

    def mostrar_deformada_rigida(self, state):
        """Muestra la deformada rígida"""
        if state == 2: # estado activo
            scale = self.options_values.get("UI_deformation_scale", {}).get(self.model.current_load_pattern, 40)
            self.model.plotter_options.UI_rigid_deformed = True
            self.model.plotter.update_rigid_deformed()
            self.model.plotter.update_displaced_nodes()
            self.model.plotter.update_supports(scale=scale)
            self.model.plotter.update_elastic_supports(scale=scale)
        elif state == 0: # estado desactivado
            self.model.plotter_options.UI_rigid_deformed = False
            self.model.plotter.update_rigid_deformed()
            self.model.plotter.update_displaced_nodes()
            if self.model.plotter_options.UI_rigid_deformed == False and self.model.plotter_options.UI_deformed == False:
                self.model.plotter.reset_supports()
                self.model.plotter.reset_elastic_supports()
        self.diagrams_and_deformed("D")

    def mostrar_soportes(self, state):
        """Muestra los soportes"""
        if state == 2: # estado activo
            self.model.plotter_options.UI_support = True
            self.model.plotter.update_supports()
            self.model.plotter.update_elastic_supports()
        elif state == 0: # estado desactivado
            self.model.plotter_options.UI_support = False
            self.model.plotter.update_supports()
            self.model.plotter.update_elastic_supports()

    def on_field_selected(self, value):
        """Cambia el campo de esfuerzos activo."""
        from milcapy.utils.types import FieldType as _FieldType, to_enum as _to_enum
        try:
            field = _to_enum(value, _FieldType)
        except ValueError:
            return
        self.model.plotter_options.stress_field = field
        self.options_values["UI_stress_field"] = field.value
        if self.model.plotter_options.UI_stress:
            self.model.plotter.update_stress_field(field=field, visibility=True)

    def mostrar_esfuerzos(self, state):
        """Muestra el mapa de esfuerzos del campo activo."""
        if state == 2:
            self.model.plotter_options.UI_stress = True
            self.options_values["UI_stress"] = True
            self.model.plotter.update_stress_field(visibility=True)
        elif state == 0:
            self.model.plotter_options.UI_stress = False
            self.options_values["UI_stress"] = False
            self.model.plotter.update_stress_field(visibility=False)

    def diagrams_and_deformed(self, type):
        """Actualiza la apariencia de los miembros en función de la deformada o deformada rígida"""
        DeformOrDiagram = (self.model.plotter_options.UI_rigid_deformed == True or self.model.plotter_options.UI_deformed == True) if type == "D" else (self.model.plotter_options.UI_axial or self.model.plotter_options.UI_shear or self.model.plotter_options.UI_moment)
        if DeformOrDiagram:
            self.model.plotter_options.UI_load = False
            if type == "D": # mientras se muestra la deformada
                if self.model.plotter_options.show_undeformed: # si se muestra la NO DEFORMADA
                    self.model.plotter.update_members(color=self.model.plotter_options.undeformed_color) # COLOR DE LA NO DEFORMADA CUANDO LA DEFORMADA SE MUESTRA
                    self.model.plotter.update_trusses(color=self.model.plotter_options.undeformed_color)
                    self.model.plotter.update_cst(color_edge=self.model.plotter_options.cst_undeformed_color_edge, color_face=self.model.plotter_options.cst_undeformed_color_face)
                    self.model.plotter.update_membrane_q3dof(color_edge=self.model.plotter_options.membrane_q3dof_undeformed_color_edge, color_face=self.model.plotter_options.membrane_q3dof_undeformed_color_face)
                    self.model.plotter.update_membrane_q2dof(color_edge=self.model.plotter_options.membrane_q2dof_undeformed_color_edge, color_face=self.model.plotter_options.membrane_q2dof_undeformed_color_face)
                    self.model.plotter.update_end_length_offset(color=self.model.plotter_options.undeformed_color)
                    self.model.plotter.update_frame_release(color=self.model.plotter_options.undeformed_color)
                else: # si no se muestra la NO DEFORMADA
                    self.model.plotter_options.UI_show_members = False
                    self.options_values["UI_show_members"] = False
                    self.model.plotter.update_members()
                    self.model.plotter.update_trusses()
                    self.model.plotter.update_cst(color_edge=self.model.plotter_options.cst_edge_color, color_face=self.model.plotter_options.cst_face_color)
                    self.model.plotter.update_membrane_q3dof(color_edge=self.model.plotter_options.membrane_q3dof_edge_color, color_face=self.model.plotter_options.membrane_q3dof_face_color)
                    self.model.plotter.update_membrane_q2dof(color_edge=self.model.plotter_options.membrane_q2dof_edge_color, color_face=self.model.plotter_options.membrane_q2dof_face_color)
                    self.model.plotter.update_end_length_offset()
                    self.model.plotter.update_frame_release()
                    self.model.plotter_options.UI_show_members = True
                    self.options_values["UI_show_members"] = True

        else:
            self.model.plotter_options.UI_load = True
            if type == "D":
                self.model.plotter.update_members(color=self.model.plotter_options.element_color)
                self.model.plotter.update_trusses(color=self.model.plotter_options.truss_color)
                self.model.plotter.update_cst(color_edge=self.model.plotter_options.cst_edge_color, color_face=self.model.plotter_options.cst_face_color)
                self.model.plotter.update_membrane_q3dof(color_edge=self.model.plotter_options.membrane_q3dof_edge_color, color_face=self.model.plotter_options.membrane_q3dof_face_color)
                self.model.plotter.update_membrane_q2dof(color_edge=self.model.plotter_options.membrane_q2dof_edge_color, color_face=self.model.plotter_options.membrane_q2dof_face_color)
                self.model.plotter.update_end_length_offset(color=self.model.plotter_options.end_length_offset_color)
                self.model.plotter.update_frame_release(color=self.model.plotter_options.frame_release_color)
                self.options_values["UI_show_members"] = True

        if self.options_values["UI_load"] == True:
            self.model.plotter.update_distributed_loads()
            self.model.plotter.update_point_load()
            self.model.plotter.update_distributed_load_labels()
            self.model.plotter.update_point_load_labels()
            self.model.plotter.update_prescribed_dofs()
            self.model.plotter.update_prescribed_dofs_labels()



        #! AJUSTES FORZADOS:
        self.model.plotter.update_cst_labels()
        self.model.plotter.update_membrane_q3dof_labels()
        self.model.plotter.update_membrane_q2dof_labels()
        self.model.plotter.update_trusses_labels()



def main_window(model: 'SystemMilcaModel'):
    """
    Inicializa y ejecuta la ventana principal del Visualizador del objeto "SystemMilcaModel".
    """
    app = QApplication.instance()
    if app is None:  # Si no existe una instancia, se crea
        app = QApplication(sys.argv)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())

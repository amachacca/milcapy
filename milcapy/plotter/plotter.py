from collections.abc import Sequence
import itertools
from typing import TYPE_CHECKING, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from milcapy.utils import InternalForceType

from milcapy.utils import rotate_xy, traslate_xy
from milcapy.plotter.suports import (
    support_ttt, support_ttf, support_tft,
    support_ftt, support_tff, support_ftf, support_fft,
    support_kt, support_kr
)
from milcapy.plotter.load import (
    graphic_one_arrow, graphic_one_arrow_dof, moment_fancy_arrow, graphic_n_arrow, redondear_si_mas_de_3_decimales
)
from milcapy.plotter.plotter_values import PlotterValues
from milcapy.plotter.options import PlotterOptions
from milcapy.plotter.stress_layer import StressLayer
from milcapy.plotter.widgets import DiagramConfig
from milcapy.plotter.utils import separate_areas, process_segments

from matplotlib.patches import Polygon
from matplotlib.text import Text
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class Plotter:
    def __init__(
        self,
        model: 'SystemMilcaModel',
    ) -> None:
        self.model = model
        self.plotter_values: Dict[str, 'PlotterValues'] = {}
        self.figure: Figure = None
        self.axes: Axes = None
        self.current_values: 'PlotterValues' = PlotterValues(self.model,
                                                             list(self.model.results.keys())[0])
        self.initialize_figure()  # Inicializar figura

        # todos los que tengan 🐍 se calculan para cada load pattern
        # todos los que tengan ✅ ya estan implementados
        # ! SOLO SE PLOTEAN LOS LOAD_PATTERN QUE ESTAN ANALIZADOS
        # ? (SOLO LOS QUE TIENEN RESULTADOS EN MODEL.RESULTS)
        # nodos
        self.nodes = {}              # ✅visibilidad, color
        # nodos deformados
        self.deformed_nodes = {}    # 🐍 interactividad al pasar el ratón

        # miembros
        self.members = {}           # ✅visibilidad, color
        # forma deformada
        """{
            load_pattern:{
                member_id: [line2D]
                cst_id: [Line2D, Polygon2D]
            }
        }"""
        self.deformed_shape = {
        }        # {load_pattern: {ele_id: [line2D, Line2D, Polygon2D]}}    ✅🐍 visibilidad, setdata, setType: colorbar
        # forma regida de la deformada
        # {load_pattern: {ele_id: [line2D, Line2D, Polygon2D]}}    ✅🐍 visibilidad, setdata, setType: colorbar
        self.rigid_deformed_shape = {}
        # cargas puntuales
        self.point_loads = {}       # ✅🐍 visibilidad
        # cargas distribuidas
        self.distributed_loads = {}  # ✅🐍 visibilidad
        # fuerzas internas (line2D: borde)
        self.internal_forces = {}   # 🐍 visibilidad, setdata, setType: colorbar
        # fillings for internal forces
        self.fillings = {}          # 🐍 visibilidad, setdata, setType: colorbar
        # apooyos
        self.supports = {}          # ✅visibilidad, color, setdata
        # apoyos dezplados
        self.displaced_supports = {}  # 🐍 visibilidad, color, setdata
        # etiquetas
        self.node_labels = {}            # ✅visibilidad, setdata
        self.member_labels = {}          # ✅visibilidad, setdata
        self.trusses_labels = {}         # ✅visibilidad, setdata
        self.point_load_labels = {}      # ✅🐍 visibilidad, setdata
        self.distributed_load_labels = {}  # ✅🐍 visibilidad, setdata
        self.internal_forces_labels = {}  # 🐍 visibilidad, setdata
        self.reactions_labels = {}       # 🐍 visibilidad, setdata
        self.displacement_labels = {}    # 🐍 visibilidad, setdata
        # reacciones
        self.reactions = {}         # 🐍 visibilidad, setdata

        # fuerzas internas (line2D: borde, polygon: relleno)
        self.axial_force = {}
        self.shear_force = {}
        self.bending_moment = {}

        self.selected_member = 1
        self.model.current_load_pattern = list(self.model.results.keys())[0]

        # end length offset {id_element: [line1, line2]}
        self.end_length_offset = {}  # ✅🐍 visibilidad, color

        # prescipcion en los DOFs
        self.prescribed_dofs = {}  # ✅🐍 visibilidad, color
        self.prescribed_dofs_labels = {}  # ✅🐍 visibilidad, setdata

        # elastic supports
        self.elastic_supports = {}  # ✅🐍 visibilidad, color

        # Support data cache
        # {node_id: [x, y, node_coords, theta]}
        self.static_support_data_cache = {}
        # {node_id: [{"ki": (x, y, theta, Line2D)}, node_coords, TextPosition]}
        self.elastic_support_data_cache = {}

        # frame release data cache
        self.frame_release_data_cache = {}  # {member_id: [plt.Scatter, ...]}

        #! CST ELEMENT:
        self.csts = {}  # {cst_id: [Line2D, Polygon2D, Text]}

        #! MEMBRANE Q6 ELEMENT:
        # {membrane_q3dof_id: [Line2D, Polygon2D, Text]}
        self.membrane_q3dof = {}

        #! MEMBRANE Q6I ELEMENT:
        # {membrane_q2dof_id: [Line2D, Polygon2D, Text]}
        self.membrane_q2dof = {}

        #! TRUSS ELEMENT:
        self.trusses = {}  # {truss_id: [Line2D]}

        #! CAPA DE ESFUERZOS (mapa de colores, solo patrón actual)
        self.stress_layer = StressLayer(self)

    @property
    def plotter_options(self) -> 'PlotterOptions':
        return self.model.plotter_options

    @plotter_options.setter
    def plotter_options(self, value: 'PlotterOptions') -> None:
        self.model.plotter_options = value

    @property
    def current_load_pattern(self) -> Optional[str]:
        return self.model.current_load_pattern

    @current_load_pattern.setter
    def current_load_pattern(self, value: Optional[str]):
        self.model.current_load_pattern = value

    @property
    def diagrams(self) -> Dict[str, DiagramConfig]:
        tol = self.plotter_options.tol
        DECIMALS = self.plotter_options.decimals

        if self.selected_member in self.trusses:
            return {
                'N(x)': DiagramConfig(
                    name='Diagrama de Fuerza Normal',
                    values=self.model.results[self.current_load_pattern].trusses[self.selected_member]["axial_forces"],
                    precision=4,
                ),
                'y(x)': DiagramConfig(
                    name='Diagrama de desplazamiento axial',
                    values=self.model.results[self.current_load_pattern].trusses[self.selected_member]["axial_displacements"],
                    precision=6,
                )
            }
        else:
            axial_forces = self.model.results[self.current_load_pattern].members[self.selected_member]["axial_forces"]
            shear_forces = self.model.results[self.current_load_pattern].members[self.selected_member]["shear_forces"]
            bending_moments = self.model.results[self.current_load_pattern].members[self.selected_member]["bending_moments"]
            slopes = self.model.results[self.current_load_pattern].members[self.selected_member]["slopes"]
            deflections = self.model.results[self.current_load_pattern].members[self.selected_member]["deflections"]
            if np.all(np.abs(np.round(axial_forces, DECIMALS)) < tol):
                axial_forces = np.zeros(axial_forces.shape, dtype=int)
            if np.all(np.abs(np.round(shear_forces, DECIMALS)) < tol):
                shear_forces = np.zeros(shear_forces.shape, dtype=int)
            if np.all(np.abs(np.round(bending_moments, DECIMALS)) < tol):
                bending_moments = np.zeros(bending_moments.shape, dtype=int)
            if np.all(np.abs(np.round(slopes, DECIMALS)) < tol):
                slopes = np.zeros(slopes.shape, dtype=int)
            if np.all(np.abs(np.round(deflections, DECIMALS)) < tol):
                deflections = np.zeros(deflections.shape, dtype=int)
            return {
                'N(x)': DiagramConfig(
                    name='Diagrama de Fuerza Normal',
                    values=axial_forces,
                    precision=4,
                ),
                'V(x)': DiagramConfig(
                    name='Diagrama de Fuerza Cortante',
                    values=shear_forces,
                    precision=4,
                ),
                'M(x)': DiagramConfig(
                    name='Diagrama de Momento Flector',
                    values=bending_moments,
                    precision=4,
                ),
                'θ(x)': DiagramConfig(
                    name='Diagrama de Rotación',
                    values=slopes,
                    precision=6,
                ),
                'y(x)': DiagramConfig(
                    name='Diagrama de Deflexión',
                    values=deflections,
                    precision=6,
                )
            }

    def initialize_plot(self):
        """Plotea por primera y unica vez (crea los objetos artist)"""
        if self.plotter_options.optCargaOP == "max":
            self.plotter_options.load_max(list(self.model.results.keys())[0])
        elif self.plotter_options.optCargaOP == "mean":
            self.plotter_options.load_mean(list(self.model.results.keys())[0])
        elif self.plotter_options.optCargaOP == "prom":
            self.plotter_options.load_max(
                list(self.model.results.keys())[0], prom=True)
        self.plot_nodes()
        self.plot_members()
        self.plot_trusses()
        self.plot_cst()
        self.plot_membrane_q3dof()
        self.plot_membrane_q2dof()
        self.plot_supports()
        self.plot_elastic_supports()
        self.plot_node_labels()
        self.plot_member_labels()
        self.plot_trusses_labels()
        self.plot_cst_labels()
        self.plot_membrane_q3dof_labels()
        self.plot_membrane_q2dof_labels()
        self.plot_end_length_offset()
        self.plot_frame_release()
        for load_pattern_name in self.model.results.keys():
            if self.plotter_options.optCargaOP == "max":
                self.plotter_options.load_max(load_pattern_name)
            elif self.plotter_options.optCargaOP == "mean":
                self.plotter_options.load_mean(load_pattern_name)
            elif self.plotter_options.optCargaOP == "prom":
                self.plotter_options.load_max(load_pattern_name, prom=True)
            self.current_load_pattern = load_pattern_name
            self.current_values = self.get_plotter_values(load_pattern_name)
            self.plot_point_loads()
            self.plot_distributed_loads()
            self.plot_rigid_deformed()
            self.plot_deformed()
            self.plot_axial_force()
            self.plot_shear_force()
            self.plot_bending_moment()
            self.plot_reactions()
            self.plot_displaced_nodes()
            self.plot_prescribed_dofs()

        # actualizar pattern actual al primero
        self.current_load_pattern = list(self.model.results.keys())[0]
        self.update_change()
        if self.plotter_options.UI_stress:
            self.update_stress_field(visibility=True)

    def plot_stress_field(self, field=None, cmap: Optional[str] = None) -> bool:
        """Dibuja el mapa de esfuerzos del patrón actual (delega a StressLayer)."""
        self.current_values = self.get_plotter_values(self.current_load_pattern)
        return self.stress_layer.show(field=field, cmap=cmap)

    def update_stress_field(self, field=None, visibility: Optional[bool] = None, cmap: Optional[str] = None) -> bool:
        """Muestra/oculta el mapa de esfuerzos según opciones y visibilidad."""
        if visibility is None:
            visibility = self.plotter_options.UI_stress
        self.plotter_options.UI_stress = bool(visibility)
        if not visibility:
            self.stress_layer.hide()
            return False
        self.current_values = self.get_plotter_values(self.current_load_pattern)
        return self.stress_layer.show(field=field, cmap=cmap)

    def update_change(self):
        """Oculta atists para todos los load patterns excepto el actual"""
        pt_cache = self.current_load_pattern
        self.stress_layer.hide(draw=False)
        for load_pattern_name in self.model.results.keys():
            self.current_load_pattern = load_pattern_name
            if load_pattern_name != pt_cache:
                self.update_point_load(visibility=False)
                self.update_point_load_labels(visibility=False)
                self.update_distributed_loads(visibility=False)
                self.update_distributed_load_labels(visibility=False)
                self.update_rigid_deformed(visibility=False)
                self.update_deformed(visibility=False)
                self.update_axial_force(visibility=False)
                self.update_shear_force(visibility=False)
                self.update_bending_moment(visibility=False)
                self.update_reactions(visibility=False)
                self.update_displaced_nodes(visibility=False)
                self.update_prescribed_dofs(visibility=False)
                self.update_prescribed_dofs_labels(visibility=False)
                # self.update_supports(visibility=True)
                # self.update_elastic_supports(visibility=True)
            elif load_pattern_name == pt_cache:
                if self.plotter_options.UI_load:
                    self.update_point_load(visibility=True)
                    self.update_point_load_labels(visibility=True)
                    self.update_distributed_loads(visibility=True)
                    self.update_distributed_load_labels(visibility=True)
                    self.update_prescribed_dofs(visibility=True)
                    self.update_prescribed_dofs_labels(visibility=True)
                if self.plotter_options.UI_rigid_deformed:
                    self.update_rigid_deformed(visibility=True)
                if self.plotter_options.UI_deformed:
                    self.update_deformed(visibility=True)
                if self.plotter_options.UI_deformed or self.plotter_options.UI_rigid_deformed:
                    self.update_displaced_nodes(visibility=True)
                if self.plotter_options.UI_axial:
                    self.update_axial_force(visibility=True)
                if self.plotter_options.UI_shear:
                    self.update_shear_force(visibility=True)
                if self.plotter_options.UI_moment:
                    self.update_bending_moment(visibility=True)
                if self.plotter_options.UI_reactions:
                    self.update_reactions(visibility=True)
                if self.plotter_options.UI_deformed or self.plotter_options.UI_rigid_deformed:
                    self.update_displaced_nodes(visibility=True)
                if self.plotter_options.UI_stress and load_pattern_name == pt_cache:
                    self.current_values = self.get_plotter_values(load_pattern_name)
                    self.stress_layer.show()
                # if self.plotter_options.UI_support:
                #     self.update_supports(visibility=False)
                #     self.update_elastic_supports(visibility=False)
        self.figure.canvas.draw_idle()
        self.current_load_pattern = pt_cache

    def get_plotter_values(self, load_pattern_name: str) -> 'PlotterValues':
        # Comprobar si ya existe en caché
        if load_pattern_name in self.plotter_values:
            return self.plotter_values[load_pattern_name]

        # Verificar que el load pattern existe
        if load_pattern_name not in self.model.load_patterns:
            raise ValueError(
                f"El load pattern '{load_pattern_name}' no se encontró")

        # Verificar que existen resultados para este load pattern
        if load_pattern_name not in self.model.results:
            raise ValueError(
                f"Los resultados para el load pattern '{load_pattern_name}' no se encontraron")

        # Crear nueva instancia de PlotterValues
        plotter_values = PlotterValues(self.model, load_pattern_name)

        # Guardar en caché
        self.plotter_values[load_pattern_name] = plotter_values

        # Actualizar valores
        self.current_values = plotter_values

        return plotter_values

    def initialize_figure(self):
        # Cerrar figuras previas
        plt.close("all")

        # Configurar estilo global
        if self.plotter_options.plot_style in plt.style.available:
            plt.style.use(self.plotter_options.plot_style)

        # Crear figura y ejes
        self.figure = plt.figure(figsize=self.plotter_options.figure_size,
                                 dpi=self.plotter_options.dpi, facecolor=self.plotter_options.UI_background_color)
        self.axes = self.figure.add_subplot(111)

        # Configurar cuadrícula
        if self.plotter_options.grid:
            self.axes.grid(True, linestyle="--", alpha=0.5)

        # Ajustar layout
        if self.plotter_options.tight_layout:
            self.figure.tight_layout()

        # Mantener proporciones iguales
        plt.axis("equal")

        # Activar los ticks secundarios en ambos ejes
        # 5 subdivisiones entre cada tick principal
        self.axes.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.axes.yaxis.set_minor_locator(AutoMinorLocator(5))

        # Activar ticks en los 4 lados (mayores y menores)
        self.axes.tick_params(
            which="both", direction="in", length=6, width=1,
            top=True, bottom=True, left=True, right=True
        )
        # Ticks menores más pequeños y rojos
        self.axes.tick_params(which="minor", length=2,
                              width=0.5, color="black")

        # Mostrar etiquetas en los 4 lados
        self.axes.tick_params(labeltop=True, labelbottom=True,
                              labelleft=True, labelright=True)

        # Asegurar que los ticks se muestran en ambos lados
        self.axes.xaxis.set_ticks_position("both")
        self.axes.yaxis.set_ticks_position("both")

        # Personalizar el color de los ejes
        for spine in ["top", "bottom", "left", "right"]:
            self.axes.spines[spine].set_color("#9bc1bc")  # Color personalizado
            self.axes.spines[spine].set_linewidth(0.5)  # Grosor del borde

        # Personalizar las etiquetas de los ejes
        plt.xticks(fontsize=8, fontfamily="serif",
                   fontstyle="italic", color="#103b58")
        plt.yticks(fontsize=8, fontfamily="serif",
                   fontstyle="italic", color="#103b58")

        # Personalizar los ticks del eje X e Y
        self.axes.tick_params(axis="x", direction="in",
                              length=3.5, width=0.7, color="#21273a")
        self.axes.tick_params(axis="y", direction="in",
                              length=3.5, width=0.7, color="#21273a")

        # Cambiar el color de fondo del área de los ejes
        # self.axes.set_facecolor("#222222")  # Fondo oscuro dentro del Axes

        # Cambiar color del fondo exterior (Canvas)
        self.figure.patch.set_facecolor("#f5f5f5")  # Color gris oscuro

    def change_background_color(self):
        # self.figure.patch.set_facecolor(self.plotter_options.UI_background_color)
        self.axes.set_facecolor(self.plotter_options.UI_background_color)
        self.figure.canvas.draw()

    def plot_nodes(self):
        """
        Dibuja los nodos de la estructura.
        """
        # if not self.plotter_options.UI_show_nodes:
        #     return
        for node_id, coord in self.current_values.nodes.items():
            x = [coord[0]]
            y = [coord[1]]

            node = self.axes.scatter(
                x, y, c=self.plotter_options.node_color, s=self.plotter_options.node_size, marker='o')
            self.nodes[node_id] = node
            node.set_visible(self.plotter_options.UI_show_nodes)
        self.figure.canvas.draw_idle()

    def update_nodes(self):
        for node in self.nodes.values():
            node.set_visible(self.plotter_options.UI_show_nodes)
        self.figure.canvas.draw_idle()

    def plot_members(self):
        """
        Dibuja los elementos de la estructura.
        """
        # if not self.plotter_options.UI_show_members:
        #     return
        for id, coord in self.current_values.members.items():
            x_coords = [coord[0][0], coord[1][0]]
            y_coords = [coord[0][1], coord[1][1]]
            line, = self.axes.plot(x_coords, y_coords, color=self.plotter_options.element_color,
                                   linewidth=self.plotter_options.element_line_width)
            self.members[id] = line
            line.set_visible(self.plotter_options.UI_show_members)
        self.figure.canvas.draw_idle()

    def update_members(self, color=None):
        for member in self.members.values():
            member.set_visible(self.plotter_options.UI_show_members)
        if color:
            for member in self.members.values():
                member.set_color(color)
        self.figure.canvas.draw_idle()

    def plot_end_length_offset(self):
        """Plotea los brazos en los elemntos si es que hubiere"""

        for ele_id, coord in self.current_values.members.items():
            brazos = []
            la = self.model.members[ele_id].la or 0
            lb = self.model.members[ele_id].lb or 0
            length = self.model.members[ele_id].length()
            length = length - la - lb
            ((xi, yi), (xj, yj)) = self.current_values.members[ele_id]
            angle_rotation = self.model.members[ele_id].angle_x()
            coords = ((xi + la*np.cos(angle_rotation), yi + la*np.sin(angle_rotation)),
                      (xj - lb*np.cos(angle_rotation), yj - lb*np.sin(angle_rotation)))
            ((xa, ya), (xb, yb)) = coords
            coords_a = [[xi, xa], [yi, ya]]
            coords_b = [[xb, xj], [yb, yj]]

            if self.model.members[ele_id].la:
                line, = self.axes.plot(coords_a[0], coords_a[1], color=self.plotter_options.end_length_offset_color,
                                       linewidth=self.plotter_options.end_length_offset_line_width)
                brazos.append(line)
                line.set_visible(self.plotter_options.UI_show_members)
            if self.model.members[ele_id].lb:
                line, = self.axes.plot(coords_b[0], coords_b[1], color=self.plotter_options.end_length_offset_color,
                                       linewidth=self.plotter_options.end_length_offset_line_width)
                brazos.append(line)
                line.set_visible(self.plotter_options.UI_show_members)
            self.end_length_offset[ele_id] = brazos
        self.figure.canvas.draw_idle()

    def update_end_length_offset(self, color=None):
        for brazos in self.end_length_offset.values():
            for brazo in brazos:
                brazo.set_visible(self.plotter_options.UI_show_members)
        if color:
            for brazos in self.end_length_offset.values():
                for brazo in brazos:
                    brazo.set_color(color)
        self.figure.canvas.draw_idle()

    def plot_supports(self):
        """
        Dibuja los apoyos de la estructura.
        """
        support_functions = {
            (True, True, True): support_ttt,
            (False, False, True): support_fft,
            (False, True, False): support_ftf,
            (True, False, False): support_tff,
            (False, True, True): support_ftt,
            (True, False, True): support_tft,
            (True, True, False): support_ttf,
            (False, False, False): None
        }

        for id, restrains in self.current_values.restraints.items():
            node_coords = self.current_values.nodes[id]
            support_func = support_functions.get(restrains)

            if self.model.nodes[id].local_axis:
                theta = self.model.nodes[id].local_axis.angle
            else:
                theta = self.plotter_options.mod_rotation_angle_conventional_supports.get(id, 0)*3.141592653589793/180.
            if support_func:
                line = support_func(
                    ax=self.axes,
                    x=node_coords[0],
                    y=node_coords[1],
                    size=self.plotter_options.support_size,
                    color=self.plotter_options.support_color,
                    zorder=3,
                    theta=theta
                )
                self.supports[id] = line
                x_data, y_data = line.get_data()
                self.static_support_data_cache[id] = [
                    x_data, y_data, node_coords, None]
        self.figure.canvas.draw_idle()

    def update_supports(self, visibility: bool | None = None, scale: float | None = None):
        """
        Actualiza la visibilidad y escala de los apoyos.
        """
        visibility = self.plotter_options.UI_support if visibility is None else visibility
        for node_id, support in self.supports.items():
            support.set_visible(visibility)
            if scale is not None:
                dx = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[
                    0]*scale
                dy = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[
                    1]*scale
                x_new = np.array(
                    self.static_support_data_cache[node_id][0]) + dx
                y_new = np.array(
                    self.static_support_data_cache[node_id][1]) + dy
                support.set_xdata(x_new)
                support.set_ydata(y_new)
        self.figure.canvas.draw_idle()

    def reset_supports(self):
        """
        Restablece la visibilidad y la posición de los apoyos.
        """
        for node_id, support in self.supports.items():
            support.set_visible(self.plotter_options.UI_support)
            x_data, y_data, *otros = self.static_support_data_cache[node_id]
            support.set_data(x_data, y_data)
        self.figure.canvas.draw_idle()

    def plot_node_labels(self):
        """
        Dibuja los etiquetas de los nodos.
        """
        for id, coord in self.current_values.nodes.items():
            # bbox = {
            #     "boxstyle": "circle",
            #     "facecolor": "lightblue",     # Color de fondo
            #     "edgecolor": "black",         # Color del borde
            #     "linewidth": 0.5,               # Grosor del borde
            #     "linestyle": "-",            # Estilo del borde
            #     "alpha": 0.8                  # Transparencia
            # }
            text = self.axes.text(coord[0], coord[1], str(id),
                                  fontsize=self.plotter_options.label_font_size,
                                  ha='left', va='bottom', color="blue",  # bbox=bbox,
                                  clip_on=True)
            self.node_labels[id] = text
            text.set_visible(self.plotter_options.UI_node_labels)
            self.figure.canvas.draw_idle()

    def update_node_labels(self):
        """
        Actualiza la visibilidad de las etiquetas de los nodos.
        """
        for text in self.node_labels.values():
            text.set_visible(self.plotter_options.UI_node_labels)
        self.figure.canvas.draw_idle()

    def plot_member_labels(self):
        """
        Dibuja las etiquetas de los elementos.
        """
        for element_id, coords in self.current_values.members.items():
            x_val = (coords[0][0] + coords[1][0]) / 2
            y_val = (coords[0][1] + coords[1][1]) / 2

            # bbox = {
            #     "boxstyle": "round,pad=0.2",  # Estilo y padding del cuadro
            #     "facecolor": "lightblue",     # Color de fondo
            #     "edgecolor": "black",         # Color del borde
            #     "linewidth": 0.5,               # Grosor del borde
            #     "linestyle": "-",            # Estilo del borde
            #     "alpha": 0.8                  # Transparencia
            # }
            text = self.axes.text(x_val, y_val, str(element_id),
                                  fontsize=self.plotter_options.label_font_size,
                                  ha='left', va='bottom', color="blue",  # bbox=bbox,
                                  clip_on=True)
            self.member_labels[element_id] = text
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def update_member_labels(self):
        """
        Actualiza la visibilidad de las etiquetas de los elementos.
        """
        for text in self.member_labels.values():
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def plot_point_loads(self) -> None:
        """
        Grafica las cargas puntuales.
        """
        # if not self.plotter_options.UI_point_load:
        #     return
        self.point_loads[self.current_load_pattern] = {}
        self.point_load_labels[self.current_load_pattern] = {}
        for id_node, load in self.current_values.point_loads.items():
            coords = self.current_values.nodes[id_node]

            arrows = []
            texts = []

            # Fuerza en dirección X
            if load["fx"] != 0:
                arrow, text = graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fx"],
                    length_arrow=self.plotter_options.point_load_length_arrow,
                    angle=0 if load["fx"] < 0 else np.pi,
                    ax=self.axes,
                    color=self.plotter_options.point_load_color,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.point_load_label_color,
                    label_font_size=self.plotter_options.point_load_label_font_size
                )
                arrows.append(arrow)
                texts.append(text)

            # Fuerza en dirección Y
            if load["fy"] != 0:
                arrow, text = graphic_one_arrow(
                    x=coords[0],
                    y=coords[1],
                    load=load["fy"],
                    length_arrow=self.plotter_options.point_load_length_arrow,
                    angle=np.pi/2 if load["fy"] < 0 else 3*np.pi/2,
                    ax=self.axes,
                    color=self.plotter_options.point_load_color,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.point_load_label_color,
                    label_font_size=self.plotter_options.point_load_label_font_size
                )
                arrows.append(arrow)
                texts.append(text)

            # Momento en Z
            if load["mz"] != 0:
                arrow, text = moment_fancy_arrow(
                    ax=self.axes,
                    x=coords[0],
                    y=coords[1],
                    moment=load["mz"],
                    radio=0.70 * self.plotter_options.point_moment_length_arrow,
                    color=self.plotter_options.point_load_color,
                    clockwise=True,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.point_load_label_color,
                    label_font_size=self.plotter_options.point_load_label_font_size
                )
                arrows.append(arrow)
                texts.append(text)

            self.point_loads[self.current_load_pattern][id_node] = arrows
            self.point_load_labels[self.current_load_pattern][id_node] = texts
        # for arrow, text in zip(arrows, texts):
        #     arrow.set_visible(self.plotter_options.point_load)
        #     text.set_visible(self.plotter_options.point_load_label)
        self.figure.canvas.draw_idle()

    def update_point_load(self, visibility: bool | None = None):
        """
        Actualiza la visibilidad de las cargas puntuales.
        """
        visibility = self.plotter_options.UI_load if visibility is None else visibility
        for arrows in self.point_loads[self.current_load_pattern].values():
            for arrow in arrows:
                arrow.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_point_load_labels(self, visibility: bool | None = None):
        """
        Actualiza la visibilidad de las etiquetas de las cargas puntuales.
        """
        visibility = self.plotter_options.UI_load if visibility is None else visibility
        for texts in self.point_load_labels[self.current_load_pattern].values():
            for text in texts:
                text.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def plot_distributed_loads(self) -> None:
        """
        Grafica las cargas distribuidas.
        """
        self.distributed_loads[self.current_load_pattern] = {}
        self.distributed_load_labels[self.current_load_pattern] = {}
        for id_element, load in self.current_values.distributed_loads.items():

            arrowslist = []
            textslist = []

            # Calcular longitud y ángulo de rotación del elemento
            if id_element in self.model.trusses:
                element = self.model.trusses[id_element]
                element.qla, element.qlb = None, None
            else:
                element = self.model.members[id_element]
            length = element.length()
            angle_rotation = element.angle_x()

            if not (element.qla and element.qlb):  # Si no hay cargas en el brazo rigido
                la = element.la or 0
                lb = element.lb or 0
                length = length - la - lb
                if id_element in self.model.trusses:
                    ((xi, yi, xj, yj),
                     disp) = self.current_values.trusses[id_element]
                else:
                    ((xi, yi), (xj, yj)
                     ) = self.current_values.members[id_element]
                coords = ((xi + la*np.cos(angle_rotation), yi + la*np.sin(angle_rotation)),
                          (xj - lb*np.cos(angle_rotation), yj - lb*np.sin(angle_rotation)))
            else:
                if id_element in self.model.trusses:
                    ((xi, yi, xj, yj),
                     disp) = self.current_values.trusses[id_element]
                else:
                    coords = self.current_values.members[id_element]

            # Cargas transversales
            if round(load["q_i"], 2) != 0 or round(load["q_j"], 2) != 0:
                arrows, texts = graphic_n_arrow(
                    x=coords[0][0],
                    y=coords[0][1],
                    load_i=-redondear_si_mas_de_3_decimales(load["q_i"]),
                    load_j=-redondear_si_mas_de_3_decimales(load["q_j"]),
                    angle=np.pi/2,
                    length=length,
                    ax=self.axes,
                    ratio_scale=self.plotter_options.scale_dist_qload[self.current_load_pattern],
                    nrof_arrows=self.plotter_options.nro_arrows(id_element),
                    color=self.plotter_options.distributed_load_color,
                    angle_rotation=angle_rotation,
                    label=self.plotter_options.distributed_load_label,
                    color_label=self.plotter_options.distributed_load_label_color,
                    label_font_size=self.plotter_options.distributed_load_label_font_size
                )
                arrowslist = arrowslist + arrows
                textslist = textslist + texts

            # Cargas axiales
            if round(load["p_i"], 2) != 0 or round(load["p_j"], 2) != 0:
                arrows, texts = graphic_n_arrow(
                    x=coords[0][0],
                    y=coords[0][1],
                    load_i=-redondear_si_mas_de_3_decimales(load["p_i"]),
                    load_j=-redondear_si_mas_de_3_decimales(load["p_j"]),
                    angle=0,
                    length=length,
                    ax=self.axes,
                    ratio_scale=self.plotter_options.scale_dist_pload[self.current_load_pattern],
                    nrof_arrows=self.plotter_options.nro_arrows(id_element),
                    color=self.plotter_options.distributed_load_color,
                    angle_rotation=angle_rotation,
                    label=self.plotter_options.distributed_load_label,
                    color_label=self.plotter_options.distributed_load_label_color,
                    label_font_size=self.plotter_options.distributed_load_label_font_size,
                    length_arrow=self.plotter_options.scale_dist_pload[self.current_load_pattern]
                )
                arrowslist = arrowslist + arrows
                textslist = textslist + texts
            # Momentos distribuidos (no implementados)
            if load["m_i"] != 0 or load["m_j"] != 0:
                raise NotImplementedError(
                    "Momentos distribuidos no implementados.")

            self.distributed_loads[self.current_load_pattern][id_element] = arrowslist
            self.distributed_load_labels[self.current_load_pattern][id_element] = textslist
        self.figure.canvas.draw_idle()

    def update_distributed_loads(self, visibility: Optional[bool] = None):
        """
        Actualiza la visibilidad de las cargas distribuidas.
        """
        visibility = bool(
            self.plotter_options.UI_load) if visibility is None else visibility
        for arrows in self.distributed_loads[self.current_load_pattern].values():
            for arrow in arrows:
                arrow.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_distributed_load_labels(self, visibility: Optional[bool] = None):
        """
        Actualiza la visibilidad de las etiquetas de las cargas distribuidas.
        """
        visibility = self.plotter_options.UI_load if visibility is None else visibility
        for texts in self.distributed_load_labels[self.current_load_pattern].values():
            for text in texts:
                text.set_visible(visibility)
        self.figure.canvas.draw_idle()

    # def plot_deformed(self, escala: float | None = None) -> None:
    #     """
    #     Grafica la forma de deformación.
    #     """
    #     self.deformed_shape[self.current_load_pattern] = {}
    #     escala = self.plotter_options.UI_deformation_scale[
    #         self.current_load_pattern] if escala is None else escala
    #     for element in self.model.members.values():
    #         x, y = self.current_values.get_deformed_shape(element.id, escala)
    #         line, = self.axes.plot(x, y, lw=self.plotter_options.deformation_line_width,
    #                                color=self.plotter_options.deformation_color)
    #         self.deformed_shape[self.current_load_pattern][element.id] = line
    #         line.set_visible(self.plotter_options.UI_deformed)
    #     self.figure.canvas.draw_idle()

    # def update_deformed(self, visibility: Optional[bool] = None, escala: float | None = None):
    #     """
    #     Actualiza la visibilidad de la forma de deformación.
    #     """
    #     visibility = self.plotter_options.UI_deformed if visibility is None else visibility
    #     for line in self.deformed_shape[self.current_load_pattern].values():
    #         line.set_visible(visibility)

    #     # HACER UN SET_DATA(X, Y) A TODOS LOS MIEMBROS DEL ACTUAL LOAD_PATTERN
    #     if escala is not None:
    #         for member in self.model.members.values():
    #             self.current_values = self.get_plotter_values(self.current_load_pattern)
    #             x, y = self.current_values.get_deformed_shape(member.id, escala)
    #             self.deformed_shape[self.current_load_pattern][member.id].set_data(x, y)

    #     self.figure.canvas.draw_idle()

    def plot_internal_forces(self, type: InternalForceType, escala: float | None = None) -> None:
        """
        Grafica los diagramas de fuerzas internas.
        """
        tol = self.plotter_options.tol
        DECIMALS = self.plotter_options.decimals
        # ESCALAS:
        if type == InternalForceType.AXIAL_FORCE:
            escala = self.plotter_options.axial_scale[self.current_load_pattern] if escala is None else escala
            self.axial_force[self.current_load_pattern] = {}
        elif type == InternalForceType.SHEAR_FORCE:
            escala = self.plotter_options.shear_scale[self.current_load_pattern] if escala is None else escala
            self.shear_force[self.current_load_pattern] = {}
        elif type == InternalForceType.BENDING_MOMENT:
            escala = self.plotter_options.moment_scale[self.current_load_pattern] if escala is None else escala
            self.bending_moment[self.current_load_pattern] = {}

        artist = []
        for member_id, member in list(self.model.members.items()) + list(self.model.trusses.items()):

            # Obtener valores del diagrama
            if type == InternalForceType.AXIAL_FORCE:
                if member_id in self.model.trusses:
                    y_val = self.model.results[self.current_load_pattern].get_truss_axial_force(
                        member_id)
                else:
                    y_val = self.model.results[self.current_load_pattern].get_member_axial_force(
                        member_id)
                y_val = np.round(y_val, DECIMALS)

                if np.all(np.abs(y_val) < tol):
                    continue
                else:
                    y_val = y_val * escala

            elif type == InternalForceType.SHEAR_FORCE:
                if member_id in self.model.trusses:
                    y_val = np.zeros(self.model.postprocessing_options.n)
                else:
                    y_val = self.model.results[self.current_load_pattern].get_member_shear_force(
                        member_id)
                y_val = np.round(y_val, DECIMALS)

                if np.all(np.abs(y_val) < tol):
                    continue
                else:
                    y_val = y_val * escala

            elif type == InternalForceType.BENDING_MOMENT:
                if member_id in self.model.trusses:
                    y_val = np.zeros(self.model.postprocessing_options.n)
                else:
                    y_val = self.model.results[self.current_load_pattern].get_member_bending_moment(
                        member_id)
                y_val = np.round(y_val, DECIMALS)

                if np.all(np.abs(y_val) < tol):
                    continue
                else:
                    y_val = y_val * escala

            # x_val = np.linspace(0, member.length(), len(y_val))
            try:
                x_val = self.model.results[self.current_load_pattern].get_member_x_val(
                    member_id)
            except:
                x_val = np.linspace(0, member.length(), len(y_val))

            # Configuración inicial
            L = member.length()
            x = x_val
            y = y_val
            yo = y[0]
            yf = y[-1]
            y3 = y[2]
            yneg3 = y[-3]
            yv = 0
            iv = 0
            if y[np.argmin(y)] < (yf and yo):
                yv = y[np.argmin(y)]
                iv = np.argmin(y)
            elif y[np.argmax(y)] > (yf and yo):
                yv = y[np.argmax(y)]
                iv = np.argmax(y)

            # [[inicial], [maximo], [final], [medio]]
            # if member_id in self.model.trusses:
            #     member.la, member.lb = None, None
            if member.la or member.lb:
                la = member.la or 0
                lb = member.lb or 0
                lab_coord = np.array(
                    [[la, y3], [iv/len(y)*L, yv], [L-lb, yneg3], [L/2, y3]])
                lab_coord = rotate_xy(lab_coord, member.angle_x(), 0, 0)
                lab_coord = traslate_xy(
                    lab_coord, *member.node_i.vertex.coordinates)
            else:
                lab_coord = np.array(
                    [[0, yo], [iv/len(y)*L, yv], [L, yf], [L/2, yo]])
                lab_coord = rotate_xy(lab_coord, member.angle_x(), 0, 0)
                lab_coord = traslate_xy(
                    lab_coord, *member.node_i.vertex.coordinates)

            # Separar y procesar áreas
            positive, negative = separate_areas(y, L, x_val)
            positive_processed = process_segments(positive, L)
            negative_processed = process_segments(negative, L)

            # areglar el borde
            val = np.stack((x, y), axis=-1)
            if val[0][0] == 0 and val[0][1] != 0:
                val = np.insert(val, 0, [[0.0, 0.0]], axis=0)
            if val[-1][0] == L and val[-1][1] != 0:
                val = np.append(val, [[L, 0.0]], axis=0)

            # Transformar coordenadas y PLOTEAR
            for area in positive_processed:
                area = rotate_xy(area, member.angle_x(), 0, 0)
                area = traslate_xy(area, *member.node_i.vertex.coordinates)
                if len(area) > 2:
                    polygon, = self.axes.fill(
                        *zip(*area), color=self.plotter_options.positive_fill_color, alpha=self.plotter_options.positive_fill_alpha)
                    artist.append(polygon)
            for area in negative_processed:
                area = rotate_xy(area, member.angle_x(), 0, 0)
                area = traslate_xy(area, *member.node_i.vertex.coordinates)
                if len(area) > 2:
                    polygon, = self.axes.fill(
                        *zip(*area), color=self.plotter_options.negative_fill_color, alpha=self.plotter_options.negative_fill_alpha)
                    artist.append(polygon)

            # PLOTEO DE LAS ETIQUETAS: o, med, f
            if self.plotter_options.internal_forces_label:
                if ((round(yo, 7) == round(yf, 7)) and (round(y3, 7) == round(yneg3, 7))) or ((round(yo, 7) == 0) and (round(yf, 7) == round(y3, 7)) or (round(yf, 7) == 0) and (round(yo, 7) == round(yneg3, 7))):    # constante
                    text = self.axes.text(
                        lab_coord[3][0], lab_coord[3][1], f'{y3/escala:.2f}', fontsize=8,)
                    artist.append(text)
                # con brazos y polinomio no constante
                elif (round(yo, 7) == round(yf, 7) == 0) and (round(y3, 7) != round(yneg3, 7)) or ((round(yo, 7) == 0) and (round(y3, 7) != 0) or (round(yf, 7) == 0) and (round(yneg3, 7) != 0)):
                    if round(y3, 7) != 0:
                        text = self.axes.text(
                            lab_coord[0][0], lab_coord[0][1], f'{y3/escala:.2f}', fontsize=8,)
                        artist.append(text)
                    if round(yneg3, 7) != 0:
                        text = self.axes.text(
                            lab_coord[2][0], lab_coord[2][1], f'{yneg3/escala:.2f}', fontsize=8,)
                        artist.append(text)

                # sin brazos en ningun endframe
                elif (round(yo, 7) != 0) and (round(yv, 7) != 0) and (round(yf, 7) != 0):
                    if round(yo, 7) != 0:
                        text = self.axes.text(
                            lab_coord[0][0], lab_coord[0][1], f'{yo/escala:.2f}', fontsize=8)
                        artist.append(text)
                    if round(yv, 7) != 0 and (abs(round(yv, 7)) > abs(round(yo, 7)) and abs(round(yv, 7)) > abs(round(yf, 7))):
                        text = self.axes.text(
                            lab_coord[1][0], lab_coord[1][1], f'{yv/escala:.2f}', fontsize=8)
                        artist.append(text)
                    if round(yf, 7) != 0:
                        text = self.axes.text(
                            lab_coord[2][0], lab_coord[2][1], f'{yf/escala:.2f}', fontsize=8)
                        artist.append(text)

            val = rotate_xy(val, member.angle_x(), 0, 0)
            val = traslate_xy(val, *member.node_i.vertex.coordinates)

            line, = self.axes.plot(
                *zip(*val), color='#424242', lw=0.5)  # Línea de la curva
            artist.append(line)

            if type == InternalForceType.AXIAL_FORCE:
                self.axial_force[self.current_load_pattern][member_id] = artist
            elif type == InternalForceType.SHEAR_FORCE:
                self.shear_force[self.current_load_pattern][member_id] = artist
            elif type == InternalForceType.BENDING_MOMENT:
                self.bending_moment[self.current_load_pattern][member_id] = artist
        # visibility
        if type == InternalForceType.AXIAL_FORCE:
            visibility = self.plotter_options.UI_axial
        elif type == InternalForceType.SHEAR_FORCE:
            visibility = self.plotter_options.UI_shear
        elif type == InternalForceType.BENDING_MOMENT:
            visibility = self.plotter_options.UI_moment
        for artist in artist:
            artist.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_internal_forces(self, type: InternalForceType, visibility: bool | None = None, scale: float | None = None) -> None:
        """
        Actualiza la visibilidad de los diagramas de fuerzas internas.
        """
        def scale_text(txt, sx, sy, origin=(0, 0)):
            x0, y0 = origin
            x, y = txt.get_position()
            new_x = x0 + sx * (x - x0)
            new_y = y0 + sy * (y - y0)
            txt.set_position((new_x, new_y))

        def scale_line(line, sx, sy, origin=(0, 0)):
            x0, y0 = origin
            x, y = line.get_data()
            new_x = x0 + sx * (x - x0)
            new_y = y0 + sy * (y - y0)
            line.set_data(new_x, new_y)

        def scale_polygon(poly, sx, sy, origin=(0, 0)):
            x0, y0 = origin
            verts = poly.get_xy()
            verts[:, 0] = x0 + sx * (verts[:, 0] - x0)
            verts[:, 1] = y0 + sy * (verts[:, 1] - y0)
            poly.set_xy(verts)

        def if_scale(ele_id, artist, scale):
            sx, sy = scale, scale
            listGlobalElements = list(
                self.members.values()) + list(self.trusses.values())
            origin = listGlobalElements[ele_id].node_i.vertex.coordinates
            if isinstance(artist, Text):
                scale_text(artist, sx, sy, origin)
            elif isinstance(artist, Line2D):
                scale_line(artist, sx, sy, origin)
            elif isinstance(artist, Polygon):
                scale_polygon(artist, sx, sy, origin)

        if type == InternalForceType.AXIAL_FORCE:
            visibility = self.plotter_options.UI_axial if visibility is None else visibility
            for ele_id, listArtist in self.axial_force[self.current_load_pattern].items():
                for artist in listArtist:
                    artist.set_visible(visibility)
                    if scale:
                        if_scale(ele_id, artist, scale)
        elif type == InternalForceType.SHEAR_FORCE:
            visibility = self.plotter_options.UI_shear if visibility is None else visibility
            for listArtist in self.shear_force[self.current_load_pattern].values():
                for artist in listArtist:
                    artist.set_visible(visibility)
                    if scale:
                        if_scale(ele_id, artist, scale)
        elif type == InternalForceType.BENDING_MOMENT:
            visibility = self.plotter_options.UI_moment if visibility is None else visibility
            for listArtist in self.bending_moment[self.current_load_pattern].values():
                for artist in listArtist:
                    artist.set_visible(visibility)
                    if scale:
                        if_scale(ele_id, artist, scale)
        self.figure.canvas.draw_idle()

    def plot_axial_force(self, escala: float | None = None) -> None:
        """
        Grafica los diagramas de fuerzas internas axiales.
        """
        self.plot_internal_forces(InternalForceType.AXIAL_FORCE, escala)

    def update_axial_force(self, visibility: Optional[bool] = None) -> None:
        """
        Actualiza la visibilidad de los diagramas de fuerzas internas axiales.
        """
        self.update_internal_forces(InternalForceType.AXIAL_FORCE, visibility)

    def plot_shear_force(self, escala: float | None = None) -> None:
        """
        Grafica los diagramas de fuerzas internas cortantes.
        """
        self.plot_internal_forces(InternalForceType.SHEAR_FORCE, escala)

    def update_shear_force(self, visibility: Optional[bool] = None) -> None:
        """
        Actualiza la visibilidad de los diagramas de fuerzas internas cortantes.
        """
        self.update_internal_forces(InternalForceType.SHEAR_FORCE, visibility)

    def plot_bending_moment(self, escala: float | None = None) -> None:
        """
        Grafica los diagramas de fuerzas internas de momento.
        """
        self.plot_internal_forces(InternalForceType.BENDING_MOMENT, escala)

    def update_bending_moment(self, visibility: Optional[bool] = None) -> None:
        """
        Actualiza la visibilidad de los diagramas de fuerzas internas de momento.
        """
        self.update_internal_forces(
            InternalForceType.BENDING_MOMENT, visibility)

    def plot_reactions(self) -> None:
        """
        Grafica las reacciones.
        """
        self.reactions[self.current_load_pattern] = {}
        artists = []
        for node in self.model.nodes.values():
            reactions = self.model.results[self.current_load_pattern].get_node_reactions(
                node.id)
            length_arrow = self.plotter_options.point_load_length_arrow
            moment_length_arrow = 0.70 * self.plotter_options.point_moment_length_arrow
            if reactions[0] != 0:
                arrowRX, textRX = graphic_one_arrow(
                    node.vertex.x, node.vertex.y, round(
                        reactions[0], 2), length_arrow,
                    0 if reactions[0] < 0 else np.pi, self.axes,
                    self.plotter_options.reactions_color, True, "blue", 8)
                artists.append(arrowRX)
                artists.append(textRX)
            if reactions[1] != 0:
                arrowRY, textRY = graphic_one_arrow(
                    node.vertex.x, node.vertex.y, round(
                        reactions[1], 2), length_arrow,
                    np.pi/2 if reactions[1] < 0 else 3*np.pi/2, self.axes,
                    self.plotter_options.reactions_color, True, "blue", 8)
                artists.append(arrowRY)
                artists.append(textRY)
            if reactions[2] != 0:
                arrowMZ, textMZ = moment_fancy_arrow(
                    self.axes, node.vertex.x, node.vertex.y, round(
                        reactions[2], 2), moment_length_arrow,
                    self.plotter_options.reactions_color, True, True, "blue", 8)
                artists.append(arrowMZ)
                artists.append(textMZ)
            for artist in artists:
                artist.set_visible(self.plotter_options.UI_reactions)
            self.reactions[self.current_load_pattern][node.id] = artists

        self.figure.canvas.draw_idle()

    def update_reactions(self, visibility: Optional[bool] = None) -> None:
        """
        Actualiza la visibilidad de las reacciones.
        """
        visibility = self.plotter_options.UI_reactions if visibility is None else visibility
        for listArtist in self.reactions[self.current_load_pattern].values():
            for artist in listArtist:
                artist.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def plot_displaced_nodes(self) -> None:
        """
        Grafica los nodos deformados.
        """
        self.deformed_nodes[self.current_load_pattern] = {}
        for node_id, coord in self.current_values.nodes.items():
            ux = self.model.results[self.current_load_pattern].get_node_displacements(
                node_id)[0]*self.plotter_options.UI_deformation_scale[self.current_load_pattern]
            vy = self.model.results[self.current_load_pattern].get_node_displacements(
                node_id)[1]*self.plotter_options.UI_deformation_scale[self.current_load_pattern]
            x = [coord[0] + ux]
            y = [coord[1] + vy]

            node = self.axes.scatter(
                x, y, c=self.plotter_options.node_color, s=self.plotter_options.node_size, marker='o')
            self.deformed_nodes[self.current_load_pattern][node_id] = node
            visibility = self.plotter_options.UI_deformed or self.plotter_options.UI_rigid_deformed
            node.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_displaced_nodes(self, visibility: Optional[bool] = None, scale: Optional[float] = None) -> None:
        """
        Actualiza la visibilidad de los nodos deformados.
        """
        vis = self.plotter_options.UI_deformed or self.plotter_options.UI_rigid_deformed
        visibility = False  # vis if visibility is None else visibility
        for node in self.deformed_nodes[self.current_load_pattern].values():
            node.set_visible(visibility)
        self.figure.canvas.draw_idle()

        # HACER UN SET_DATA(X, Y) A TODOS LOS NODOS DEL ACTUAL LOAD_PATTERN
        if scale is not None:
            for node_id, node in self.deformed_nodes[self.current_load_pattern].items():
                ux = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[
                    0]*scale
                vy = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[
                    1]*scale
                x = self.model.nodes[node_id].vertex.x + ux
                y = self.model.nodes[node_id].vertex.y + vy
                node.set_offsets([x, y])
            self.figure.canvas.draw_idle()


################################# TOPICOS ###################################################

    def plot_prescribed_dofs(self) -> None:
        """
        Grafica las prescipciones en los DOFs.
        """
        # if not self.plotter_options.UI_prescribed_dofs:
        #     return
        self.prescribed_dofs[self.current_load_pattern] = {}
        self.prescribed_dofs_labels[self.current_load_pattern] = {}
        for id_node, PDOF in self.current_values.prescribed_dofs.items():
            coords = self.current_values.nodes[id_node]

            arrows = []
            texts = []
            arrowstyle = "-|>"
            # dezplamientos en dirección X
            if PDOF.ux != 0 and PDOF.ux is not None:
                if self.model.nodes[id_node].local_axis is not None:
                    if PDOF.ux > 0:
                        angle = self.model.nodes[id_node].local_axis.angle
                    else:
                        angle = self.model.nodes[id_node].local_axis.angle + np.pi
                else:
                    angle = np.pi if PDOF.ux < 0 else 0
                arrow, text = graphic_one_arrow_dof(
                    x=coords[0],
                    y=coords[1],
                    load=PDOF.ux,
                    length_arrow=self.plotter_options.point_load_length_arrow,
                    angle=angle,
                    ax=self.axes,
                    color=self.plotter_options.disp_pre_color,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.disp_pre_label_color,
                    label_font_size=self.plotter_options.disp_pre_label_font_size,
                    lw=self.plotter_options.disp_pre_length_width,
                    arrowstyle=arrowstyle
                )
                arrows.append(arrow)
                texts.append(text)

            # Fuerza en dirección Y
            if PDOF.uy != 0 and PDOF.uy is not None:
                if self.model.nodes[id_node].local_axis is not None:
                    if PDOF.uy > 0:
                        angle = np.pi/2 + \
                            self.model.nodes[id_node].local_axis.angle
                    else:
                        angle = 3*np.pi/2 + \
                            self.model.nodes[id_node].local_axis.angle
                else:
                    angle = np.pi/2 if PDOF.uy > 0 else 3*np.pi/2
                arrow, text = graphic_one_arrow_dof(
                    x=coords[0],
                    y=coords[1],
                    load=PDOF.uy,
                    length_arrow=self.plotter_options.point_load_length_arrow,
                    angle=angle,  # np.pi/2 if PDOF.uy < 0 else 3*np.pi/2,
                    ax=self.axes,
                    color=self.plotter_options.disp_pre_color,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.disp_pre_label_color,
                    label_font_size=self.plotter_options.disp_pre_label_font_size,
                    lw=self.plotter_options.disp_pre_length_width,
                    arrowstyle=arrowstyle
                )
                arrows.append(arrow)
                texts.append(text)

            # Momento en Z
            if PDOF.rz != 0 and PDOF.rz is not None:
                arrow, text = moment_fancy_arrow(
                    ax=self.axes,
                    x=coords[0],
                    y=coords[1],
                    moment=PDOF.rz,
                    radio=0.50 * self.plotter_options.point_moment_length_arrow,
                    color=self.plotter_options.disp_pre_color,
                    clockwise=True,
                    label=self.plotter_options.point_load_label,
                    color_label=self.plotter_options.disp_pre_label_color,
                    label_font_size=self.plotter_options.disp_pre_label_font_size,
                    lw=self.plotter_options.disp_pre_length_width,
                    arrowstyle=arrowstyle
                )
                arrows.append(arrow)
                texts.append(text)

            self.prescribed_dofs[self.current_load_pattern][id_node] = arrows
            self.prescribed_dofs_labels[self.current_load_pattern][id_node] = texts
        # for arrow, text in zip(arrows, texts):
        #     arrow.set_visible(self.plotter_options.point_load)
        #     text.set_visible(self.plotter_options.point_load_label)
        self.figure.canvas.draw_idle()

    def update_prescribed_dofs(self, visibility: bool | None = None):
        """
        Actualiza la visibilidad de los valores prescritas en los grados de libertad.
        """
        visibility = self.plotter_options.UI_load if visibility is None else visibility
        for arrows in self.prescribed_dofs[self.current_load_pattern].values():
            for arrow in arrows:
                arrow.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def update_prescribed_dofs_labels(self, visibility: bool | None = None):
        """
        Actualiza la visibilidad de los valores prescritas en los grados de libertad.
        """
        visibility = self.plotter_options.UI_load if visibility is None else visibility
        for texts in self.prescribed_dofs_labels[self.current_load_pattern].values():
            for text in texts:
                text.set_visible(visibility)
        self.figure.canvas.draw_idle()

    def plot_elastic_supports(self):
        """
        Dibuja los apoyos elásticos de la estructura.
        """
        theta_krz = getattr(self.plotter_options, "mod_krz_rotation_angle", 135)
        if isinstance(theta_krz, (list, tuple)):
            theta_iter = iter(theta_krz)
        elif isinstance(theta_krz, (int, float)):
            theta_iter = itertools.repeat(theta_krz)
        elif isinstance(theta_krz, dict):
            pass  # not implemented yet
        else:
            theta = 135  # Default value if the input is invalid

        for node_id, elastic_support in self.current_values.elastic_supports.items():
            node_coords = self.current_values.nodes[node_id]
            kx, ky, krz = elastic_support.get_elastic_supports()
            LA = self.model.nodes[node_id].local_axis
            if LA is not None:
                T = LA.get_transformation_matrix()
            else:
                T = np.eye(3)
            kx, ky, krz = T @ np.array([kx or 0, ky or 0, krz or 0])
            artist = []
            data_artist = {}
            if redondear_si_mas_de_3_decimales(kx) != 0:
                theta = 0  # if kx < 0 else 180
                line = support_kt(
                    self.axes,
                    node_coords[0],
                    node_coords[1],
                    self.plotter_options.support_size,
                    self.plotter_options.support_color,
                    theta=theta
                )
                x_data, y_data = line.get_data()
                artist.append(line)
                coord_label = [
                    node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                coord_label = rotate_xy(
                    coord_label, theta, node_coords[0], node_coords[1])
                if self.plotter_options.elastic_support_label:
                    text = self.axes.text(coord_label[0], coord_label[1], f"kx = {redondear_si_mas_de_3_decimales(kx)}",
                                          fontsize=self.plotter_options.label_font_size, color=self.plotter_options.node_label_color)
                    artist.append(text)
                    data_artist["kx"] = (x_data, y_data, theta, line, text)
                else:
                    data_artist["kx"] = (x_data, y_data, theta, line)

            if redondear_si_mas_de_3_decimales(ky) != 0:
                theta = 90  # if ky < 0 else 270
                line = support_kt(
                    self.axes,
                    node_coords[0],
                    node_coords[1],
                    self.plotter_options.support_size,
                    self.plotter_options.support_color,
                    theta=theta
                )
                x_data, y_data = line.get_data()
                artist.append(line)
                coord_label = [
                    node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                coord_label = rotate_xy(
                    coord_label, theta, node_coords[0], node_coords[1])
                if self.plotter_options.elastic_support_label:
                    text = self.axes.text(coord_label[0], coord_label[1], f"ky = {redondear_si_mas_de_3_decimales(ky)}",
                                          fontsize=self.plotter_options.label_font_size, color=self.plotter_options.node_label_color)
                    artist.append(text)
                    data_artist["ky"] = (x_data, y_data, theta, line, text)
                else:
                    data_artist["ky"] = (x_data, y_data, theta, line)

            if redondear_si_mas_de_3_decimales(krz) != 0:

                if isinstance(theta_krz, dict):
                    theta = theta_krz.get(node_id, 135)
                else:
                    try:
                        theta = next(theta_iter)
                    except StopIteration:
                        theta = 135

                line = support_kr(
                    self.axes,
                    node_coords[0],
                    node_coords[1],
                    self.plotter_options.support_size,
                    self.plotter_options.support_color,
                    theta=theta
                )

                x_data, y_data = line.get_data()
                artist.append(line)
                coord_label = [
                    node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                coord_label = rotate_xy(
                    coord_label, theta, node_coords[0], node_coords[1])
                if self.plotter_options.elastic_support_label:
                    text = self.axes.text(coord_label[0], coord_label[1], f"krz = {redondear_si_mas_de_3_decimales(krz)}",
                                          fontsize=self.plotter_options.label_font_size, color=self.plotter_options.node_label_color)
                    artist.append(text)
                    data_artist["krz"] = (x_data, y_data, theta, line, text)
                else:
                    data_artist["krz"] = (x_data, y_data, theta, line)

            self.elastic_supports[node_id] = artist
            self.elastic_support_data_cache[node_id] = [
                data_artist, node_coords]
        self.figure.canvas.draw_idle()

    def update_elastic_supports(self, visibility: bool | None = None, scale: float | None = None):
        """
        Actualiza la visibilidad de los apoyos elásticos.
        """
        visibility = self.plotter_options.UI_support if visibility is None else visibility
        for node_id, artists in self.elastic_supports.items():
            for artist in artists:
                artist.set_visible(visibility)
                if scale is not None:
                    dx = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[
                        0]*scale
                    dy = self.model.results[self.current_load_pattern].get_node_displacements(node_id)[
                        1]*scale
                    if isinstance(artist, plt.Line2D):
                        if artist is self.elastic_support_data_cache[node_id][0].get("kx", [0, 0, 0, 0, 0])[3]:
                            x_new = np.array(
                                self.elastic_support_data_cache[node_id][0].get("kx")[0]) + dx
                            y_new = np.array(
                                self.elastic_support_data_cache[node_id][0].get("kx")[1]) + dy
                            artist.set_xdata(x_new)
                            artist.set_ydata(y_new)
                        elif artist is self.elastic_support_data_cache[node_id][0].get("ky", [0, 0, 0, 0, 0])[3]:
                            x_new = np.array(
                                self.elastic_support_data_cache[node_id][0].get("ky")[0]) + dx
                            y_new = np.array(
                                self.elastic_support_data_cache[node_id][0].get("ky")[1]) + dy
                            artist.set_xdata(x_new)
                            artist.set_ydata(y_new)
                        elif artist is self.elastic_support_data_cache[node_id][0].get("krz", [0, 0, 0, 0, 0])[3]:
                            x_new = np.array(
                                self.elastic_support_data_cache[node_id][0].get("krz")[0]) + dx
                            y_new = np.array(
                                self.elastic_support_data_cache[node_id][0].get("krz")[1]) + dy
                            artist.set_xdata(x_new)
                            artist.set_ydata(y_new)
                    elif isinstance(artist, plt.Text):
                        node_coords = self.current_values.nodes[node_id]
                        if self.elastic_support_data_cache[node_id][0].get("kx"):
                            theta = 0  # if kx < 0 else 180
                            coord_label = [
                                node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                            coord_label = rotate_xy(
                                coord_label, theta, node_coords[0], node_coords[1])
                            coord_label = [coord_label[0] +
                                           dx, coord_label[1] + dy]
                            if artist is self.elastic_support_data_cache[node_id][0].get("kx", [0, 0, 0, 0, 0])[4]:
                                artist.set_position(
                                    (coord_label[0], coord_label[1]))
                        if self.elastic_support_data_cache[node_id][0].get("ky"):
                            theta = 90  # if ky < 0 else 270
                            coord_label = [
                                node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                            coord_label = rotate_xy(
                                coord_label, theta, node_coords[0], node_coords[1])
                            coord_label = [coord_label[0] +
                                           dx, coord_label[1] + dy]
                            if artist is self.elastic_support_data_cache[node_id][0].get("ky", [0, 0, 0, 0, 0])[4]:
                                artist.set_position(
                                    (coord_label[0], coord_label[1]))
                        if self.elastic_support_data_cache[node_id][0].get("krz"):
                            theta = 135
                            coord_label = [
                                node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                            coord_label = rotate_xy(
                                coord_label, theta, node_coords[0], node_coords[1])
                            coord_label = [coord_label[0] +
                                           dx, coord_label[1] + dy]
                            if artist is self.elastic_support_data_cache[node_id][0].get("krz", [0, 0, 0, 0, 0])[4]:
                                artist.set_position(
                                    (coord_label[0], coord_label[1]))
        self.figure.canvas.draw_idle()

    def reset_elastic_supports(self):
        """
        Restablece los apoyos elásticos.
        """
        for node_id, artists in self.elastic_supports.items():
            for artist in artists:
                artist.set_visible(self.plotter_options.UI_support)
                if isinstance(artist, plt.Line2D):
                    if artist is self.elastic_support_data_cache[node_id][0].get("kx", [0, 0, 0, 0, []])[3]:
                        x_new = np.array(
                            self.elastic_support_data_cache[node_id][0].get("kx")[0])
                        y_new = np.array(
                            self.elastic_support_data_cache[node_id][0].get("kx")[1])
                        artist.set_xdata(x_new)
                        artist.set_ydata(y_new)
                    elif artist is self.elastic_support_data_cache[node_id][0].get("ky", [0, 0, 0, 0, []])[3]:
                        x_new = np.array(
                            self.elastic_support_data_cache[node_id][0].get("ky")[0])
                        y_new = np.array(
                            self.elastic_support_data_cache[node_id][0].get("ky")[1])
                        artist.set_xdata(x_new)
                        artist.set_ydata(y_new)
                    elif artist is self.elastic_support_data_cache[node_id][0].get("krz", [0, 0, 0, 0, []])[3]:
                        x_new = np.array(
                            self.elastic_support_data_cache[node_id][0].get("krz")[0])
                        y_new = np.array(
                            self.elastic_support_data_cache[node_id][0].get("krz")[1])
                        artist.set_xdata(x_new)
                        artist.set_ydata(y_new)
                elif isinstance(artist, plt.Text):
                    node_coords = self.current_values.nodes[node_id]
                    if self.elastic_support_data_cache[node_id][0].get("kx"):
                        theta = 0  # if kx < 0 else 180
                        coord_label = [
                            node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                        coord_label = rotate_xy(
                            coord_label, theta, node_coords[0], node_coords[1])
                        if artist is self.elastic_support_data_cache[node_id][0].get("kx", [0, 0, 0, 0, 0])[4]:
                            artist.set_position(
                                (coord_label[0], coord_label[1]))
                    if self.elastic_support_data_cache[node_id][0].get("ky"):
                        theta = 90  # if ky < 0 else 270
                        coord_label = [
                            node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                        coord_label = rotate_xy(
                            coord_label, theta, node_coords[0], node_coords[1])
                        if artist is self.elastic_support_data_cache[node_id][0].get("ky", [0, 0, 0, 0, 0])[4]:
                            artist.set_position(
                                (coord_label[0], coord_label[1]))
                    if self.elastic_support_data_cache[node_id][0].get("krz"):
                        theta = 135
                        coord_label = [
                            node_coords[0]-self.plotter_options.support_size/2, node_coords[1]]
                        coord_label = rotate_xy(
                            coord_label, theta, node_coords[0], node_coords[1])
                        if artist is self.elastic_support_data_cache[node_id][0].get("krz", [0, 0, 0, 0, 0])[4]:
                            artist.set_position(
                                (coord_label[0], coord_label[1]))
        self.figure.canvas.draw_idle()

    def plot_cst(self):
        """
        Dibuja los CST de la estructura.
        """
        for id, (coord, disp) in self.current_values.cst.items():
            x_coords = coord[[0, 2, 4, 0]]
            y_coords = coord[[1, 3, 5, 1]]
            artists = []

            line, = self.axes.plot(x_coords, y_coords,
                                   color=self.plotter_options.cst_edge_color,
                                   linewidth=self.plotter_options.cst_element_line_width)

            polygon = self.axes.fill(x_coords, y_coords,
                                     color=self.plotter_options.cst_face_color,
                                     alpha=self.plotter_options.cst_alpha)[0]

            artists.append(line)
            artists.append(polygon)
            self.csts[id] = artists

            for artist in artists:
                artist.set_visible(self.plotter_options.UI_show_members)

        self.figure.canvas.draw_idle()

    def update_cst(self, color_edge: str | None = None, color_face: str | None = None):
        """
        Actualiza la visibilidad de los CST de la estructura.
        """
        color_edge = self.plotter_options.cst_edge_color if color_edge is None else color_edge
        color_face = self.plotter_options.cst_face_color if color_face is None else color_face
        for cst in self.csts.values():
            for artist in cst:
                artist.set_visible(self.plotter_options.UI_show_members)
                if isinstance(artist, plt.Line2D):
                    artist.set_color(color_edge)
                elif isinstance(artist, plt.Polygon):
                    artist.set_facecolor(color_face)
        self.figure.canvas.draw_idle()

    def plot_cst_labels(self):
        """
        Dibuja las etiquetas de los CST de la estructura.
        """
        for cst_id, (coords, disp) in self.current_values.cst.items():
            x_val = np.mean(coords[[0, 2, 4]])
            y_val = np.mean(coords[[1, 3, 5]])

            text = self.axes.text(x_val, y_val, f"CST: {cst_id}",
                                  fontsize=self.plotter_options.label_font_size,
                                  ha='left', va='bottom', color="blue",  # bbox=bbox,
                                  clip_on=True)
            self.csts[cst_id].append(text)
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def update_cst_labels(self):
        """
        Actualiza la visibilidad de las etiquetas de los CST de la estructura.
        """
        for cst in self.csts.values():
            for artist in cst:
                if isinstance(artist, plt.Text):
                    artist.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def plot_membrane_q3dof(self):
        """
        Dibuja los Membrane Q3DOF de la estructura.
        """
        for id, (coord, disp) in self.current_values.membrane_q3dof.items():
            x_coords = coord[[0, 2, 4, 6, 0]]
            y_coords = coord[[1, 3, 5, 7, 1]]
            artists = []

            line, = self.axes.plot(x_coords, y_coords,
                                   color=self.plotter_options.membrane_q3dof_edge_color,
                                   linewidth=self.plotter_options.membrane_q3dof_element_line_width)

            polygon = self.axes.fill(x_coords, y_coords,
                                     color=self.plotter_options.membrane_q3dof_face_color,
                                     alpha=self.plotter_options.membrane_q3dof_alpha)[0]

            artists.append(line)
            artists.append(polygon)
            self.membrane_q3dof[id] = artists

            for artist in artists:
                artist.set_visible(self.plotter_options.UI_show_members)

        self.figure.canvas.draw_idle()

    def update_membrane_q3dof(self, color_edge: str | None = None, color_face: str | None = None):
        """
        Actualiza la visibilidad de los Membrane Q3DOF de la estructura.
        """
        color_edge = self.plotter_options.membrane_q3dof_edge_color if color_edge is None else color_edge
        color_face = self.plotter_options.membrane_q3dof_face_color if color_face is None else color_face
        for membrane_q3dof in self.membrane_q3dof.values():
            for artist in membrane_q3dof:
                artist.set_visible(self.plotter_options.UI_show_members)
                if isinstance(artist, plt.Line2D):
                    artist.set_color(color_edge)
                elif isinstance(artist, plt.Polygon):
                    artist.set_facecolor(color_face)
        self.figure.canvas.draw_idle()

    def plot_membrane_q3dof_labels(self):
        """
        Dibuja las etiquetas de los Membrane Q3DOF de la estructura.
        """
        for membrane_q3dof_id, (coords, disp) in self.current_values.membrane_q3dof.items():
            x_val = np.mean(coords[[0, 2, 4]])
            y_val = np.mean(coords[[1, 3, 5]])

            text = self.axes.text(x_val, y_val, f"MEMBRANA Q3DOF: {membrane_q3dof_id}",
                                  fontsize=self.plotter_options.label_font_size,
                                  ha='left', va='bottom', color="blue",  # bbox=bbox,
                                  clip_on=True)
            self.membrane_q3dof[membrane_q3dof_id].append(text)
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def update_membrane_q3dof_labels(self):
        """
        Actualiza la visibilidad de las etiquetas de los Membrane Q3DOF de la estructura.
        """
        for membrane_q3dof in self.membrane_q3dof.values():
            for artist in membrane_q3dof:
                if isinstance(artist, plt.Text):
                    artist.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def plot_membrane_q2dof(self):
        """
        Dibuja los Membrane Q2DOF de la estructura.
        """
        for id, (coord, disp) in self.current_values.membrane_q2dof.items():
            x_coords = coord[[0, 2, 4, 6, 0]]
            y_coords = coord[[1, 3, 5, 7, 1]]
            artists = []

            line, = self.axes.plot(x_coords, y_coords,
                                   color=self.plotter_options.membrane_q2dof_edge_color,
                                   linewidth=self.plotter_options.membrane_q2dof_element_line_width)

            polygon = self.axes.fill(x_coords, y_coords,
                                     color=self.plotter_options.membrane_q2dof_face_color,
                                     alpha=self.plotter_options.membrane_q2dof_alpha)[0]

            artists.append(line)
            artists.append(polygon)
            self.membrane_q2dof[id] = artists

            for artist in artists:
                artist.set_visible(self.plotter_options.UI_show_members)

        self.figure.canvas.draw_idle()

    def update_membrane_q2dof(self, color_edge: str | None = None, color_face: str | None = None):
        """
        Actualiza la visibilidad de los Membrane Q2DOF de la estructura.
        """
        color_edge = self.plotter_options.membrane_q2dof_edge_color if color_edge is None else color_edge
        color_face = self.plotter_options.membrane_q2dof_face_color if color_face is None else color_face
        for membrane_q2dof in self.membrane_q2dof.values():
            for artist in membrane_q2dof:
                artist.set_visible(self.plotter_options.UI_show_members)
                if isinstance(artist, plt.Line2D):
                    artist.set_color(color_edge)
                elif isinstance(artist, plt.Polygon):
                    artist.set_facecolor(color_face)
        self.figure.canvas.draw_idle()

    def plot_membrane_q2dof_labels(self):
        """
        Dibuja las etiquetas de los Membrane Q2DOF de la estructura.
        """
        for membrane_q2dof_id, (coords, disp) in self.current_values.membrane_q2dof.items():
            x_val = np.mean(coords[[0, 2, 4]])
            y_val = np.mean(coords[[1, 3, 5]])

            text = self.axes.text(x_val, y_val, f"MEMBRANA Q2DOF: {membrane_q2dof_id}",
                                  fontsize=self.plotter_options.label_font_size,
                                  ha='left', va='bottom', color="blue",  # bbox=bbox,
                                  clip_on=True)
            self.membrane_q2dof[membrane_q2dof_id].append(text)
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def update_membrane_q2dof_labels(self):
        """
        Actualiza la visibilidad de las etiquetas de los Membrane Q2DOF de la estructura.
        """
        for membrane_q2dof in self.membrane_q2dof.values():
            for artist in membrane_q2dof:
                if isinstance(artist, plt.Text):
                    artist.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def plot_trusses(self):
        """
        Dibuja los elementos de la estructura.
        """
        # if not self.plotter_options.UI_show_members:
        #     return
        for id, (coord, disp) in self.current_values.trusses.items():
            x_coords = [coord[0], coord[2]]
            y_coords = [coord[1], coord[3]]
            line, = self.axes.plot(x_coords, y_coords, color=self.plotter_options.truss_color,
                                   linewidth=self.plotter_options.element_line_width)
            self.trusses[id] = line
            line.set_visible(self.plotter_options.UI_show_members)
        self.figure.canvas.draw_idle()

    def update_trusses(self, color=None):
        for truss in self.trusses.values():
            truss.set_visible(self.plotter_options.UI_show_members)
        if color:
            for truss in self.trusses.values():
                truss.set_color(color)
        self.figure.canvas.draw_idle()

    def plot_deformed(self, escala: float | None = None) -> None:
        """
        Dibuja la forma deformada de la estructura.
        Actualiza la visibilidad de la forma deformada de la estructura.
        forma de los datos de la propiedad de la forma deformada:
            {
            load_pattern:{
                member_id: [line2D]
                truss_id: [line2D]
                cst_id: [Line2D, Polygon2D]
                membrane_q3dof_id: [Line2D, Polygon2D]
                membrane_q2dof_id: [Line2D, Polygon2D]
                        }
            }
        """
        self.deformed_shape[self.current_load_pattern] = {}

        escala = self.plotter_options.UI_deformation_scale[
            self.current_load_pattern] if escala is None else escala

        for element in self.model.members.values():
            self.deformed_shape[self.current_load_pattern][element.id] = []
            x, y = self.current_values.get_deformed_shape(element.id, escala)
            line, = self.axes.plot(x, y, lw=self.plotter_options.deformation_line_width,
                                   color=self.plotter_options.deformation_color, zorder=70)
            self.deformed_shape[self.current_load_pattern][element.id].append(
                line)
            line.set_visible(self.plotter_options.UI_deformed)

        for truss in self.model.trusses.values():
            self.deformed_shape[self.current_load_pattern][truss.id] = []
            crd, disp = self.current_values.trusses[truss.id]
            disp = np.array(disp)*escala
            x, y = [crd[0]+disp[0], crd[2]+disp[2]
                    ], [crd[1]+disp[1], crd[3]+disp[3]]
            line, = self.axes.plot(x, y, lw=self.plotter_options.deformation_line_width,
                                   color=self.plotter_options.truss_deformed_color, zorder=70)
            self.deformed_shape[self.current_load_pattern][truss.id].append(
                line)
            line.set_visible(self.plotter_options.UI_deformed)

        for cst in self.model.csts.values():
            self.deformed_shape[self.current_load_pattern][cst.id] = []
            coordinates, displacements = self.current_values.cst[cst.id]
            displacements = np.array(displacements)*escala
            x, y = coordinates[[0, 2, 4]]+displacements[[0, 2, 4]
                                                        ], coordinates[[1, 3, 5]]+displacements[[1, 3, 5]]
            line, = self.axes.plot(np.hstack((x, [x[0]])), np.hstack((y, [y[0]])), lw=self.plotter_options.cst_element_line_width,
                                   color=self.plotter_options.cst_deformed_color_edge, zorder=70)
            polygon, = self.axes.fill(x, y, color=self.plotter_options.cst_deformed_color_face,
                                      alpha=self.plotter_options.cst_alpha, zorder=70)
            self.deformed_shape[self.current_load_pattern][cst.id].append(line)
            self.deformed_shape[self.current_load_pattern][cst.id].append(
                polygon)

            line.set_visible(self.plotter_options.UI_deformed)
            polygon.set_visible(self.plotter_options.UI_deformed)

        for membrane_q3dof in self.model.membrane_q3dof.values():
            self.deformed_shape[self.current_load_pattern][membrane_q3dof.id] = [
            ]
            coordinates, displacements = self.current_values.membrane_q3dof[membrane_q3dof.id]
            displacements = np.array(displacements)*escala
            x, y = coordinates[[0, 2, 4, 6]]+displacements[[0, 3, 6, 9]
                                                           ], coordinates[[1, 3, 5, 7]]+displacements[[1, 4, 7, 10]]
            line, = self.axes.plot(np.hstack((x, [x[0]])), np.hstack((y, [y[0]])), lw=self.plotter_options.membrane_q3dof_element_line_width,
                                   color=self.plotter_options.membrane_q3dof_deformed_color_edge, zorder=70)
            polygon, = self.axes.fill(x, y, color=self.plotter_options.membrane_q3dof_deformed_color_face,
                                      alpha=self.plotter_options.membrane_q3dof_alpha, zorder=70)
            self.deformed_shape[self.current_load_pattern][membrane_q3dof.id].append(
                line)
            self.deformed_shape[self.current_load_pattern][membrane_q3dof.id].append(
                polygon)

            line.set_visible(self.plotter_options.UI_deformed)
            polygon.set_visible(self.plotter_options.UI_deformed)

        for membrane_q2dof in self.model.membrane_q2dof.values():
            self.deformed_shape[self.current_load_pattern][membrane_q2dof.id] = [
            ]
            coordinates, displacements = self.current_values.membrane_q2dof[membrane_q2dof.id]
            displacements = np.array(displacements)*escala
            x, y = coordinates[[0, 2, 4, 6]]+displacements[[0, 2, 4, 6]
                                                           ], coordinates[[1, 3, 5, 7]]+displacements[[1, 3, 5, 7]]
            line, = self.axes.plot(np.hstack((x, [x[0]])), np.hstack((y, [y[0]])), lw=self.plotter_options.membrane_q2dof_element_line_width,
                                   color=self.plotter_options.membrane_q2dof_deformed_color_edge, zorder=70)
            polygon, = self.axes.fill(x, y, color=self.plotter_options.membrane_q2dof_deformed_color_face,
                                      alpha=self.plotter_options.membrane_q2dof_alpha, zorder=70)
            self.deformed_shape[self.current_load_pattern][membrane_q2dof.id].append(
                line)
            self.deformed_shape[self.current_load_pattern][membrane_q2dof.id].append(
                polygon)

            line.set_visible(self.plotter_options.UI_deformed)
            polygon.set_visible(self.plotter_options.UI_deformed)

        self.figure.canvas.draw_idle()

    def update_deformed(self, visibility: Optional[bool] = None, escala: Optional[float] = None):
        """
        Actualiza la visibilidad de la forma deformada de la estructura.
        forma de los datos de la propiedad de la forma deformada:
            {
            load_pattern:{
                member_id: [line2D]
                truss_id: [line2D]
                cst_id: [Line2D, Polygon2D]
                membrane_q3dof_id: [Line2D, Polygon2D]
                membrane_q2dof_id: [Line2D, Polygon2D]
                        }
            }
        """
        visibility = self.plotter_options.UI_deformed if visibility is None else visibility
        for artists in self.deformed_shape[self.current_load_pattern].values():
            for artist in artists:
                artist.set_visible(visibility)

        # HACER UN SET_DATA(X, Y) A TODOS LOS MIEMBROS DEL ACTUAL LOAD_PATTERN
        if escala is not None:
            self.current_values = self.get_plotter_values(
                self.current_load_pattern)
            for member in self.model.members.values():
                x, y = self.current_values.get_deformed_shape(
                    member.id, escala)
                self.deformed_shape[self.current_load_pattern][member.id][0].set_data(
                    x, y)

            for truss in self.model.trusses.values():
                crd, disp = self.current_values.trusses[truss.id]
                disp = np.array(disp)*escala
                x, y = [crd[0]+disp[0], crd[2]+disp[2]
                        ], [crd[1]+disp[1], crd[3]+disp[3]]
                self.deformed_shape[self.current_load_pattern][truss.id][0].set_data(
                    x, y)

            for cst in self.model.csts.values():
                coordinates, displacements = self.current_values.cst[cst.id]
                displacements = np.array(displacements)*escala
                x, y = coordinates[[0, 2, 4]]+displacements[[0, 2, 4]
                                                            ], coordinates[[1, 3, 5]]+displacements[[1, 3, 5]]
                # Une x,y en pares [(x0,y0), (x1,y1), (x2,y2)]
                coords = np.column_stack((x, y))

                self.deformed_shape[self.current_load_pattern][cst.id][0].set_data(
                    np.hstack((x, [x[0]])), np.hstack((y, [y[0]])))
                self.deformed_shape[self.current_load_pattern][cst.id][1].set_xy(
                    coords)

            for membrane_q3dof in self.model.membrane_q3dof.values():
                coordinates, displacements = self.current_values.membrane_q3dof[membrane_q3dof.id]
                displacements = np.array(displacements)*escala
                x, y = coordinates[[0, 2, 4, 6]]+displacements[[0, 3, 6, 9]
                                                               ], coordinates[[1, 3, 5, 7]]+displacements[[1, 4, 7, 10]]
                # Une x,y en pares [(x0,y0), (x1,y1), (x2,y2)]
                coords = np.column_stack((x, y))

                self.deformed_shape[self.current_load_pattern][membrane_q3dof.id][0].set_data(
                    np.hstack((x, [x[0]])), np.hstack((y, [y[0]])))
                self.deformed_shape[self.current_load_pattern][membrane_q3dof.id][1].set_xy(
                    coords)

            for membrane_q2dof in self.model.membrane_q2dof.values():
                coordinates, displacements = self.current_values.membrane_q2dof[membrane_q2dof.id]
                displacements = np.array(displacements)*escala
                x, y = coordinates[[0, 2, 4, 6]]+displacements[[0, 2, 4, 6]
                                                               ], coordinates[[1, 3, 5, 7]]+displacements[[1, 3, 5, 7]]
                # Une x,y en pares [(x0,y0), (x1,y1), (x2,y2)]
                coords = np.column_stack((x, y))

                self.deformed_shape[self.current_load_pattern][membrane_q2dof.id][0].set_data(
                    np.hstack((x, [x[0]])), np.hstack((y, [y[0]])))
                self.deformed_shape[self.current_load_pattern][membrane_q2dof.id][1].set_xy(
                    coords)

        self.figure.canvas.draw_idle()

    def plot_rigid_deformed(self, escala: float | None = None):
        """
        Grafica la forma de deformación rígida.
        """
        self.rigid_deformed_shape[self.current_load_pattern] = {}
        escala = self.plotter_options.UI_deformation_scale[
            self.current_load_pattern] if escala is None else escala
        for member_id in self.model.members.keys():
            self.rigid_deformed_shape[self.current_load_pattern][member_id] = [
            ]
            x, y = self.current_values.rigid_deformed(member_id, escala)
            line, = self.axes.plot(
                x, y, color=self.plotter_options.rigid_deformed_color, lw=0.7, ls='--', zorder=60)
            self.rigid_deformed_shape[self.current_load_pattern][member_id].append(
                line)
            line.set_visible(self.plotter_options.UI_rigid_deformed)

        for truss in self.model.trusses.values():
            self.rigid_deformed_shape[self.current_load_pattern][truss.id] = []
            crd, disp = self.current_values.trusses[truss.id]
            disp = np.array(disp)*escala
            x, y = [crd[0]+disp[0], crd[2]+disp[2]
                    ], [crd[1]+disp[1], crd[3]+disp[3]]
            line, = self.axes.plot(
                x, y, lw=0.7, color=self.plotter_options.rigid_deformed_color, ls='--', zorder=60)
            self.rigid_deformed_shape[self.current_load_pattern][truss.id].append(
                line)
            line.set_visible(self.plotter_options.UI_rigid_deformed)

        for cst in self.model.csts.values():
            self.rigid_deformed_shape[self.current_load_pattern][cst.id] = []
            coordinates, displacements = self.current_values.cst[cst.id]
            displacements = np.array(displacements)*escala
            x, y = coordinates[[0, 2, 4]]+displacements[[0, 2, 4]
                                                        ], coordinates[[1, 3, 5]]+displacements[[1, 3, 5]]
            line, = self.axes.plot(np.hstack((x, [x[0]])), np.hstack((y, [y[0]])), lw=0.7,
                                   color=self.plotter_options.rigid_deformed_color, zorder=60)
            self.rigid_deformed_shape[self.current_load_pattern][cst.id].append(
                line)
            line.set_visible(self.plotter_options.UI_rigid_deformed)

        for membrane_q3dof in self.model.membrane_q3dof.values():
            self.rigid_deformed_shape[self.current_load_pattern][membrane_q3dof.id] = [
            ]
            coordinates, displacements = self.current_values.membrane_q3dof[membrane_q3dof.id]
            displacements = np.array(displacements)*escala
            x, y = coordinates[[0, 2, 4, 6]]+displacements[[0, 3, 6, 9]
                                                           ], coordinates[[1, 3, 5, 7]]+displacements[[1, 4, 7, 10]]
            line, = self.axes.plot(np.hstack((x, [x[0]])), np.hstack((y, [y[0]])), lw=0.7,
                                   color=self.plotter_options.rigid_deformed_color, zorder=60)
            self.rigid_deformed_shape[self.current_load_pattern][membrane_q3dof.id].append(
                line)
            line.set_visible(self.plotter_options.UI_rigid_deformed)

        for membrane_q2dof in self.model.membrane_q2dof.values():
            self.rigid_deformed_shape[self.current_load_pattern][membrane_q2dof.id] = [
            ]
            coordinates, displacements = self.current_values.membrane_q2dof[membrane_q2dof.id]
            displacements = np.array(displacements)*escala
            x, y = coordinates[[0, 2, 4, 6]]+displacements[[0, 2, 4, 6]
                                                           ], coordinates[[1, 3, 5, 7]]+displacements[[1, 3, 5, 7]]
            line, = self.axes.plot(np.hstack((x, [x[0]])), np.hstack((y, [y[0]])), lw=0.7,
                                   color=self.plotter_options.rigid_deformed_color, zorder=60)
            self.rigid_deformed_shape[self.current_load_pattern][membrane_q2dof.id].append(
                line)
            line.set_visible(self.plotter_options.UI_rigid_deformed)

        self.figure.canvas.draw_idle()

    def update_rigid_deformed(self, visibility: Optional[bool] = None, escala: float | None = None) -> None:
        """
        Actualiza la visibilidad de la forma de deformación rígida.
        """
        visibility = self.plotter_options.UI_rigid_deformed if visibility is None else visibility
        for line in self.rigid_deformed_shape[self.current_load_pattern].values():
            for artist in line:
                artist.set_visible(visibility)

        # HACER UN SET_DATA(X, Y) A TODOS LOS MIEMBROS DEL ACTUAL LOAD_PATTERN
        if escala is not None:
            self.current_values = self.get_plotter_values(
                self.current_load_pattern)
            for member in self.model.members.values():
                x, y = self.current_values.rigid_deformed(member.id, escala)
                for artist in self.rigid_deformed_shape[self.current_load_pattern][member.id]:
                    artist.set_data(x, y)

            for truss in self.model.trusses.values():
                crd, disp = self.current_values.trusses[truss.id]
                disp = np.array(disp)*escala
                x, y = [crd[0]+disp[0], crd[2]+disp[2]
                        ], [crd[1]+disp[1], crd[3]+disp[3]]
                for artist in self.rigid_deformed_shape[self.current_load_pattern][truss.id]:
                    artist.set_data(x, y)

            for cst in self.model.csts.values():
                coordinates, displacements = self.current_values.cst[cst.id]
                displacements = np.array(displacements)*escala
                x, y = coordinates[[0, 2, 4]]+displacements[[0, 2, 4]
                                                            ], coordinates[[1, 3, 5]]+displacements[[1, 3, 5]]

                self.rigid_deformed_shape[self.current_load_pattern][cst.id][0].set_data(
                    np.hstack((x, [x[0]])), np.hstack((y, [y[0]])))

            for membrane_q3dof in self.model.membrane_q3dof.values():
                coordinates, displacements = self.current_values.membrane_q3dof[membrane_q3dof.id]
                displacements = np.array(displacements)*escala
                x, y = coordinates[[0, 2, 4, 6]]+displacements[[0, 3, 6, 9]
                                                               ], coordinates[[1, 3, 5, 7]]+displacements[[1, 4, 7, 10]]

                self.rigid_deformed_shape[self.current_load_pattern][membrane_q3dof.id][0].set_data(
                    np.hstack((x, [x[0]])), np.hstack((y, [y[0]])))

            for membrane_q2dof in self.model.membrane_q2dof.values():
                coordinates, displacements = self.current_values.membrane_q2dof[membrane_q2dof.id]
                displacements = np.array(displacements)*escala
                x, y = coordinates[[0, 2, 4, 6]]+displacements[[0, 2, 4, 6]
                                                               ], coordinates[[1, 3, 5, 7]]+displacements[[1, 3, 5, 7]]

                self.rigid_deformed_shape[self.current_load_pattern][membrane_q2dof.id][0].set_data(
                    np.hstack((x, [x[0]])), np.hstack((y, [y[0]])))

        self.figure.canvas.draw_idle()

    def plot_frame_release(self):
        """
        Dibuja las liberaciones de los miembros.
        """
        for member in self.model.members.values():
            if member.release:
                length = member.length()
                offset = self.plotter_options.frame_release_length_offset * length
                release = member.release.get_dof_release()
                scatter = []

                # Nodo i: liberaciones en 0, 1 o 2
                if any(r in release for r in (0, 1, 2)):
                    UnitVector = (member.node_j.vertex.coordinates -
                                  member.node_i.vertex.coordinates) / length
                    point = UnitVector * offset + member.node_i.vertex.coordinates
                    scatter.append(
                        self.axes.scatter(
                            point[0], point[1],
                            s=self.plotter_options.frame_release_point_size,
                            color=self.plotter_options.frame_release_color,
                            zorder=80
                        )
                    )

                # Nodo j: liberaciones en 3, 4 o 5
                if any(r in release for r in (3, 4, 5)):
                    UnitVector = (member.node_j.vertex.coordinates -
                                  member.node_i.vertex.coordinates) / length
                    point = -UnitVector * offset + member.node_j.vertex.coordinates
                    scatter.append(
                        self.axes.scatter(
                            point[0], point[1],
                            s=self.plotter_options.frame_release_point_size,
                            color=self.plotter_options.frame_release_color,
                            zorder=80
                        )
                    )

                self.frame_release_data_cache[member.id] = scatter

    def update_frame_release(self, color: Optional[str] = None):
        """
        Actualiza la visibilidad de las liberaciones de los miembros.
        """
        visibility = self.plotter_options.UI_show_members
        for artists in self.frame_release_data_cache.values():
            for artist in artists:
                artist.set_visible(visibility)
                if color:
                    artist.set_color(color)
        self.figure.canvas.draw_idle()

    def plot_trusses_labels(self):
        """
        Dibuja las etiquetas de los trusses.
        """
        for element_id, (coords, displacements) in self.current_values.trusses.items():
            x_val = (coords[0] + coords[2]) / 2
            y_val = (coords[1] + coords[3]) / 2

            # bbox = {
            #     "boxstyle": "round,pad=0.2",  # Estilo y padding del cuadro
            #     "facecolor": "lightblue",     # Color de fondo
            #     "edgecolor": "black",         # Color del borde
            #     "linewidth": 0.5,               # Grosor del borde
            #     "linestyle": "-",            # Estilo del borde
            #     "alpha": 0.8                  # Transparencia
            # }
            text = self.axes.text(x_val, y_val, str(element_id),
                                  fontsize=self.plotter_options.label_font_size,
                                  ha='left', va='bottom', color="blue",  # bbox=bbox,
                                  clip_on=True)
            self.trusses_labels[element_id] = text
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

    def update_trusses_labels(self):
        """
        Actualiza la visibilidad de las etiquetas de los trusses.
        """
        for text in self.trusses_labels.values():
            text.set_visible(self.plotter_options.UI_member_labels)
        self.figure.canvas.draw_idle()

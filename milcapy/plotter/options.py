from typing import (
    TYPE_CHECKING,
    List,
    Tuple,
    Sequence,
    Union,
    Dict
)

from numpy import mean
if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
class PlotterOptions:        # ✅✅✅
    """Clase que define las opciones de visualización"""

    def __init__(self, model: 'SystemMilcaModel'):
        """
        Inicializa las opciones de visualización con valores predeterminados.
        """
        self.model = model
        # self.values_calculator: GraphicOptionCalculator = graphic_option_calculator
        # Opciones
        # GENERALES
        self.figure_size = (10, 8)      ##
        self.dpi = 100                  ### ✅✅✅
        self.save_fig_dpi = 400         ### ✅✅✅
        self.UI_background_color = 'none' ### ✅✅✅
        self.grid = False                ##
        # 'default', 'dark_background', 'ggplot', etc.
        self.plot_style = 'ggplot'     ##

        # Opciones para visualización de estructura
        # NODOS:
        self.UI_show_nodes = False     # ✅✅✅      #####
        self.node_size = 4              #####
        self.node_color = 'blue'            #####
        # ELEMENTOS:
        self.UI_show_members = True        #####
        self.element_line_width = 2.0   #####
        self.element_color = 'blue'    #####
        # APOYOS:
        self.UI_support = True               ###########
        self.elastic_support_label = True
        self.support_size = 0.5        #####
        self.support_color = '#00ff00'    #####
        # OPCIONES DE SELECCIÓN:
        self.highlight_selected = True
        self.selected_color = 'red'
        # ETIQUETAS DE NODOS Y ELEMENTOS:
        self.UI_node_labels = False              ###########
        self.UI_member_labels = False           ###########
        self.label_font_size = 8                ##########
        self.node_label_color = '#ed0808'        ###########
        self.member_label_color = '#26a699'     ###########


        self.UI_load = True               ###########

        # CARGAS PUNTUALES
        self.point_load = self.UI_load #True               ###########
        self.point_load_color = 'black'           ###########
        self.point_load_length_arrow = 0.2 ############
        self.point_moment_length_arrow = 0.5 ########
        # ETIQUETAS DE CARGAS PUNTUALES
        self.point_load_label = self.UI_load #True                    ##########
        self.point_load_label_color = '#ff0892'         ###########
        self.point_load_label_font_size = 8             ###########

        # CARGAS DISTRIBUIDAS
        self.distributed_load = self.UI_load #True              #######
        self.scale_dist_qload = {}                     ####
        self.scale_dist_pload = {}                     ####
        self.distributed_load_color = '#831f7a'         ########
        # ETIQUETAS DE CARGAS DISTRIBUIDAS
        self.distributed_load_label = self.UI_load #True              #######
        self.distributed_load_label_color = '#511f74'   #######
        self.distributed_load_label_font_size = 8       #######

        self.UI_reactions = False              #######
        self.reactions_color = '#b62ded'           #######

        # DEFORMADA
        self.UI_deformation_scale = {} ########### # Factor de escala para deformaciones
        self.UI_deformed = False ########### # mostrar la deformada
        self.UI_rigid_deformed = False ########### # Color para deformaciones
        self.rigid_deformed_color = '#e81f64' ########### # Color para deformaciones
        self.deformation_line_width = 2.0 ######### # Ancho de línea para deformaciones
        self.deformation_color = 'blue' # #007acc' ########### # Color para deformacione
        # CON ESTOS DATOS DE ACTUALIZA DE FORMA SIN DEFORMAR automatixcamente, y se REVIERTE CON EL BOTON DE DEFORMADA
        self.show_undeformed = True    ########## # Mostrar estructura sin deformar ################################################################################ Undeformed
        self.undeformed_color = '#cccccc'   ######### Color para estructura sin deformar
        self.disp_nodes_decimals = 7


        # ANOTACIONES DE LOS DEZPLAZAMIENTOS EN NODOS
        self.disp_nodes = True    ########## # Mostrar desplazamientos en nodos
        self.disp_nodes_color = 'black'  ########## # Color para desplazamientos en nodos
        self.disp_nodes_font_size = 8    ########## # Tamaño de fuente para desplazamientos en nodos

        # FUERZAS INTERNAS
        self.moment_on_tension_side = True     # (C| ---|Ɔ)
        self.internal_forces_label = False                                                        ############################################################################### LABEL FI
        self.internal_forces_line_width = 1.0           # Ancho de línea de contorno para diagramas de esfuerzos
        self.UI_axial = False
        self.axial_scale = {}
        self.UI_shear = False
        self.shear_scale = {}
        self.UI_moment = False
        self.moment_scale = {}
        self.UI_slope = False
        self.slope_scale = {}
        self.UI_deflection = False
        self.deflection_scale = {}

        colorOption = 2 # 2 = color, 1 = blanco
        self.internal_forces_ratio_scale = 0.2
        self.positive_fill_color = '#7f7fff' if colorOption == 2 else 'white'
        self.positive_fill_alpha = 1
        self.negative_fill_color = '#ff7f7f' if colorOption == 2 else 'white'
        self.negative_fill_alpha = 1

        # RELLENOS Y CONTORNOS
        self.UI_filling_type = 'Barcolor'      # 'solid', 'barcolor'
        self.alpha_filling = 0.7               # Transparencia para relleno
        self.UI_colormap = 'jet'               # 'jet', 'viridis', 'coolwarm', etc.
        self.UI_show_colorbar = True           # Mostrar barra de colores
        self.border_contour = True             # Mostrar contornos
        self.edge_color = 'black'              # Color de bordes en contornos
        self.alpha = 0.7                       # Transparencia para contornos
        self.num_contours = 20                 # Número de niveles en contornos

        # OPCIONES DE GUARDADO
        self.save_dpi = 300                    # DPI para guardar imágenes
        self.tight_layout = True               # Ajuste automático de layout

        # OTROS:
        self.label_size = 8                    # Tamaño de fuente para etiquetas
        self.relsult_label_size = 8            # Tamaño de fuente para etiquetas de resultados



        # TOPICOS:
        self.end_length_offset = True          # Mostrar brazos en los elementos
        self.end_length_offset_color = 'black'   # Color para brazos en los elementos
        self.end_length_offset_line_width = 3.0  # Ancho de línea para brazos en los elementos

        # DESPLAMIENTOS PRESCRITOS:
        self.disp_pre_color = 'red'
        self.disp_pre_label_color = 'black'
        self.disp_pre_label_font_size = 8
        self.disp_pre_length_width = 2


        #! ELEMENTO CST:
        self.cst_alpha = 0.7
        self.cst_element_line_width = 0.7
        self.cst_element_label_color = 'black'
        self.cst_element_label_font_size = 8

        self.cst_edge_color = '#800000'
        self.cst_face_color = "#ff7f7f"
        self.cst_deformed_color_edge = '#800000'
        self.cst_deformed_color_face = '#ff7f7f'
        self.cst_undeformed_color_edge = '#b6f9ff'
        self.cst_undeformed_color_face = '#e3eaf3'


        #! ELEMENTO MEMBRANA Q3DOF:
        self.membrane_q3dof_alpha = 0.7
        self.membrane_q3dof_element_line_width = 0.7
        self.membrane_q3dof_element_label_color = 'black'
        self.membrane_q3dof_element_label_font_size = 8

        self.membrane_q3dof_edge_color = '#800000'
        self.membrane_q3dof_face_color = "#ff7f7f"
        self.membrane_q3dof_deformed_color_edge = '#800000'
        self.membrane_q3dof_deformed_color_face = '#ff7f7f'
        self.membrane_q3dof_undeformed_color_edge = '#b6f9ff'
        self.membrane_q3dof_undeformed_color_face = '#e3eaf3'

        #! ELEMENTO MEMBRANA Q2DOF:
        self.membrane_q2dof_alpha = 0.7
        self.membrane_q2dof_element_line_width = 0.7
        self.membrane_q2dof_element_label_color = 'black'
        self.membrane_q2dof_element_label_font_size = 8

        self.membrane_q2dof_edge_color = '#800000'
        self.membrane_q2dof_face_color = "#ff7f7f"
        self.membrane_q2dof_deformed_color_edge = '#800000'
        self.membrane_q2dof_deformed_color_face = '#ff7f7f'
        self.membrane_q2dof_undeformed_color_edge = '#b6f9ff'
        self.membrane_q2dof_undeformed_color_face = '#e3eaf3'


        #! ELEMENTO ARMADURA:
        self.truss_color = '#39a532'
        self.truss_label_color = 'black'
        self.truss_label_font_size = 8

        self.truss_deformed_color = '#39a532'
        self.truss_undeformed_color = '#b6f9ff'


        #! FRAME RELEASE:
        self.frame_release_color = '#00ff00'
        self.frame_release_point_size = 20
        self.frame_release_length_offset = 0.07 # 5% de la longitud del miembro



        self.optCargaOP = "prom" # "max", "mean", "prom"



        #! TOLERANCIAS
        self.tol = 1e-5
        self.decimals = 6



        #! MODIFICADORES
        self.mod_support_size = 1.0
        self.mod_scale_internal_forces = 1.0
        self.mod_scale_deformation = 1.0
        self.mod_scale_dist_qload = 1.0
        self.mod_krz_rotation_angle: Union[float, int, Tuple, List, Sequence, Dict] = 135 # Grados | si es lista [cada soporte creado en el orden idn se girará]
        self.mod_rotation_angle_conventional_supports: Dict = {} # Grados
        self.mod_scale_point_load = 1.0




    def _calculate_params_max(self, pattern_name: str):
        """
        Calcula el máximo de los resultados para un patrón.
        """
        length_max = 0
        length_max_mq3dof = 0
        length_max_mq2dof = 0
        length_max_cst = 0
        length_max_truss = 0
        length_min = 100000000000000
        q_max = 0
        p_max = 0
        for member in self.model.members.values():
            load_pt = self.model.load_patterns.get(pattern_name)
            if load_pt:
                dist_load = load_pt.distributed_loads.get(member.id)
                q_m = (abs(dist_load.q_i), abs(dist_load.q_j))
                p_m = (abs(dist_load.p_i), abs(dist_load.p_j))

                q_max = max(q_m) if max(q_m) > q_max else q_max
                p_max = max(p_m) if max(p_m) > p_max else p_max

            length_max = member.length() if member.length() > length_max else length_max
            length_min = member.length() if member.length() < length_min else length_min

        for membrane_q3dof in self.model.membrane_q3dof.values():
            minxx = min(membrane_q3dof.node1.vertex.distance_to(membrane_q3dof.node3.vertex), membrane_q3dof.node2.vertex.distance_to(membrane_q3dof.node4.vertex))
            maxxx = max(membrane_q3dof.node1.vertex.distance_to(membrane_q3dof.node3.vertex), membrane_q3dof.node2.vertex.distance_to(membrane_q3dof.node4.vertex))
            length_max_mq3dof = maxxx if maxxx > length_max_mq3dof else length_max_mq3dof
            length_min = minxx if minxx < length_min else length_min

        for membrane_q2dof in self.model.membrane_q2dof.values():
            maxxx = max(membrane_q2dof.node1.vertex.distance_to(membrane_q2dof.node3.vertex), membrane_q2dof.node2.vertex.distance_to(membrane_q2dof.node4.vertex))
            minxx = min(membrane_q2dof.node1.vertex.distance_to(membrane_q2dof.node3.vertex), membrane_q2dof.node2.vertex.distance_to(membrane_q2dof.node4.vertex))
            length_max_mq2dof = maxxx if maxxx > length_max_mq2dof else length_max_mq2dof
            length_min = minxx if minxx < length_min else length_min

        for cst in self.model.csts.values():
            maxxx = max(cst.node1.vertex.distance_to(cst.node2.vertex), cst.node2.vertex.distance_to(cst.node3.vertex), cst.node3.vertex.distance_to(cst.node1.vertex))
            minxx = min(cst.node1.vertex.distance_to(cst.node2.vertex), cst.node2.vertex.distance_to(cst.node3.vertex), cst.node3.vertex.distance_to(cst.node1.vertex))
            length_max_cst = maxxx if maxxx > length_max_cst else length_max_cst
            length_min = minxx if minxx < length_min else length_min

        for truss in self.model.trusses.values():
            length = truss.length()
            length_max_truss = length if length > length_max_truss else length_max_truss
            length_min = length if length < length_min else length_min

        length = (max(length_max, length_max_mq3dof, length_max_mq2dof, length_max_cst, length_max_truss) + length_min)/2

        return (length,
                q_max,
                p_max)

    def _calculate_params_mean(self, pattern_name: str):
        """
        Calcula la media de los resultados para un patrón.
        """
        length_mean = 0
        length_mean_mq3dof = 0
        length_mean_mq2dof = 0
        length_mean_cst = 0
        length_mean_truss = 0
        q_mean = 0
        p_mean = 0
        n = len(self.model.members) + len(self.model.membrane_q3dof) + len(self.model.membrane_q2dof) + len(self.model.csts) + len(self.model.trusses)
        for member in self.model.members.values():
            load_pt = self.model.load_patterns.get(pattern_name)
            if load_pt:
                dist_load = load_pt.distributed_loads.get(member.id, None)
                if dist_load:
                    q_mean += (abs(dist_load.q_i) + abs(dist_load.q_j))/2
                    p_mean += (abs(dist_load.p_i) + abs(dist_load.p_j))/2

            length_mean += member.length()

        for membrane_q3dof in self.model.membrane_q3dof.values():
            length_mean_mq3dof += mean([membrane_q3dof.node1.vertex.distance_to(membrane_q3dof.node3.vertex), membrane_q3dof.node2.vertex.distance_to(membrane_q3dof.node4.vertex)])

        for membrane_q2dof in self.model.membrane_q2dof.values():
            length_mean_mq2dof += mean([membrane_q2dof.node1.vertex.distance_to(membrane_q2dof.node3.vertex), membrane_q2dof.node2.vertex.distance_to(membrane_q2dof.node4.vertex)])

        for cst in self.model.csts.values():
            length_mean_cst += mean([cst.node1.vertex.distance_to(cst.node2.vertex), cst.node2.vertex.distance_to(cst.node3.vertex), cst.node3.vertex.distance_to(cst.node1.vertex)])

        for truss in self.model.trusses.values():
            length_mean_truss += truss.length()

        length = (length_mean + length_mean_mq3dof + length_mean_mq2dof + length_mean_cst + length_mean_truss)/n if n != 0 else 0
        q_mean = q_mean/n if n != 0 else 0
        p_mean = p_mean/n if n != 0 else 0
        return (length,
                q_mean,
                p_mean)




    def reset(self, pattern_name: str):
        """
        Reinicia todas las opciones a sus valores predeterminados.
        """
        self.__init__(self.model)
        self.UI_label_font_size = self.label_size
        self.point_load_label_font_size = self.label_size
        self.distributed_load_label_font_size = self.label_size
        self.disp_nodes_font_size = self.relsult_label_size
        self.reactions_font_size = self.relsult_label_size
        if self.optCargaOP == "max":
            self.load_max(pattern_name)
        elif self.optCargaOP == "mean":
            self.load_mean(pattern_name)
        elif self.optCargaOP == "prom":
            self.load_max(pattern_name, prom=True)

    def load_mean(self, pattern_name: str):
        """
        Calcula y asigna la media de los resultados para un patrón.
        """
        from numpy import nan, inf
        if self.model.membrane_q3dof != {} or self.model.membrane_q2dof != {} or self.model.csts != {}:
            fact = 0.1
        else:
            fact = 0.7
        val = self._calculate_mean(pattern_name)
        self.support_size = self.mod_support_size*0.10 * val["length_mean"] if round(val["length_mean"], 2) not in [0, None, nan, inf] else 0.4
        self.scale_dist_qload[pattern_name] = self.mod_scale_dist_qload * 0.07 * val["length_mean"] / val["q_mean"] if val["q_mean"] not in [0, None, nan, inf] else 0
        self.scale_dist_pload[pattern_name] = 0.02 * val["length_mean"] / val["p_mean"] if val["p_mean"] not in [0, None, nan, inf] else 0
        self.point_load_length_arrow = self.mod_scale_point_load * 0.15 * val["length_mean"]
        self.point_moment_length_arrow = self.mod_scale_point_load  * 0.075 * val["length_mean"]
        self.axial_scale[pattern_name] =  self.mod_scale_internal_forces * fact * self.internal_forces_ratio_scale * val["length_mean"] / val["axial_mean"] if val["axial_mean"] not in [0, None, nan, inf] else 1
        self.shear_scale[pattern_name] =  self.mod_scale_internal_forces * fact * self.internal_forces_ratio_scale * val["length_mean"] / val["shear_mean"] if val["shear_mean"] not in [0, None, nan, inf] else 1
        self.moment_scale[pattern_name] = self.mod_scale_internal_forces * fact * self.internal_forces_ratio_scale * val["length_mean"] / val["bending_mean"] if val["bending_mean"] not in [0, None, nan, inf] else 1
        self.slope_scale[pattern_name] =  self.mod_scale_internal_forces * fact * self.internal_forces_ratio_scale * val["length_mean"] / val["slope_mean"] if val["slope_mean"] not in [0, None, nan, inf] else 1
        # self.deflection_scale[pattern_name] = 0.15 * val["length_mean"] / val["deflection_mean"]
        self.UI_deformation_scale[pattern_name] = self.mod_scale_deformation * fact * self.internal_forces_ratio_scale * val["length_mean"] / val["deflection_mean"] if val["deflection_mean"] not in [0, None, nan, inf] else 1*self.mod_scale_deformation

        if self.model.members == {} and self.model.membrane_q3dof == {} and self.model.membrane_q2dof == {} and self.model.csts == {}:
            axial_max = 0
            for member in self.model.trusses.values():
                axial_max = max(self.model.results[pattern_name].get_truss_axial_force(member.id)) if max(self.model.results[pattern_name].get_truss_axial_force(member.id)) > axial_max else axial_max
            def_max = max(self.model.results[pattern_name].get_model_displacements())
            self.scale_dist_pload[pattern_name] = 0.05 * val["length_max"] / axial_max if axial_max not in [0, None, nan, inf] else 1
            self.axial_scale[pattern_name] = 0.1 * val["length_max"] / axial_max if axial_max not in [0, None, nan, inf] else 0.001
            self.shear_scale[pattern_name] = 0.1
            self.moment_scale[pattern_name] = 0.1
            self.slope_scale[pattern_name] = 0.03 * val["length_max"] / def_max if def_max not in [0, None, nan, inf] else 1
            self.UI_deformation_scale[pattern_name] = 0.03 * val["length_max"] / def_max if def_max not in [0, None, nan, inf] else 1

    def load_max(self, pattern_name: str, prom: bool = False):
        """
        Calcula y asigna el máximo de los resultados para un patrón.
        """
        from numpy import nan, inf
        if self.model.membrane_q3dof != {} or self.model.membrane_q2dof != {} or self.model.csts != {}: # Si hay elementos de membrana
            fact = 0.3
        else:
            fact = 1
        val = self._calculate_max(pattern_name)
        if prom:
            lmin = val["length_min"]
            lmax = val["length_max"]
            if 0 < lmin < 0.3*lmax:
                val["length_max"] = 0.65 * (lmax + lmin) / 2
            elif 0.3*lmax < lmin < 0.4*lmax:
                val["length_max"] = 0.8 * (lmax + lmin) / 2
            elif 0.4*lmax < lmin < 0.5*lmax:
                val["length_max"] = 0.95 * (lmax + lmin) / 2
            elif 0.5*lmax < lmin < 0.6*lmax:
                val["length_max"] = 1.1 * (lmax + lmin) / 2
            elif 0.6*lmax < lmin < 0.7*lmax:
                val["length_max"] = 1.25 * (lmax + lmin) / 2
            elif 0.7*lmax < lmin < 0.8*lmax:
                val["length_max"] = 1.4 * (lmax + lmin) / 2
            elif 0.8*lmax < lmin < 0.9*lmax:
                val["length_max"] = 1.55 * (lmax + lmin) / 2
            elif 0.9*lmax < lmin < 1*lmax:
                val["length_max"] = 1.7 * (lmax + lmin) / 2
            else:
                val["length_max"] = lmax
        self.support_size = self.mod_support_size*0.10 * val["length_max"] if round(val["length_max"], 2) not in [0, None, nan, inf] else 0.4
        self.scale_dist_qload[pattern_name] = self.mod_scale_dist_qload * 0.15 * val["length_max"] / val["q_max"] if val["q_max"] not in [0, None, nan, inf] else 1
        self.scale_dist_pload[pattern_name] = 0.1 * val["length_max"] if val["length_max"] not in [0, None, nan, inf] else 1
        self.point_load_length_arrow = self.mod_scale_point_load  * 0.15 * val["length_max"]
        self.point_moment_length_arrow = self.mod_scale_point_load  * 0.075 * val["length_max"]
        self.axial_scale[pattern_name] =  self.mod_scale_internal_forces * fact * self.internal_forces_ratio_scale * val["length_max"] / val["axial_max"] if val["axial_max"] not in [0, None, nan, inf] else 1
        self.shear_scale[pattern_name] =  self.mod_scale_internal_forces * fact * self.internal_forces_ratio_scale * val["length_max"] / val["shear_max"] if val["shear_max"] not in [0, None, nan, inf] else 1
        self.moment_scale[pattern_name] = self.mod_scale_internal_forces * fact * self.internal_forces_ratio_scale * val["length_max"] / val["bending_max"] if val["bending_max"] not in [0, None, nan, inf] else 1
        self.slope_scale[pattern_name] =  self.mod_scale_internal_forces * fact * self.internal_forces_ratio_scale * val["length_max"] / val["slope_max"] if val["slope_max"] not in [0, None, nan, inf] else 1
        # self.deflection_scale[pattern_name] = 0.15 * val["length_max"] / val["deflection_max"] if val["deflection_max"] not in [0, None, np.nan] else 0
        self.UI_deformation_scale[pattern_name] = self.mod_scale_deformation * fact * self.internal_forces_ratio_scale * val["length_max"] / val["deflection_max"] if val["deflection_max"] not in [0, None, nan, inf] else 1*self.mod_scale_deformation

        if self.model.members == {} and self.model.csts == {} and self.model.trusses == {}:
            self.UI_deformation_scale[pattern_name] = self.mod_scale_deformation * fact * (0.1) * self.internal_forces_ratio_scale * val["length_max"] if val["length_max"] not in [0, None, nan, inf] else 1*self.mod_scale_deformation



        if self.model.members == {} and self.model.membrane_q3dof == {} and self.model.membrane_q2dof == {} and self.model.csts == {}:
            axial_max = 0
            for member in self.model.trusses.values():
                axial_max = max(self.model.results[pattern_name].get_truss_axial_force(member.id)) if max(self.model.results[pattern_name].get_truss_axial_force(member.id)) > axial_max else axial_max
            def_max = max(self.model.results[pattern_name].get_model_displacements())
            self.scale_dist_pload[pattern_name] = 0.05 * val["length_max"] / axial_max if axial_max not in [0, None, nan, inf] else 1
            self.axial_scale[pattern_name] = 0.1 * val["length_max"] / axial_max if axial_max not in [0, None, nan, inf] else 0.001
            self.shear_scale[pattern_name] = 0.1
            self.moment_scale[pattern_name] = 0.1
            self.slope_scale[pattern_name] = 0.03 * val["length_max"] / def_max if def_max not in [0, None, nan, inf] else 1
            self.UI_deformation_scale[pattern_name] = 0.03 * val["length_max"] / def_max if def_max not in [0, None, nan, inf] else 1

    def _calculate_max(self, pattern_name: str):
        """
        Calcula el máximo de los resultados para un patrón.
        """
        length_max = 0
        length_max_mq3dof = 0
        length_max_mq2dof = 0
        length_max_cst = 0
        length_max_truss = 0
        length_min = 100000000000000
        q_max = 0
        p_max = 0
        deflection_max = 0
        slope_max = 0
        bending_max = 0
        shear_max = 0
        axial_max = 0
        for member, results in zip(self.model.members.values(), self.model.results[pattern_name].members.values()):
            dist_load = member.get_distributed_load(pattern_name)
            q_m = (abs(dist_load.q_i), abs(dist_load.q_j))
            p_m = (abs(dist_load.p_i), abs(dist_load.p_j))
            axial_m = abs(results["axial_forces"]).max()
            shear_m = abs(results["shear_forces"]).max()
            bending_m = abs(results["bending_moments"]).max()
            slope_m = abs(results["slopes"]).max()
            deflection_m = abs(results["deflections"]).max()

            length_max = member.length() if member.length() > length_max else length_max
            length_min = member.length() if member.length() < length_min else length_min
            q_max = max(q_m) if max(q_m) > q_max else q_max
            p_max = max(p_m) if max(p_m) > p_max else p_max
            deflection_max = deflection_m if deflection_m > deflection_max else deflection_max
            slope_max = slope_m if slope_m > slope_max else slope_max
            bending_max = bending_m if bending_m > bending_max else bending_max
            shear_max = shear_m if shear_m > shear_max else shear_max
            axial_max = axial_m if axial_m > axial_max else axial_max

        for membrane_q3dof in self.model.membrane_q3dof.values():
            minxx = min(membrane_q3dof.node1.vertex.distance_to(membrane_q3dof.node3.vertex), membrane_q3dof.node2.vertex.distance_to(membrane_q3dof.node4.vertex))
            maxxx = max(membrane_q3dof.node1.vertex.distance_to(membrane_q3dof.node3.vertex), membrane_q3dof.node2.vertex.distance_to(membrane_q3dof.node4.vertex))
            length_max_mq3dof = maxxx if maxxx > length_max_mq3dof else length_max_mq3dof
            length_min = minxx if minxx < length_min else length_min

        for membrane_q2dof in self.model.membrane_q2dof.values():
            maxxx = max(membrane_q2dof.node1.vertex.distance_to(membrane_q2dof.node3.vertex), membrane_q2dof.node2.vertex.distance_to(membrane_q2dof.node4.vertex))
            minxx = min(membrane_q2dof.node1.vertex.distance_to(membrane_q2dof.node3.vertex), membrane_q2dof.node2.vertex.distance_to(membrane_q2dof.node4.vertex))
            length_max_mq2dof = maxxx if maxxx > length_max_mq2dof else length_max_mq2dof
            length_min = minxx if minxx < length_min else length_min

        for cst in self.model.csts.values():
            maxxx = max(cst.node1.vertex.distance_to(cst.node2.vertex), cst.node2.vertex.distance_to(cst.node3.vertex), cst.node3.vertex.distance_to(cst.node1.vertex))
            minxx = min(cst.node1.vertex.distance_to(cst.node2.vertex), cst.node2.vertex.distance_to(cst.node3.vertex), cst.node3.vertex.distance_to(cst.node1.vertex))
            length_max_cst = maxxx if maxxx > length_max_cst else length_max_cst
            length_min = minxx if minxx < length_min else length_min

        for truss in self.model.trusses.values():
            length = truss.length()
            length_max_truss = length if length > length_max_truss else length_max_truss
            length_min = length if length < length_min else length_min

        return {
            "length_max": max(length_max, length_max_mq3dof, length_max_mq2dof, length_max_cst, length_max_truss),
            "length_min": length_min,
            "q_max": q_max,
            "p_max": p_max,
            "deflection_max": deflection_max,
            "slope_max": slope_max,
            "bending_max": bending_max,
            "shear_max": shear_max,
            "axial_max": axial_max
        }

    def _calculate_mean(self, pattern_name: str):
        """
        Calcula la media de los resultados para un patrón.
        """
        length_mean = 0
        length_mean_mq3dof = 0
        length_mean_mq2dof = 0
        length_mean_cst = 0
        length_mean_truss = 0
        q_mean = 0
        p_mean = 0
        deflection_mean = 0
        slope_mean = 0
        bending_mean = 0
        shear_mean = 0
        axial_mean = 0
        n = len(self.model.members) + len(self.model.membrane_q3dof) + len(self.model.membrane_q2dof) + len(self.model.csts) + len(self.model.trusses)
        for member, results in zip(self.model.members.values(), self.model.results[pattern_name].members.values()):
            dist_load = member.get_distributed_load(pattern_name)
            length_mean += member.length()
            q_mean += (abs(dist_load.q_i) + abs(dist_load.q_j))/2
            p_mean += (abs(dist_load.p_i) + abs(dist_load.p_j))/2
            deflection_mean += abs(results["deflections"]).max()
            slope_mean += abs(results["slopes"]).max()
            bending_mean += abs(results["bending_moments"]).max()
            shear_mean += abs(results["shear_forces"]).max()
            axial_mean += abs(results["axial_forces"]).max()

        for membrane_q3dof in self.model.membrane_q3dof.values():
            length_mean_mq3dof += mean([membrane_q3dof.node1.vertex.distance_to(membrane_q3dof.node3.vertex), membrane_q3dof.node2.vertex.distance_to(membrane_q3dof.node4.vertex)])

        for membrane_q2dof in self.model.membrane_q2dof.values():
            length_mean_mq2dof += mean([membrane_q2dof.node1.vertex.distance_to(membrane_q2dof.node3.vertex), membrane_q2dof.node2.vertex.distance_to(membrane_q2dof.node4.vertex)])

        for cst in self.model.csts.values():
            length_mean_cst += mean([cst.node1.vertex.distance_to(cst.node2.vertex), cst.node2.vertex.distance_to(cst.node3.vertex), cst.node3.vertex.distance_to(cst.node1.vertex)])

        for truss in self.model.trusses.values():
            length_mean_truss += truss.length()
        return {
            "length_mean": (length_mean + length_mean_mq3dof + length_mean_mq2dof + length_mean_cst + length_mean_truss)/n,
            "q_mean": q_mean/n,
            "p_mean": p_mean/n,
            "deflection_mean": deflection_mean/n,
            "slope_mean": slope_mean/n,
            "bending_mean": bending_mean/n,
            "shear_mean": shear_mean/n,
            "axial_mean": axial_mean/n
        }

    def nro_arrows(self, member_id: int):
        """
        Calcula el número de flechas para un miembro.
        """
        dx = self.support_size
        if member_id in self.model.trusses:
            dxx = int(self.model.trusses[member_id].length()/dx)
        else:
            dxx = int(self.model.members[member_id].length()/dx)
        return dxx if dxx > 0 else 1

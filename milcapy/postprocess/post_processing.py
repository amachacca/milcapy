from milcapy.postprocess.segment_member import BeamSeg
from milcapy.postprocess.CST_pp import PP_CST
from milcapy.postprocess import membrane_pp
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.core.results import Results


@dataclass
class PostProcessingOptions:
    """Opciones para el post-procesamiento de resultados estructurales.

    Attributes:
        factor (float): Factor de escala para la visualización de resultados.
        n (int): Número de puntos para discretizar los elementos.
    """
    factor: float
    n: int


class PostProcessing:   # para un solo load pattern
    """Clase para el post-procesamiento de resultados estructurales."""

    def __init__(
        self,
        model: "SystemMilcaModel",
        results: "Results",                 # ya viene con U, R del modelo para el LP activo
        options: "PostProcessingOptions",
        load_pattern_name: str
    ) -> None:
        """
        Inicializa el post-procesamiento para un sistema estructural.

        Args:
        model: Sistema estructural analizado
        results: Resultados del análisis
        options: Opciones de post-procesamiento
        load_pattern_name: Nombre del patrón de carga
    """
        self.model = model
        self.results = results
        self.options = options
        self.load_pattern_name = load_pattern_name

        self.reactions = self.results.model["reactions"]
        self.displacements = self.results.model["displacements"]

    def process_displacements_for_nodes(self) -> None:
        """Almacena las desplazamientos de los nodos en el objeto Results en sistema GLOBAL."""
        for id, node in self.model.nodes.items():
            if node.local_axis is None:
                array_displacements = self.displacements[node.dofs-1]
            if node.local_axis is not None:
                array_displacements = self.displacements[node.dofs-1]
                Tlg = node.local_axis.get_transformation_matrix()
                array_displacements = Tlg @ array_displacements
            self.results.set_node_displacements(id, array_displacements)

    def process_reactions_for_nodes(self) -> None:
        """Almacena las reacciones de los nodos en el objeto Results en sistema GLOBAL."""
        for id, node in self.model.nodes.items():
            if np.all(self.reactions[node.dofs - 1] == 0):
                pass
            else:
                if node.local_axis is None:
                    array_reactions = self.reactions[node.dofs-1]
                if node.local_axis is not None:
                    array_reactions = self.reactions[node.dofs-1]
                    Tlg = node.local_axis.get_transformation_matrix()
                    array_reactions = Tlg @ array_reactions
                self.results.set_node_reactions(id, array_reactions)

    def process_displacements_for_members(self) -> None:
        """Almacena las desplazamientos de los miembros en sistema LOCAL en el objeto Results."""
        for id, member in self.model.members.items():
            global_displacements = np.hstack((self.results.get_node_displacements(member.node_i.id), self.results.get_node_displacements(member.node_j.id)))
            array_displacements = np.dot(member.transformation_matrix(), global_displacements)
            self.results.set_member_displacements(id, array_displacements)

    def process_internal_forces_for_members(self) -> None:
        """Almacena las fuerzas internas de los miembros en sistema LOCAL en el objeto Results."""
        for id, member in self.model.members.items():
            local_displacements = self.results.get_member_displacements(id)
            load_vector = member.local_load_vector()
            stiffness_matrix = member.local_stiffness_matrix()
            array_internal_forces = np.dot(stiffness_matrix, local_displacements) - load_vector
            self.results.set_member_internal_forces(id, array_internal_forces)

    def process_displacements_for_cst(self) -> None:
        for id, cst in self.model.csts.items():
            global_displacements = np.hstack((self.results.get_node_displacements(cst.node1.id)[:2], self.results.get_node_displacements(cst.node2.id)[:2], self.results.get_node_displacements(cst.node3.id)[:2]))
            self.results.set_cst_displacements(id, global_displacements)

    def process_displacements_for_membrane_q3dof(self) -> None:
        for id, membrane_q3dof in self.model.membrane_q3dof.items():
            global_displacements = np.hstack((self.results.get_node_displacements(membrane_q3dof.node1.id), self.results.get_node_displacements(membrane_q3dof.node2.id), self.results.get_node_displacements(membrane_q3dof.node3.id), self.results.get_node_displacements(membrane_q3dof.node4.id)))
            self.results.set_membrane_q3dof_displacements(id, global_displacements)

    def process_displacements_for_membrane_q2dof(self) -> None:
        for id, membrane_q2dof in self.model.membrane_q2dof.items():
            global_displacements = np.hstack((self.results.get_node_displacements(membrane_q2dof.node1.id)[:2], self.results.get_node_displacements(membrane_q2dof.node2.id)[:2], self.results.get_node_displacements(membrane_q2dof.node3.id)[:2], self.results.get_node_displacements(membrane_q2dof.node4.id)[:2]))
            self.results.set_membrane_q2dof_displacements(id, global_displacements)

    def process_displacements_for_trusses(self) -> None:
        for id, truss in self.model.trusses.items():
            global_displacements = np.hstack((self.results.get_node_displacements(truss.node_i.id)[:2], self.results.get_node_displacements(truss.node_j.id)[:2]))
            Tlg = truss.transformation_matrix()
            local_displacements = Tlg @ global_displacements
            self.results.set_truss_displacements(id, local_displacements)

    def process_internal_forces_for_trusses(self) -> None:
        for id, truss in self.model.trusses.items():
            local_displacements = self.results.get_truss_displacements(id)
            load_vector = truss.q_local()
            stiffness_matrix = truss.local_stiffness_matrix()
            array_internal_forces = np.dot(stiffness_matrix, local_displacements) - load_vector
            self.results.set_truss_internal_forces(id, array_internal_forces)


    def post_process_for_members(self) -> None:
        """Almacena todos los resultados para cada miembro en el objeto Results (solo parte flexible)."""

        n = self.options.n
        calculator = BeamSeg()

        for id, member in self.model.members.items():

            result = self.results.get_results_member(id)
            calculator.process_builder(member, result, self.load_pattern_name)
            calculator.coefficients()

            if member.la or member.lb:
                # si hay end length offsets
                la = member.la or 0
                lb = member.lb or 0
                L = member.length()
                x_val = np.linspace(la, L-lb, n - 4)
                dx = L/100000
                x_val = np.hstack(([0, la-dx], x_val, [L-lb + dx, L]))
                self.results.set_x_val(id, x_val)
            else:
                x_val = np.linspace(0, member.length(), n)
                self.results.set_x_val(id, x_val)

            array_axial_force = np.zeros(n)
            array_axial_displacement = np.zeros(n)
            array_shear_force = np.zeros(n)
            array_bending_moment = np.zeros(n)
            array_slope = np.zeros(n)
            array_deflection = np.zeros(n)

            for i, x in enumerate(x_val):
                # Calcular fuerzas axiales
                array_axial_force[i] = calculator.axial(x)

                # Calcular los dezplazamientos axiales
                array_axial_displacement[i] = calculator.axial_displacement(x)

                # Calcular fuerzas cortantes
                array_shear_force[i] = calculator.shear(x)

                # Calcular momentos de flexión
                array_bending_moment[i] = calculator.moment(x)

                # Calcular pendientes
                array_slope[i] = calculator.slope(x)

                # Calcular deflexiones
                array_deflection[i] = calculator.deflection(x)

            self.results.set_member_axial_force(id, array_axial_force)
            self.results.set_member_axial_displacement(id, array_axial_displacement)
            self.results.set_member_shear_force(id, array_shear_force)
            self.results.set_member_bending_moment(id, array_bending_moment)
            self.results.set_member_slope(id, array_slope)
            self.results.set_member_deflection(id, array_deflection)


    def post_process_for_cst(self) -> None:
        """Almacena deformaciones y tensiones (centroide + réplica nodal)."""
        calculator = PP_CST()

        for id, cst in self.model.csts.items():
            displacements = self.results.get_cst_displacements(id)
            calculator.process_builder(cst, displacements)

            strains = calculator.strains()
            stresses = calculator.stresses()
            self.results.set_cst_strains(id, strains)
            self.results.set_cst_stresses(id, stresses)
            # Réplica nodal (3,3) para consumo uniforme del FieldService
            self.results.CST[id]["strains_nodes"] = np.tile(np.asarray(strains).ravel()[:3], (3, 1))
            self.results.CST[id]["stresses_nodes"] = np.tile(np.asarray(stresses).ravel()[:3], (3, 1))

    def post_process_for_membrane_q3dof(self) -> None:
        """Tensiones/deformaciones en Gauss extrapoladas a los 4 nodos (Q6 / MQ6IMod)."""
        from milcapy.elements.MQ6IMod import MembraneQuad6IMod
        from milcapy.elements.MQ6 import MembraneQuad6

        for id, ele in self.model.membrane_q3dof.items():
            u = self.results.get_membrane_q3dof_displacements(id)
            if isinstance(ele, MembraneQuad6IMod):
                strains_n, stresses_n, strains_g, stresses_g = membrane_pp.recover_q6imod(ele, u)
            elif isinstance(ele, MembraneQuad6):
                strains_n, stresses_n, strains_g, stresses_g = membrane_pp.recover_q6(ele, u)
            else:  # pragma: no cover - bucket heterogéneo defensivo
                continue
            self.results.set_membrane_q3dof_strains(id, strains_n, strains_g)
            self.results.set_membrane_q3dof_stresses(id, stresses_n, stresses_g)

    def post_process_for_membrane_q2dof(self) -> None:
        """Tensiones/deformaciones en Gauss extrapoladas a los 4 nodos (Q4/Q6I/Q8)."""
        from milcapy.elements.quad4 import MembraneQuad4
        from milcapy.elements.MQ6I import MembraneQuad6I
        from milcapy.elements.quad8 import MembraneQuad8

        for id, ele in self.model.membrane_q2dof.items():
            u = self.results.get_membrane_q2dof_displacements(id)
            if isinstance(ele, MembraneQuad4):
                strains_n, stresses_n, strains_g, stresses_g = membrane_pp.recover_q4(ele, u)
            elif isinstance(ele, MembraneQuad6I):
                strains_n, stresses_n, strains_g, stresses_g = membrane_pp.recover_q6i(ele, u)
            elif isinstance(ele, MembraneQuad8):
                strains_n, stresses_n, strains_g, stresses_g = membrane_pp.recover_q8(ele, u)
            else:  # pragma: no cover - bucket heterogéneo defensivo
                continue
            self.results.set_membrane_q2dof_strains(id, strains_n, strains_g)
            self.results.set_membrane_q2dof_stresses(id, stresses_n, stresses_g)

    def post_process_for_trusses(self) -> None:
        """Almacena todos los resultados para cada truss en el objeto Results."""

        n = self.options.n
        calculator = BeamSeg()

        for id, truss in self.model.trusses.items():
            results = self.results.get_results_truss(id)
            calculator.process_builder_for_truss(truss, results, self.load_pattern_name)
            calculator.coefficients()
            x_val = np.linspace(0, truss.length(), n)
            array_axial_force = np.zeros(n)
            array_axial_displacement = np.zeros(n)
            for i, x in enumerate(x_val):
                array_axial_force[i] = calculator.axial(x)
                array_axial_displacement[i] = calculator.axial_displacement(x)


            self.results.set_truss_axial_force(id, array_axial_force)
            self.results.set_truss_axial_displacement(id, array_axial_displacement)

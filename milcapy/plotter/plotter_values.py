from typing import Dict, Tuple, List, TYPE_CHECKING
import numpy as np
from milcapy.postprocess.segment_member import deformed_shape
from milcapy.analysis.linear_static import LinearStaticAnalysis
from milcapy.utils import rotation_matrix

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel

class PlotterValues:
    """
    Clase que prepara los valores calculados para ser representados gráficamente.
    Mantiene una referencia a datos estáticos compartidos entre todas las instancias.
    """

    _static_data: dict = None

    @classmethod
    def initialize_static_data(cls, system: 'SystemMilcaModel') -> None:
        """
        Inicializa los datos estáticos de la estructura una sola vez.

        los datos son de la forma:
        {
            'nodes': {node_id: (x, y)},
            'members': {member_id: ((x1, y1), (x2, y2))},
            'trusses': {member_id: ((x1, y1), (x2, y2))},
            'restraints': {node_id: (bool, bool, bool)}
        }
        """
        if cls._static_data is None:
            # Coordenadas de nodos
            nodes = {node.id: (node.vertex.x, node.vertex.y)
                    for node in system.nodes.values()}

            # Coordenadas de elementos
            members = {}
            for member in system.members.values():
                node_i = member.node_i
                node_j = member.node_j
                members[member.id] = (
                    (node_i.vertex.x, node_i.vertex.y),
                    (node_j.vertex.x, node_j.vertex.y)
                )

            # Coordenadas de trusses
            trusses = {}
            for truss in system.trusses.values():
                node_i = truss.node_i
                node_j = truss.node_j
                trusses[truss.id] = (
                    (node_i.vertex.x, node_i.vertex.y),
                    (node_j.vertex.x, node_j.vertex.y)
                )

            # Restricciones de nodos
            restraints = {}
            for node in system.nodes.values():
                if node.restraints != (False, False, False):
                    restraints[node.id] = node.restraints

            cls._static_data = {
                'nodes': nodes,
                'members': members,
                'trusses': trusses,
                'restraints': restraints
            }

    def __init__(
        self,
        model: 'SystemMilcaModel',
        current_load_pattern: str,
    ) -> None:
        """
        Inicializa los valores para un load pattern específico.

        Args:
            system: Sistema estructural analizado
            current_load_pattern: Nombre del load pattern
        """
        # Inicializar datos estáticos si aún no se ha hecho
        self.initialize_static_data(model)

        self.model = model
        self.current_load_pattern = current_load_pattern
        self.results = model.results[current_load_pattern]
        self.load_pattern = model.load_patterns.get(current_load_pattern, None)
        self.distributed_loads = {} # {member_id: {member_id: {qi, qj, pi, pj, mi, mj}}}
        self.point_loads       = {} # {node_id: {node_id: {fx, fy, mz}}}
        self.prescribed_dofs   = {} # {node_id: PrescribedDOF}
        self.elastic_supports  = {} # {node_id: ElasticSupport}
        self.cst               = {} # {cst_id: (coords, displacements)}
        self.membrane_q3dof       = {} # {membrane_q3dof_id: (coords, displacements)}
        self.membrane_q2dof      = {} # {membrane_q2dof_id: (coords, displacements)}
        self.trusses           = {} # {truss_id: (coords, displacements)}
        self._field_cache = {} # {(pattern, field_value): (x, y, tris, vals)}

        if not self.load_pattern:
            raise ValueError(
                f"Load pattern con nombre '{current_load_pattern}' no encontrado")

        # Procesamiento de datos dinámicos específicos del load pattern
        self._process_load_data()
        self._process_prescribed_dofs()
        self._process_elastic_supports()
        self._process_cst_element()
        self._process_membrane_q3dof()
        self._process_membrane_q2dof()
        self._process_trusses()

    def _process_load_data(self) -> None:
        """Procesa los datos de cargas para el load pattern actual."""
        # Cargas distribuidas
        for member_id, loads in self.load_pattern.distributed_loads.items():
            self.distributed_loads[member_id] = loads.to_dict()

        # Cargas puntuales
        for node_id, loads in self.load_pattern.point_loads.items():
            self.point_loads[node_id] = loads.to_dict()

    # Propiedades para acceder a datos estáticos
    @property
    def nodes(self) -> Dict[int, Tuple[float, float]]:
        """Devuelve las coordenadas de los nodos."""
        return self._static_data['nodes']

    @property
    def members(self) -> Dict[int, List[Tuple[float, float]]]:
        """Devuelve las coordenadas de los miembros."""
        return self._static_data['members']

    @property
    def restraints(self) -> Dict[int, Tuple[bool, bool, bool]]:
        """Devuelve las restricciones de los nodos."""
        return self._static_data['restraints']

    def rigid_deformed(self, member_id: int, escale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        member = self.model.members[member_id]
        MT = member.transformation_matrix()
        arraydisp = self.results.members[member_id]['displacements']
        arraydisp = np.dot(MT.T, arraydisp)
        x_val = np.array([
            member.node_i.vertex.x + arraydisp[0] * escale,
            member.node_j.vertex.x + arraydisp[3] * escale
        ])
        y_val = np.array([
            member.node_i.vertex.y + arraydisp[1] * escale,
            member.node_j.vertex.y + arraydisp[4] * escale
        ])
        return x_val, y_val

    def get_deformed_shape(self, member_id: int, escale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:

        """
        Obtiene la deformada de un miembro en coordenadas globales.
        """

        member = self.model.members[member_id]

        # Obtener la deformada en coordenadas LOCAL
        x_val, y_val = deformed_shape(member, self.results.members[member_id], escale)

        # Rotar el vector de deflexiones
        deformada_local = np.column_stack((x_val, y_val))
        deformada_global = np.dot(deformada_local, rotation_matrix(member.angle_x()).T) + member.node_i.vertex.coordinates

        x_val = deformada_global[:, 0]
        y_val = deformada_global[:, 1]

        return x_val, y_val

    def _process_prescribed_dofs(self) -> None:
        """Mapea los deslazamientos prescritos."""

        for node in self.model.nodes.values():
            PDOF = node.prescribed_dofs.get(self.model.current_load_pattern)
            if PDOF is not None:
                self.prescribed_dofs[node.id] = PDOF

    def _process_elastic_supports(self) -> None:
        """Mapea los apoyos elásticos."""
        for node in self.model.nodes.values():
            if node.elastic_supports is not None:
                self.elastic_supports[node.id] = node.elastic_supports

    def _process_cst_element(self) -> None:
        """
        Obtiene la coordenadas y desplazamientos de un CST en coordenadas globales.
        """
        for cst in self.model.csts.values():
            coordinates =  np.concatenate(([cst.node1.vertex.coordinates, cst.node2.vertex.coordinates, cst.node3.vertex.coordinates]))
            displacements = self.model.results[self.current_load_pattern].get_cst_displacements(cst.id)
            self.cst[cst.id] = (coordinates, displacements)

    def _process_membrane_q3dof(self) -> None:
        """
        Obtiene la coordenadas y desplazamientos de un MembraneQ6 en coordenadas globales.
        """
        for membrane_q3dof in self.model.membrane_q3dof.values():
            coordinates =  np.concatenate(([membrane_q3dof.node1.vertex.coordinates, membrane_q3dof.node2.vertex.coordinates, membrane_q3dof.node3.vertex.coordinates, membrane_q3dof.node4.vertex.coordinates]))
            displacements = self.model.results[self.current_load_pattern].get_membrane_q3dof_displacements(membrane_q3dof.id)
            self.membrane_q3dof[membrane_q3dof.id] = (coordinates, displacements)

    def _process_membrane_q2dof(self) -> None:
        """
        Obtiene la coordenadas y desplazamientos de un MembraneQ6i en coordenadas globales.
        """
        for membrane_q2dof in self.model.membrane_q2dof.values():
            coordinates =  np.concatenate(([membrane_q2dof.node1.vertex.coordinates, membrane_q2dof.node2.vertex.coordinates, membrane_q2dof.node3.vertex.coordinates, membrane_q2dof.node4.vertex.coordinates]))
            displacements = self.model.results[self.current_load_pattern].get_membrane_q2dof_displacements(membrane_q2dof.id)
            self.membrane_q2dof[membrane_q2dof.id] = (coordinates, displacements)

    def _process_trusses(self) -> None:
        """
        Obtiene la coordenadas y desplazamientos de un Truss en coordenadas globales.
        """
        for truss in self.model.trusses.values():
            coordinates =  np.concatenate((truss.node_i.vertex.coordinates, truss.node_j.vertex.coordinates))
            displacements = np.concatenate((self.results.get_node_displacements(truss.node_i.id)[:2], self.results.get_node_displacements(truss.node_j.id)[:2]))
            self.trusses[truss.id] = (coordinates, displacements)

    def nodal_field(self, field) -> tuple:
        """Campo nodal promediado (x, y, tris, vals) con caché por patrón+campo.

        Delega el cálculo a ``field_service``; esta clase solo cachea valores
        para presentación.
        """
        from milcapy.postprocess.field_service import nodal_field as _nodal_field
        from milcapy.utils.types import FieldType, to_enum

        key = field.value if isinstance(field, FieldType) else str(field)
        cache_key = (self.current_load_pattern, key)
        if cache_key not in self._field_cache:
            f = field if isinstance(field, FieldType) else to_enum(str(field), FieldType)
            self._field_cache[cache_key] = _nodal_field(self.model, self.current_load_pattern, f)
        return self._field_cache[cache_key]
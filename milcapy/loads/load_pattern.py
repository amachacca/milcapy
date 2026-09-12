from typing import Dict, Optional, TYPE_CHECKING, List, Tuple
import numpy as np
import warnings

from milcapy.loads import PointLoad, DistributedLoad, PrescribedDOF, CSTLoad
from milcapy.utils import CoordinateSystemType, StateType, DirectionType, LoadType

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel


def loads_to_global_system(load: PointLoad, angle: float) -> PointLoad:
    """
    Transforma una carga puntual del sistema local al sistema global.

    Args:
        load: Carga puntual a transformar.
        angle: Ángulo de rotación en radianes.

    Returns:
        PointLoad: Carga transformada al sistema global.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return PointLoad(
        fx=load.fx * cos_a - load.fy * sin_a,
        fy=load.fx * sin_a + load.fy * cos_a,
        mz=load.mz
    )


def distributed_load_to_local_system(
    system: "SystemMilcaModel",
    load_start: float,
    load_end: float,
    csys: CoordinateSystemType,
    direction: DirectionType,
    load_type: LoadType,
    member_id: int
) -> DistributedLoad:
    """
    Transforma una carga distribuida del sistema global al sistema local del miembro.

    Args:
        system: Modelo estructural que contiene el miembro.
        load_start: Valor inicial de la carga distribuida.
        load_end: Valor final de la carga distribuida.
        csys: Sistema de coordenadas de la carga.
        direction: Dirección de aplicación de la carga.
        load_type: Tipo de carga (fuerza o momento).
        member_id: Identificador del miembro.

    Returns:
        DistributedLoad: Carga transformada al sistema local del miembro.

    Raises:
        ValueError: Si la dirección especificada no es válida.
    """
    if csys == CoordinateSystemType.LOCAL:
        if load_type == LoadType.FORCE:
            if direction == DirectionType.LOCAL_1:
                return DistributedLoad(
                    q_i=0, q_j=0,
                    p_i=load_start, p_j=load_end,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.LOCAL_2:
                return DistributedLoad(
                    q_i=load_start, q_j=load_end,
                    p_i=0, p_j=0,
                    m_i=0, m_j=0
                )
            else:
                raise ValueError(f"Dirección de carga no válida: {direction}")
        elif load_type == LoadType.MOMENT:
            if direction == DirectionType.LOCAL_3:
                return DistributedLoad(
                    q_i=0, q_j=0,
                    p_i=0, p_j=0,
                    m_i=load_start, m_j=load_end
                )
            else:
                raise ValueError(f"Dirección de carga no válida para momento: {direction}")

    elif csys == CoordinateSystemType.GLOBAL:
        angle = system.members.get(member_id).angle_x()
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        li, lj = load_start, load_end

        if load_type == LoadType.FORCE:
            if direction == DirectionType.X:
                return DistributedLoad(
                    q_i=-li * sin_a, q_j=-lj * sin_a,
                    p_i=li * cos_a, p_j=lj * cos_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.Y:
                return DistributedLoad(
                    q_i=li * cos_a, q_j=lj * cos_a,
                    p_i=li * sin_a, p_j=lj * sin_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.GRAVITY:
                return DistributedLoad(
                    q_i=-li * cos_a, q_j=-lj * cos_a,
                    p_i=-li * sin_a, p_j=-lj * sin_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.X_PROJ:
                return DistributedLoad(
                    q_i=-li * sin_a * sin_a, q_j=-lj * sin_a * sin_a,
                    p_i=li * cos_a * sin_a, p_j=lj * cos_a * sin_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.Y_PROJ:
                return DistributedLoad(
                    q_i=li * cos_a * cos_a, q_j=lj * cos_a * cos_a,
                    p_i=li * sin_a * cos_a, p_j=lj * sin_a * cos_a,
                    m_i=0, m_j=0
                )
            elif direction == DirectionType.GRAVITY_PROJ:
                return DistributedLoad(
                    q_i=-li * cos_a * cos_a, q_j=-lj * cos_a * cos_a,
                    p_i=-li * sin_a * cos_a, p_j=-lj * sin_a * cos_a,
                    m_i=0, m_j=0
                )
            else:
                raise ValueError(f"Dirección de carga no válida: {direction}")

        elif load_type == LoadType.MOMENT and direction == DirectionType.MOMENT:
            return DistributedLoad(
                q_i=0, q_j=0,
                p_i=0, p_j=0,
                m_i=li, m_j=lj
            )
        else:
            raise ValueError(f"Combinación no válida de tipo de carga y dirección: {load_type}, {direction}")


from dataclasses import dataclass, field

@dataclass
class SelfWeightControl:
    multiplier: int = None
    excluded_elements: list[int] = field(default_factory=list)



class LoadPattern:
    """
    Representa un patrón de carga en un modelo estructural.

    Esta clase gestiona la aplicación y transformación de cargas puntuales y distribuidas
    en un sistema estructural, permitiendo su manipulación en diferentes sistemas de coordenadas.
    """

    def __init__(
        self,
        name: str,
        self_weight_multiplier: float|None = None,
        state: StateType = StateType.ACTIVE,
        system: Optional["SystemMilcaModel"] = None,
    ) -> None:
        """
        Inicializa un nuevo patrón de carga con los parámetros especificados.

        Args:
            name: Nombre identificativo del patrón de carga.
            self_weight_multiplier: Factor multiplicador para el peso propio.
            state: Estado inicial del patrón de carga.
            system: Sistema estructural al que pertenece el patrón de carga.

        Raises:
            ValueError: Si el nombre está vacío o el multiplicador es negativo.
        """
        if not name.strip():
            raise ValueError("El nombre del patrón de carga no puede estar vacío")
        if self_weight_multiplier is not None and self_weight_multiplier < 0:
            raise ValueError("El multiplicador de peso propio no puede ser negativo")

        self._system = system

        self.name = name
        # self.self_weight_multiplier: Tuple[int, List[int]] = (self_weight_multiplier, list([]))
        self.self_weight = SelfWeightControl(self_weight_multiplier)


        self.state = state
        self.analyzed = False

        self.point_loads: Dict[int, PointLoad] = {}
        self.distributed_loads: Dict[int, DistributedLoad] = {}
        self.prescribed_dof: Dict[int, PrescribedDOF] = {}
        self.cst_loads: Dict[int, CSTLoad] = {}

    def add_point_load(
        self,
        node_id: int,
        forces: PointLoad,
        csys: CoordinateSystemType = CoordinateSystemType.GLOBAL,
        angle_rot: Optional[float] = None,
        replace: bool = False,
    ) -> None:
        """
        Agrega o actualiza una carga puntual en un nodo específico.

        Args:
            node_id: Identificador del nodo objetivo.
            forces: Carga puntual a aplicar.
            csys: Sistema de coordenadas de la carga.
            angle_rot: Ángulo de rotación en radianes (solo para sistema LOCAL).
            replace: Si True, reemplaza cualquier carga existente en el nodo.

        Raises:
            TypeError: Si forces no es del tipo PointLoad.
            ValueError: Si el sistema de coordenadas es inválido o el ángulo es incorrecto.
        """
        if not isinstance(forces, PointLoad):
            raise TypeError("forces debe ser una instancia de PointLoad")

        if csys not in (CoordinateSystemType):
            raise ValueError("Sistema de coordenadas debe ser GLOBAL o LOCAL")

        transformed_forces = forces
        if csys == CoordinateSystemType.GLOBAL:
            if angle_rot is not None:
                warnings.warn("El ángulo de rotación se ignora en sistema GLOBAL")
        elif csys == CoordinateSystemType.LOCAL:
            if angle_rot is None:
                raise ValueError("Se debe indicar el ángulo de rotación en sistema LOCAL")
            transformed_forces = loads_to_global_system(forces, angle_rot)

        if replace:
            self.point_loads[node_id] = transformed_forces
        else:
            existing_load = self.point_loads.get(node_id, PointLoad())
            self.point_loads[node_id] = existing_load + transformed_forces

    def add_distributed_load(
        self,
        member_id: int,
        load_start: float,
        load_end: float,
        csys: CoordinateSystemType = CoordinateSystemType.LOCAL,
        direction: DirectionType = DirectionType.LOCAL_2,
        load_type: LoadType = LoadType.FORCE,
        replace: bool = False,
    ) -> None:
        """
        Agrega o actualiza una carga distribuida en un miembro específico.

        Args:
            member_id: Identificador del miembro objetivo.
            load_start: Valor inicial de la carga distribuida.
            load_end: Valor final de la carga distribuida.
            csys: Sistema de coordenadas de la carga.
            direction: Dirección de aplicación de la carga.
            load_type: Tipo de carga (fuerza o momento).
            replace: Si True, reemplaza cualquier carga existente en el miembro.

        Raises:
            ValueError: Si el sistema de coordenadas es inválido.
        """
        if self._system is None:
            raise ValueError("Se requiere un sistema válido para transformar cargas distribuidas")

        transformed_load = distributed_load_to_local_system(
            system=self._system,
            load_start=load_start,
            load_end=load_end,
            csys=csys,
            direction=direction,
            load_type=load_type,
            member_id=member_id
        )

        if replace:
            self.distributed_loads[member_id] = transformed_load
        else:
            existing_load = self.distributed_loads.get(member_id, DistributedLoad())
            self.distributed_loads[member_id] = existing_load + transformed_load

    def add_prescribed_dof(
        self,
        node_id: int,
        prescribed_dof: PrescribedDOF,
        replace: bool = False,
    ) -> None:
        """
        Agrega o actualiza un desplazamiento prescindido en un nodo específico.

        Args:
            node_id: Identificador del nodo objetivo.
            prescribed_dof: Desplazamiento prescindido a aplicar.
            replace: Si True, reemplaza cualquier desplazamiento prescindido existente en el nodo.

        Raises:
            TypeError: Si prescribed_dof no es del tipo PrescribedDOF.
        """
        if not isinstance(prescribed_dof, PrescribedDOF):
            raise TypeError("prescribed_dof debe ser una instancia de PrescribedDOF")

        if replace:
            self.prescribed_dof[node_id] = prescribed_dof
        else:
            existing_dof = self.prescribed_dof.get(node_id, PrescribedDOF())
            self.prescribed_dof[node_id] = existing_dof + prescribed_dof

    def add_cst_uniform_temperature_load(self, cst_id: int, dt: float) -> None:
        """
        agrega un incremento de temperatura al elemento CST
        """
        if self._system is None:
            raise ValueError("Se requiere un sistema válido para transformar cargas distribuidas")

        cst = self._system.csts.get(cst_id)
        if cst is None:
            raise ValueError(f"CST {cst_id} no encontrado en el sistema")

        if self.cst_loads.get(cst_id) is None:
            self.cst_loads[cst_id] = self.__create_cst_load(cst_id)

        self.cst_loads[cst_id].add_uniform_temperature_load(dt)

    def add_cst_uniform_distributed_load(self, cst_id: int, qx: float, qy: float) -> None:
        """
        agrega una carga distribuida uniforme superficial en su plano del elemento CST
        """
        if self._system is None:
            raise ValueError("Se requiere un sistema válido para transformar cargas distribuidas")

        cst = self._system.csts.get(cst_id)
        if cst is None:
            raise ValueError(f"CST {cst_id} no encontrado en el sistema")

        if self.cst_loads.get(cst_id) is None:
            self.cst_loads[cst_id] = self.__create_cst_load(cst_id)

        self.cst_loads[cst_id].add_uniform_distributed_load(qx, qy)

    def add_cst_uniform_edge_load(self, cst_id: int, q: float, edge: int) -> None:
        """
        agrega una carga distribuida uniforme en el borde del elemento CST
        """
        if self._system is None:
            raise ValueError("Se requiere un sistema válido para transformar cargas distribuidas")

        cst = self._system.csts.get(cst_id)
        if cst is None:
            raise ValueError(f"CST {cst_id} no encontrado en el sistema")

        if self.cst_loads.get(cst_id) is None:
            self.cst_loads[cst_id] = self.__create_cst_load(cst_id)

        self.cst_loads[cst_id].add_uniform_edge_load(q, edge)

    def add_cst_linear_edge_load(self, cst_id: int, qi: float, qj: float, edge: int) -> None:
        """
        agrega una carga distribuida lineal en el borde del elemento CST
        """
        if self._system is None:
            raise ValueError("Se requiere un sistema válido para transformar cargas distribuidas")

        cst = self._system.csts.get(cst_id)
        if cst is None:
            raise ValueError(f"CST {cst_id} no encontrado en el sistema")

        if self.cst_loads.get(cst_id) is None:
            self.cst_loads[cst_id] = self.__create_cst_load(cst_id)

        self.cst_loads[cst_id].add_linear_edge_load(qi, qj, edge)

    def __create_cst_load(self, cst_id: int) -> CSTLoad:
        """
        Crea un objeto CSTLoad para el elemento CST especificado.
        """
        if self._system is None:
            raise ValueError("Se requiere un sistema válido para transformar cargas distribuidas")

        cst = self._system.csts.get(cst_id)
        if cst is None:
            raise ValueError(f"CST {cst_id} no encontrado en el sistema")

        cd1 = cst.node1.vertex.coordinates
        cd2 = cst.node2.vertex.coordinates
        cd3 = cst.node3.vertex.coordinates
        E = cst.section.E()
        v = cst.section.v()
        t = cst.section.t
        beta = cst.section.beta()
        return CSTLoad(np.array([cd1, cd2, cd3]), E, v, t, beta)

    def assign_loads_to_nodes(self) -> None:
        """
        Asigna las cargas puntuales a los nodos correspondientes del sistema.
        """
        for node_id, load in self.point_loads.items():
            node = self._system.nodes.get(node_id)
            if node:
                node.set_load(load)
            else:
                warnings.warn(f"Nodo {node_id} no encontrado en el sistema")

    def assign_loads_to_members(self) -> None:
        """
        Asigna las cargas distribuidas a los miembros correspondientes del sistema.
        """

        for member_id, load in self.distributed_loads.items():
            members = self._system.members | self._system.trusses
            member = members.get(member_id)
            if member:
                member.set_distributed_load(load)
            else:
                warnings.warn(f"Miembro {member_id} no encontrado en el sistema")

        for cst_id, load in self.cst_loads.items():
            cst = self._system.csts.get(cst_id)
            if cst:
                cst.set_load(load)
            else:
                warnings.warn(f"CST {cst_id} no encontrado en el sistema")

    def assign_prescribed_dofs_to_nodes(self) -> None:
        """
        Asigna los desplazamientos prescindidos a los nodos correspondientes del sistema.
        """
        for node_id, dof in self.prescribed_dof.items():
            node = self._system.nodes.get(node_id)
            if node:
                node.set_prescribed_dof(dof)
            else:
                warnings.warn(f"Nodo {node_id} no encontrado en el sistema")


    def add_self_weight(self) -> None:
        if self.self_weight.multiplier:
            for member in self._system.members.values():
                if member.id not in self.self_weight.excluded_elements:
                    self._system.add_distributed_load(
                        member_id=member.id,
                        load_pattern_name=self.name,
                        load_start=member.section.A()/member.section.kA * member.section.g() * self.self_weight.multiplier,
                        load_end=member.section.A()/member.section.kA * member.section.g() * self.self_weight.multiplier,
                        CSys="GLOBAL",
                        direction="GRAVITY",
                        load_type="FORCE",
                    )

    def set_no_weight(self, ele_id: int) -> None:
        self.self_weight.excluded_elements.append(ele_id)
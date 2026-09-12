from typing import Dict, Optional
from milcapy.core.node import Node
from milcapy.section.section import ShellSection
import numpy as np
from milcapy.loads.load import CSTLoad
from milcapy.utils.types import ConstitutiveModelType

class MembraneTriangle:
    """
    Elemento de membrana triangular (elemento mas basico de FEM, constante).
    """
    def __init__(
        self,
        id: int,
        node1: Node,
        node2: Node,
        node3: Node,
        section: ShellSection,
        state: ConstitutiveModelType,
        ) -> None:
        """
        Inicializa el elemento de membrana triangular.

        Args:
            id (int): Identificador del elemento.
            node1 (Node): Primer nodo.
            node2 (Node): Segundo nodo.
            node3 (Node): Tercer nodo.
            section (ShellSection): Sección del elemento tipo area (shell).
            state (ConstitutiveModelType): Estado constitutivo del elemento.
        """
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.section = section
        self.state = state
        self.dofs = np.concatenate((node1.dofs[:2], node2.dofs[:2], node3.dofs[:2]))
        self.current_load_pattern: Optional[str] = None
        self.loads: Dict[str, CSTLoad] = {} # {lp_name: CSTLoad}

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna las coordenadas de los nodos del elemento.
        """
        x = np.array([self.node1.vertex.x, self.node2.vertex.x, self.node3.vertex.x])
        y = np.array([self.node1.vertex.y, self.node2.vertex.y, self.node3.vertex.y])
        return x, y

    @property
    def A(self) -> float:
        """
        Area del elemento.
        """
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates

        a1 = x2*y3 - x3*y2
        a2 = x3*y1 - x1*y3
        a3 = x1*y2 - x2*y1

        return (a1 + a2 + a3)/2

    @property
    def B(self) -> np.ndarray:
        """
        Matriz de deformacion del elemento.
        """
        x1, y1 = self.node1.vertex.coordinates
        x2, y2 = self.node2.vertex.coordinates
        x3, y3 = self.node3.vertex.coordinates

        a1 = x2*y3 - x3*y2
        a2 = x3*y1 - x1*y3
        a3 = x1*y2 - x2*y1

        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2

        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        a = (a1 + a2 + a3)/2

        B = (1/(2*a))*np.array([
            [b1, 0,  b2, 0,  b3, 0],
            [0,  c1, 0,  c2, 0,  c3],
            [c1, b1, c2, b2, c3, b3],
        ])

        return B

    @property
    def D(self) -> np.ndarray:
        """
        Matriz constitutiva de esfuerzo plano.
        """
        v = self.section.v()
        E = self.section.E()
        # ! T E N S I O N    C O N S T A N T E
        if self.state == ConstitutiveModelType.PLANE_STRESS:
            k = E/(1-v**2)
            return k*np.array([
                            [1, v, 0],
                            [v, 1, 0],
                            [0, 0, (1-v)/2],
                            ])

        # ! D E F O R M A C I O N    C O N S T A N T E
        if self.state == ConstitutiveModelType.PLANE_STRAIN:
            k = (E*(1-v))/((1 + v)*(1 - 2*v))
            return k * np.array([
                [1,                 v/(1-v), 0],
                [v/(1-v),           1,       0],
                [0,                 0,       (1-2*v)/(2*(1-v))],
            ])

    def global_stiffness_matrix(self) -> np.ndarray:
        """
        Matriz de rigidez del elemento.
        """
        return self.B.T @ self.D.T @ self.B * self.section.t * self.A

    def set_load(self, load: CSTLoad) -> None:
        """
        Asigna una carga al elemento para el patrón de carga actual.
        """
        self.loads[self.current_load_pattern] = load

    def get_load(self) -> Optional[CSTLoad]:
        """
        Obtiene la carga asignada al elemento para el patron de carga actual.
        """
        return self.loads.get(self.current_load_pattern, None)

    def get_load_vector(self) -> Optional[np.ndarray]:
        """
        Obtiene el vector de cargas equivalentes en los nodos para el patrond e carga actual.
        """
        cstLoad = self.get_load()
        return cstLoad.Feq if cstLoad else np.zeros(6)
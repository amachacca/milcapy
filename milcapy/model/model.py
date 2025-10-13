from typing import TYPE_CHECKING, Dict, Optional, Union
from milcapy.material.material import Material, GenericMaterial
from milcapy.section.section import Section, RectangularSection, GenericSection, CircularSection, ShellSection
from milcapy.core.node import Node
from milcapy.core.results import Results
from milcapy.elements.member import Member
from milcapy.elements.truss import TrussElement
from milcapy.elements.CST import MembraneTriangle
from milcapy.elements.MQ6 import MembraneQuad6
from milcapy.elements.MQ6I import MembraneQuad6I
from milcapy.elements.MQ6IMod import MembraneQuad6IMod
from milcapy.elements.quad4 import MembraneQuad4
from milcapy.elements.quad8 import MembraneQuad8
from milcapy.plotter.plotter import Plotter, PlotterOptions
from milcapy.analysis.manager import AnalysisManager
from milcapy.postprocess.post_processing import PostProcessingOptions
from milcapy.loads import LoadPattern, PointLoad, PrescribedDOF, ElasticSupport, LocalAxis
from milcapy.utils.geometry import Vertex
from milcapy.utils.types import (
    CoordinateSystemType,
    StateType,
    MemberType,
    BeamTheoriesType,
    DirectionType,
    ShearCoefficientMethodType,
    LoadType,
    to_enum,
)
from milcapy.utils.types import (
    FieldType,
    InternalForceType,
    ConstitutiveModelType,
    IntegrationType,
    MembraneQuadElementType,
)
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from milcapy.utils.types import Restraints





class SystemMilcaModel:
    """
    Clase que representa el modelo estructural.
    Colecciona:
    - Materiales
    - Secciones
    - Nodos
    - Modelos de elementos finitos
    - Patrones de Carga
    - Resultados
    """

    def __init__(self) -> None:
        """
        Inicializa un nuevo modelo estructural vacío.
        """
        # Propiedades de los elementos [UNIQUE]
        self.materials: Dict[str, Material] = {}
        self.sections: Dict[str, Section] = {}

        # Elementos del modelo [UNIQUE / ADD]
        self.nodes: Dict[int, Node] = {}
        self.members: Dict[int, Member] = {}
        self.csts: Dict[int, MembraneTriangle] = {}
        self.membrane_q3dof: Dict[int, Union[MembraneQuad6, MembraneQuad6IMod]] = {}
        self.membrane_q2dof: Dict[int, Union[MembraneQuad6I, MembraneQuad4, MembraneQuad8]] = {}
        self.trusses: Dict[int, TrussElement] = {}
        # self.membrane_q6imod: Dict[int, MembraneQuad6IMod] = {} $ POR MIENTRAS SE AGREGARA A Q6
        # Patrones de carga con las asiganciones de carga en los miembros y nodos
        # {pattern_name: load pattern}
        self.load_patterns: Dict[str, LoadPattern] = {}
        self.current_load_pattern: Optional[str] = None

        # Coleccion de resultados incluyendo postprocesamiento [ADD]
        self.results: Dict[str, Results] = {}  # {pattern_name: results}

        # Analisis [UNIQUE]
        self.analysis: Optional[AnalysisManager] = None

        # Visualización [UNIQUE]
        self.plotter: Optional[Plotter] = None

        # Opciones del modelo [UNIQUE]
        self.plotter_options: "PlotterOptions" = PlotterOptions(self)
        self.postprocessing_options: "PostProcessingOptions" = PostProcessingOptions(
            factor=1, n=17)

        self.global_stiffness_matrix: Optional[np.ndarray] = None
        self.global_load_vector: Optional[Dict[str, np.ndarray]] = {}

    #! MATERIALES ##########################################################
    def add_material(
        self,
        name: str,
        modulus_elasticity: float,
        poisson_ratio: float,
        specific_weight: float = 0.0
    ) -> Material:
        """
        Agrega un material al modelo.

        Args:
            name (str): Nombre del material.
            modulus_elasticity (float): Módulo de elasticidad (E).
            poisson_ratio (float): Coeficiente de Poisson.
            specific_weight (float, opcional): Peso específico o densidad. Default es 0.0.

        Returns:
            Material: El material creado.

        Raises:
            ValueError: Si ya existe un material con el mismo nombre.
        """
        if name in self.materials:
            raise ValueError(f"Ya existe un material con el nombre '{name}'")

        material = GenericMaterial(
            name, modulus_elasticity, poisson_ratio, specific_weight)
        self.materials[name] = material
        return material

    #! SECCIONES ###########################################################
    def add_rectangular_section(
        self,
        name: str,
        material_name: str,
        base: float,
        height: float
    ) -> Section:
        """
        Agrega una sección rectangular al modelo.

        Args:
            name (str): Nombre de la sección.
            material_name (str): Nombre del material asociado (ya agregado).
            base (float): Base de la sección.
            height (float): Altura de la sección.

        Returns:
            Section: La sección creada.

        Raises:
            ValueError: Si ya existe una sección con el mismo nombre o si no existe el material.
        """
        if name in self.sections:
            raise ValueError(f"Ya existe una sección con el nombre '{name}'")

        if material_name not in self.materials:
            raise ValueError(
                f"No existe un material con el nombre '{material_name}'")

        section = RectangularSection(
            name=name,
            material=self.materials[material_name],
            base=base,
            height=height,
            shear_method=ShearCoefficientMethodType.TIMOSHENKO
        )
        self.sections[name] = section
        return section

    def add_circular_section(
        self,
        name: str,
        material_name: str,
        diameter: float,
    ) -> Section:
        """
        Agrega una sección circular al modelo.

        Args:
            name (str): Nombre de la sección.
            material_name (str): Nombre del material asociado (ya agregado).
            diameter (float): Diámetro de la sección.
        Raises:
            ValueError: Si el diámetro es menor o igual a cero.
        """
        if diameter <= 0:
            raise ValueError("El diámetro debe ser positivo.")
        if material_name not in self.materials:
            raise ValueError(
                f"No existe un material con el nombre '{material_name}'")
        section = CircularSection(
            name=name,
            material=self.materials[material_name],
            radius=diameter/2,
            shear_method=ShearCoefficientMethodType.TIMOSHENKO
        )
        self.sections[name] = section
        return section

    def add_generic_section(
        self,
        name: str,
        material_name: str,
        area: float,
        inertia: float,
        k_factor: float
    ) -> Section:
        """
        Agrega una sección rectangular al modelo.

        Args:
            name (str): Nombre de la sección.
            material_name (str): Nombre del material asociado (ya agregado).
            area (float): Área de la sección.
            inertia (float): Momento de inercia de la sección.
            k_factor (float): Coeficiente de corte de la sección.

        Returns:
            Section: La sección creada.

        Raises:
            ValueError: Si ya existe una sección con el mismo nombre o si no existe el material.
        """
        if name in self.sections:
            raise ValueError(f"Ya existe una sección con el nombre '{name}'")

        if material_name not in self.materials:
            raise ValueError(
                f"No existe un material con el nombre '{material_name}'")

        section = GenericSection(
            name=name,
            material=self.materials[material_name],
            area=area,
            inertia=inertia,
            k_factor=k_factor,
        )
        self.sections[name] = section
        return section

    def add_shell_section(
        self,
        name: str,
        material_name: str,
        thickness: float
    ) -> ShellSection:
        """
        Agrega una sección de cascaras al modelo.

        Args:
            name (str): Nombre de la sección.
            material_name (str): Nombre del material asociado (ya agregado).
            thickness (float): Grosor de la sección.

        Returns:
            ShellSection: La sección creada.

        Raises:
            ValueError: Si ya existe una sección con el mismo nombre o si no existe el material.
        """
        if name in self.sections:
            raise ValueError(f"Ya existe una sección con el nombre '{name}'")

        if material_name not in self.materials:
            raise ValueError(
                f"No existe un material con el nombre '{material_name}'")

        section = ShellSection(
            name=name,
            material=self.materials[material_name],
            thickness=thickness,
        )
        self.sections[name] = section
        return section

    def set_property_modifiers(
        self,
        section_name: str,
        axial_area: Optional[float] = 1,
        shear_area: Optional[float] = 1,
        moment_inertia: Optional[float] = 1,
        weight: Optional[float] = 1
    ) -> None:
        """
        Establece los modificadores de propiedad de una sección.

        Args:
            section_name (str): Nombre de la sección.
            axial_area (float): Modificador de área transversal.
            shear_area (float): Modificador de área de corte.
            moment_inertia (float): Modificador de momento de inercia.
            weight (float): Modificador de peso.
        """
        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        section = self.sections[section_name]
        section.kA = axial_area
        section.kAs = shear_area
        section.kI = moment_inertia
        section.kg = weight

    #! NODOS ###############################################################
    def add_node(
        self,
        id: int,
        x: float,
        y: float
    ) -> Node:
        """
        Agrega un nodo al modelo.

        Args:
            id (int): Identificador del nodo.
            x (float): Coordenada x del nodo.
            y (float): Coordenada y del nodo.

        Returns:
            Node: El nodo creado.

        Raises:
            ValueError: Si ya existe un nodo con el mismo ID.
        """
        if id in self.nodes:
            raise ValueError(f"Ya existe un nodo con el ID {id}")

        node = Node(id, Vertex(x, y))
        self.nodes[id] = node
        return node

    #! ELEMENTOS ###########################################################
    def add_member(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        section_name: str,
        beam_theory: Union[str,
                           BeamTheoriesType] = BeamTheoriesType.TIMOSHENKO,
        # member_type: Union[str, MemberType] = MemberType.FRAME
    ) -> Member:
        """
        Agrega un miembro estructural al modelo, por defecto usa la teoría de Timoshenko para vigas.

        Args:
            id (int): Identificador del miembro.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            beam_theory (str, opcional): Teoría de la viga. Por defecto es Timoshenko.

        Returns:
            Member: El miembro creado.

        Raises:
            ValueError: Si ya existe un miembro con el mismo ID, o si no existen los nodos o la sección.
        """
        self.__id_verifier(id, node_i_id, node_j_id)

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        # # Convertir a enum si es string
        # if isinstance(member_type, str):
        #     member_type = to_enum(member_type, MemberType)
        member_type = MemberType.FRAME

        if isinstance(beam_theory, str):
            beam_theory = to_enum(beam_theory, BeamTheoriesType)

        element = Member(
            id=id,
            node_i=self.nodes[node_i_id],
            node_j=self.nodes[node_j_id],
            section=self.sections[section_name],
            member_type=member_type,
            beam_theory=beam_theory,
        )
        self.members[id] = element
        return element

    def add_elastic_timoshenko_beam(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        section_name: str,
    ) -> Member:
        """
        Agrega una viga de Timoshenko al modelo con rigidez axial.

        Args:
            id (int): Identificador del miembro.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.

        Returns:
            Member: El miembro creado.

        Raises:
            ValueError: Si ya existe un elemento con el mismo ID, o si no existen los nodos o la sección.
        """
        self.__id_verifier(id, node_i_id, node_j_id)

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        beam_theory = BeamTheoriesType.TIMOSHENKO

        element = Member(
            id=id,
            node_i=self.nodes[node_i_id],
            node_j=self.nodes[node_j_id],
            section=self.sections[section_name],
            member_type=MemberType.FRAME,
            beam_theory=beam_theory,
        )
        self.members[id] = element
        return element

    def add_elastic_euler_bernoulli_beam(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        section_name: str,
    ) -> Member:
        """
        Agrega una viga de Euler-Bernoulli al modelo con rigidez axial.

        Args:
            id (int): Identificador del miembro.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.

        Returns:
            Member: El miembro creado.

        Raises:
            ValueError: Si ya existe un elemento con el mismo ID, o si no existen los nodos o la sección.
        """
        self.__id_verifier(id, node_i_id, node_j_id)

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        beam_theory = BeamTheoriesType.EULER_BERNOULLI

        element = Member(
            id=id,
            node_i=self.nodes[node_i_id],
            node_j=self.nodes[node_j_id],
            section=self.sections[section_name],
            member_type=MemberType.FRAME,
            beam_theory=beam_theory,
        )
        self.members[id] = element
        return element

    def add_truss(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        section_name: str
    ) -> TrussElement:
        """
        Agrega un elemento de trusse al modelo.

        Args:
            id (int): Identificador del elemento.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.

        Returns:
            TrussElement: El elemento creado.

        Raises:
            ValueError: Si ya existe un elemento con el mismo ID, o si no existen los nodos o la sección.
        """
        self.__id_verifier(id, node_i_id, node_j_id)

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        TE = TrussElement(
            id=id,
            node_i=self.nodes[node_i_id],
            node_j=self.nodes[node_j_id],
            section=self.sections[section_name],
        )
        self.trusses[id] = TE
        return TE


    #! ELEMENTOS FINITOS ###################################################
    def add_cst(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        node_k_id: int,
        section_name: str,
        state: Union[ConstitutiveModelType, str] = ConstitutiveModelType.PLANE_STRESS,
    ) -> MembraneTriangle:
        """
        Agrega un triángulo de membrana al modelo.

        Args:
            id (int): Identificador del triángulo.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            node_k_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            state (Union[ConstitutiveModelType, str]): Estado constitutivo del triángulo.

        Returns:
            MembraneTriangle: El triángulo creado.

        Raises:
            ValueError: Si ya existe un elemento con el mismo ID, o si no existen los nodos o el material.
            ValueError: Si el estado constitutivo no es 'PLANE_STRESS' o 'PLANE_STRAIN'.
            ValueError: Si los nodos no forman un polígono en sentido antihorario.

        Notes:
            El elemento agregado es un CONSTANT STRAIN TRIANGLE (CST).
            La enumeracion (orden) de los nodos debe ser en sentido antihorario.
        """
        self.__id_verifier(id, node_i_id, node_j_id, node_k_id)

        if self.__is_counterclockwise(node_i_id, node_j_id, node_k_id):
            raise ValueError("Los nodos no forman un triángulo en sentido antihorario")

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        if isinstance(state, str):
            state = to_enum(state, ConstitutiveModelType)
        if state not in [ConstitutiveModelType.PLANE_STRESS, ConstitutiveModelType.PLANE_STRAIN]:
            raise ValueError(
                f"El estado constitutivo debe ser 'PLANE_STRESS' o 'PLANE_STRAIN'")

        cst = MembraneTriangle(
            id=id,
            node1=self.nodes[node_i_id],
            node2=self.nodes[node_j_id],
            node3=self.nodes[node_k_id],
            section=self.sections[section_name],
            state=state,
        )
        self.csts[id] = cst
        return cst

    def add_membrane_q6(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        node_k_id: int,
        node_l_id: int,
        section_name: str,
        state: Union[ConstitutiveModelType, str] = ConstitutiveModelType.PLANE_STRESS,
    ) -> MembraneQuad6:
        """
        Agrega un cuadrilátero de membrana de 6 nodos al modelo con rigidez a la perforación (3dof/nodo).

        Args:
            id (int): Identificador del cuadrilátero.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            node_k_id (int): ID del nodo final.
            node_l_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            state (Union[ConstitutiveModelType, str]): Estado constitutivo del cuadrilátero.

        Returns:
            MembraneQuad6: El cuadrilátero creado.

        Raises:
            ValueError: Si ya existe un cuadrilátero con el mismo ID, o si no existen los nodos o la sección.
            ValueError: Si el estado constitutivo no es 'PLANE_STRESS' o 'PLANE_STRAIN'.
            ValueError: Si los nodos no forman un cuadrilátero en sentido antihorario.

        Notes:
            El elemento agregado es un Q6 el mismo elemento que platea el dr. wilson en su libro "analisis estatico y dinamico de estructuras" membranas con grados de libertad de perforación.
            La enumeracion (orden) de los nodos debe ser en sentido antihorario.
        """
        self.__id_verifier(id, node_i_id, node_j_id, node_k_id, node_l_id)

        if self.__is_counterclockwise(node_i_id, node_j_id, node_k_id, node_l_id):
            raise ValueError("Los nodos no forman un cuadrilátero en sentido antihorario")

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        if isinstance(state, str):
            state = to_enum(state, ConstitutiveModelType)
        if state not in [ConstitutiveModelType.PLANE_STRESS, ConstitutiveModelType.PLANE_STRAIN]:
            raise ValueError(
                f"El estado constitutivo debe ser 'PLANE_STRESS' o 'PLANE_STRAIN'")

        MQ6 = MembraneQuad6(
            id=id,
            node1=self.nodes[node_i_id],
            node2=self.nodes[node_j_id],
            node3=self.nodes[node_k_id],
            node4=self.nodes[node_l_id],
            section=self.sections[section_name],
            state=state,
        )
        self.membrane_q3dof[id] = MQ6
        return MQ6

    def add_membrane_q6i(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        node_k_id: int,
        node_l_id: int,
        section_name: str,
        state: Union[ConstitutiveModelType, str] = ConstitutiveModelType.PLANE_STRESS,
    ) -> MembraneQuad6I:
        """
        Agrega un rectangular de membrana de 4 nodos y con modos incompatibles al modelo, sin rigidez a la perforación.

        Args:
            id (int): Identificador del cuadrilátero.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            node_k_id (int): ID del nodo final.
            node_l_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            state (Union[ConstitutiveModelType, str]): Estado constitutivo del cuadrilátero.

        Returns:
            MembraneQuad6I: El cuadrilátero creado.

        Raises:
            ValueError: Si ya existe un cuadrilátero con el mismo ID, o si no existen los nodos o la sección.
            ValueError: Si el estado constitutivo no es 'PLANE_STRESS' o 'PLANE_STRAIN'.
            ValueError: Si los nodos no forman un cuadrilátero en sentido antihorario.

        Notes:
            El elemento agregado es un Q4I el mismo modelo que usa ETABS, SAP2000.
            La enumeracion (orden) de los nodos debe ser en sentido antihorario.
        """

        self.__id_verifier(id, node_i_id, node_j_id, node_k_id, node_l_id)

        # # verificar si es rectangular
        # if not self.__is_rectangular(node_i_id, node_j_id, node_k_id, node_l_id):
        #     raise ValueError(
        #         "Los nodos no forman un rectángulo en sentido antihorario")

        if self.__is_counterclockwise(node_i_id, node_j_id, node_k_id, node_l_id):
            raise ValueError(
                "Los nodos no forman un rectángulo en sentido antihorario")

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        if isinstance(state, str):
            state = to_enum(state, ConstitutiveModelType)
        if state not in [ConstitutiveModelType.PLANE_STRESS, ConstitutiveModelType.PLANE_STRAIN]:
            raise ValueError(
                f"El estado constitutivo debe ser 'PLANE_STRESS' o 'PLANE_STRAIN'")

        MQ6I = MembraneQuad6I(
            id=id,
            node1=self.nodes[node_i_id],
            node2=self.nodes[node_j_id],
            node3=self.nodes[node_k_id],
            node4=self.nodes[node_l_id],
            section=self.sections[section_name],
            state=state,
        )
        self.membrane_q2dof[id] = MQ6I
        return MQ6I

    def add_membrane_q6i_mod(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        node_k_id: int,
        node_l_id: int,
        section_name: str,
        state: Union[ConstitutiveModelType, str] = ConstitutiveModelType.PLANE_STRESS,
        ele_type: Union[MembraneQuadElementType, str] = MembraneQuadElementType.MQ6I,
    ) -> MembraneQuad6IMod:
        """
        Agrega un cuadrilátero de membrana de 4 nodos y con modos incompatibles al modelo, con rigidez a la perforación.

        Args:
            id (int): Identificador del cuadrilátero.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            node_k_id (int): ID del nodo final.
            node_l_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            state (Union[ConstitutiveModelType, str]): Estado constitutivo del cuadrilátero.
            ele_type (Union[MembraneQuadElementType, str]): Tipo de elemento de membrana.

        Returns:
            MembraneQuad6IMod: El cuadrilátero creado.

        Raises:
            ValueError: Si ya existe un cuadrilátero con el mismo ID, o si no existen los nodos o la sección.
            ValueError: Si los nodos no forman un cuadrilátero en sentido antihorario.
            ValueError: Si los nodos no forman un rectángulo.

        Notes:
            El elemento agregado es un QUAD6I el mismo modelo que usa ETABS, SAP2000.
            La enumeracion (orden) de los nodos debe ser en sentido antihorario.
            OJO: es un modelo rigidez combinada no esta documentado en ningun libro (no confiar)
        """

        self.__id_verifier(id, node_i_id, node_j_id, node_k_id, node_l_id)
        # if not self.__is_counterclockwise(node_i_id, node_j_id, node_k_id, node_l_id):
        #     raise ValueError(
        #         "Los nodos no forman un rectángulo en sentido antihorario")

        # if not self.__is_rectangular(node_i_id, node_j_id, node_k_id, node_l_id):
        #     raise ValueError(
        #         "Los nodos no forman un rectángulo")

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        if isinstance(state, str):
            state = to_enum(state, ConstitutiveModelType)
        if state not in [ConstitutiveModelType.PLANE_STRESS, ConstitutiveModelType.PLANE_STRAIN]:
            raise ValueError(
                f"El estado constitutivo debe ser 'PLANE_STRESS' o 'PLANE_STRAIN'")

        if isinstance(ele_type, str):
            ele_type = to_enum(ele_type, MembraneQuadElementType)
        if ele_type not in [MembraneQuadElementType.MQ4, MembraneQuadElementType.MQ6, MembraneQuadElementType.MQ6I, MembraneQuadElementType.MQ8Reduced, MembraneQuadElementType.MQ8Complete]:
            raise ValueError(
                f"El tipo de elemento debe ser 'MQ4', 'MQ6', 'MQ6I', 'MQ8Reduced' o 'MQ8Complete'")

        MQ6IMod = MembraneQuad6IMod(
            id=id,
            node1=self.nodes[node_i_id],
            node2=self.nodes[node_j_id],
            node3=self.nodes[node_k_id],
            node4=self.nodes[node_l_id],
            section=self.sections[section_name],
            state=state,
            ele_type=ele_type,
        )
        self.membrane_q3dof[id] = MQ6IMod
        return MQ6IMod

    def add_membrane_q4(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        node_k_id: int,
        node_l_id: int,
        section_name: str,
        state: Union[ConstitutiveModelType, str] = ConstitutiveModelType.PLANE_STRESS,
    ) -> MembraneQuad4:
        """
        Agrega un cuadrilátero de membrana de 4 nodos al modelo (2dof/nodo).

        Args:
            id (int): Identificador del cuadrilátero.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            node_k_id (int): ID del nodo final.
            node_l_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            state (Union[ConstitutiveModelType, str], optional): Estado constitutivo del cuadrilátero. Defaults to ConstitutiveModelType.PLANE_STRESS.

        Raises:
            ValueError: Si ya existe un cuadrilátero con el mismo ID, o si no existen los nodos o la sección.
            ValueError: Si los nodos no forman un cuadrilátero en sentido antihorario.
            ValueError: Si los nodos no forman un rectángulo.

        Returns:
            MembraneQuad4: El cuadrilátero creado.

        Notes:
            El elemento agregado es un Q4.
            La enumeracion (orden) de los nodos debe ser en sentido antihorario.
            Tiene muy poca convergencia, recomiendo usar Q8.
        """
        self.__id_verifier(id, node_i_id, node_j_id, node_k_id, node_l_id)

        if self.__is_counterclockwise(node_i_id, node_j_id, node_k_id, node_l_id):
            raise ValueError("Los nodos no forman un cuadrilátero en sentido antihorario")

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        if isinstance(state, str):
            state = to_enum(state, ConstitutiveModelType)
        if state not in [ConstitutiveModelType.PLANE_STRESS, ConstitutiveModelType.PLANE_STRAIN]:
            raise ValueError(
                f"El estado constitutivo debe ser 'PLANE_STRESS' o 'PLANE_STRAIN'")

        MQ4 = MembraneQuad4(
            id=id,
            node1=self.nodes[node_i_id],
            node2=self.nodes[node_j_id],
            node3=self.nodes[node_k_id],
            node4=self.nodes[node_l_id],
            section=self.sections[section_name],
            state=state,
        )
        self.membrane_q2dof[id] = MQ4
        return MQ4

    def add_membrane_q8(
        self,
        id: int,
        node_i_id: int,
        node_j_id: int,
        node_k_id: int,
        node_l_id: int,
        section_name: str,
        state: Union[ConstitutiveModelType, str] = ConstitutiveModelType.PLANE_STRESS,
        integration: Union[IntegrationType, str] = IntegrationType.COMPLETE,
    ) -> MembraneQuad8:
        """
        Agrega un cuadrilátero de membrana de 8 nodos al modelo con rigidez a la perforación (2dof/nodo).

        Args:
            id (int): Identificador del cuadrilátero.
            node_i_id (int): ID del nodo inicial.
            node_j_id (int): ID del nodo final.
            node_k_id (int): ID del nodo final.
            node_l_id (int): ID del nodo final.
            section_name (str): Nombre de la sección asociada.
            state (Union[ConstitutiveModelType, str]): Estado constitutivo del cuadrilátero.
            integration (Union[IntegrationType, str]): Tipo de integración.

        Returns:
            MembraneQuad6: El cuadrilátero creado.

        Raises:
            ValueError: Si ya existe un cuadrilátero con el mismo ID, o si no existen los nodos o la sección.
            ValueError: Si el estado constitutivo no es 'PLANE_STRESS' o 'PLANE_STRAIN'.
            ValueError: Si los nodos no forman un cuadrilátero en sentido antihorario.

        Notes:
            La enumeracion (orden) de los nodos debe ser en sentido antihorario.
        """
        self.__id_verifier(id, node_i_id, node_j_id, node_k_id, node_l_id)

        if self.__is_counterclockwise(node_i_id, node_j_id, node_k_id, node_l_id):
            raise ValueError("Los nodos no forman un cuadrilátero en sentido antihorario")

        if section_name not in self.sections:
            raise ValueError(
                f"No existe una sección con el nombre '{section_name}'")

        if isinstance(state, str):
            state = to_enum(state, ConstitutiveModelType)
        if state not in [ConstitutiveModelType.PLANE_STRESS, ConstitutiveModelType.PLANE_STRAIN]:
            raise ValueError(
                f"El estado constitutivo debe ser 'PLANE_STRESS' o 'PLANE_STRAIN'")

        if isinstance(integration, str):
            integration = to_enum(integration, IntegrationType)
        if integration not in [IntegrationType.COMPLETE, IntegrationType.REDUCED]:
            raise ValueError(
                f"El tipo de integración debe ser 'COMPLETE' o 'REDUCED'")

        MQ8 = MembraneQuad8(
            id=id,
            node1=self.nodes[node_i_id],
            node2=self.nodes[node_j_id],
            node3=self.nodes[node_k_id],
            node4=self.nodes[node_l_id],
            section=self.sections[section_name],
            state=state,
            integration=integration,
        )
        self.membrane_q2dof[id] = MQ8
        return MQ8

    #! CONDICIONES DE FRONTERA #############################################

    def add_restraint(
        self,
        node_id: int,
        ux: bool,
        uy: bool,
        rz: bool,
    ) -> None:
        """
        Asigna restricciones (condiciones de frontera) a un nodo.

        Args:
            node_id (int): Identificador del nodo.
            ux (bool): Restricción de traslación en el eje x.
            uy (bool): Restricción de traslación en el eje y.
            rz (bool): Restricción de rotación en el eje z.

        Raises:
            ValueError: Si no existe el nodo.
        """
        if node_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_id}")
        if not all(isinstance(x, bool) for x in [ux, uy, rz]):
            raise ValueError(
                "ux, uy y rz deben ser booleanos")

        self.nodes[node_id].set_restraints((ux, uy, rz))


    #! PATRONES DE CARGA ###################################################
    def add_load_pattern(
        self,
        name: str,
        self_weight_multiplier: float = 0.0,
        state: Union[str, StateType] = StateType.ACTIVE
    ) -> LoadPattern:
        """
        Agrega un patrón de carga al modelo.

        Args:
            name (str): Nombre del patrón de carga.
            self_weight_multiplier (float, opcional): Multiplicador del peso propio. Default es 0.0.
            state (str o State, opcional): Estado del patrón. Default es ACTIVE.

        Returns:
            LoadPattern: El patrón de carga creado.

        Raises:
            ValueError: Si ya existe un patrón con el mismo nombre.
        """
        if name in self.load_patterns:
            raise ValueError(
                f"Ya existe un patrón de carga con el nombre '{name}'")

        if isinstance(state, str):
            state = to_enum(state, StateType)

        load_pattern = LoadPattern(
            name=name,
            self_weight_multiplier=self_weight_multiplier,
            state=state,
            system=self,
        )
        self.load_patterns[name] = load_pattern
        return load_pattern

    def set_state_load_pattern(
        self,
        load_pattern_name: str,
        state: Union[str, StateType]
    ) -> None:
        """
        Cambia el estado de un patrón de carga.

        Args:
            load_pattern_name (str): Nombre del patrón de carga.
            state (str o State, opcional): Estado del patrón. Default es ACTIVE.

        Raises:
            ValueError: Si no existe el patrón de carga.
        """
        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")

        if isinstance(state, str):
            state = to_enum(state, StateType)

        self.load_patterns[load_pattern_name].state = state

    #! CARGAS ##############################################################
    def add_point_load(
        self,
        node_id: int,
        load_pattern_name: str,
        fx: float = 0.0,
        fy: float = 0.0,
        mz: float = 0.0,
        CSys: Union[str, CoordinateSystemType] = "GLOBAL",
        replace: bool = False
    ) -> None:
        """
        Asigna una carga puntual a un nodo dentro de un patrón de carga.

        Args:
            node_id (int): Identificador del nodo.
            load_pattern_name (str): Nombre del patrón de carga.
            CSys (str o CoordinateSystemType, opcional): Sistema de coordenadas. Default es "GLOBAL".
            fx (float, opcional): Fuerza en X. Default es 0.0.
            fy (float, opcional): Fuerza en Y. Default es 0.0.
            mz (float, opcional): Momento en Z. Default es 0.0.
            angle_rot (float, opcional): Ángulo de rotación en radianes. Default es None.
            replace (bool, opcional): Si se reemplaza la carga existente. Default es False.

        Raises:
            ValueError: Si no existe el nodo o el patrón de carga.
        """
        if node_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_id}")

        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")

        # Convertir a enum si es string
        if isinstance(CSys, str):
            csys_enum = to_enum(CSys, CoordinateSystemType)
        else:
            csys_enum = CSys

        if csys_enum == CoordinateSystemType.LOCAL:
            angle_rot = self.nodes[node_id].local_axis.angle
        else:
            angle_rot = None

        self.load_patterns[load_pattern_name].add_point_load(
            node_id=node_id,
            forces=PointLoad(fx=fx, fy=fy, mz=mz),
            csys=csys_enum,
            angle_rot=angle_rot,
            replace=replace
        )

    def add_distributed_load(
        self,
        member_id: int,
        load_pattern_name: str,
        load_start: float = 0.0,
        load_end: float = 0.0,
        CSys: Union[str, CoordinateSystemType] = "LOCAL",
        direction: Union[str, DirectionType] = "LOCAL_2",
        load_type: Union[str, LoadType] = "FORCE",
        replace: bool = False,
    ) -> None:
        """
        Asigna una carga distribuida a un miembro dentro de un patrón de carga.

        Args:
            member_id (int): Identificador del miembro.
            load_pattern_name (str): Nombre del patrón de carga.
            load_start (float, opcional): Magnitud de la carga en el inicio. Default es 0.0.
            load_end (float, opcional): Magnitud de la carga en el final. Default es 0.0.
            CSys (str o CoordinateSystemType, opcional): Sistema de coordenadas. Default es "LOCAL".
            direction (str o DirectionType, opcional): Dirección de la carga. Default es "LOCAL_2".
            load_type (str o LoadType, opcional): Tipo de carga. Default es "FORCE".
            replace (bool, opcional): Si se reemplaza la carga existente. Default es False.

        Raises:
            ValueError: Si no existe el miembro o el patrón de carga.
        """
        if member_id not in list(self.members.keys()) + list(self.trusses.keys()):
            raise ValueError(f"No existe un miembro con el ID {member_id}")

        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")

        # Convertir a enum si son strings
        if isinstance(CSys, str):
            csys_enum = to_enum(CSys, CoordinateSystemType)
        else:
            csys_enum = CSys

        if isinstance(direction, str):
            direction_enum = to_enum(direction, DirectionType)
        else:
            direction_enum = direction

        if isinstance(load_type, str):
            load_type_enum = to_enum(load_type, LoadType)
        else:
            load_type_enum = load_type

        if isinstance(self.trusses.get(member_id), TrussElement) and direction_enum != DirectionType.LOCAL_1:
            raise ValueError("Solo se puede aplicar carga distribuida axial en Armaduras osea solo en la direccion LOCAL_1")

        self.load_patterns[load_pattern_name].add_distributed_load(
            member_id=member_id,
            load_start=load_start,
            load_end=load_end,
            csys=csys_enum,
            direction=direction_enum,
            load_type=load_type_enum,
            replace=replace,
        )

    def add_self_weight(
        self,
        load_pattern_name: str,
        factor: float = 1
    ) -> None:
        """
        Asigna el peso propio a todos los miembros del modelo dentro de un patrón de carga.

        Args:
            load_pattern_name (str): Nombre del patrón de carga.
            factor (float, opcional): Factor de escala para el peso propio. Default es 1.

        Raises:
            ValueError: Si no existe el patrón de carga.
        """
        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")

        self.load_patterns[load_pattern_name].self_weight.multiplier = factor

    def add_cst_uniform_temperature_load(
            self,
            cst_id: int,
            load_pattern_name: str,
            dt: float) -> None:
        """
        Asigna una carga uniforme de temperatura a un CST dentro de un patrón de carga.

        Args:
            cst_id (int): ID del CST.
            load_pattern_name (str): Nombre del patrón de carga.
            dt (float): Incremento de temperatura.

        Raises:
            ValueError: Si no existe el CST o el patrón de carga.
        """
        if cst_id not in self.csts:
            raise ValueError(f"No existe un miembro con el ID {cst_id}")
        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")

        self.load_patterns[load_pattern_name].add_cst_uniform_temperature_load(
            cst_id=cst_id,
            dt=dt
        )

    def add_cst_uniform_distributed_load(
        self,
        cst_id: int,
        load_pattern_name: str,
        qx: float,
        qy: float,
    ) -> None:
        """
        Asigna una carga distribuida uniforme a un CST dentro de un patrón de carga.

        Args:
            cst_id (int): ID del CST.
            load_pattern_name (str): Nombre del patrón de carga.
            qx (float): Carga distribuida en X.
            qy (float): Carga distribuida en Y.

        Raises:
            ValueError: Si no existe el CST o el patrón de carga.
        """
        if cst_id not in self.csts:
            raise ValueError(f"No existe un miembro con el ID {cst_id}")
        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")

        self.load_patterns[load_pattern_name].add_cst_uniform_distributed_load(
            cst_id=cst_id,
            qx=qx,
            qy=qy
        )

    def add_cst_uniform_edge_load(
        self,
        cst_id: int,
        load_pattern_name: str,
        q: float,
        edge: int
    ) -> None:
        """
        Asigna una carga uniforme a una cara de un CST dentro de un patrón de carga.

        Args:
            cst_id (int): ID del CST.
            load_pattern_name (str): Nombre del patrón de carga.
            q (float): Carga uniforme.
            edge (int): Cara del CST (1, 2, o 3).

        Raises:
            ValueError: Si no existe el CST o el patrón de carga, o si la cara no es válida.
        """
        if cst_id not in self.csts:
            raise ValueError(f"No existe un miembro con el ID {cst_id}")
        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")
        if edge not in [1, 2, 3]:
            raise ValueError(f"Las caras(edge) solo puenden ser [1, 2, 3]")

        self.load_patterns[load_pattern_name].add_cst_uniform_edge_load(
            cst_id=cst_id,
            q=q,
            edge=edge
        )

    def add_cst_linear_edge_load(
        self,
        cst_id: int,
        load_pattern_name: str,
        qi: float,
        qj: float,
        edge: int
    ) -> None:
        """
        Asigna una carga lineal a una cara de un CST dentro de un patrón de carga.

        Args:
            cst_id (int): ID del CST.
            load_pattern_name (str): Nombre del patrón de carga.
            qi (float): Carga en el inicio.
            qj (float): Carga en el final.
            edge (int): Cara del CST (1, 2, o 3).

        Raises:
            ValueError: Si no existe el CST o el patrón de carga, o si la cara no es válida.
        """
        if cst_id not in self.csts:
            raise ValueError(f"No existe un miembro con el ID {cst_id}")
        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")
        if edge not in [1, 2, 3]:
            raise ValueError(f"Las caras(edge) solo puenden ser [1, 2, 3]")

        self.load_patterns[load_pattern_name].add_cst_uniform_edge_load(
            cst_id=cst_id,
            qi=qi,
            qj=qj,
            edge=edge
        )


    #! TOPICO ESPECIALES DE ANALISIS #######################################
    def add_end_length_offset(
        self,
        member_id: int,
        la: float = 0,
        lb: float = 0,
        qla: bool = True,
        qlb: bool = True,
        fla: float = 1,
        flb: float = 1
    ) -> None:
        """
        Asigna un desplazamiento de longitud final a un miembro dentro de un patrón de carga.

        Args:
            member_id (int): ID del miembro.
            la (float): Desface de longitud final en el nodo inicial.
            lb (float): Desface de longitud final en el nodo final.
            qla (bool, opcional): Si se aplica el carga en el brazo inicial. Default es True.
            qlb (bool, opcional): Si se aplica el carga en el brazo final. Default es True.
            fla (float, opcional): Factor de zona rigida del brazo inicial. Default es 1.
            flb (float, opcional): Factor de zona rigida del brazo final. Default es 1.

        Raises:
            ValueError: Si no existe el miembro.
        """
        if member_id not in self.members:
            raise ValueError(f"No existe un miembro con el ID {member_id}")
        self.members[member_id].la = la
        self.members[member_id].lb = lb
        self.members[member_id].qla = qla
        self.members[member_id].qlb = qlb
        self.members[member_id].fla = fla
        self.members[member_id].flb = flb

    def add_prescribed_dof(
        self,
        node_id: int,
        load_pattern_name: str,
        ux: Optional[float] = None,
        uy: Optional[float] = None,
        rz: Optional[float] = None,
        CSys: Union[str, CoordinateSystemType] = "GLOBAL"
    ) -> None:
        """
        Asigna un desplazamiento prescrito a un nodo dentro de un patrón de carga.

        Args:
            node_id (int): Identificador del nodo.
            load_pattern_name (str): Nombre del patrón de carga.
            ux (float, opcional): Desplazamiento en X. Default es None.
            uy (float, opcional): Desplazamiento en Y. Default es None.
            rz (float, opcional): Desplazamiento en Z. Default es None.
            CSys (str, opcional): Sistema de coordenadas. Default es "GLOBAL".
        Raises:
            ValueError: Si no existe el nodo o el patrón de carga.
        """
        if node_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_id}")
        if load_pattern_name not in self.load_patterns:
            raise ValueError(
                f"No existe un patrón de carga con el nombre '{load_pattern_name}'")
        if isinstance(CSys, str):
            CSys = to_enum(CSys, CoordinateSystemType)
        if CSys not in [CoordinateSystemType.LOCAL, CoordinateSystemType.GLOBAL]:
            raise ValueError(f"CSys debe ser 'LOCAL' o 'GLOBAL'")
        if CSys == CoordinateSystemType.GLOBAL and self.nodes[node_id].local_axis is not None:
            angle = self.nodes[node_id].local_axis.angle
            T = np.array([
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ]).T
            ux, uy, rz = np.dot(T, np.array([ux or 0, uy or 0, rz or 0]))
            ux = ux if ux != 0 else None
            uy = uy if uy != 0 else None
            rz = rz if rz != 0 else None
        self.load_patterns[load_pattern_name].add_prescribed_dof(
            node_id, PrescribedDOF(ux=ux, uy=uy, rz=rz))

    def add_elastic_support(
        self,
        node_id: int,
        kx: Optional[float] = None,
        ky: Optional[float] = None,
        krz: Optional[float] = None,
        CSys: Union[str, CoordinateSystemType] = "GLOBAL"
    ) -> None:
        """
        Asigna un apoyo elástico a un nodo.

        Args:
            node_id (int): Identificador del nodo.
            kx (float, opcional): Constante de rigidez en X. Default es None.
            ky (float, opcional): Constante de rigidez en Y. Default es None.
            krz (float, opcional): Constante de rigidez en Z. Default es None.
            CSys (str, opcional): Sistema de coordenadas. Default es "GLOBAL".
        Raises:
            ValueError: Si no existe el nodo.
        """
        if node_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_id}")
        if isinstance(CSys, str):
            CSys = to_enum(CSys, CoordinateSystemType)
        if CSys not in [CoordinateSystemType.LOCAL, CoordinateSystemType.GLOBAL]:
            raise ValueError(f"CSys debe ser 'LOCAL' o 'GLOBAL'")
        if CSys == CoordinateSystemType.GLOBAL and self.nodes[node_id].local_axis is not None:
            angle = self.nodes[node_id].local_axis.angle
            T = np.array([
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            kx, ky, krz = np.dot(T, np.array([kx or 0, ky or 0, krz or 0]))
            kx = kx if kx != 0 else None
            ky = ky if ky != 0 else None
            krz = krz if krz != 0 else None
        self.nodes[node_id].set_elastic_support(
            ElasticSupport(kx=kx, ky=ky, krz=krz))

    def add_local_axis_for_node(
        self,
        node_id: int,
        angle: float
    ) -> None:
        """
        Asigna un eje local a un nodo.
        >> con esto se puede hacer soportes inclinados y obtener resultados en el eje local del NODO
        Args:
            node_id (int): Identificador del nodo.
            angle (float): Ángulo del eje local en radianes.

        Raises:
            ValueError: Si no existe el nodo.
        """
        if node_id not in self.nodes:
            raise ValueError(f"No existe un nodo con el ID {node_id}")
        local_axis = LocalAxis(angle)
        self.nodes[node_id].set_local_axis(local_axis)

    def add_releases(
        self,
        member_id: int,
        pi: bool = False,
        vi: bool = False,
        mi: bool = False,
        pj: bool = False,
        vj: bool = False,
        mj: bool = False
    ):
        """
        Asigna liberaciones a un miembro.

        Args:
            member_id (int): Identificador del miembro.
            pi (bool, opcional): Liberación de fuerza axial del nodo inicial. Default es False.
            vi (bool, opcional): Liberación de cortante del nodo inicial. Default es False.
            mi (bool, opcional): Liberación de momento del nodo inicial. Default es False.
            pj (bool, opcional): Liberación de fuerza axial del nodo final. Default es False.
            vj (bool, opcional): Liberación de cortante del nodo final. Default es False.
            mj (bool, opcional): Liberación de momento del nodo final. Default es False.

        Raises:
            ValueError: Si no existe el miembro.
        """
        if member_id not in self.members:
            raise ValueError(f"No existe un miembro con el ID {member_id}")

        # Reglas de exclusión
        if pi and pj:
            raise ValueError(
                "Liberación inválida: no se pueden liberar P en ambos extremos (pi y pj).")
        if vi and vj:
            raise ValueError(
                "Liberación inválida: no se pueden liberar V en ambos extremos (vi y vj).")

        self.members[member_id].add_releases(
            uxi=pi, uyi=vi, rzi=mi, uxj=pj, uyj=vj, rzj=mj)

    #! ANALISIS ##############################################################
    def solve(
        self,
        load_pattern_name: list[str] | None = None
    ) -> Dict[str, Results]:
        """
        Resuelve el sistema estructural aplicando el método de rigidez:
        - Asigna las cargas a nodos y elementos.
        - Calcula el vector de fuerzas y la matriz de rigidez global.
        - Resuelve el sistema de ecuaciones para obtener los desplazamientos y reacciones.

        Returns:
            Dict[str, Results]: Objeto con los resultados del análisis.

        Raises:
            ValueError: Si no hay patrones de carga definidos.
        """
        # Verificar que exista al menos un patrón de carga
        if not self.load_patterns:
            raise ValueError(
                "No hay patrones de carga definidos. Agregue al menos uno para resolver el sistema.")

        # Inicializar análisis
        self.analysis = AnalysisManager(self)
        self.analysis.analyze_for_list_load_pattern(load_pattern_name)

        return self.results


    #! SETERS Y GETERS #######################################################
    def get_global_stiffness_matrix(
        self
    ) -> NDArray:
        """
        Obtiene la matriz de rigidez global del modelo.

        Returns:
            np.ndarray: Matriz de rigidez global.
        """
        return self.global_stiffness_matrix

    def get_global_load_vector(
        self,
        load_pattern_name: str
    ) -> NDArray:
        """
        Obtiene el vector de fuerzas global del modelo.

        Returns:
            np.ndarray: Vector de fuerzas global.
        """
        return self.global_load_vector.get(load_pattern_name)


    #! VISUALIZACION EN VENTANA INTERACTIVA ##################################
    def show(
        self
    ):
        """
        Muestra la interfaz gráfica para visualizar el modelo.
        """
        if any(list(x.analyzed for x in self.load_patterns.values())):
            pass
        else:
            print("No hay patrones de carga analizados. Por favor, analice el modelo antes de mostrar la visualización.")
            return
        import time
        start_time = time.time()
        from milcapy.plotter.UIdisplay import main_window
        self.plotter = Plotter(self)
        self.plotter.initialize_plot()
        end_time = time.time()
        print(f"Tiempo de generacion de la visualizacion: {end_time - start_time}")
        main_window(self)

    def plot_model(
        self,
        load_pattern: Optional[str] = None,
        node_labels: bool = True,
        member_labels: bool = True,
    ):
        """
        Grafica el modelo estructural.
        """
        from milcapy.plotter.plot_model import plot_model
        plot_model(self, load_pattern, node_labels, member_labels)


    #! RESULTADOS ###########################################################
    def get_results(
        self,
        load_pattern_name: str
    ):
        """
        Obtiene los resultados del análisis.

        Returns:
            Results: Objeto con los resultados del análisis.
        """
        return self.results.get(load_pattern_name)

    def get_results_excel(self, load_pattern_name: str) -> None:
        """
        Obtiene los resultados del análisis en formato .xlsx.

        Returns:
            Object: Resultados del análisis en formato .xlsx.
        """
        import pandas as pd
        nodedf, memberdf = self.results.get(load_pattern_name).get_dataframes()

        # Crear el archivo Excel
        with pd.ExcelWriter(f'{load_pattern_name}.xlsx') as writer:
            nodedf.to_excel(writer, sheet_name='Nodos', index=False)
            memberdf.to_excel(writer, sheet_name='Miembros', index=False)

    #! PLOT DE ELEMENTOS FINITOS #############################################

    def plot_model2(
        self,
        load_pattern: Optional[str] = None,
        type: Optional[InternalForceType] = None,
    ):
        """
        Grafica el modelo estructural.
        """

        import matplotlib.pyplot as plt
        from milcapy.plotter.UIdisplay import MainWindow, QApplication, sys
        __Plotter = Plotter(self)#, load_pattern)
        __Plotter.initialize_plot()#type)

        # __Plotter.plotter_options.UI_axial = True
        # __Plotter.update_axial_force(visibility=True)

        plt.show()

        # app = QApplication.instance()
        # if app is None:  # Si no existe una instancia, se crea
        #     app = QApplication(sys.argv)
        # window = MainWindow(self)
        # window.show()



    def plot_field(
        self,
        field: Union[FieldType, str],
        cmap: str = "jet"
    ):
        """
        Grafica un campo de esfuerzos o deformaciones en el modelo.

        Args:
            field (Union[FieldType, str]): Campo de esfuerzos o deformaciones a graficar.
            cmap (str, opcional): Mapa de colores para la visualización. Default es "jet".

        Raises:
            ValueError: Si el campo no es válido.
        """
        # coordenadas nodales
        x = [node.vertex.x for node in self.nodes.values()]
        y = [node.vertex.y for node in self.nodes.values()]

        triangles = []
        node_values = {nid: [] for nid in self.nodes.keys()}

        for cst in self.csts.values():
            if isinstance(cst, MembraneTriangle):
                n1, n2, n3 = cst.node1, cst.node2, cst.node3
                triangles.append([n1.id - 1, n2.id - 1, n3.id - 1])

                # esfuerzos y deformaciones (al centroide)
                stresses = self.results[self.current_load_pattern].get_cst_stresses(
                    cst.id)
                strains = self.results[self.current_load_pattern].get_cst_strains(
                    cst.id)

                if field == FieldType.SX:
                    val = stresses[0]
                elif field == FieldType.SY:
                    val = stresses[1]
                elif field == FieldType.SXY:
                    val = stresses[2]
                elif field == FieldType.EX:
                    val = strains[0]
                elif field == FieldType.EY:
                    val = strains[1]
                elif field == FieldType.EXY:
                    val = strains[2]
                else:
                    val = None

                if val is not None:
                    # asignar valor del centroide a nodos
                    for n in [n1, n2, n3]:
                        node_values[n.id].append(val)

        # --- NUEVO: desplazamientos nodales ---
        if field in [FieldType.UX, FieldType.UY, FieldType.UMAG]:
            for nid, node in self.nodes.items():
                ux, uy = self.results[self.current_load_pattern].get_node_displacements(nid)[
                    :2]
                if field == FieldType.UX:
                    node_values[nid] = [ux]
                elif field == FieldType.UY:
                    node_values[nid] = [uy]
                elif field == FieldType.UMAG:
                    node_values[nid] = [np.sqrt(ux**2 + uy**2)]

        # promediar valores en los nodos
        nodal_field = np.zeros(len(self.nodes))
        for nid, vals in node_values.items():
            if vals:
                nodal_field[nid - 1] = np.mean(vals)

        triang = tri.Triangulation(x, y, triangles)

        # --- GRAFICADO CON tricontourf ---
        plt.figure(figsize=(15, 8))
        tcf = plt.tricontourf(triang, nodal_field,
                              levels=20, cmap=cmap)  # suavizado
        plt.colorbar(tcf, label=f"{field.value}")
        # --- GRAFICANDO LOS MIEMBROS ESTRUCTURALES ----
        for member in self.members.values():
            coord = np.hstack((member.node_i.vertex.coordinates,
                              member.node_j.vertex.coordinates))
            x = coord[[0, 2]]
            y = coord[[1, 3]]
            plt.plot(x, y, color="b")
        plt.axis('equal')
        plt.title(f"Mapa de calor - {field.value}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()

    def _plot_ux(self):
        self.plot_field(FieldType.UX)

    def _plot_uy(self):
        self.plot_field(FieldType.UY)

    def _plot_umag(self):
        self.plot_field(FieldType.UMAG)

    def _plot_sx(self):
        self.plot_field(FieldType.SX)

    def _plot_sy(self):
        self.plot_field(FieldType.SY)

    def _plot_sxy(self):
        self.plot_field(FieldType.SXY)

    def _plot_ex(self):
        self.plot_field(FieldType.EX)

    def _plot_ey(self):
        self.plot_field(FieldType.EY)

    def _plot_exy(self):
        self.plot_field(FieldType.EXY)


    #! METODOS PRIVADOS ######################################################
    def __id_verifier(self, ele_id: int, *id_nodes: int):
        """
        Verifica que el ID del elemento no exista y que los nodos existan.

        Args:
            ele_id (int): ID del elemento.
            *id_nodes (int): IDs de los nodos.

        Raises:
            ValueError: Si el ID del elemento ya existe o si no existen los nodos.
        """
        if (ele_id in self.members
            or ele_id in self.csts
            or ele_id in self.membrane_q3dof
                or ele_id in self.membrane_q2dof
                    or ele_id in self.trusses):
            raise ValueError(f"Ya existe un elemento con el ID {ele_id}")

        for node_id in id_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"No existe un nodo con el ID {node_id}")

    def __get_vector(self, nA: Node, nB: Node) -> np.ndarray:
        """Vector de A hacia B."""
        return np.array([nB.x - nA.x, nB.y - nA.y], dtype=float)

    def __is_right_angle(self, v1: np.ndarray, v2: np.ndarray, tol: float = 1e-8) -> bool:
        """Verifica si dos vectores son ortogonales (producto punto ≈ 0)."""
        return abs(np.dot(v1, v2)) < tol

    def __is_same_length(self, v1: np.ndarray, v2: np.ndarray, tol: float = 1e-8) -> bool:
        """Verifica si dos vectores tienen la misma longitud."""
        return abs(np.linalg.norm(v1) - np.linalg.norm(v2)) < tol

    def __is_rectangular(self, n1_id: int, n2_id: int, n3_id: int, n4_id: int) -> bool:
        # """
        # Verifica si los nodos forman un rectángulo en sentido antihorario.
        # Orden esperado: n1 → n2 → n3 → n4.
        # """
        # try:
        #     n1, n2, n3, n4 = self.nodes[n1_id], self.nodes[n2_id], self.nodes[n3_id], self.nodes[n4_id]
        # except KeyError as e:
        #     raise ValueError(f"Nodo no encontrado: {e}")

        # # Vectores de lados
        # v12 = self.__get_vector(n1, n2)
        # v23 = self.__get_vector(n2, n3)
        # v34 = self.__get_vector(n3, n4)
        # v41 = self.__get_vector(n4, n1)

        # # Condiciones:
        # cond1 = self.__is_right_angle(v12, v23) and self.__is_right_angle(v23, v34) \
        #         and self.__is_right_angle(v34, v41) and self.__is_right_angle(v41, v12)

        # cond2 = self.__is_same_length(v12, v34) and self.__is_same_length(v23, v41)

        # return cond1 and cond2
        pass

    def __is_counterclockwise(self, *node_ids: int) -> bool:
        # """
        # Verifica si los nodos dados están en sentido antihorario (CCW).
        # Args:
        #     *node_ids: IDs de nodos en el orden del polígono.
        # Returns:
        #     bool: True si el polígono definido por los nodos está en sentido CCW.
        # """
        # if len(node_ids) < 3:
        #     raise ValueError("Se requieren al menos 3 nodos para definir un polígono.")

        # try:
        #     coords = [self.nodes[nid].vertex.coordinates for nid in node_ids]
        # except KeyError as e:
        #     raise ValueError(f"Nodo no encontrado: {e}")

        # area_signed = 0.0
        # n = len(coords)
        # for i in range(n):
        #     x1, y1 = coords[i]
        #     x2, y2 = coords[(i + 1) % n]  # siguiente nodo (cierra el polígono)
        #     area_signed += (x2 - x1) * (y2 + y1)  # variante del shoelace para orientación

        # # Otra opción equivalente:
        # # area_signed = sum(coords[i][0]*coords[(i+1)%n][1] - coords[(i+1)%n][0]*coords[i][1] for i in range(n))

        # return area_signed < 0  # <0 CCW, >0 CW (depende de convención)
        pass

    # ! HELPERS #####################################################################################################
    def set_no_weight(self, load_pattern_name, ele_id)->None:
        """Setea el peso propio de un elemento a cero."""
        self.load_patterns[load_pattern_name].set_no_weight(ele_id)
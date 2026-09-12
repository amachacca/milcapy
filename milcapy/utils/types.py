from typing import Type, TypeVar
from typing import Tuple
from enum import Enum, auto

# Definición de restricciones
Restraints = Tuple[bool, bool, bool]


class ShearCoefficientMethodType(Enum):
    """Métodos para el cálculo del coeficiente de corte."""
    TIMOSHENKO = 'TIMOSHENKO'
    COWPER = 'COWPER'
    NONE = 'NONE'


class MemberType(Enum):
    """Tipos de miembros en la estructura."""
    FRAME = 'FRAME'
    TRUSS = 'TRUSS'
    BEAM = 'BEAM'
    BRACE = 'BRACE'
    MEMBRANE_2DOF = 'MEMBRANE_2DOF'
    MEMBRANE_3DOF = 'MEMBRANE_3DOF'


class BeamTheoriesType(Enum):
    """Tipos de teorías de vigas."""
    TIMOSHENKO = 'TIMOSHENKO'
    EULER_BERNOULLI = 'EULER_BERNOULLI'


class CoordinateSystemType(Enum):
    """Tipos de sistemas de coordenadas."""
    GLOBAL = 'GLOBAL'
    LOCAL = 'LOCAL'


class DirectionType(Enum):
    """Direcciones en el sistema global y local."""
    X = 'X'
    Y = 'Y'
    X_PROJ = 'X_PROJ'
    Y_PROJ = 'Y_PROJ'
    GRAVITY = 'GRAVITY'
    GRAVITY_PROJ = 'GRAVITY_PROJ'
    MOMENT = 'MOMENT'
    LOCAL_1 = 'LOCAL_1'  # AXIAL
    LOCAL_2 = 'LOCAL_2'  # SHEAR
    LOCAL_3 = 'LOCAL_3'  # MOMENT


class StateType(Enum):
    """Estado de los elementos o componentes."""
    ACTIVE = 'ACTIVE'
    INACTIVE = 'INACTIVE'


class LoadType(Enum):
    """Tipos de cargas (fuerzas o momentos)."""
    FORCE = 'FORCE'
    MOMENT = 'MOMENT'


class InternalForceType(Enum):
    """Enumeración de tipos de Fuerzas Internas."""
    AXIAL_FORCE = "AXIAL_FORCE"
    SHEAR_FORCE = "SHEAR_FORCE"
    BENDING_MOMENT = "BENDING_MOMENT"
    SLOPE = "SLOPE"
    DEFLECTION = "DEFLECTION"
    DEFORMED = "DEFORMED"
    RIGID_DEFORMED = "RIGID_DEFORMED"


class FieldType(Enum):
    SX = "SX"
    SY = "SY"
    SXY = "SXY"
    EX = "EX"
    EY = "EY"
    EXY = "EXY"
    UX = "UX"
    UY = "UY"
    UMAG = "UMAG"
    VM = "VM"
    S1 = "S1"
    S2 = "S2"


ENUM = TypeVar("ENUM", bound=Enum)
def to_enum(key: str, enum: Type[ENUM]) -> ENUM:
    # def to_enum(key: str, enum: Enum) -> Enum:
    """Convierte un string a un miembro de un Enum."""
    if isinstance(key, enum):
        return key
    assert isinstance(key, str), 'La clave debe ser un string.'
    assert issubclass(
        enum, Enum), 'El segundo argumento debe ser una clase Enum.'
    try:
        return enum(key)
    except ValueError:
        raise ValueError(
            f'La clave "{key}" no se encuentra en el Enum "{enum.__name__}".')


class ConstitutiveModelType(Enum):
    PLANE_STRESS = 'PLANE_STRESS'     # Esfuerzo plano
    PLANE_STRAIN = 'PLANE_STRAIN'     # Deformación plana


class IntegrationType(Enum):
    """Tipos de integración."""
    COMPLETE = 'COMPLETE'
    REDUCED = 'REDUCED'


class MembraneQuadElementType(Enum):
    """Tipos de elementos de membrana."""
    MQ4 = "MQ4"
    MQ6 = "MQ6"
    MQ6I = "MQ6I"
    MQ8Reduced = "MQ8Reduced"
    MQ8Complete = "MQ8Complete"

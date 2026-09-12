from math import nan
from typing import Optional, Dict, Union, Tuple, List
import numpy as np
from abc import ABC, abstractmethod

class Load(ABC):
    """
    Clase base abstracta que define la interfaz común para diferentes tipos de cargas estructurales.

    Esta clase implementa operaciones básicas como comparación, inversión y conversión,
    que son comunes a todos los tipos de cargas.

    Attributes:
        id (Optional[int]): Identificador único de la carga. Por defecto es None.
    """

    def __init__(
        self,
        id: Optional[int] = None
        ) -> None:

        """
        Inicializa una nueva instancia de carga.

        Args:
            id: Identificador único opcional para la carga.
        """

        self.id = id

    @property
    @abstractmethod
    def components(
        self
        ) -> np.ndarray:
        """
        Obtiene los componentes de la carga como un array NumPy.

        Returns:
            np.ndarray: Array con los componentes de la carga.
        """
        pass

    @abstractmethod
    def to_dict(
        self
        ) -> Dict[str, Union[int, float, None]]:
        """
        Convierte la carga a un diccionario serializable.

        Returns:
            Dict[str, Union[int, float, None]]: Diccionario con los atributos de la carga.
        """
        pass

    def __eq__(
        self,
        other: object
        ) -> bool:
        """
        Compara si dos cargas son aproximadamente iguales.

        Args:
            other: Objeto a comparar.

        Returns:
            bool: True si las cargas son aproximadamente iguales, False en caso contrario.
        """
        if not isinstance(other, Load):
            return NotImplemented
        return np.allclose(self.components, other.components)

    def __neg__(
        self
        ) -> 'Load':
        """
        Devuelve una nueva instancia con todos los componentes de carga invertidos.

        Returns:
            Load: Nueva instancia con cargas negativas.
        """
        return self.__class__(*(-self.components), id=self.id)

    def __pos__(
        self
        ) -> 'Load':
        """
        Devuelve la misma instancia sin modificaciones (operador unario +).

        Returns:
            Load: La misma instancia sin cambios.
        """
        return self


class PointLoad(Load):
    """
    Representa una carga puntual en un sistema estructural 2D.

    Esta clase modela una carga concentrada con:
    - Fuerza de corte perpendicular al eje de la viga (fx)
    - Fuerza axial a lo largo del eje x de la viga (fy)
    - Momento que genera rotación alrededor del eje z (mz)

    Attributes:
        fx (float): Fuerza de corte perpendicular al eje de la viga.
        fy (float): Fuerza axial a lo largo del eje x de la viga.
        mz (float): Momento alrededor del eje z.
        id (Optional[int]): Identificador único de la carga.
    """

    __slots__ = ('fx', 'fy', 'mz', 'id')
    ZERO = np.zeros(3, dtype=np.float64)

    def __init__(
        self,
        fx: float = 0.0,
        fy: float = 0.0,
        mz: float = 0.0,
        id: Optional[int] = None
        ) -> None:
        """
        Inicializa una nueva carga puntual.

        Args:
            fx: Fuerza de corte perpendicular al eje de la viga.
            fy: Fuerza axial a lo largo del eje x de la viga.
            mz: Momento que genera rotación alrededor del eje z.
            id: Identificador único opcional.
        """
        super().__init__(id)
        self.fx = float(fx)
        self.fy = float(fy)
        self.mz = float(mz)

    @property
    def components(
        self
        ) -> np.ndarray:
        """
        Obtiene los componentes de la carga como un array NumPy.

        Returns:
            np.ndarray: Array con [fx, fy, mz].
        """
        return np.array([self.fx, self.fy, self.mz], dtype=np.float64)

    def to_dict(
        self
        ) -> Dict[str, Union[int, float, None]]:
        """
        Convierte la carga puntual a un diccionario serializable.

        Returns:
            Dict[str, Union[int, float, None]]: Diccionario con los componentes de la carga.
        """
        return {"fx": self.fx, "fy": self.fy, "mz": self.mz, "id": self.id}

    def __add__(
        self,
        other: "PointLoad"
        ) -> "PointLoad":
        """
        Suma dos cargas puntuales.

        Args:
            other: Otra carga puntual a sumar.

        Returns:
            PointLoad: Nueva carga puntual resultante de la suma.

        Raises:
            TypeError: Si other no es una instancia de PointLoad.
        """
        if not isinstance(other, PointLoad):
            return NotImplemented
        return PointLoad(
            self.fx + other.fx,
            self.fy + other.fy,
            self.mz + other.mz,
            self.id or other.id
        )

    def __sub__(
        self,
        other: "PointLoad"
        ) -> "PointLoad":
        """
        Resta dos cargas puntuales.

        Args:
            other: Carga puntual a restar.

        Returns:
            PointLoad: Nueva carga puntual resultante de la resta.

        Raises:
            TypeError: Si other no es una instancia de PointLoad.
        """
        if not isinstance(other, PointLoad):
            return NotImplemented
        return PointLoad(
            self.fx - other.fx,
            self.fy - other.fy,
            self.mz - other.mz,
            self.id or other.id
        )

    def __mul__(
        self,
        scalar: Union[float, int]
        ) -> "PointLoad":
        """
        Multiplica la carga por un escalar.

        Args:
            scalar: Factor de escala.

        Returns:
            PointLoad: Nueva carga puntual escalada.

        Raises:
            TypeError: Si scalar no es un número.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return PointLoad(
            self.fx * scalar,
            self.fy * scalar,
            self.mz * scalar,
            self.id
        )

    __rmul__ = __mul__

    def __truediv__(
        self,
        scalar: Union[float, int]
        ) -> "PointLoad":
        """
        Divide la carga por un escalar.

        Args:
            scalar: Divisor.

        Returns:
            PointLoad: Nueva carga puntual dividida.

        Raises:
            TypeError: Si scalar no es un número.
            ZeroDivisionError: Si scalar es cero.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("No se puede dividir por cero.")
        return PointLoad(
            self.fx / scalar,
            self.fy / scalar,
            self.mz / scalar,
            self.id
        )


class DistributedLoad(Load):
    """
    Representa una carga distribuida en un elemento estructural 2D.

    Esta clase modela cargas distribuidas con valores iniciales (i) y finales (j):
    - q: Carga distribuida perpendicular al eje de la viga
    - p: Carga distribuida axial a lo largo del eje de la viga
    - m: Momento distribuido que genera rotación alrededor del eje z

    Attributes:
        q_i (float): Carga de corte inicial, perpendicular al eje de la viga, eje y.
        q_j (float): Carga de corte final, perpendicular al eje de la viga, eje y.
        p_i (float): Carga axial inicial, a lo largo del eje x de la viga.
        p_j (float): Carga axial final, a lo largo del eje x de la viga.
        m_i (float): Momento inicial alrededor del eje z.
        m_j (float): Momento final alrededor del eje z.
        id (Optional[int]): Identificador único de la carga.
    """

    __slots__ = ('q_i', 'q_j', 'p_i', 'p_j', 'm_i', 'm_j', 'id')
    ZERO = np.zeros(6, dtype=np.float64)

    def __init__(
        self,
        q_i: float = 0.0,
        q_j: float = 0.0,
        p_i: float = 0.0,
        p_j: float = 0.0,
        m_i: float = 0.0,
        m_j: float = 0.0,
        id: Optional[int] = None
        ) -> None:
        """
        Inicializa una nueva carga distribuida.

        Args:
            q_i: Carga de corte inicial, perpendicular al eje de la viga.
            q_j: Carga de corte final, perpendicular al eje de la viga.
            p_i: Carga axial inicial, a lo largo del eje x de la viga.
            p_j: Carga axial final, a lo largo del eje x de la viga.
            m_i: Momento inicial alrededor del eje z.
            m_j: Momento final alrededor del eje z.
            id: Identificador único opcional.

        Raises:
            TypeError: Si alguno de los valores no es un número real.
        """
        super().__init__(id)

        # Validación de tipos y conversión
        load_params = {'q_i': q_i, 'q_j': q_j, 'p_i': p_i,
                    'p_j': p_j, 'm_i': m_i, 'm_j': m_j}

        for name, value in load_params.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"El parámetro {name} debe ser un número real")
            setattr(self, name, float(value))

    @property
    def components(
        self
        ) -> np.ndarray:
        """
        Obtiene los componentes de la carga como un array NumPy.

        Returns:
            np.ndarray: Array con [q_i, q_j, p_i, p_j, m_i, m_j].
        """
        return np.array([self.q_i, self.q_j, self.p_i, self.p_j, self.m_i, self.m_j],
                       dtype=np.float64)

    def to_dict(
        self
        ) -> Dict[str, Union[int, float, None]]:
        """
        Convierte la carga distribuida a un diccionario serializable.

        Returns:
            Dict[str, Union[int, float, None]]: Diccionario con los componentes de la carga.
        """
        return {
            "q_i": self.q_i, "q_j": self.q_j,
            "p_i": self.p_i, "p_j": self.p_j,
            "m_i": self.m_i, "m_j": self.m_j,
            "id": self.id
        }

    def __add__(
        self,
        other: "DistributedLoad"
        ) -> "DistributedLoad":
        """
        Suma dos cargas distribuidas.

        Args:
            other: Otra carga distribuida a sumar.

        Returns:
            DistributedLoad: Nueva carga distribuida resultante de la suma.

        Raises:
            TypeError: Si other no es una instancia de DistributedLoad.
        """
        if not isinstance(other, DistributedLoad):
            return NotImplemented
        return DistributedLoad(
            self.q_i + other.q_i,
            self.q_j + other.q_j,
            self.p_i + other.p_i,
            self.p_j + other.p_j,
            self.m_i + other.m_i,
            self.m_j + other.m_j,
            self.id or other.id
        )

    def __sub__(
        self,
        other: "DistributedLoad"
        ) -> "DistributedLoad":
        """
        Resta dos cargas distribuidas.

        Args:
            other: Carga distribuida a restar.

        Returns:
            DistributedLoad: Nueva carga distribuida resultante de la resta.

        Raises:
            TypeError: Si other no es una instancia de DistributedLoad.
        """
        if not isinstance(other, DistributedLoad):
            return NotImplemented
        return DistributedLoad(
            self.q_i - other.q_i,
            self.q_j - other.q_j,
            self.p_i - other.p_i,
            self.p_j - other.p_j,
            self.m_i - other.m_i,
            self.m_j - other.m_j,
            self.id or other.id
        )

    def __mul__(
        self,
        scalar: Union[float, int]
        ) -> "DistributedLoad":
        """
        Multiplica la carga por un escalar.

        Args:
            scalar: Factor de escala.

        Returns:
            DistributedLoad: Nueva carga distribuida escalada.

        Raises:
            TypeError: Si scalar no es un número.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return DistributedLoad(
            self.q_i * scalar,
            self.q_j * scalar,
            self.p_i * scalar,
            self.p_j * scalar,
            self.m_i * scalar,
            self.m_j * scalar,
            self.id
        )

    __rmul__ = __mul__

    def __truediv__(
        self,
        scalar: Union[float, int]
        ) -> "DistributedLoad":
        """
        Divide la carga por un escalar.

        Args:
            scalar: Divisor.

        Returns:
            DistributedLoad: Nueva carga distribuida dividida.

        Raises:
            TypeError: Si scalar no es un número.
            ZeroDivisionError: Si scalar es cero.
        """
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("No se puede dividir por cero.")
        return DistributedLoad(
            self.q_i / scalar,
            self.q_j / scalar,
            self.p_i / scalar,
            self.p_j / scalar,
            self.m_i / scalar,
            self.m_j / scalar,
            self.id
        )

class PrescribedDOF:
    """
    Representa los desplazamientos prescritos en los grados de libertad de un nodo.
    """
    def __init__(
        self,
        ux: Optional[float] = None,
        uy: Optional[float] = None,
        rz: Optional[float] = None
        ) -> None:
        """
        Inicializa una nueva instancia de PrescribedDOF.

        Args:
            ux: Desplazamiento en el eje x. Por defecto es None.
            uy: Desplazamiento en el eje y. Por defecto es None.
            rz: Rotación en el eje z. Por defecto es None.
        """
        self.ux = ux
        self.uy = uy
        self.rz = rz

    def set_prescribed_dofs(
        self,
        ux: Optional[float] = None,
        uy: Optional[float] = None,
        rz: Optional[float] = None
        ) -> None:
        """
        Establece los desplazamientos prescritos en los grados de libertad del nodo.

        Args:
            ux: Desplazamiento en el eje x. Por defecto es None.
            uy: Desplazamiento en el eje y. Por defecto es None.
            rz: Rotación en el eje z. Por defecto es None.
        """
        self.ux = ux
        self.uy = uy
        self.rz = rz

    def get_prescribed_dofs(
        self
        ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Obtiene los desplazamientos prescritos en los grados de libertad del nodo.

        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]: Tupla con los desplazamientos prescritos en los grados de libertad del nodo.
        """
        return self.ux, self.uy, self.rz

    def __add__(
        self,
        other: "PrescribedDOF"
        ) -> "PrescribedDOF":
        """
        Suma dos desplazamientos prescritos.

        Args:
            other: Otra instancia de PrescribedDOF a sumar.

        Returns:
            PrescribedDOF: Nueva instancia de PrescribedDOF resultante de la suma.
        """
        if self.ux is None and other.ux is None:
            ux = None
        elif self.ux is None or other.ux is None:
            ux = self.ux or other.ux
        else:
            ux = self.ux + other.ux
        if self.uy is None and other.uy is None:
            uy = None
        elif self.uy is None or other.uy is None:
            uy = self.uy or other.uy
        else:
            uy = self.uy + other.uy
        if self.rz is None and other.rz is None:
            rz = None
        elif self.rz is None or other.rz is None:
            rz = self.rz or other.rz
        else:
            rz = self.rz + other.rz
        return PrescribedDOF(
            ux,
            uy,
            rz
        )

    def __sub__(
        self,
        other: "PrescribedDOF"
        ) -> "PrescribedDOF":
        """
        Resta dos desplazamientos prescritos.

        Args:
            other: Otra instancia de PrescribedDOF a restar.

        Returns:
            PrescribedDOF: Nueva instancia de PrescribedDOF resultante de la resta.
        """
        if self.ux is None and other.ux is None:
            ux = None
        elif self.ux is None or other.ux is None:
            ux = self.ux or other.ux
        else:
            ux = self.ux - other.ux
        if self.uy is None and other.uy is None:
            uy = None
        elif self.uy is None or other.uy is None:
            uy = self.uy or other.uy
        else:
            uy = self.uy - other.uy
        if self.rz is None and other.rz is None:
            rz = None
        elif self.rz is None or other.rz is None:
            rz = self.rz or other.rz
        else:
            rz = self.rz - other.rz
        return PrescribedDOF(
            ux,
            uy,
            rz
        )

    def __pos__(
        self
        ) -> "PrescribedDOF":
        """
        Obtiene una copia de la instancia actual.

        Returns:
            PrescribedDOF: Nueva instancia de PrescribedDOF con los mismos valores.
        """
        return PrescribedDOF(
            self.ux,
            self.uy,
            self.rz
        )

    def __neg__(
        self
        ) -> "PrescribedDOF":
        """
        Obtiene la negación de la instancia actual.

        Returns:
            PrescribedDOF: Nueva instancia de PrescribedDOF con los valores negados.
        """
        if self.ux is None:
            ux = None
        else:
            ux = -self.ux
        if self.uy is None:
            uy = None
        else:
            uy = -self.uy
        if self.rz is None:
            rz = None
        else:
            rz = -self.rz
        return PrescribedDOF(
            ux,
            uy,
            rz
        )

    def __str__(
        self
        ) -> str:
        """
        Obtiene una representación en cadena de la instancia actual.

        Returns:
            str: Representación en cadena de la instancia actual.
        """
        return f"PrescribedDOF(ux={self.ux}, uy={self.uy}, rz={self.rz})"

class ElasticSupport:
    """
    Representa los soportes elásticos en los grados de libertad de un nodo.
    """
    def __init__(
        self,
        kx: Optional[float] = None,
        ky: Optional[float] = None,
        krz: Optional[float] = None
        ) -> None:
        """
        Inicializa una nueva instancia de ElasticSupport.

        Args:
            kx: Constante de rigidez en X. Por defecto es None.
            ky: Constante de rigidez en Y. Por defecto es None.
            krz: Constante de rigidez en Z. Por defecto es None.
        """
        self.kx = kx
        self.ky = ky
        self.krz = krz

    def set_elastic_supports(
        self,
        kx: Optional[float] = None,
        ky: Optional[float] = None,
        krz: Optional[float] = None
        ) -> None:
        """
        Establece los soportes elásticos en los grados de libertad del nodo.

        Args:
            kx: Constante de rigidez en X. Por defecto es None.
            ky: Constante de rigidez en Y. Por defecto es None.
            krz: Constante de rigidez en Z. Por defecto es None.
        """
        self.kx = kx
        self.ky = ky
        self.krz = krz

    def get_elastic_supports(
        self
        ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Obtiene los soportes elásticos en los grados de libertad del nodo.

        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]: Tupla con los soportes elásticos en los grados de libertad del nodo.
        """
        return self.kx, self.ky, self.krz

    def __str__(
        self
        ) -> str:
        """
        Obtiene una representación en cadena de la instancia actual.

        Returns:
            str: Representación en cadena de la instancia actual.
        """
        return f"ElasticSupport(kx={self.kx}, ky={self.ky}, krz={self.krz})"

    def __add__(
        self,
        other: "ElasticSupport"
        ) -> "ElasticSupport":
        """
        Suma dos soportes elásticos.

        Args:
            other: Otra instancia de ElasticSupport a sumar.

        Returns:
            ElasticSupport: Nueva instancia de ElasticSupport resultante de la suma.
        """
        if self.kx is None and other.kx is None:
            kx = None
        elif self.kx is None or other.kx is None:
            kx = self.kx or other.kx
        else:
            kx = self.kx + other.kx
        if self.ky is None and other.ky is None:
            ky = None
        elif self.ky is None or other.ky is None:
            ky = self.ky or other.ky
        else:
            ky = self.ky + other.ky
        if self.krz is None and other.krz is None:
            krz = None
        elif self.krz is None or other.krz is None:
            krz = self.krz or other.krz
        else:
            krz = self.krz + other.krz
        return ElasticSupport(
            kx,
            ky,
            krz
        )

    def __sub__(
        self,
        other: "ElasticSupport"
        ) -> "ElasticSupport":
        """
        Resta dos soportes elásticos.

        Args:
            other: Otra instancia de ElasticSupport a restar.

        Returns:
            ElasticSupport: Nueva instancia de ElasticSupport resultante de la resta.
        """
        if self.kx is None and other.kx is None:
            kx = None
        elif self.kx is None or other.kx is None:
            kx = self.kx or other.kx
        else:
            kx = self.kx - other.kx
        if self.ky is None and other.ky is None:
            ky = None
        elif self.ky is None or other.ky is None:
            ky = self.ky or other.ky
        else:
            ky = self.ky - other.ky
        if self.krz is None and other.krz is None:
            krz = None
        elif self.krz is None or other.krz is None:
            krz = self.krz or other.krz
        else:
            krz = self.krz - other.krz
        return ElasticSupport(
            kx,
            ky,
            krz
        )

    def __pos__(
        self
        ) -> "ElasticSupport":
        """
        Obtiene una copia de la instancia actual.

        Returns:
            ElasticSupport: Nueva instancia de ElasticSupport con los mismos valores.
        """
        return ElasticSupport(
            self.kx,
            self.ky,
            self.krz
        )

    def __neg__(
        self
        ) -> "ElasticSupport":
        """
        Obtiene la negación de la instancia actual.

        Returns:
            ElasticSupport: Nueva instancia de ElasticSupport con los valores negados.
        """
        if self.kx is None:
            kx = None
        else:
            kx = -self.kx
        if self.ky is None:
            ky = None
        else:
            ky = -self.ky
        if self.krz is None:
            krz = None
        else:
            krz = -self.krz
        return ElasticSupport(
            kx,
            ky,
            krz
        )

class LocalAxis:
    """
    Representa el eje local de un Nodo.
    """
    def __init__(
        self,
        angle: float
        ) -> None:
        """
        Args:
            angle (float): angulo  desde  el  sistema  de  coordenadas  global  hasta  el  plano  inclinado
        """
        self.angle = angle

    def get_transformation_matrix(
        self
        ) -> np.ndarray:
        """
        Obtiene la matriz de transformación del eje local.

        Returns:
            np.ndarray: Matriz de transformación del eje local.
        """
        return np.array([
            [np.cos(self.angle), -np.sin(self.angle), 0],
            [np.sin(self.angle),  np.cos(self.angle), 0],
            [0,                   0,                  1]
        ])

class EndLengthOffset:
    """
    Representa el desplazamiento de los brazos rigidos de un elemento.
    """
    def __init__(
        self,
        la:  Optional[float]=None,
        lb:  Optional[float]=None,
        qla: Optional[bool] =None,
        qlb: Optional[bool] =None,
        fla: Optional[float]=None,
        flb: Optional[float]=None
        ) -> None:
        self.la:  Optional[float] = la    # Longitud del brazo rigido inicial
        self.lb:  Optional[float] = lb    # Longitud del brazo rigido final
        self.qla: Optional[bool]  = qla   # Indica si hay cargas en el brazo rigido inicial
        self.qlb: Optional[bool]  = qlb   # Indica si hay cargas en el brazo rigido final
        self.fla: Optional[float] = fla   # Factor de longitud del brazo rigido inicial
        self.flb: Optional[float] = flb   # Factor de longitud del brazo rigido final

class CSTLoad:
    """
    Representa la carga constante de un elemento.
    """
    def __init__(
        self,
        coordinates: np.ndarray,
        E: float,
        v: float,
        t: float,
        beta: float
        ) -> None:
        """
        Args:
            coordinates (np.ndarray): coordenadas de los nodos del elemento
            E (float): Young's modulus
            v (float): Poisson's ratio
            t (float): thickness
            beta (float): angle of the inclined plane
        """
        self.coordinates = coordinates # [[x1, y1], [x2, y2], [x3, y3]]
        self.E = E
        self.v = v
        self.t = t
        self.beta = beta

        self.Feq = np.zeros(6)

    def A(
        self
        ) -> float:
        """
        Obtiene el area del elemento.

        Returns:
            float: Area del elemento.
        """
        x1, y1 = self.coordinates[0]
        x2, y2 = self.coordinates[1]
        x3, y3 = self.coordinates[2]

        a1 = x2*y3 - x3*y2
        a2 = x3*y1 - x1*y3
        a3 = x1*y2 - x2*y1

        return (a1 + a2 + a3)/2

    def add_uniform_temperature_load(
        self,
        dt: float
        ) -> np.ndarray:
        """
        Args:
            dt (float): diferencia de temperatura
        """
        x1, y1 = self.coordinates[0]
        x2, y2 = self.coordinates[1]
        x3, y3 = self.coordinates[2]

        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2

        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        k = self.E * self.A() * self.t * self.beta * dt / (1 - self.v)
        Feq = k * np.array([b1, c1, b2, c2, b3, c3])

        self.Feq += Feq
        return Feq

    def add_uniform_distributed_load(
        self,
        qx: float,
        qy: float
        ) -> np.ndarray:
        """
        Añade una carga distribuida uniforme superficial en su plano del elemento
        """
        A = self.A()
        Feq = np.array([qx, qy, qx, qy, qx, qy]) * A/3 * self.t
        self.Feq += Feq
        return Feq

    def add_uniform_edge_load(
        self,
        q: float,
        edge: int
        ) -> np.ndarray:
        """
        Añade una carga distribuida uniforme en el borde del elemento
        """
        if edge not in (1, 2, 3):
            raise ValueError(f"edge debe ser 1, 2 o 3, recibido {edge}")
        xa, ya = self.coordinates[edge - 1]
        xb, yb = self.coordinates[edge % 3]
        beta = np.pi - np.arctan((yb - ya) / (xb - xa))
        length = np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
        qx = q*np.sin(beta)
        qy = q*np.cos(beta)
        if edge == 1:
            Feq = np.array([qx, qy, qx, qy, 0, 0]) * length/2
        elif edge == 2:
            Feq = np.array([0, 0, qx, qy, qx, qy]) * length/2
        elif edge == 3:
            Feq = np.array([qx, qy, 0, 0, qx, qy]) * length/2
        self.Feq += Feq
        return Feq

    def add_linear_edge_load(
        self,
        qi: float,
        qj: float,
        edge: int
        ) -> np.ndarray:
        """
        Añade una carga distribuida lineal en el borde del elemento
        Note: el sentido es en orden de los nodos (antihorario)
        """
        if edge not in (1, 2, 3):
            raise ValueError(f"edge debe ser 1, 2 o 3, recibido {edge}")
        xa, ya = self.coordinates[edge - 1]
        xb, yb = self.coordinates[edge % 3]
        beta = np.pi - np.arctan((yb - ya) / (xb - xa))
        length = np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
        qxi = qi*np.sin(beta)
        qyi = qi*np.cos(beta)
        qxj = qj*np.sin(beta)
        qyj = qj*np.cos(beta)
        if edge == 1:
            Feq = np.array([qxi, qyi, qxj, qyj, 0, 0]) * length/2
        elif edge == 2:
            Feq = np.array([0, 0, qxi, qyi, qxj, qyj]) * length/2
        elif edge == 3:
            Feq = np.array([qxj, qyj, 0, 0, qxi, qyi]) * length/2
        self.Feq += Feq
        return Feq



class FrameRelease:
    """
    Representa el estado de liberación de un elemento.
    """
    def __init__(
        self,
        uxi: bool=False,
        uyi: bool=False,
        rzi: bool=False,
        uxj: bool=False,
        uyj: bool=False,
        rzj: bool=False
        ) -> None:
        """
        Args:
            uxi (bool, optional): liberación en el nodo inicial en la dirección x. Defaults to False.
            uyi (bool, optional): liberación en el nodo inicial en la dirección y. Defaults to False.
            rzi (bool, optional): liberación en el nodo inicial en la dirección z. Defaults to False.
            uxj (bool, optional): liberación en el nodo final en la dirección x. Defaults to False.
            uyj (bool, optional): liberación en el nodo final en la dirección y. Defaults to False.
            rzj (bool, optional): liberación en el nodo final en la dirección z. Defaults to False.
        """
        self.uxi = uxi
        self.uyi = uyi
        self.rzi = rzi
        self.uxj = uxj
        self.uyj = uyj
        self.rzj = rzj

    def set_release(
        self, 
        uxi: bool=False, 
        uyi: bool=False, 
        rzi: bool=False, 
        uxj: bool=False, 
        uyj: bool=False, 
        rzj: bool=False
        ) -> None:
        """
        Args:
            uxi (bool, optional): liberación en el nodo inicial en la dirección x. Defaults to False.
            uyi (bool, optional): liberación en el nodo inicial en la dirección y. Defaults to False.
            rzi (bool, optional): liberación en el nodo inicial en la dirección z. Defaults to False.
            uxj (bool, optional): liberación en el nodo final en la dirección x. Defaults to False.
            uyj (bool, optional): liberación en el nodo final en la dirección y. Defaults to False.
            rzj (bool, optional): liberación en el nodo final en la dirección z. Defaults to False.
        """
        self.uxi = uxi
        self.uyi = uyi
        self.rzi = rzi
        self.uxj = uxj
        self.uyj = uyj
        self.rzj = rzj

    def get_release(
        self
        ) -> Tuple[bool, bool, bool, bool, bool, bool]:
        """
        Returns:
            Tuple[bool, bool, bool, bool, bool, bool]: liberación en el nodo inicial en la dirección x, y, z y liberación en el nodo final en la dirección x, y, z.
        """
        return self.uxi, self.uyi, self.rzi, self.uxj, self.uyj, self.rzj

    def get_dof_release(
        self
        ) -> List[int]:
        """
        Returns:
            List[int]: grados de libertad liberados.
        """
        release = []
        if self.uxi:
            release.append(0)
        if self.uyi:
            release.append(1)
        if self.rzi:
            release.append(2)
        if self.uxj:
            release.append(3)
        if self.uyj:
            release.append(4)
        if self.rzj:
            release.append(5)
        return release

    def __str__(
        self
        ) -> str:
        """
        Returns:
            str: representación en cadena de la liberación del elemento.
        """
        return f"FrameRelease(uxi={self.uxi}, uyi={self.uyi}, rzi={self.rzi}, uxj={self.uxj}, uyj={self.uyj}, rzj={self.rzj})"

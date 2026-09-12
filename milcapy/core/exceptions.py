"""Base de excepciones del dominio milcapy.

Fase 2 (alcance mínimo): centralizar la taxonomía de errores para que
el dominio no use ``ValueError``/``KeyError`` crudos de forma dispersa.

Por ahora solo se definen; los módulos migrarán progresivamente.
La API pública existente no cambia.
"""


class MilcapyError(Exception):
    """Error base de milcapy."""


class ModelError(MilcapyError):
    """Modelo inconsistente (nodos/elementos/secciones)."""


class InputError(MilcapyError, ValueError):
    """Input inválido del usuario. Hereda de ValueError por compatibilidad."""


class ConfigurationError(MilcapyError):
    """Configuración inválida del análisis."""


class NumericalError(MilcapyError):
    """Error numérico (matriz singular, Jacobiano no positivo, etc.)."""


class ConvergenceError(NumericalError):
    """Falta de convergencia / sistema singular."""


class ResultsError(MilcapyError):
    """Acceso a resultados inexistentes o inconsistentes."""

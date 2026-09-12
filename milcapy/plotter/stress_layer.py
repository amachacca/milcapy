"""Capa de visualización de campos de esfuerzos/deformaciones (solo presentación).

``StressLayer`` es propiedad de ``Plotter`` y traduce el estado
(``PlotterOptions.UI_stress / stress_field / stress_colormap`` + patrón
actual) a artists matplotlib. El cálculo vive en
``PlotterValues.nodal_field`` / ``field_service``; aquí solo dibujo.

Se dibuja bajo demanda para el patrón actual (``show``) y se retira con
``hide``. No se pre-construye por patrón: el campo puede cambiar sin
reconstruir geometría.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from milcapy.plotter.plotter import Plotter
    from milcapy.utils.types import FieldType


class StressLayer:
    """Dibuja el mapa de colores del campo activo sobre los ejes del Plotter."""

    def __init__(self, plotter: "Plotter") -> None:
        self._plotter = plotter
        self._collection = None
        self._colorbar = None
        self._pattern: Optional[str] = None
        self._field_key: Optional[str] = None

    # -- estado ---------------------------------------------------------
    @property
    def visible(self) -> bool:
        return self._collection is not None

    @property
    def info(self) -> dict:
        return {"pattern": self._pattern, "field": self._field_key}

    # -- API ------------------------------------------------------------
    def show(self, field: "FieldType | str | None" = None, cmap: Optional[str] = None) -> bool:
        """Dibuja el campo para el patrón actual. Retorna False si no hay malla."""
        import matplotlib.tri as tri

        from milcapy.utils.types import FieldType, to_enum

        plotter = self._plotter
        options = plotter.plotter_options
        if field is None:
            field = options.stress_field
        if isinstance(field, str):
            field = to_enum(field, FieldType)
        cmap = cmap or options.stress_colormap

        self.hide(draw=False)
        x, y, tris, vals = plotter.current_values.nodal_field(field)
        if tris.shape[0] == 0 or np.all(~np.isfinite(vals)):
            plotter.figure.canvas.draw_idle()
            return False
        triangulation = tri.Triangulation(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), tris)
        masked = np.ma.masked_invalid(np.asarray(vals, dtype=float))
        self._collection = plotter.axes.tripcolor(
            triangulation, masked, shading="gouraud",
            cmap=cmap, alpha=options.stress_alpha, zorder=1)
        self._colorbar = plotter.figure.colorbar(
            self._collection, ax=plotter.axes, label=f"{field.value}")
        self._pattern = plotter.current_load_pattern
        self._field_key = field.value
        plotter.figure.canvas.draw_idle()
        return True

    def hide(self, draw: bool = True) -> None:
        plotter = self._plotter
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None
        if self._collection is not None:
            try:
                self._collection.remove()
            except Exception:
                pass
            self._collection = None
        self._pattern = None
        self._field_key = None
        if draw:
            try:
                plotter.figure.canvas.draw_idle()
            except Exception:
                pass

    def refresh(self) -> bool:
        """Redibuja si la opción UI_stress está activa; si no, oculta."""
        options = self._plotter.plotter_options
        if options.UI_stress:
            return self.show()
        self.hide()
        return False

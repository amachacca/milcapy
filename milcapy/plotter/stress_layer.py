"""Capa de visualización de campos de esfuerzos/deformaciones (solo presentación).

``StressLayer`` es propiedad de ``Plotter`` y traduce el estado
(``PlotterOptions.UI_stress / stress_field / stress_colormap`` + patrón
actual) a artists matplotlib. El cálculo vive en
``PlotterValues.nodal_field`` / ``field_service``; aquí solo dibujo.

Dibujo: ``tricontourf`` continuo sobre la malla unificada (triángulos CST +
quads divididos) con valores nodales promediados ponderados por área.
Colorbar única flotante dentro del canvas (inset axes).

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
        self._artists: list = []
        self._colorbar = None
        self._cax = None
        self._pattern: Optional[str] = None
        self._field_key: Optional[str] = None

    # -- estado ---------------------------------------------------------
    @property
    def visible(self) -> bool:
        return len(self._artists) > 0

    @property
    def info(self) -> dict:
        return {"pattern": self._pattern, "field": self._field_key}

    # -- API ------------------------------------------------------------
    def show(self, field: "FieldType | str | None" = None, cmap: Optional[str] = None) -> bool:
        """Dibuja el campo para el patrón actual. Retorna False si no hay malla."""
        import matplotlib.tri as tri
        from matplotlib.colors import Normalize

        from milcapy.postprocess.field_service import split_field
        from milcapy.utils.types import FieldType, to_enum

        plotter = self._plotter
        options = plotter.plotter_options
        if field is None:
            field = options.stress_field
        if isinstance(field, str):
            field = to_enum(field, FieldType)
        cmap = cmap or options.stress_colormap
        levels = max(int(getattr(options, "stress_levels", 20)), 2)

        self.hide(draw=False)
        model = plotter.model
        pattern = plotter.current_load_pattern
        x, y, tris, nodal = split_field(model, pattern, field)

        finite = np.asarray(nodal, dtype=float)
        finite = finite[np.isfinite(finite)]
        if tris.shape[0] == 0 or finite.size == 0:
            plotter.figure.canvas.draw_idle()
            return False
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmax - vmin <= 0:
            eps = max(abs(vmax) * 1e-9, 1e-12)
            vmin, vmax = vmin - eps, vmax + eps
        norm = Normalize(vmin=vmin, vmax=vmax)
        alpha = float(getattr(options, "stress_alpha", 0.8))

        mask = [bool(np.any(~np.isfinite(nodal[t]))) for t in tris]
        triangulation = tri.Triangulation(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), tris)
        triangulation.set_mask(mask)
        cs = plotter.axes.tricontourf(
            triangulation, np.asarray(nodal, dtype=float),
            levels=levels, cmap=cmap, norm=norm, alpha=alpha, zorder=2)
        self._artists.append(cs)
        mappable = cs

        # Colorbar flotante dentro del canvas
        self._cax = plotter.axes.inset_axes([0.88, 0.12, 0.035, 0.76])
        self._colorbar = plotter.figure.colorbar(mappable, cax=self._cax, label=f"{field.value}")
        self._pattern = pattern
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
        if self._cax is not None:
            try:
                self._cax.remove()
            except Exception:
                pass
            self._cax = None
        for artist in self._artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._artists = []
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

"""Capa de visualización de campos de esfuerzos/deformaciones (solo presentación).

``StressLayer`` es propiedad de ``Plotter`` y traduce el estado
(``PlotterOptions.UI_stress / stress_field / stress_colormap`` + patrón
actual) a artists matplotlib. El cálculo vive en
``PlotterValues.nodal_field`` / ``field_service``; aquí solo dibujo.

Dibujo:
  - Quads (Q4/Q6/Q6I/Q8): ``tricontourf`` (contour suave, niveles configurables).
  - CST: ``tripcolor`` plano, un solo color por triángulo (constant strain).
  - Colorbar única flotante dentro del canvas (inset axes), norma compartida.

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
        x, y, cst_tris, cst_face, quad_tris, quad_nodal = split_field(model, pattern, field)

        fin_cst = cst_face[np.isfinite(cst_face)] if cst_face.size else np.zeros(0)
        fin_quad = quad_nodal[np.isfinite(quad_nodal)] if quad_nodal.size else np.zeros(0)
        if fin_cst.size + fin_quad.size == 0:
            plotter.figure.canvas.draw_idle()
            return False
        vmin = float(min(fin_cst.min() if fin_cst.size else np.inf,
                         fin_quad.min() if fin_quad.size else np.inf))
        vmax = float(max(fin_cst.max() if fin_cst.size else -np.inf,
                         fin_quad.max() if fin_quad.size else -np.inf))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            plotter.figure.canvas.draw_idle()
            return False
        if vmax - vmin <= 0:
            eps = max(abs(vmax) * 1e-9, 1e-12)
            vmin, vmax = vmin - eps, vmax + eps
        norm = Normalize(vmin=vmin, vmax=vmax)
        alpha = float(getattr(options, "stress_alpha", 0.8))

        mappable = None
        # Quads: contour suave
        if quad_tris.shape[0] > 0 and np.any(np.isfinite(quad_nodal)):
            qmask = [bool(np.any(~np.isfinite(quad_nodal[t]))) for t in quad_tris]
            triangulation = tri.Triangulation(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float), quad_tris)
            triangulation.set_mask(qmask)
            cs = plotter.axes.tricontourf(
                triangulation, np.asarray(quad_nodal, dtype=float),
                levels=levels, cmap=cmap, norm=norm, alpha=alpha, zorder=2)
            self._artists.append(cs)
            mappable = cs
        # CST: un color plano por elemento (constant strain)
        if cst_tris.shape[0] > 0 and np.any(np.isfinite(cst_face)):
            masked_face = np.ma.masked_invalid(np.asarray(cst_face, dtype=float))
            coll = plotter.axes.tripcolor(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                np.asarray(cst_tris, dtype=int), facecolors=masked_face,
                shading="flat", cmap=cmap, norm=norm, alpha=alpha, zorder=2)
            self._artists.append(coll)
            if mappable is None:
                mappable = coll

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

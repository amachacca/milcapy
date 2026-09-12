"""Servicio de campos para visualización (negocio puro, sin matplotlib/Qt).

Responsabilidades:
  - Traducir (patrón, FieldType) a campo nodal global promediado.
  - Promediado nodal ponderado por área entre elementos adyacentes.
  - Campos derivados: VM, S1, S2, UMAG.
  - Mallas de dibujo: triángulos (CST directo, quads divididos en 2).

Convenciones:
  - Strains/stresses por elemento están en Results como nodales (nn_elem,3)
    en orden [ex/ey/exy o sx/sy/sxy] (ver membrane_pp).
  - CST aporta réplicas nodales (3,3) + centroides (3,).
  - IDs de nodo arbitrarios: se usa mapa id->índice, nunca ``id-1``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from milcapy.postprocess.membrane_pp import quad_area, von_mises, principal_stresses
from milcapy.utils.types import FieldType

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.core.results import Results

__all__ = ["nodal_field", "split_field", "element_field_data", "supported_fields", "field_label"]

_STRESS = {"SX": 0, "SY": 1, "SXY": 2}
_STRAIN = {"EX": 0, "EY": 1, "EXY": 2}


def _coerce_field(field) -> FieldType:
    if isinstance(field, FieldType):
        return field
    try:
        return FieldType(field)
    except Exception as exc:
        raise ValueError(f"Campo no válido: {field}") from exc


def _derived_value(field: FieldType, sx, sy, sxy):
    if field == FieldType.VM:
        return von_mises(sx, sy, sxy)
    s1, s2 = principal_stresses(sx, sy, sxy)
    return s1 if field == FieldType.S1 else s2


def supported_fields() -> list[FieldType]:
    return [f for f in FieldType]


def field_label(field: FieldType) -> str:
    return field.value if isinstance(field, FieldType) else str(field)


def nodal_field(model: "SystemMilcaModel", pattern: str, field: FieldType):
    """Campo nodal promediado ponderado por área.

    Returns:
        x, y (n_nodes,), tris (nt,3) índices 0-based, vals (n_nodes,).
        Nodos sin aporte reciben nan.
    """
    if not isinstance(field, FieldType):
        try:
            field = FieldType(field)
        except Exception as exc:
            raise ValueError(f"Campo no válido: {field}") from exc
    if pattern not in model.results:
        raise KeyError(f"Sin resultados para el patrón '{pattern}'")
    results = model.results[pattern]

    node_ids = list(model.nodes.keys())
    index = {nid: k for k, nid in enumerate(node_ids)}
    n = len(node_ids)
    x = np.array([model.nodes[nid].vertex.x for nid in node_ids], dtype=float)
    y = np.array([model.nodes[nid].vertex.y for nid in node_ids], dtype=float)

    acc = np.zeros(n)
    wacc = np.zeros(n)
    tris: list[list[int]] = []

    def _add_quad(n1, n2, n3, n4):
        tris.append([index[n1], index[n2], index[n3]])
        tris.append([index[n1], index[n3], index[n4]])

    def _accumulate(nids, vals, area):
        a = max(float(area), 1e-12)
        for nid, v in zip(nids, vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            k = index[nid]
            acc[k] += float(v) * a
            wacc[k] += a

    def _derived(sx, sy, sxy):
        if field == FieldType.VM:
            return von_mises(sx, sy, sxy)
        s1, s2 = principal_stresses(sx, sy, sxy)
        return s1 if field == FieldType.S1 else s2

    # --- CST ---
    for eid, cst in model.csts.items():
        data = results.CST.get(eid)
        if not data:
            continue
        nids = [cst.node1.id, cst.node2.id, cst.node3.id]
        tris.append([index[i] for i in nids])
        area = abs(float(cst.A))
        if field in (FieldType.SX, FieldType.SY, FieldType.SXY):
            arr = np.asarray(data.get("stresses_nodes", np.tile(np.asarray(data["stresses"]).ravel()[:3], (3, 1))), dtype=float)
            _accumulate(nids, arr[:, _STRESS[field.name]], area)
        elif field in (FieldType.EX, FieldType.EY, FieldType.EXY):
            arr = np.asarray(data.get("strains_nodes", np.tile(np.asarray(data["strains"]).ravel()[:3], (3, 1))), dtype=float)
            _accumulate(nids, arr[:, _STRAIN[field.name]], area)
        elif field in (FieldType.VM, FieldType.S1, FieldType.S2):
            arr = np.asarray(data.get("stresses_nodes", np.tile(np.asarray(data["stresses"]).ravel()[:3], (3, 1))), dtype=float)
            _accumulate(nids, [_derived(*row) for row in arr], area)
        # UX/UY/UMAG se resuelven abajo (nodales directos)

    # --- Quads 12 DOF (Q6 / MQ6IMod) ---
    for eid, ele in model.membrane_q3dof.items():
        data = results.membrane_q3dof.get(eid)
        if not data or "stresses" not in data:
            continue
        nids = [ele.node1.id, ele.node2.id, ele.node3.id, ele.node4.id]
        _add_quad(*nids)
        area = quad_area(ele)
        if field in (FieldType.SX, FieldType.SY, FieldType.SXY):
            _accumulate(nids, np.asarray(data["stresses"], dtype=float)[:, _STRESS[field.name]], area)
        elif field in (FieldType.EX, FieldType.EY, FieldType.EXY):
            _accumulate(nids, np.asarray(data["strains"], dtype=float)[:, _STRAIN[field.name]], area)
        elif field in (FieldType.VM, FieldType.S1, FieldType.S2):
            arr = np.asarray(data["stresses"], dtype=float)
            _accumulate(nids, [_derived(*row) for row in arr], area)

    # --- Quads 8 DOF (Q4 / Q6I / Q8) ---
    for eid, ele in model.membrane_q2dof.items():
        data = results.membrane_q2dof.get(eid)
        if not data or "stresses" not in data:
            continue
        nids = [ele.node1.id, ele.node2.id, ele.node3.id, ele.node4.id]
        _add_quad(*nids)
        area = quad_area(ele)
        if field in (FieldType.SX, FieldType.SY, FieldType.SXY):
            _accumulate(nids, np.asarray(data["stresses"], dtype=float)[:, _STRESS[field.name]], area)
        elif field in (FieldType.EX, FieldType.EY, FieldType.EXY):
            _accumulate(nids, np.asarray(data["strains"], dtype=float)[:, _STRAIN[field.name]], area)
        elif field in (FieldType.VM, FieldType.S1, FieldType.S2):
            arr = np.asarray(data["stresses"], dtype=float)
            _accumulate(nids, [_derived(*row) for row in arr], area)

    vals = np.full(n, np.nan)
    mask = wacc > 0
    vals[mask] = acc[mask] / wacc[mask]

    if field in (FieldType.UX, FieldType.UY, FieldType.UMAG):
        vals = np.full(n, np.nan)
        for nid in node_ids:
            try:
                d = results.get_node_displacements(nid)[:2]
            except KeyError:
                continue
            if field == FieldType.UX:
                vals[index[nid]] = d[0]
            elif field == FieldType.UY:
                vals[index[nid]] = d[1]
            else:
                vals[index[nid]] = float(np.sqrt(d[0] ** 2 + d[1] ** 2)

            )
    return x, y, np.asarray(tris, dtype=int).reshape(-1, 3) if tris else np.zeros((0, 3), dtype=int), vals


def split_field(model: "SystemMilcaModel", pattern: str, field: FieldType):
    """Malla unificada con promedio nodal ponderado por área (CST incluido).

    A diferencia del diseño anterior (CST plano por elemento), aquí el CST
    aporta sus valores nodales (réplica constant-strain) al promedio, por lo
    que el dibujo es un contour continuo sobre todos los elementos, como en
    los postprocesadores comerciales. El suavizado puede ocultar
    discontinuidades inter-elemento en mallas gruesas.

    Returns:
        x, y (n,): coordenadas nodales.
        tris (nt,3): triángulos CST + quads divididos en 2.
        nodal (n,): promedio ponderado por área (nan donde no aportan).
    """
    field = _coerce_field(field)
    if pattern not in model.results:
        raise KeyError(f"Sin resultados para el patrón '{pattern}'")
    results = model.results[pattern]

    node_ids = list(model.nodes.keys())
    index = {nid: k for k, nid in enumerate(node_ids)}
    n = len(node_ids)
    x = np.array([model.nodes[nid].vertex.x for nid in node_ids], dtype=float)
    y = np.array([model.nodes[nid].vertex.y for nid in node_ids], dtype=float)

    acc = np.zeros(n)
    wacc = np.zeros(n)
    tris: list[list[int]] = []

    def _acc(nids, vals, area):
        a = max(float(area), 1e-12)
        for nid, v in zip(nids, vals):
            if v is None:
                continue
            fv = float(v)
            if np.isnan(fv):
                continue
            k = index[nid]
            acc[k] += fv * a
            wacc[k] += a

    def _vals(data, nids, area):
        if field in (FieldType.SX, FieldType.SY, FieldType.SXY):
            arr = np.asarray(data["stresses"], dtype=float)
            if arr.ndim == 1:
                arr = np.tile(arr.ravel()[:3], (len(nids), 1))
            _acc(nids, arr[:, _STRESS[field.name]], area)
        elif field in (FieldType.EX, FieldType.EY, FieldType.EXY):
            arr = np.asarray(data["strains"], dtype=float)
            if arr.ndim == 1:
                arr = np.tile(arr.ravel()[:3], (len(nids), 1))
            _acc(nids, arr[:, _STRAIN[field.name]], area)
        elif field in (FieldType.VM, FieldType.S1, FieldType.S2):
            arr = np.asarray(data["stresses"], dtype=float)
            if arr.ndim == 1:
                arr = np.tile(arr.ravel()[:3], (len(nids), 1))
            _acc(nids, [_derived_value(field, *row) for row in arr], area)

    # --- CST: aporta réplica nodal al promedio (ya no plano por elemento) ---
    for eid, cst in model.csts.items():
        data = results.CST.get(eid)
        if not data or "stresses" not in data or "strains" not in data:
            continue
        nids = [cst.node1.id, cst.node2.id, cst.node3.id]
        tris.append([index[i] for i in nids])
        _vals(data, nids, abs(float(cst.A)))

    # --- Quads (suave): Q6/MQ6IMod y Q4/Q6I/Q8 ---
    for eid, ele in model.membrane_q3dof.items():
        data = results.membrane_q3dof.get(eid)
        if not data or "stresses" not in data or "strains" not in data:
            continue
        nids = [ele.node1.id, ele.node2.id, ele.node3.id, ele.node4.id]
        tris.append([index[nids[0]], index[nids[1]], index[nids[2]]])
        tris.append([index[nids[0]], index[nids[2]], index[nids[3]]])
        _vals(data, nids, quad_area(ele))
    for eid, ele in model.membrane_q2dof.items():
        data = results.membrane_q2dof.get(eid)
        if not data or "stresses" not in data or "strains" not in data:
            continue
        nids = [ele.node1.id, ele.node2.id, ele.node3.id, ele.node4.id]
        tris.append([index[nids[0]], index[nids[1]], index[nids[2]]])
        tris.append([index[nids[0]], index[nids[2]], index[nids[3]]])
        _vals(data, nids, quad_area(ele))

    nodal = np.full(n, np.nan)
    mask = wacc > 0
    nodal[mask] = acc[mask] / wacc[mask]

    if field in (FieldType.UX, FieldType.UY, FieldType.UMAG):
        nodal = np.full(n, np.nan)
        for nid in node_ids:
            try:
                d = results.get_node_displacements(nid)[:2]
            except KeyError:
                continue
            if field == FieldType.UX:
                nodal[index[nid]] = d[0]
            elif field == FieldType.UY:
                nodal[index[nid]] = d[1]
            else:
                nodal[index[nid]] = float(np.sqrt(d[0] ** 2 + d[1] ** 2))

    return (
        x, y,
        np.asarray(tris, dtype=int).reshape(-1, 3) if tris else np.zeros((0, 3), dtype=int),
        nodal,
    )


def _quad_phys(corner_xy: np.ndarray, xi: float, eta: float) -> tuple[float, float]:
    """Posición física bilineal desde esquinas (evita element.coordinates roto en MQ6)."""
    N = np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                  (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4.0
    return float(N @ corner_xy[:, 0]), float(N @ corner_xy[:, 1])


def element_field_data(model: "SystemMilcaModel", pattern: str, kind: str, ele_id: int) -> dict:
    """Detalle por elemento para inspección: nodos + TODOS los puntos de Gauss.

    Args:
        kind: "cst" | "q3" (Q6/MQ6IMod) | "q2" (Q4/Q6I/Q8).
    Returns:
        dict con label, node_ids, node_xy (nn,2), strains_nodes/stresses_nodes
        (nn,3), gauss: lista de {xi, eta, x, y, strains (3,), stresses (3,)},
        ngauss, state, thickness.
    """
    if pattern not in model.results:
        raise KeyError(f"Sin resultados para el patrón '{pattern}'")
    results = model.results[pattern]
    if kind == "cst":
        ele = model.csts[ele_id]
        data = results.CST[ele_id]
        nodes = [ele.node1, ele.node2, ele.node3]
        label = f"CST {ele_id}"
    elif kind == "q3":
        ele = model.membrane_q3dof[ele_id]
        data = results.membrane_q3dof[ele_id]
        nodes = [ele.node1, ele.node2, ele.node3, ele.node4]
        label = f"Membrana Q3DOF {ele_id} ({type(ele).__name__})"
    elif kind == "q2":
        ele = model.membrane_q2dof[ele_id]
        data = results.membrane_q2dof[ele_id]
        nodes = [ele.node1, ele.node2, ele.node3, ele.node4]
        label = f"Membrana Q2DOF {ele_id} ({type(ele).__name__})"
    else:
        raise ValueError(f"kind debe ser cst/q3/q2, recibido {kind!r}")

    node_ids = [nd.id for nd in nodes]
    node_xy = np.array([nd.vertex.coordinates for nd in nodes], dtype=float)
    try:
        node_disp = np.array([results.get_node_displacements(i)[:2] for i in node_ids], dtype=float)
    except KeyError:
        node_disp = np.full((len(node_ids), 2), np.nan)
    strains_n = np.asarray(data.get("strains_nodes", data["strains"]), dtype=float).reshape(-1, 3)
    stresses_n = np.asarray(data.get("stresses_nodes", data["stresses"]), dtype=float).reshape(-1, 3)

    gauss = []
    if kind == "cst":
        cx, cy = node_xy.mean(axis=0)
        e = np.asarray(data["strains"], dtype=float).ravel()[:3]
        s = np.asarray(data["stresses"], dtype=float).ravel()[:3]
        gauss.append({"xi": None, "eta": None, "x": float(cx), "y": float(cy),
                      "strains": e, "stresses": s})
    else:
        corner_xy = node_xy[:4]
        for i, (xi, eta) in enumerate(zip(np.asarray(ele.xi).ravel(), np.asarray(ele.eta).ravel())):
            gx, gy = _quad_phys(corner_xy, float(xi), float(eta))
            gauss.append({"xi": float(xi), "eta": float(eta), "x": gx, "y": gy,
                          "strains": np.asarray(data["strains_gauss"][i], dtype=float).ravel()[:3],
                          "stresses": np.asarray(data["stresses_gauss"][i], dtype=float).ravel()[:3]})
    return {"label": label, "kind": kind, "ele_id": ele_id,
            "node_ids": node_ids, "node_xy": node_xy, "node_disp": node_disp,
            "strains_nodes": strains_n, "stresses_nodes": stresses_n,
            "gauss": gauss, "ngauss": len(gauss),
            "state": getattr(ele.state, "value", str(ele.state)),
            "thickness": float(ele.section.t)}

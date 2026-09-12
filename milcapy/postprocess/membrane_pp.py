"""Postproceso de membranas: deformaciones/tensiones en Gauss extrapoladas a nodos.

Flujo tipo comercial:
  1. u_global -> u_elemento (ya en Results por PostProcessing).
  2. En cada punto de Gauss: eps = B(xi,eta) @ u_full, sig = D @ eps.
  3. Extrapolación Gauss -> nodos del elemento (pinv de funciones de forma
     evaluadas en Gauss; exacta para Q4 bilineal, mínimos cuadrados si
     ng != nn).
  4. El promediado nodal global ponderado por área vive en FieldService.

Casos especiales:
  - CST: B constante; nodal = réplica del valor del centroide.
  - Q6 (drilling): u en Results está intercalado [ux,uy,rz]*4; B() espera
    bloque [u(8), r(4)]; se aplica permutación inversa documentada en MQ6.Ki.
  - Q6I: u en Results es el condensado (8); se recuperan los 4 modos
    incompatibles con alpha = -inv(kss) @ ksc @ u.
  - Q8: u en Results es el condensado (8 esquinas); se recuperan los 8 DOF
    medios con u_mid = -inv(Kii) @ Kic @ u_corners. Nodos de esquina por
    evaluación directa B(+-1,+-1) (extrapolar 4 Gauss a 8 nodos es
    rango-deficiente; se documenta).
  - MQ6IMod: se delega al elemento de desplazamientos según ele_type
    (MQ6 usa este mismo objeto).
"""
from __future__ import annotations

import numpy as np

from milcapy.core.exceptions import NumericalError

# Esquinas en orden nodal 1..4
_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

# Permutación MQ6 bloque -> intercalado (ver MembraneQuad6.Ki)
_MQ6_PERM = [0, 1, 8, 2, 3, 9, 4, 5, 10, 6, 7, 11]
_MQ6_INV_PERM = list(np.argsort(_MQ6_PERM))


def von_mises(sx: float, sy: float, sxy: float) -> float:
    """Von Mises en tensión plana."""
    return float(np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3.0 * sxy ** 2))


def principal_stresses(sx: float, sy: float, sxy: float) -> tuple[float, float]:
    """Tensiones principales S1 >= S2 en tensión plana."""
    avg = 0.5 * (sx + sy)
    rad = np.sqrt(((sx - sy) / 2.0) ** 2 + sxy ** 2)
    return float(avg + rad), float(avg - rad)


def quad_area(element) -> float:
    """Área del cuadrilátero (esquinas) por shoelace. Para ponderar promedios."""
    x, y = element.get_coordinates()
    x = np.asarray(x, dtype=float).ravel()[:4]
    y = np.asarray(y, dtype=float).ravel()[:4]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _gauss_points(element) -> tuple[np.ndarray, np.ndarray]:
    xi = np.asarray(element.xi, dtype=float).ravel()
    eta = np.asarray(element.eta, dtype=float).ravel()
    return xi, eta


def _extrapolation_matrix(element, nnodes: int = 4) -> np.ndarray:
    """Matriz E (nn x ng): nodal = E @ gauss.

    G[i,j] = N_j(gauss_i); E = pinv(G). Para Q4 4x4 es la inversa exacta.
    """
    xi, eta = _gauss_points(element)
    ng = xi.shape[0]
    G = np.zeros((ng, nnodes))
    for i in range(ng):
        N = element.shape_function(float(xi[i]), float(eta[i]))
        G[i, :] = np.asarray(N, dtype=float).ravel()[:nnodes]
    return np.linalg.pinv(G)


def _strains_stresses_at_gauss(element, u_full: np.ndarray, D: np.ndarray, Bfun=None):
    xi, eta = _gauss_points(element)
    ng = xi.shape[0]
    strains = np.zeros((ng, 3))
    stresses = np.zeros((ng, 3))
    Bfun = Bfun or element.B
    for i in range(ng):
        B = np.asarray(Bfun(float(xi[i]), float(eta[i])), dtype=float)
        e = B @ u_full
        strains[i, :] = e.ravel()[:3]
        stresses[i, :] = (D @ e).ravel()[:3]
    return strains, stresses


def recover_cst(cst, u6: np.ndarray):
    """CST: (strains_nodes (3,3), stresses_nodes (3,3), strains_gauss (1,3), stresses_gauss (1,3))."""
    u = np.asarray(u6, dtype=float).ravel()[:6]
    e = np.asarray(cst.B @ u, dtype=float).ravel()[:3]
    s = np.asarray(cst.D @ e, dtype=float).ravel()[:3]
    return np.tile(e, (3, 1)), np.tile(s, (3, 1)), e.reshape(1, 3), s.reshape(1, 3)


def recover_q4(element, u8: np.ndarray):
    u = np.asarray(u8, dtype=float).ravel()[:8]
    D = np.asarray(element.D(), dtype=float)
    strains_g, stresses_g = _strains_stresses_at_gauss(element, u, D)
    E = _extrapolation_matrix(element, 4)
    return E @ strains_g, E @ stresses_g, strains_g, stresses_g


def recover_q6(element, u12_intercalated: np.ndarray):
    """Q6 con drilling: des-permuta u a bloque antes de B (3x12)."""
    u_inter = np.asarray(u12_intercalated, dtype=float).ravel()[:12]
    u = u_inter[_MQ6_INV_PERM]
    D = np.asarray(element.D(), dtype=float)
    strains_g, stresses_g = _strains_stresses_at_gauss(element, u, D, Bfun=element.B)
    E = _extrapolation_matrix(element, 4)
    return E @ strains_g, E @ stresses_g, strains_g, stresses_g


def recover_q6i(element, u8: np.ndarray):
    """Q6I: recupera modos incompatibles condensados y luego extrapola."""
    u_c = np.asarray(u8, dtype=float).ravel()[:8]
    # Reconstruye bloques de la matriz completa 12x12
    Kfull = np.zeros((12, 12))
    for i in range(len(element.xi)):
        Kfull += np.asarray(element.phiKi(element.xi[i], element.eta[i]), dtype=float) * element.w[i]
    kcc, kcs = Kfull[:8, :8], Kfull[:8, 8:]
    ksc, kss = Kfull[8:, :8], Kfull[8:, 8:]
    try:
        alpha = -np.linalg.solve(kss, ksc @ u_c)
    except np.linalg.LinAlgError as exc:
        raise NumericalError(f"Q6I: bloque incompatible singular en elemento {element.id}") from exc
    u12 = np.concatenate([u_c, alpha])
    D = np.asarray(element.D(), dtype=float)
    strains_g, stresses_g = _strains_stresses_at_gauss(element, u12, D, Bfun=element.B)
    E = _extrapolation_matrix(element, 4)
    return E @ strains_g, E @ stresses_g, strains_g, stresses_g


def recover_q8(element, u8corners: np.ndarray):
    """Q8: recupera nodos medios condensados; esquinas por evaluación directa.

    u_full (16) = [corners(8); mids(8)] con u_mid = -inv(Kii) @ Kic @ u_c.
    """
    u_c = np.asarray(u8corners, dtype=float).ravel()[:8]
    Kfull = np.zeros((16, 16))
    for i in range(len(element.xi)):
        Kfull += np.asarray(element.phiKi(element.xi[i], element.eta[i]), dtype=float) * element.w[i]
    c = [0, 1, 2, 3, 4, 5, 6, 7]
    m = [8, 9, 10, 11, 12, 13, 14, 15]
    Kcc = Kfull[np.ix_(c, c)]
    Kci = Kfull[np.ix_(c, m)]
    Kic = Kfull[np.ix_(m, c)]
    Kii = Kfull[np.ix_(m, m)]
    try:
        u_mid = -np.linalg.solve(Kii, Kic @ u_c)
    except np.linalg.LinAlgError as exc:
        raise NumericalError(f"Q8: bloque de nodos medios singular en elemento {element.id}") from exc
    u16 = np.concatenate([u_c, u_mid])
    D = np.asarray(element.D(), dtype=float)
    strains_g, stresses_g = _strains_stresses_at_gauss(element, u16, D, Bfun=element.B)
    # Evaluación directa en las 4 esquinas (documentado: extrapolar 4 Gauss
    # a 8 nodos es rango-deficiente en integración reducida).
    strains_n = np.zeros((4, 3))
    stresses_n = np.zeros((4, 3))
    for k, (xi, eta) in enumerate(_CORNERS):
        B = np.asarray(element.B(float(xi), float(eta)), dtype=float)
        e = B @ u16
        strains_n[k, :] = e.ravel()[:3]
        stresses_n[k, :] = (D @ e).ravel()[:3]
    return strains_n, stresses_n, strains_g, stresses_g


def recover_q6imod(element, u12_intercalated: np.ndarray):
    """Wrapper MQ6IMod: usa el elemento de desplazamientos según ele_type."""
    from milcapy.utils.types import MembraneQuadElementType

    ele_type = getattr(element, "ele_type", MembraneQuadElementType.MQ6)
    if ele_type == MembraneQuadElementType.MQ6:
        return recover_q6(element, u12_intercalated)
    disp = element.disp_element()
    u = np.asarray(u12_intercalated, dtype=float).ravel()[:12]
    # Solo parte de desplazamientos del vector intercalado
    u8 = u[[0, 1, 3, 4, 6, 7, 9, 10]]
    if ele_type == MembraneQuadElementType.MQ4:
        return recover_q4(disp, u8)
    if ele_type == MembraneQuadElementType.MQ6I:
        return recover_q6i(disp, u8)
    # MQ8*: el wrapper guarda el elemento en q3dof (12 DOF con drilling);
    # la parte de desplazamientos se postprocesa como Q8 condensado.
    return recover_q8(disp, u8)

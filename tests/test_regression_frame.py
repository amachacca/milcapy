"""Regresión numérica mínima: voladizo elástico vs solución analítica.

Viga en voladizo L=5, sección 0.3x0.5, E=2.1e6, carga vertical P=-10 en extremo.
Solución Euler-Bernoulli: dy = P*L^3/(3EI), giro = P*L^2/(2EI).
Se acepta tolerancia 3% por deformación de corte (Timoshenko por defecto).
"""
import numpy as np

from milcapy import SystemModel


def build_cantilever():
    m = SystemModel()
    m.add_material("c", 2.1e6, 0.2)
    m.add_rectangular_section("v", "c", 0.3, 0.5)
    m.add_node(1, 0, 0)
    m.add_node(2, 5, 0)
    m.add_member(1, 1, 2, "v")
    m.add_restraint(1, True, True, True)
    m.add_restraint(2, False, False, False)
    m.add_load_pattern("D")
    m.add_point_load(2, "D", fy=-10)
    return m


def test_cantilever_tip_deflection():
    m = build_cantilever()
    m.solve()
    r = m.get_results("D")
    d2 = r.get_node_displacements(2)
    E, L, P = 2.1e6, 5.0, -10.0
    I = 0.3 * 0.5 ** 3 / 12.0
    dy_ref = P * L ** 3 / (3 * E * I)
    th_ref = P * L ** 2 / (2 * E * I)
    assert abs(d2[1] - dy_ref) / abs(dy_ref) < 0.03
    assert abs(d2[2] - th_ref) / abs(th_ref) < 0.03


def test_cantilever_reactions_equilibrium():
    m = build_cantilever()
    m.solve()
    r = m.get_results("D")
    R = r.get_model_reactions()
    # equilibrio vertical y momento en empotramiento
    assert abs(R[1] - 10.0) < 1e-6
    assert abs(R[2] - 50.0) < 1e-3

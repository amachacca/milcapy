"""Regresión: bugs verificados en Fase 0 (sin cambiar API pública)."""
import numpy as np

from milcapy import SystemModel
from milcapy.loads.load_pattern import LoadPattern


def test_load_pattern_default_multiplier_no_crash():
    lp = LoadPattern("X")
    assert lp.self_weight.multiplier is None
    # negativo sí debe fallar
    try:
        LoadPattern("Y", self_weight_multiplier=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("multiplicador negativo debe lanzar ValueError")


def test_results_get_results_estructura():
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
    m.solve()
    r = m.get_results("D")
    d = r.get_results()
    assert set(d.keys()) == {
        "model", "nodes", "members", "CST",
        "membrane_q3dof", "membrane_q2dof", "trusses",
    }
    assert 2 in d["nodes"]


def test_add_cst_linear_edge_load_delega_correcto():
    m = SystemModel()
    m.add_material("c", 2e6, 0.2)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.add_node(3, 0, 1)
    m.add_cst(1, 1, 2, 3, "s")
    m.add_load_pattern("D")
    # antes lanzaba TypeError por delegar al método uniforme
    m.add_cst_linear_edge_load(1, "D", 5.0, 10.0, 1)
    feq = m.load_patterns["D"].cst_loads[1].Feq
    assert np.any(feq != 0)


def test_cst_edge_3_no_index_error():
    m = SystemModel()
    m.add_material("c", 2e6, 0.2)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.add_node(3, 0, 1)
    m.add_cst(1, 1, 2, 3, "s")
    m.add_load_pattern("D")
    m.add_cst_uniform_edge_load(1, "D", 5.0, 3)
    m.add_cst_linear_edge_load(1, "D", 5.0, 10.0, 3)
    feq = m.load_patterns["D"].cst_loads[1].Feq
    assert np.all(np.isfinite(feq))

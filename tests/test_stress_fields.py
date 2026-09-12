"""Campos de esfuerzos en membranas: Gauss -> nodos -> promedio ponderado."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from milcapy import SystemModel
from milcapy.postprocess import membrane_pp
from milcapy.postprocess.field_service import nodal_field
from milcapy.utils.types import FieldType

E, V = 2e6, 0.2
EX_REF = 1e-3
SX_REF = E * EX_REF  # tracción uniaxial, tensión plana


def _tension_model(build, cst=False):
    m = SystemModel()
    m.add_material("c", E, V)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.add_node(3, 1, 1)
    m.add_node(4, 0, 1)
    build(m)
    for nid in (1, 2, 3, 4):
        m.add_restraint(nid, True, False, True)
    m.add_load_pattern("D")
    m.add_prescribed_dof(1, "D", ux=0.0)
    m.add_prescribed_dof(4, "D", ux=0.0)
    m.add_prescribed_dof(2, "D", ux=EX_REF)
    m.add_prescribed_dof(3, "D", ux=EX_REF)
    m.solve()
    return m


def test_von_mises_and_principals_math():
    assert membrane_pp.von_mises(2000.0, 0.0, 0.0) == pytest.approx(2000.0)
    s1, s2 = membrane_pp.principal_stresses(2000.0, 0.0, 0.0)
    assert (s1, s2) == pytest.approx((2000.0, 0.0))
    assert membrane_pp.von_mises(100.0, 100.0, 0.0) == pytest.approx(100.0)


@pytest.mark.parametrize("name,build,get", [
    ("CST", lambda m: [m.add_cst(1, 1, 2, 3, "s"), m.add_cst(2, 1, 3, 4, "s")], "cst"),
    ("Q4", lambda m: m.add_membrane_q4(1, 1, 2, 3, 4, "s"), "q2"),
    ("Q6", lambda m: m.add_membrane_q6(1, 1, 2, 3, 4, "s"), "q3"),
    ("Q6I", lambda m: m.add_membrane_q6i(1, 1, 2, 3, 4, "s"), "q2"),
])
def test_patch_tension_exact(name, build, get):
    m = _tension_model(build)
    r = m.get_results("D")
    if get == "cst":
        s = np.array([r.get_cst_stresses(1), r.get_cst_stresses(2)])
        e = np.array([r.get_cst_strains(1), r.get_cst_strains(2)])
    elif get == "q3":
        s = np.asarray(r.get_membrane_q3dof_stresses(1))
        e = np.asarray(r.get_membrane_q3dof_strains(1))
        assert s.shape == (4, 3) and e.shape == (4, 3)
        assert np.asarray(r.get_membrane_q3dof_stresses_gauss(1)).shape[1] == 3
    else:
        s = np.asarray(r.get_membrane_q2dof_stresses(1))
        e = np.asarray(r.get_membrane_q2dof_strains(1))
        assert s.shape == (4, 3) and e.shape == (4, 3)
        assert np.asarray(r.get_membrane_q2dof_stresses_gauss(1)).shape[1] == 3
    assert np.allclose(s[:, 0], SX_REF, rtol=1e-9), f"{name} SX"
    assert np.allclose(s[:, 1], 0.0, atol=1e-6), f"{name} SY"
    assert np.allclose(s[:, 2], 0.0, atol=1e-6), f"{name} SXY"
    assert np.allclose(e[:, 0], EX_REF, rtol=1e-9), f"{name} EX"


def test_q8_produces_finite_fields():
    m = SystemModel()
    m.add_material("c", E, V)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.add_node(3, 1, 1)
    m.add_node(4, 0, 1)
    m.add_membrane_q8(1, 1, 2, 3, 4, "s", integration="COMPLETE")
    for nid in (1, 2, 3, 4):
        m.add_restraint(nid, True, False, True)
    m.add_load_pattern("D")
    m.add_prescribed_dof(1, "D", ux=0.0)
    m.add_prescribed_dof(4, "D", ux=0.0)
    m.add_prescribed_dof(2, "D", ux=EX_REF)
    m.add_prescribed_dof(3, "D", ux=EX_REF)
    m.solve()
    r = m.get_results("D")
    s = np.asarray(r.get_membrane_q2dof_stresses(1))
    assert s.shape == (4, 3) and np.all(np.isfinite(s))


def test_field_service_nodal_and_derived():
    m = _tension_model(lambda m: m.add_membrane_q4(1, 1, 2, 3, 4, "s"))
    x, y, tris, sx = nodal_field(m, "D", FieldType.SX)
    assert tris.shape == (2, 3) and np.allclose(sx, SX_REF, rtol=1e-9)
    _, _, _, vm = nodal_field(m, "D", "VM")  # coerción desde str
    _, _, _, s1 = nodal_field(m, "D", FieldType.S1)
    _, _, _, s2 = nodal_field(m, "D", FieldType.S2)
    assert np.allclose(vm, SX_REF, rtol=1e-9)
    assert np.allclose(s1, SX_REF, rtol=1e-9) and np.allclose(s2, 0.0, atol=1e-6)
    with pytest.raises(KeyError):
        nodal_field(m, "NOPE", FieldType.SX)
    with pytest.raises(ValueError):
        nodal_field(m, "D", "CAMPO_X")


def test_plot_field_wrapper_headless():
    m = _tension_model(lambda m: m.add_membrane_q4(1, 1, 2, 3, 4, "s"))
    m.plot_field("SX")
    plt.close("all")
    m.plot_field(FieldType.VM)
    plt.close("all")


def test_stress_layer_headless():
    from milcapy.plotter.plotter import Plotter
    m = _tension_model(lambda m: m.add_membrane_q4(1, 1, 2, 3, 4, "s"))
    m.plotter = Plotter(m)
    m.plotter.initialize_plot()
    assert m.plotter.update_stress_field(visibility=True) is True
    assert m.plotter.stress_layer.visible
    assert m.plotter.update_stress_field(field="VM", visibility=True) is True
    assert m.plotter.stress_layer.info["field"] == "VM"
    assert m.plotter.update_stress_field(visibility=False) is False
    assert not m.plotter.stress_layer.visible


def test_q6imod_ele_type_str_and_postprocess():
    m = SystemModel()
    m.add_material("c", E, V)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.add_node(3, 1, 1)
    m.add_node(4, 0, 1)
    ele = m.add_membrane_q6i_mod(1, 1, 2, 3, 4, "s", ele_type="MQ4")
    assert ele.ele_type.value == "MQ4"
    for nid in (1, 2, 3, 4):
        m.add_restraint(nid, True, False, True)
    m.add_load_pattern("D")
    m.add_prescribed_dof(1, "D", ux=0.0)
    m.add_prescribed_dof(4, "D", ux=0.0)
    m.add_prescribed_dof(2, "D", ux=EX_REF)
    m.add_prescribed_dof(3, "D", ux=EX_REF)
    m.solve()
    s = np.asarray(m.get_results("D").get_membrane_q3dof_stresses(1))
    assert s.shape == (4, 3) and np.all(np.isfinite(s))

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


def _mixed_model():
    from milcapy.plotter.plotter_values import PlotterValues
    PlotterValues._static_data = None  # caché estática global por proceso
    m = SystemModel()
    m.add_material("c", E, V)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.add_node(3, 1, 1)
    m.add_node(4, 0, 1)
    m.add_node(5, 2, 0)
    m.add_node(6, 2, 1)
    m.add_membrane_q4(1, 1, 2, 3, 4, "s")
    m.add_cst(2, 2, 5, 6, "s")
    m.add_cst(3, 2, 6, 3, "s")
    for nid in m.nodes:
        m.add_restraint(nid, True, False, True)
    m.add_load_pattern("D")
    for nid in m.nodes:
        m.add_prescribed_dof(nid, "D", ux=EX_REF * m.nodes[nid].vertex.x)
    m.solve()
    return m


def test_split_field_unified_with_cst_averaged():
    from milcapy.postprocess.field_service import split_field
    m = _mixed_model()
    x, y, tris, nodal = split_field(m, "D", FieldType.SX)
    assert tris.shape == (4, 3)  # 2 CST + quad dividido en 2
    assert nodal.shape == (6,)
    assert np.all(np.isfinite(nodal))


def test_cst_nodes_are_area_weighted_averages():
    """El nodo compartido por 2 CST es el promedio ponderado por áreas."""
    from milcapy.postprocess.field_service import split_field
    m = SystemModel()
    m.add_material("c", E, V)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 2, 0)
    m.add_node(3, 2, 1)
    m.add_node(4, 0, 1)
    m.add_cst(1, 1, 2, 3, "s")  # área 1.0
    m.add_cst(2, 1, 3, 4, "s")  # área 1.0
    for nid in m.nodes:
        m.add_restraint(nid, True, False, True)
    m.add_load_pattern("D")
    m.add_point_load(2, "D", fx=500.0, fy=-300.0)
    m.solve()
    r = m.get_results("D")
    s1 = np.asarray(r.get_cst_stresses(1)).ravel()
    s2 = np.asarray(r.get_cst_stresses(2)).ravel()
    assert not np.allclose(s1, s2)  # estados distintos -> promedio no trivial
    _, _, _, nodal = split_field(m, "D", FieldType.SX)
    idx = {nid: k for k, nid in enumerate(m.nodes.keys())}
    a1, a2 = abs(float(m.csts[1].A)), abs(float(m.csts[2].A))
    assert nodal[idx[1]] == pytest.approx((s1[0] * a1 + s2[0] * a2) / (a1 + a2))
    assert nodal[idx[2]] == pytest.approx(s1[0])  # solo CST 1
    assert nodal[idx[4]] == pytest.approx(s2[0])  # solo CST 2


def test_stress_layer_single_contour_with_floating_colorbar():
    from matplotlib.contour import ContourSet
    from milcapy.plotter.plotter import Plotter
    m = _mixed_model()
    m.plotter = Plotter(m)
    m.plotter.initialize_plot()
    assert m.plotter.update_stress_field(visibility=True) is True
    artists = m.plotter.stress_layer._artists
    assert len(artists) == 1 and isinstance(artists[0], ContourSet)  # un solo contour
    assert len(m.plotter.axes.child_axes) == 1  # colorbar flotante en el canvas
    assert m.plotter.update_stress_field(field="VM", visibility=True) is True
    assert len(m.plotter.stress_layer._artists) == 1
    m.plotter.update_stress_field(visibility=False)
    assert not m.plotter.stress_layer.visible
    assert len(m.plotter.axes.child_axes) == 0
    assert len(m.plotter.figure.axes) == 1
    plt.close("all")


def _gauss_model(kind):
    """Modelo de tracción uniaxial con un solo elemento del tipo pedido."""
    m = SystemModel()
    m.add_material("c", E, V)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.add_node(3, 1, 1)
    m.add_node(4, 0, 1)
    if kind == "cst":
        m.add_cst(1, 1, 2, 3, "s")
        m.add_cst(2, 1, 3, 4, "s")
        ele = m.csts[1]
    elif kind == "q4":
        m.add_membrane_q4(1, 1, 2, 3, 4, "s")
        ele = m.membrane_q2dof[1]
    elif kind == "q6":
        m.add_membrane_q6(1, 1, 2, 3, 4, "s")
        ele = m.membrane_q3dof[1]
    elif kind == "q6i":
        m.add_membrane_q6i(1, 1, 2, 3, 4, "s")
        ele = m.membrane_q2dof[1]
    elif kind == "q8r":
        m.add_membrane_q8(1, 1, 2, 3, 4, "s", integration="REDUCED")
        ele = m.membrane_q2dof[1]
    elif kind == "q8c":
        m.add_membrane_q8(1, 1, 2, 3, 4, "s", integration="COMPLETE")
        ele = m.membrane_q2dof[1]
    elif kind == "q6mod":
        m.add_membrane_q6i_mod(1, 1, 2, 3, 4, "s", ele_type="MQ4")
        ele = m.membrane_q3dof[1]
    for nid in (1, 2, 3, 4):
        m.add_restraint(nid, True, False, True)
    m.add_load_pattern("D")
    m.add_prescribed_dof(1, "D", ux=0.0)
    m.add_prescribed_dof(4, "D", ux=0.0)
    m.add_prescribed_dof(2, "D", ux=EX_REF)
    m.add_prescribed_dof(3, "D", ux=EX_REF)
    m.solve()
    return m, ele


@pytest.mark.parametrize("kind,bucket,ng", [
    ("cst", "cst", 1),
    ("q4", "q2", 4),
    ("q6", "q3", 4),
    ("q6i", "q2", 4),
    ("q8r", "q2", 4),
    ("q8c", "q2", 9),
    ("q6mod", "q3", 4),
])
def test_gauss_points_complete(kind, bucket, ng):
    """Todos los puntos de Gauss calculados: 1/4/9 según integración."""
    from milcapy.postprocess.field_service import element_field_data
    m, ele = _gauss_model(kind)
    if kind != "cst":
        assert len(ele.xi) == len(ele.eta) == len(ele.w) == ng
    d = element_field_data(m, "D", bucket, 1)
    assert d["ngauss"] == ng == len(d["gauss"])
    for g in d["gauss"]:
        assert np.all(np.isfinite(g["strains"])) and np.all(np.isfinite(g["stresses"]))
        assert set(g.keys()) == {"xi", "eta", "x", "y", "strains", "stresses"}
    # coherencia con lo guardado en Results
    r = m.get_results("D")
    store = {"cst": r.CST, "q3": r.membrane_q3dof, "q2": r.membrane_q2dof}[bucket][1]
    assert np.asarray(store["strains_gauss"]).shape == (ng, 3)
    assert np.asarray(store["stresses_gauss"]).shape == (ng, 3)


@pytest.mark.parametrize("kind,bucket", [
    ("cst", "cst"), ("q4", "q2"), ("q6", "q3"), ("q6i", "q2"),
])
def test_gauss_values_exact_in_tension(kind, bucket):
    """En tracción uniforme cada Gauss da SX=E·ex (formulaciones exactas)."""
    from milcapy.postprocess.field_service import element_field_data
    m, _ = _gauss_model(kind)
    d = element_field_data(m, "D", bucket, 1)
    for g in d["gauss"]:
        assert g["stresses"][0] == pytest.approx(SX_REF, rel=1e-9)
        assert g["stresses"][1] == pytest.approx(0.0, abs=1e-6)
        assert g["stresses"][2] == pytest.approx(0.0, abs=1e-6)


def test_element_field_data_errors_and_meta():
    from milcapy.postprocess.field_service import element_field_data
    m, _ = _gauss_model("q4")
    with pytest.raises(KeyError):
        element_field_data(m, "NOPE", "q2", 1)
    with pytest.raises(ValueError):
        element_field_data(m, "D", "frame", 1)
    d = element_field_data(m, "D", "q2", 1)
    assert d["node_ids"] == [1, 2, 3, 4]
    assert d["node_xy"].shape == (4, 2) and d["node_disp"].shape == (4, 2)
    assert d["strains_nodes"].shape == (4, 3) and d["stresses_nodes"].shape == (4, 3)
    assert d["thickness"] == pytest.approx(0.2) and d["state"] == "PLANE_STRESS"


def test_1d_diagram_widget_builds():
    """Regresión: el popup de barras 1D debe construirse (create_grid_layout)."""
    import matplotlib
    tk = pytest.importorskip("tkinter")
    try:
        root_probe = tk.Tk()
        root_probe.withdraw()
    except tk.TclError:
        pytest.skip("sin display para Tk")
        return
    root_probe.destroy()
    patched = False
    try:
        tk.Tk.mainloop = lambda self: None
        patched = True
        from milcapy.plotter.widgets import DiagramConfig, InternalForceDiagramWidget
        m = SystemModel()
        m.add_material("c", E, V)
        m.add_rectangular_section("v", "c", 0.3, 0.5)
        m.add_node(1, 0, 0)
        m.add_node(2, 5, 0)
        m.add_member(1, 1, 2, "v")
        m.add_restraint(1, True, True, True)
        m.add_load_pattern("D")
        m.add_point_load(2, "D", fy=-10)
        m.solve()
        r = m.get_results("D")
        w = InternalForceDiagramWidget(
            m.members[1],
            {"N(x)": DiagramConfig("Axial", r.get_member_axial_force(1))},
            r.get_member_x_val(1))
        assert "N(x)" in w.interactive_elements
        w.on_closing()
    finally:
        if patched:
            del tk.Tk.mainloop  # restaura el mainloop original heredado


def test_membrane_picking_uses_fill_not_edge():
    """El picking debe activarse dentro del elemento, no en la arista."""
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.patches import Polygon as MplPolygon
    from milcapy.plotter.plotter import Plotter
    from milcapy.plotter.plotter_values import PlotterValues
    PlotterValues._static_data = None
    m = SystemModel()
    m.add_material("c", E, V)
    m.add_shell_section("s", "c", 0.2)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.add_node(3, 0, 1)
    m.add_cst(1, 1, 2, 3, "s")
    m.add_restraint(1, True, True, True)
    m.add_load_pattern("D")
    m.add_point_load(2, "D", fx=100.0)
    m.solve()
    m.plotter = Plotter(m)
    m.plotter.initialize_plot()
    m.plotter.figure.canvas.draw()
    ax = m.plotter.axes
    arts = m.plotter.csts[1]
    fill = next(a for a in arts if isinstance(a, MplPolygon))

    def at(x, y):
        px, py = ax.transData.transform((x, y))
        return MouseEvent("button_press_event", m.plotter.figure.canvas, px, py)

    assert fill.contains(at(0.2, 0.2))[0] is True    # dentro
    assert arts[0].contains(at(0.2, 0.2))[0] is False  # la arista ya no dispara
    assert fill.contains(at(0.9, 0.9))[0] is False   # fuera
    plt.close("all")


class _StubLabel:
    def __init__(self):
        self.text = ""

    def configure(self, text=""):
        self.text = text


class _StubCanvas:
    def draw(self):
        pass


def test_popup_draw_uses_contour_headless():
    """El popup dibuja contour (nada plano) sin necesitar display."""
    import matplotlib
    from matplotlib.contour import ContourSet
    from milcapy.postprocess.field_service import element_field_data
    from milcapy.plotter.widgets import MembraneStressWidget
    m, _ = _gauss_model("q4")
    data = element_field_data(m, "D", "q2", 1)
    w = MembraneStressWidget.__new__(MembraneStressWidget)
    w.data = data
    w.field, w.cmap, w.levels = "SX", "jet", 8
    w.show_gauss, w.show_nodes = True, True
    w._cbar, w._cax = None, None
    w.fig = matplotlib.pyplot.figure()
    w.ax = w.fig.add_subplot(111)
    w.canvas, w.minmax_label, w.status_label = _StubCanvas(), _StubLabel(), _StubLabel()
    w._draw()
    kinds = [type(a).__name__ for a in w.ax.collections]
    assert any(isinstance(a, ContourSet) for a in w.ax.collections), kinds
    assert "Min" in w.minmax_label.text and "Max" in w.minmax_label.text
    # hover: dentro -> valor, fuera -> "—"
    cx, cy = np.asarray(data["node_xy"]).mean(axis=0)
    w._on_hover(type("E", (), {"inaxes": w.ax, "xdata": cx, "ydata": cy})())
    assert "SX=" in w.status_label.text
    w._on_hover(type("E", (), {"inaxes": w.ax, "xdata": 1e6, "ydata": 1e6})())
    assert w.status_label.text == "—"
    # CST también en contour
    m2, _ = _gauss_model("cst")
    w.data = element_field_data(m2, "D", "cst", 1)
    w._draw()
    assert any(isinstance(a, ContourSet) for a in w.ax.collections)
    matplotlib.pyplot.close("all")

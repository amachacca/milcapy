from milcapy import SystemMilcaModel

E = 2534560  # MPa
v = 0.20

l = 3  # mm

h = 0.6  # mm
t = 0.25  # mm

fy = 6
model = SystemMilcaModel()
model.add_material("concreto", E, v)
model.add_shell_section("muro", "concreto", t)
model.add_node(1, 0, 0)
model.add_node(2, l, 0)
model.add_node(3, l, h)
model.add_node(4, 0, h)
model.add_membrane_q6(1, 1, 2, 3, 4, "muro")
model.add_restraint(1, *(True, True, False))
model.add_restraint(4, *(True, True, False))
model.add_load_pattern("carga")
model.add_point_load(2, "carga", fy=-0.5*fy)
model.add_point_load(3, "carga", fy=-0.5*fy)
model.solve()
model.plotter_options.mod_scale_deformation = 1.2
model.show()

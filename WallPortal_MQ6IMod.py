from milcapy import SystemModel
from milcapy.utils.types import MembraneQuadElementType

E = 2e6 # Modulo de Young
v = 0.3 # Coeficiente de Poisson

h = 2.4 # Altura de la pared
b = 1.2 # Ancho de la pared
l = 2.4 # Longitud de la vigas

t = 0.4 # Espesor de la pared
sec = [0.4, 0.4] # Sección de la vigas
F = 100 # Carga aplicada
ele_type = MembraneQuadElementType.MQ8Complete

wallPortal = SystemModel()
wallPortal.add_material("concreto", E, v)
wallPortal.add_rectangular_section("vig40x40", "concreto", *sec)
wallPortal.add_shell_section("muro", "concreto", t)
wallPortal.add_node(1, 0, 0)
wallPortal.add_node(2, b, 0)
wallPortal.add_node(3, 0, h)
wallPortal.add_node(4, b, h)
wallPortal.add_node(5, 0, 2*h)
wallPortal.add_node(6, b, 2*h)
wallPortal.add_node(7, 0, 3*h)
wallPortal.add_node(8, b, 3*h)
wallPortal.add_node(9, b+l, 0)
wallPortal.add_node(10, b+l+b, 0)
wallPortal.add_node(11, b+l, h)
wallPortal.add_node(12, b+l+b, h)
wallPortal.add_node(13, b+l, 2*h)
wallPortal.add_node(14, b+l+b, 2*h)
wallPortal.add_node(15, b+l, 3*h)
wallPortal.add_node(16, b+l+b, 3*h)
wallPortal.add_membrane_q6i_mod(1, 1, 2, 4, 3, "muro", ele_type=ele_type)
wallPortal.add_membrane_q6i_mod(2, 3, 4, 6, 5, "muro", ele_type=ele_type)
wallPortal.add_membrane_q6i_mod(3, 5, 6, 8, 7, "muro", ele_type=ele_type)
wallPortal.add_membrane_q6i_mod(4, 9, 10, 12, 11, "muro", ele_type=ele_type)
wallPortal.add_membrane_q6i_mod(5, 11, 12, 14, 13, "muro", ele_type=ele_type)
wallPortal.add_membrane_q6i_mod(6, 13, 14, 16, 15, "muro", ele_type=ele_type)
wallPortal.add_elastic_timoshenko_beam(7, 4, 11, "vig40x40")
wallPortal.add_elastic_timoshenko_beam(8, 6, 13, "vig40x40")
wallPortal.add_elastic_timoshenko_beam(9, 8, 15, "vig40x40")
wallPortal.add_restraint(1, *(True, True, True))
wallPortal.add_restraint(2, *(True, True, True))
wallPortal.add_restraint(9, *(True, True, True))
wallPortal.add_restraint(10, *(True, True, True))
wallPortal.add_load_pattern("carga")
wallPortal.add_point_load(3, "carga", fx=1*F)
wallPortal.add_point_load(5, "carga", fx=2*F)
wallPortal.add_point_load(7, "carga", fx=3*F)
wallPortal.solve()
wallPortal.show()
# wallPortal.plot_model("carga")

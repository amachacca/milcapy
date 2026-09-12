from milcapy import SystemModel, ConstitutiveModelType

E = 2e6
v = 0.3
h = 2.4
b = 1.2
l = 2.4
t = 0.4
sec = [0.4, 0.4]
F = 100
STATE = ConstitutiveModelType.PLANE_STRAIN

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
wallPortal.add_cst(1, 1, 2, 4, "muro", STATE)
wallPortal.add_cst(2, 1, 4, 3, "muro", STATE)
wallPortal.add_cst(3, 3, 4, 6, "muro", STATE)
wallPortal.add_cst(4, 3, 6, 5, "muro", STATE)
wallPortal.add_cst(5, 5, 6, 8, "muro", STATE)
wallPortal.add_cst(6, 5, 8, 7, "muro", STATE)
wallPortal.add_cst(7, 9, 10, 12, "muro", STATE)
wallPortal.add_cst(8, 9, 12, 11, "muro", STATE)
wallPortal.add_cst(9, 11, 12, 14, "muro", STATE)
wallPortal.add_cst(10, 11, 14, 13, "muro", STATE)
wallPortal.add_cst(11, 13, 14, 16, "muro", STATE)
wallPortal.add_cst(12, 13, 16, 15, "muro", STATE)
wallPortal.add_elastic_euler_bernoulli_beam(13, 4, 11, "vig40x40")
wallPortal.add_elastic_euler_bernoulli_beam(14, 6, 13, "vig40x40")
wallPortal.add_elastic_euler_bernoulli_beam(15, 8, 15, "vig40x40")
wallPortal.add_restraint(1, *(True, True, True))
wallPortal.add_restraint(2, *(True, True, True))
wallPortal.add_restraint(9, *(True, True, True))
wallPortal.add_restraint(10, *(True, True, True))
wallPortal.add_load_pattern("carga")
wallPortal.add_point_load(3, "carga", fx=1*F)
wallPortal.add_point_load(5, "carga", fx=2*F)
wallPortal.add_point_load(7, "carga", fx=3*F)
wallPortal.solve()
print(f'A {STATE.value} \n{list(wallPortal.csts.values())[1].D}')
wallPortal.show()



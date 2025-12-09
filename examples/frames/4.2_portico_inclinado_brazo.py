from milcapy import SystemModel
from math import atan, cos


a, bp, c = 1, 1.7, 0.5
L1, L2 = 7, 1.7
H1, H2 = 3, 4
e, d = 0.4, 0.7
q = 4
F1, F2, M = 10, 5, 2

t_vig = atan(H2/(L1+0.5*(a+bp)))
lbri = 0.5*a/cos(t_vig)
lbrf = 0.5*bp/cos(t_vig)
lpvol = (L2+0.5*bp-0.5*c)/cos(t_vig)
lf2 = L2+0.5*bp
Peq = -(F2+q*lpvol)
Meq = -q*lpvol**2*0.5*cos(t_vig) + M - F2*lf2

portico = SystemModel()

portico.add_material("concreto", 2509980.08, 0.25)
portico.add_rectangular_section("sec1", "concreto", e, a)
portico.add_rectangular_section("sec2", "concreto", e, d)
portico.add_rectangular_section("sec3", "concreto", e, bp)
portico.add_node(1, 0, 0)
portico.add_node(2, 0, H1)
portico.add_node(3, L1+0.5*(a+bp), H1+H2)
portico.add_node(4, L1+0.5*(a+bp), 0)
portico.add_member(1, 1, 2, "sec1")
portico.add_member(2, 3, 2, "sec2")
portico.add_member(3, 4, 3, "sec3")
portico.add_restraint(1, *(True, True, True))
portico.add_restraint(4, *(True, True, True))
portico.add_load_pattern("Live Load")
portico.add_distributed_load(2, "Live Load", q, q, "GLOBAL", "GRAVITY")
portico.add_end_length_offset(2, lbri, lbrf)
portico.add_point_load(2, "Live Load", F1, 0, 0, "GLOBAL")
portico.add_point_load(3, "Live Load", Peq, 0, Meq, "GLOBAL")
portico.postprocessing_options.n = 100
portico.solve()
portico.show()

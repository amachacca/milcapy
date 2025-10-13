from milcapy import SystemModel
nt = 17
l = 0.1
beam = SystemModel()
beam.add_material("concreto", 2e6, 0.15, 2.4)
beam.add_rectangular_section("vig", "concreto", 1.0, 0.1)

for i in range(nt+1):
    beam.add_node(i+1, i*l, 0)
    if i > 0 and i < nt:
        beam.add_elastic_support(i+1, ky=100)

for i in range(nt):
    beam.add_member(i+1, i+1, i+2, "vig")

beam.add_restraint(1, *(True, True, False))
beam.add_restraint(nt+1, *(True, True, False))
beam.add_elastic_support(nt+1, krz=100)
beam.add_elastic_support(1, krz=100)


beam.add_load_pattern("Live Load")
beam.add_self_weight("Live Load")
for idb in beam.members.keys():
    beam.add_distributed_load(idb, "Live Load", -100, -100)
beam.solve()

beam.plotter_options.mod_support_size = 7
beam.plotter_options.elastic_support_label = False
beam.plotter_options.mod_scale_internal_forces = 4
beam.plotter_options.mod_scale_deformation = 2
beam.plotter_options.mod_scale_dist_qload = 4
beam.plotter_options.internal_forces_label = True
beam.plotter_options.mod_krz_rotation_angle = {1: 0, nt+1: 180}

beam.show()
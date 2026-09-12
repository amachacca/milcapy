from milcapy import SystemModel, FieldType, ConstitutiveModelType

# Units: kg, cm
E = 219498.39                    # Young's Modulus
v = 0.167                        # Poisson's Ratio
t = 100                          # Thickness
# Create a new model instance for Plane Stress
new_model = SystemModel()
new_model.add_material('concreto', E, v)
new_model.add_shell_section('muro', 'concreto', t)
# Add nodes from the "Tabla de coordenadas de los nodos"
nodes_data = {
    1: (0, 0), 2: (1.8, 0), 3: (3.6, 0), 4: (0, 0.6),
    5: (1, 0.6), 6: (1.6, 0.6), 7: (2.6, 0.6), 8: (3.6, 0.6),
    9: (1.06, 1.68), 10: (1.6, 1.68), 11: (1.12, 2.76), 12: (1.6, 2.76),
    13: (1.18, 3.84), 14: (1.6, 3.84), 15: (1.24, 4.92), 16: (1.6, 4.92),
    17: (1.3, 6), 18: (1.6, 6)
}

for node_id, (x_coord, y_coord) in nodes_data.items():
    new_model.add_node(node_id, x_coord*100, y_coord*100)
    # new_model.add_restraint(node_id, False, False, True)

# Add elements from the "Tabla de sentido antihorario de numeración de los nodos"
# Note: The table has 16 elements.
elements_data = [
    (1, 1, 5, 4), (2, 1, 2, 5), (3, 5, 2, 6), (4, 6, 2, 7),
    (5, 7, 2, 3), (6, 7, 3, 8), (7, 5, 6, 9), (8, 9, 6, 10),
    (9, 9, 10, 11), (10, 11, 10, 12), (11, 11, 12, 13), (12, 13, 12, 14),
    (13, 13, 14, 15), (14, 15, 14, 16), (15, 15, 16, 17), (16, 17, 16, 18)
]

for elem_id, node_i, node_j, node_k in elements_data:
    new_model.add_cst(elem_id, node_i, node_j, node_k, 'muro', ConstitutiveModelType.PLANE_STRAIN)
# Add restraints based on the image (nodes 1, 2, and 3 are fixed)
new_model.add_restraint(1, True, True, False)
new_model.add_restraint(2, True, True, False)
new_model.add_restraint(3, True, True, False)

# Add forces based on the image (node 18 has a vertical force of -10000)
tonf = 1000
new_model.add_load_pattern('Live Load')
new_model.add_point_load(18, 'Live Load', -5*tonf, 0)
new_model.add_point_load(16, 'Live Load', -10*tonf, 0)
new_model.add_point_load(14, 'Live Load', -15*tonf, 0)
new_model.add_point_load(12, 'Live Load', -20*tonf, 0)
new_model.add_point_load(10, 'Live Load', -25*tonf, 0)
new_model.add_point_load(6, 'Live Load', -30*tonf, 0)


# Solve the model
new_model.solve()
# new_model.plot_model('Live Load')

# show
new_model.plotter_options.mod_support_size = 3
new_model.show()

# # Plot the deformed shape
# new_model.plot_model()
# # new_model.plot_deformed(100, 10000, undeformed=True, label=True)

# new_model.plot_field(FieldType.EX)
# new_model.plot_field(FieldType.EY)
# new_model.plot_field(FieldType.EXY)
# new_model.plot_field(FieldType.SX)
# new_model.plot_field(FieldType.SY)
# new_model.plot_field(FieldType.SXY)
# new_model.plot_field(FieldType.UX)
# new_model.plot_field(FieldType.UY)
# new_model.plot_field(FieldType.UMAG)

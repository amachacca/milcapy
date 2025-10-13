from typing import TYPE_CHECKING, Tuple, Dict
import numpy as np
import time
from milcapy.loads.load import PrescribedDOF

if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel

class LinearStaticAnalysis:
    """
    Solucionador basado en el Método de Rigidez Directa.
    Resuelve sistemas de ecuaciones estructurales K*u = F usando métodos de solución directa.
    Usa numpy.linalg.solve(K, F)
    """

    def __init__(
        self,
        model: "SystemMilcaModel",
        ) -> None:
        """
        Inicializa el solucionador con el modelo y el método de solución.

        Args:
            model: Modelo estructural que contiene K_global y F_global.
        """
        self.model = model

        # Rendimiento y diagnóstico
        self.solution_time = 0.0
        self.assembly_time = 0.0


    def _dof_map(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
        """Mapea los grados de libertad del modelo para trabajar con numpy.
            [0, 1, 2, ] pertenece al nodo 1 y así sucesivamente."""
        nn = len(self.model.nodes)
        restraints = np.zeros(nn * 3, dtype=bool)

        prescribed_dofs = {} # {dof: value}

        for node in self.model.nodes.values():
            preDOF: PrescribedDOF = node.prescribed_dofs.get(self.model.current_load_pattern, PrescribedDOF())

            restraints[node.dofs[0] - 1] = node.restraints[0] #or preDOF.ux is not None
            restraints[node.dofs[1] - 1] = node.restraints[1] #or preDOF.uy is not None
            restraints[node.dofs[2] - 1] = node.restraints[2] #or preDOF.rz is not None

            if preDOF.ux is not None:
                prescribed_dofs[node.dofs[0] - 1] = preDOF.ux
            if preDOF.uy is not None:
                prescribed_dofs[node.dofs[1] - 1] = preDOF.uy
            if preDOF.rz is not None:
                prescribed_dofs[node.dofs[2] - 1] = preDOF.rz

        free_dofs = np.where(~restraints)[0]
        restrained_dofs = np.where(restraints)[0]

        return free_dofs, restrained_dofs, prescribed_dofs

    def assemble_global_load_vector(
        self,
    ) -> np.ndarray:
        """Calcula el vector de carga global del sistema.

        Returns:
            np.ndarray: Vector de carga global.
        """
        start_time = time.time()

        nn = len(self.model.nodes)
        F = np.zeros(nn * 3)

        # Asignar fuerzas nodales almacenadas en los nodos
        for node in self.model.nodes.values():
            f = node.load_vector()
            dofs = node.dofs
            F[dofs - 1] += f

        # Agregar el vector de fuerzas globales almacenadas en los miembros
        for member in self.model.members.values():
            f = member.global_load_vector()
            dofs = member.dofs
            F[dofs - 1] += f

        # Agregar cargas de los elementos CST
        for cst in self.model.csts.values():
            f = cst.get_load_vector()
            dofs = cst.dofs
            F[dofs - 1] += f

        # Agregar cargas de los elementos Truss
        for truss in self.model.trusses.values():
            f = truss.global_load_vector()
            dofs = truss.dofs
            F[dofs - 1] += f

        end_time = time.time()
        self.assembly_time += (end_time - start_time)

        return F

    def assemble_global_stiffness_matrix(
        self,
    ) -> np.ndarray:
        """Ensamblaje de la matriz de rigidez global.

        Returns:
            np.ndarray: Matriz de rigidez global.
        """
        start_time = time.time()
        nn = len(self.model.nodes)
        K = np.zeros((nn * 3, nn * 3))

        # Ensamblar la matriz de rigidez global
        # Rigides de los elmentos 1D
        for member in self.model.members.values():
            k = member.global_stiffness_matrix()
            dofs = member.dofs

            rows = np.repeat(dofs - 1, 6)
            cols = np.tile(dofs - 1, 6)
            values = k.flatten()

            for i, j, val in zip(rows, cols, values):
                K[i, j] += val

        # Rigidez de los elementos CST
        for cst in self.model.csts.values():
            k = cst.global_stiffness_matrix()
            dofs = cst.dofs

            rows = np.repeat(dofs - 1, 6)
            cols = np.tile(dofs - 1, 6)
            values = k.flatten()

            for i, j, val in zip(rows, cols, values):
                K[i, j] += val

        # Rigidez de los elementos Membrane de 3 grados de libertad por nodo
        for membQ3dof in self.model.membrane_q3dof.values():
            k = membQ3dof.global_stiffness_matrix()
            dofs = membQ3dof.dofs

            rows = np.repeat(dofs - 1, 12)
            cols = np.tile(dofs - 1, 12)
            values = k.flatten()

            for i, j, val in zip(rows, cols, values):
                K[i, j] += val


        # Rigidez de los elementos Membrana de 2 grados de libertad por nodo
        for membQ2dof in self.model.membrane_q2dof.values():
            k = membQ2dof.global_stiffness_matrix()
            dofs = membQ2dof.dofs

            rows = np.repeat(dofs - 1, 8) # 8 es el numero de grados de libertad del elemento
            cols = np.tile(dofs - 1, 8)
            values = k.flatten()

            for i, j, val in zip(rows, cols, values):
                K[i, j] += val


        # Rigidez de los elementos Armadura
        for truss in self.model.trusses.values():
            k = truss.global_stiffness_matrix()
            dofs = truss.dofs

            rows = np.repeat(dofs - 1, 4)
            cols = np.tile(dofs - 1, 4)
            values = k.flatten()

            for i, j, val in zip(rows, cols, values):
                K[i, j] += val


        # aplicar las condiciones de ejes locales de los nodos
        # for node in self.model.nodes.values():
        #     if node.local_axis is not None:
        #         dofs = node.dofs
        #         Tlg = node.local_axis.get_transformation_matrix()
        #         kfc = Tlg.T @ K[dofs - 1, dofs - 1] @ Tlg
        #         kfila = np.zeros((3, nn * 3))
        #         kcolumna = np.zeros((nn * 3, 3))
        #         for node in self.model.nodes.values():
        #             dofi = node.dofs
        #             kfila[:, dofi - 1] += Tlg.T @ K[dofs - 1, dofi - 1]
        #             kcolumna[dofi - 1, :] += K[dofi - 1, dofs - 1] @ Tlg

        #         K[dofs - 1, :] = kfila
        #         K[:, dofs - 1] = kcolumna
        #         K[dofs - 1, dofs - 1] = kfc


        K = self.apply_local_axes(K, self.model)

        # Kold = K.copy()
        # for node in self.model.nodes.values():
        #     if node.local_axis is not None:
        #         dofs = node.dofs
        #         Tlg = node.local_axis.get_transformation_matrix()

        #         # diagonal
        #         kfc = Tlg.T @ Kold[dofs - 1, dofs - 1] @ Tlg

        #         # fila y columna
        #         kfila = np.zeros((3, nn * 3))
        #         kcolumna = np.zeros((nn * 3, 3))
        #         for nodej in self.model.nodes.values():
        #             dofj = nodej.dofs
        #             kfila[:, dofj - 1] = Tlg.T @ Kold[dofs - 1, dofj - 1]
        #             kcolumna[dofj - 1, :] = Kold[dofs - 1, dofj - 1] @ Tlg

        #         # escribir en la nueva K
        #         K[dofs - 1, :] = kfila
        #         K[:, dofs - 1] = kcolumna
        #         K[dofs - 1, dofs - 1] = kfc

                # for i, j, val in zip(rows, cols, values):
                #     K[i, j] += val

        end_time = time.time()
        self.assembly_time += (end_time - start_time)

        return K

    @staticmethod
    def apply_local_axes(
        K: np.ndarray,
        model: "SystemMilcaModel"
    ) -> np.ndarray:
        """
        Aplica las transformaciones locales de cada nodo a la matriz de rigidez global K. (nota: aplicar nodo por nodo)
        """
        Kold = K.copy()
        Knew = Kold.copy()

        for node in model.nodes.values():
            if node.local_axis is None:
                continue

            dofs = node.dofs - 1  # índices base 0
            T = node.local_axis.get_transformation_matrix()
            kij = Kold[np.ix_(dofs, dofs)]

            # fila
            for nodei in model.nodes.values():
                dofi = nodei.dofs - 1
                Knew[np.ix_(dofs, dofi)] = T.T @ Kold[np.ix_(dofs, dofi)]

            # columna
            for nodej in model.nodes.values():
                dofj = nodej.dofs - 1
                Knew[np.ix_(dofj, dofs)] = Kold[np.ix_(dofj, dofs)] @ T

            # diagonal
            Knew[np.ix_(dofs, dofs)] = T.T @ kij @ T
            # Knew[np.ix_(dofs, dofs)] = T.T @ Kold[np.ix_(dofs, dofs)] @ T

            Kold = Knew.copy()
            Knew = Kold.copy()

        return Knew



    def apply_boundary_conditions(
        self,
        K: np.ndarray,
        F: np.ndarray,
        free_dofs: np.ndarray,
        restrained_dofs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aplica las condiciones de frontera.

        Args:
            K (np.ndarray): Matriz de rigidez global.
            F (np.ndarray): Vector de fuerzas global.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - K_d: Matriz de rigidez para grados de libertad libres.
                - K_dc: Matriz de rigidez que relaciona GDL libres con restringidos.
                - K_cd: Matriz de rigidez que relaciona GDL restringidos con libres.
                - K_c: Matriz de rigidez para GDL restringidos.
                - F_d: Vector de fuerzas para GDL libres.
                - F_c: Vector de fuerzas para GDL restringidos.
        """

        # Aplicar las condiciones de apoyos elásticos
        nn = len(self.model.nodes)
        K_kk = np.zeros((nn * 3, nn * 3))
        for node in self.model.nodes.values():
            if node.elastic_supports is not None:
                kx, ky, krz = node.elastic_supports.get_elastic_supports()
                if kx is not None:
                    K_kk[node.dofs[0] - 1, node.dofs[0] - 1] += kx
                if ky is not None:
                    K_kk[node.dofs[1] - 1, node.dofs[1] - 1] += ky
                if krz is not None:
                    K_kk[node.dofs[2] - 1, node.dofs[2] - 1] += krz
        K = K + K_kk

        # Reducir la matriz de rigidez (d: desconocidos, c: conocidos)
        #     | Kd   Kdc |
        # K = |          |
        #     | Kcd   Kc |
        K_d  = K[np.ix_(free_dofs, free_dofs)]
        K_dc = K[np.ix_(free_dofs, restrained_dofs)]
        K_cd = K[np.ix_(restrained_dofs, free_dofs)]
        K_c  = K[np.ix_(restrained_dofs, restrained_dofs)]

        # Reducir el vector de fuerzas
        #     | Fd |
        # F = |    |
        #     | Fc |
        F_d = F[free_dofs]
        F_c = F[restrained_dofs]

        return K_d, K_dc, K_cd, K_c, F_d, F_c, K_kk

    def solve(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resuelve el sistema de ecuaciones F = KU.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Desplazamientos nodales
                - Reacciones en los apoyos.
        """

        start_time = time.time()
        # Obtener la matriz de rigidez global y el vector de fuerzas global
        K = self.assemble_global_stiffness_matrix()
        F = self.assemble_global_load_vector()
        self.model.global_stiffness_matrix = K
        self.model.global_load_vector[self.model.current_load_pattern] = F

        # Aplicar las condiciones de frontera
        free_dofs, restrained_dofs, prescribed_dofs = self._dof_map()
        K_d, K_dc, K_cd, K_c, F_d, F_c, K_kk = self.apply_boundary_conditions(K, F, free_dofs, restrained_dofs)

        # Resolver el sistema de ecuaciones
        #     |Ud|
        # U = |  | = displacements
        #     |Uc|

        # Colocar los desplazamientos en los grados de libertad libres
        nn = len(self.model.nodes)
        # completar el vector de desplazamientos
        displacements = np.zeros(nn * 3)
        # Aplicar las condiciones de prescindidos
        displacements[list(prescribed_dofs.keys())] = list(prescribed_dofs.values())
        Uc = np.copy(displacements[restrained_dofs])
        Upc = np.copy(displacements[free_dofs])

        # resolver el sistema de ecuaciones
        if np.linalg.det(K_d) == 0:
            try:
                factorDeRegularizacion = 1e-7
                Warning(f"La matriz de rigidez global es singular, por lo que para resolver se le sumara un I[ndof. ndof] * {factorDeRegularizacion} a la matriz de rigidez global")
                K_d += np.eye(K_d.shape[0]) * factorDeRegularizacion
                U_d = np.linalg.solve(K_d, F_d - K_dc @ Uc - K_d @ Upc)
            except:
                raise ValueError("La matriz de rigidez global es singular y no se puede resolver")
        else:
            U_d = np.linalg.solve(K_d, F_d - K_dc @ Uc - K_d @ Upc)


        displacements[free_dofs] = U_d + Upc

        # Calcular las reacciones en los apoyos
        R = K_cd @ (U_d + Upc) + K_c @ Uc - F_c
        # Completar el vector de reacciones
        reactions = np.zeros(nn * 3)
        reactions[restrained_dofs] = R
        # corregir las reacciones por las condiciones de apoyos elásticos
        reactions -= K_kk @ displacements

        # Otra forma de alcular las reacciones en los apoyos, para no estar completando el vector de reacciones
        #  R  =  K_global * U_global - F_global
        # reactions = K @ displacements - F

        end_time = time.time()
        self.solution_time = (end_time - start_time)

        ## Tiempo de solucion
        print(f"Tiempo de solucion: {self.solution_time} para el Load Pattern: {self.model.current_load_pattern}")

        return displacements, reactions
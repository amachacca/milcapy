from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

class Results:
    """Clase que almacena los resultados de un análisis de un Load Pattern.
    La estructura de los resultados es la siguiente:

    self.model: {
        "displacements": np.ndarray,
        "reactions": np.ndarray
            }

    self.nodes: {
        node_id: {
            "displacements": np.ndarray,
            "reactions": np.ndarray
                }
            }

    self.members: {
        "x_val": np.ndarray,
        member_id: {
            "displacements": np.ndarray,
            "internal_forces": np.ndarray,
            "axial_forces": np.ndarray,
            "axial_displacements": np.ndarray,
            "shear_forces": np.ndarray,
            "bending_moments": np.ndarray,
            "slopes": np.ndarray,
            "deflections": np.ndarray
                    }
            }

    self.CST: {
        "displacements": np.ndarray,
        "strains": np.ndarray,
        "stresses": np.ndarray,
        }

    self.membrane_q3dof: {
        "displacements": np.ndarray,
        "strains": np.ndarray,
        "stresses": np.ndarray,
        }

    self.membrane_q2dof: {
        "displacements": np.ndarray,
        "strains": np.ndarray,
        "stresses": np.ndarray,
        }

    self.trusses: {
        "displacements": np.ndarray,
        "internal_forces": np.ndarray,
        "axial_forces": np.ndarray,
        "axial_displacements": np.ndarray,
            }
    """

    def __init__(self):
        self.nodes: Dict[int, Dict[str, np.ndarray]] = {}
        self.members: Dict[int, Dict[str, np.ndarray]] = {}
        self.CST: Dict[int, np.ndarray] = {}
        self.membrane_q3dof: Dict[int, np.ndarray] = {}
        self.membrane_q2dof: Dict[int, np.ndarray] = {}
        self.trusses: Dict[int, Dict[str, np.ndarray]] = {}
        self.model: Dict[str, np.ndarray] = {}

    def get_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Construye dos DataFrames:
        - Object 1: Resultados por nodo (desplazamientos y reacciones).
        - Object 2: Resultados por elemento (desplazamientos y fuerzas internas).
        """

        # ============ OBJECT 1 (NODOS) ============
        displacements = self.get_model_displacements()  # vector largo múltiplo de 3
        reactions = self.get_model_reactions()          # vector largo múltiplo de 3

        n_nodes = len(displacements) // 3
        rows_nodes = []

        for i in range(n_nodes):
            node_id = i + 1
            Ux, Uy, Rz = displacements[i*3:(i+1)*3]
            ReacFx, ReacFy, ReacMz = reactions[i*3:(i+1)*3]
            rows_nodes.append([node_id, Ux, Uy, Rz, ReacFx, ReacFy, ReacMz])

        df_nodes = pd.DataFrame(
            rows_nodes,
            columns=["node_id", "Ux", "Uy", "Rz", "ReacFx", "ReacFy", "ReacMz"]
        )

        # ============ OBJECT 2 (ELEMENTOS) ============
        rows_elements = []

        idEle = list(self.members.keys()) + list(self.trusses.keys())
        for ele_id in idEle:
            if ele_id in self.trusses:
                disp = self.get_truss_displacements(ele_id)       # [Uxi, Uxj]
                int_forces = self.get_truss_internal_forces(ele_id)  # [Ni, Nj]

                # completar con 0
                row = [ele_id,
                    disp[0], 0.0, 0.0,   # Uxi, Uyi=0, Rzi=0
                    disp[1], 0.0, 0.0,   # Uxj, Uyj=0, Rzj=0
                    int_forces[0], 0.0, 0.0,   # Ni, Vi=0, Mi=0
                    int_forces[1], 0.0, 0.0]   # Nj, Vj=0, Mj=0

            else:
                disp = self.get_member_displacements(ele_id)       # 6 elementos
                int_forces = self.get_member_internal_forces(ele_id)  # 6 elementos

                row = [ele_id,
                    disp[0], disp[1], disp[2],   # Uxi, Uyi, Rzi
                    disp[3], disp[4], disp[5],   # Uxj, Uyj, Rzj
                    int_forces[0], int_forces[1], int_forces[2],   # Ni, Vi, Mi
                    int_forces[3], int_forces[4], int_forces[5]]   # Nj, Vj, Mj

            rows_elements.append(row)

        df_elements = pd.DataFrame(
            rows_elements,
            columns=[
                "ele_id",
                "Uxi", "Uyi", "Rzi", "Uxj", "Uyj", "Rzj",
                "Ni", "Vi", "Mi", "Nj", "Vj", "Mj"
            ]
        )

        return df_nodes, df_elements


    def set_model_displacements(self, displacements: np.ndarray) -> None:
        self.model["displacements"] = displacements

    def set_model_reactions(self, reactions: np.ndarray) -> None:
        self.model["reactions"] = reactions

    def set_node_displacements(self, node_id: int, displacement: np.ndarray) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"displacements": np.zeros(3), "reactions": np.zeros(3)}
        self.nodes[node_id]["displacements"] = displacement

    def set_node_reactions(self, node_id: int, reaction: np.ndarray) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"displacements": np.zeros(3), "reactions": np.zeros(3)}
        self.nodes[node_id]["reactions"] = reaction

    def set_member_displacements(self, member_id: int, displacement: np.ndarray) -> None:
        if member_id not in self.members:
            self.members[member_id] = {"displacements": np.zeros(6), "internal_forces": np.zeros(6)}
        self.members[member_id]["displacements"] = displacement

    def set_member_internal_forces(self, member_id: int, internal_forces: np.ndarray) -> None:
        if member_id not in self.members:
            self.members[member_id] = {"displacements": np.zeros(6), "internal_forces": np.zeros(6)}
        self.members[member_id]["internal_forces"] = internal_forces

    def set_x_val(self, member_id: int, x_val: np.ndarray) -> None:
        self.members[member_id]["x_val"] = x_val

    def set_member_axial_force(self, member_id: int, axial_force: np.ndarray) -> None:
        self.members[member_id]["axial_forces"] = axial_force

    def set_member_shear_force(self, member_id: int, shear_force: np.ndarray) -> None:
        self.members[member_id]["shear_forces"] = shear_force

    def set_member_bending_moment(self, member_id: int, bending_moment: np.ndarray) -> None:
        self.members[member_id]["bending_moments"] = bending_moment

    def set_member_deflection(self, member_id: int, deflection: np.ndarray) -> None:
        self.members[member_id]["deflections"] = deflection

    def set_member_slope(self, member_id: int, slope: np.ndarray) -> None:
        self.members[member_id]["slopes"] = slope

    def set_member_axial_displacement(self, member_id: int, axial_displacement: np.ndarray) -> None:
        self.members[member_id]["axial_displacements"] = axial_displacement

    def get_model_displacements(self) -> np.ndarray:
        return self.model["displacements"]

    def get_model_reactions(self) -> np.ndarray:
        return self.model["reactions"]

    def get_node_displacements(self, node_id: int) -> np.ndarray:
        return self.nodes[node_id]["displacements"]

    def get_node_reactions(self, node_id: int) -> np.ndarray:
        return self.nodes[node_id]["reactions"]

    def get_member_displacements(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["displacements"]

    def get_member_internal_forces(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["internal_forces"]

    def get_member_x_val(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["x_val"]

    def get_member_axial_force(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["axial_forces"]

    def get_member_shear_force(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["shear_forces"]

    def get_member_bending_moment(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["bending_moments"]

    def get_member_deflection(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["deflections"]

    def get_member_slope(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["slopes"]

    def get_member_axial_displacement(self, member_id: int) -> np.ndarray:
        return self.members[member_id]["axial_displacements"]

    def get_results_node(self, node_id: int) -> Dict[str, np.ndarray]:
        return self.nodes[node_id]

    def get_results_member(self, member_id: int) -> Dict[str, np.ndarray]:
        return self.members[member_id]

    def get_results_model(self) -> Dict[str, np.ndarray]:
        return self.model

    def get_results(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            "model": self.model,
            "nodes": self.nodes,
            "members": self.members,
            "CST": self.CST,
            "membrane_q3dof": self.membrane_q3dof,
            "membrane_q2dof": self.membrane_q2dof,
            "trusses": self.trusses,
        }

    def set_cst_displacements(self, cst_id: int, displacement: np.ndarray) -> None:
        if cst_id not in self.CST:
            self.CST[cst_id] = {"displacements": np.zeros(6)}
        self.CST[cst_id]["displacements"] = displacement

    def get_cst_displacements(self, cst_id: int) -> np.ndarray:
        return self.CST[cst_id]["displacements"]

    def set_cst_strains(self, cst_id: int, strains: np.ndarray) -> None:
        if cst_id not in self.CST:
            self.CST[cst_id] = {"displacements": np.zeros(6)}
        self.CST[cst_id]["strains"] = strains

    def set_cst_stresses(self,  cst_id: int, stresses: np.ndarray) -> None:
            if cst_id not in self.CST:
                self.CST[cst_id] = {"displacements": np.zeros(6)}
            self.CST[cst_id]["stresses"] = stresses

    def get_cst_strains(self, cst_id: int) -> np.ndarray:
        return self.CST[cst_id]["strains"]

    def get_cst_stresses(self, cst_id: int) -> np.ndarray:
        return self.CST[cst_id]["stresses"]

    def set_membrane_q3dof_displacements(self, membrane_q3dof_id: int, displacement: np.ndarray) -> None:
        if membrane_q3dof_id not in self.membrane_q3dof:
            self.membrane_q3dof[membrane_q3dof_id] = {"displacements": np.zeros(12)}
        self.membrane_q3dof[membrane_q3dof_id]["displacements"] = displacement

    def get_membrane_q3dof_displacements(self, membrane_q3dof_id: int) -> np.ndarray:
        return self.membrane_q3dof[membrane_q3dof_id]["displacements"]

    def set_membrane_q3dof_strains(self, membrane_q3dof_id: int, strains_nodes: np.ndarray, strains_gauss: np.ndarray | None = None) -> None:
        """Guarda deformaciones: nodales (4,3) y opcionalmente en Gauss (ng,3)."""
        if membrane_q3dof_id not in self.membrane_q3dof:
            self.membrane_q3dof[membrane_q3dof_id] = {"displacements": np.zeros(12)}
        self.membrane_q3dof[membrane_q3dof_id]["strains"] = strains_nodes
        if strains_gauss is not None:
            self.membrane_q3dof[membrane_q3dof_id]["strains_gauss"] = strains_gauss

    def set_membrane_q3dof_stresses(self, membrane_q3dof_id: int, stresses_nodes: np.ndarray, stresses_gauss: np.ndarray | None = None) -> None:
        """Guarda tensiones: nodales (4,3) y opcionalmente en Gauss (ng,3)."""
        if membrane_q3dof_id not in self.membrane_q3dof:
            self.membrane_q3dof[membrane_q3dof_id] = {"displacements": np.zeros(12)}
        self.membrane_q3dof[membrane_q3dof_id]["stresses"] = stresses_nodes
        if stresses_gauss is not None:
            self.membrane_q3dof[membrane_q3dof_id]["stresses_gauss"] = stresses_gauss

    def get_membrane_q3dof_strains(self, membrane_q3dof_id: int) -> np.ndarray:
        return self.membrane_q3dof[membrane_q3dof_id]["strains"]

    def get_membrane_q3dof_stresses(self, membrane_q3dof_id: int) -> np.ndarray:
        return self.membrane_q3dof[membrane_q3dof_id]["stresses"]

    def get_membrane_q3dof_strains_gauss(self, membrane_q3dof_id: int) -> np.ndarray | None:
        return self.membrane_q3dof[membrane_q3dof_id].get("strains_gauss")

    def get_membrane_q3dof_stresses_gauss(self, membrane_q3dof_id: int) -> np.ndarray | None:
        return self.membrane_q3dof[membrane_q3dof_id].get("stresses_gauss")

    def set_membrane_q2dof_displacements(self, membrane_q2dof_id: int, displacement: np.ndarray) -> None:
        if membrane_q2dof_id not in self.membrane_q2dof:
            self.membrane_q2dof[membrane_q2dof_id] = {"displacements": np.zeros(8)}
        self.membrane_q2dof[membrane_q2dof_id]["displacements"] = displacement

    def get_membrane_q2dof_displacements(self, membrane_q2dof_id: int) -> np.ndarray:
        return self.membrane_q2dof[membrane_q2dof_id]["displacements"]

    def set_membrane_q2dof_strains(self, membrane_q2dof_id: int, strains_nodes: np.ndarray, strains_gauss: np.ndarray | None = None) -> None:
        """Guarda deformaciones: nodales (4,3) y opcionalmente en Gauss (ng,3)."""
        if membrane_q2dof_id not in self.membrane_q2dof:
            self.membrane_q2dof[membrane_q2dof_id] = {"displacements": np.zeros(8)}
        self.membrane_q2dof[membrane_q2dof_id]["strains"] = strains_nodes
        if strains_gauss is not None:
            self.membrane_q2dof[membrane_q2dof_id]["strains_gauss"] = strains_gauss

    def set_membrane_q2dof_stresses(self, membrane_q2dof_id: int, stresses_nodes: np.ndarray, stresses_gauss: np.ndarray | None = None) -> None:
        """Guarda tensiones: nodales (4,3) y opcionalmente en Gauss (ng,3)."""
        if membrane_q2dof_id not in self.membrane_q2dof:
            self.membrane_q2dof[membrane_q2dof_id] = {"displacements": np.zeros(8)}
        self.membrane_q2dof[membrane_q2dof_id]["stresses"] = stresses_nodes
        if stresses_gauss is not None:
            self.membrane_q2dof[membrane_q2dof_id]["stresses_gauss"] = stresses_gauss

    def get_membrane_q2dof_strains(self, membrane_q2dof_id: int) -> np.ndarray:
        return self.membrane_q2dof[membrane_q2dof_id]["strains"]

    def get_membrane_q2dof_stresses(self, membrane_q2dof_id: int) -> np.ndarray:
        return self.membrane_q2dof[membrane_q2dof_id]["stresses"]

    def get_membrane_q2dof_strains_gauss(self, membrane_q2dof_id: int) -> np.ndarray | None:
        return self.membrane_q2dof[membrane_q2dof_id].get("strains_gauss")

    def get_membrane_q2dof_stresses_gauss(self, membrane_q2dof_id: int) -> np.ndarray | None:
        return self.membrane_q2dof[membrane_q2dof_id].get("stresses_gauss")

    def set_truss_displacements(self, truss_id: int, displacement: np.ndarray) -> None:
        if truss_id not in self.trusses:
            self.trusses[truss_id] = {"displacements": np.zeros(6)}
        self.trusses[truss_id]["displacements"] = displacement

    def get_truss_displacements(self, truss_id: int) -> np.ndarray:
        return self.trusses[truss_id]["displacements"]

    def set_truss_axial_force(self, truss_id: int, axial_force: np.ndarray) -> None:
        self.trusses[truss_id]["axial_forces"] = axial_force

    def get_truss_axial_force(self, truss_id: int) -> np.ndarray:
        return self.trusses[truss_id]["axial_forces"]

    def set_truss_internal_forces(self, truss_id: int, internal_forces: np.ndarray) -> None:
        self.trusses[truss_id]["internal_forces"] = internal_forces

    def get_truss_internal_forces(self, truss_id: int) -> np.ndarray:
        return self.trusses[truss_id]["internal_forces"]

    def get_results_truss(self, truss_id: int) -> Dict[str, np.ndarray]:
        return self.trusses[truss_id]

    def set_truss_axial_displacement(self, truss_id: int, axial_displacement: np.ndarray) -> None:
        self.trusses[truss_id]["axial_displacements"] = axial_displacement

    def get_truss_axial_displacement(self, truss_id: int) -> np.ndarray:
        return self.trusses[truss_id]["axial_displacements"]


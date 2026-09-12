from collections.abc import ValuesView
from typing import TYPE_CHECKING
from milcapy.analysis.linear_static import LinearStaticAnalysis
from milcapy.postprocess.post_processing import PostProcessing
from milcapy.core.results import Results
from milcapy.utils.types import StateType
if TYPE_CHECKING:
    from milcapy.model.model import SystemMilcaModel
    from milcapy.loads.load_pattern import LoadPattern



class AnalysisManager:
    """Clase mananger para el análisis estructural.
    maneja los tipos de análisis y opciones de análisis estructural.
    realiza el análisis estructural para todos las condiciones de carga."""

    def __init__(
        self,
        model: "SystemMilcaModel",
    ) -> None:
        """
        Inicializa el análisis estructural.

        Args:
            model: Sistema estructural a analizar.
        """
        self.model = model
        self.load_patterns = model.load_patterns
        self.results = model.results
        self.current_load_pattern: str | None = None


    def notify_all(
        self,
    ) -> None:
        """Notifica a todos los componentes del modelo."""
        # Notificar a todos los miembros del modelo
        for member in self.model.members.values():
            member.set_current_load_pattern(self.current_load_pattern)
        # Notificar a todos los cerchas del modelo
        for truss in self.model.trusses.values():
            truss.set_current_load_pattern(self.current_load_pattern)
        # Notificar a todos los nodos del modelo
        for node in self.model.nodes.values():
            node.set_current_load_pattern(self.current_load_pattern)
        # Notificar al modelo
        self.model.current_load_pattern = self.current_load_pattern

    def analyze_for_list_load_pattern(
        self,
        load_pattern_name: list[str] | None = None,
    ) -> None:
        """Ejecuta el análisis estructural para todas las condiciones de carga."""

        patterns_to_analyze: list[LoadPattern] | ValuesView[LoadPattern] = []

        if load_pattern_name is not None:
            patterns_to_analyze = [self.load_patterns[name] for name in load_pattern_name]
        else:
            patterns_to_analyze = self.load_patterns.values()

        # solucionar para cada load pattern
        for load_pattern in patterns_to_analyze:
            if load_pattern.state == StateType.INACTIVE:
                continue
            # SYSTEM NOTIFICATION: notificar a todos los componentes del modelo
            self.current_load_pattern = load_pattern.name
            self.notify_all()

            # Asignar las cargas a los nodos y miembros almacenados en el patrón de carga
            load_pattern.add_self_weight()
            load_pattern.assign_loads_to_nodes()
            load_pattern.assign_loads_to_members()
            load_pattern.assign_prescribed_dofs_to_nodes()

            # resolver el modelo
            analysis = LinearStaticAnalysis(self.model)
            displacements, reactions = analysis.solve()

            # Crear un objeto de resultados para almacenar los resultados del análisis
            self.results[load_pattern.name] = Results()
            self.results[load_pattern.name].set_model_displacements(displacements)
            self.results[load_pattern.name].set_model_reactions(reactions)


            # crear un objeto de post-procesamiento para procesar los resultados
            post_processing = PostProcessing(self.model, self.results[load_pattern.name], self.model.postprocessing_options, load_pattern.name)
            post_processing.process_displacements_for_nodes()
            post_processing.process_reactions_for_nodes()
            post_processing.process_displacements_for_members()
            post_processing.process_internal_forces_for_members()
            post_processing.process_displacements_for_cst()
            post_processing.post_process_for_cst()
            post_processing.process_displacements_for_membrane_q3dof()
            post_processing.process_displacements_for_membrane_q2dof()
            post_processing.post_process_for_membrane_q3dof()
            post_processing.post_process_for_membrane_q2dof()
            post_processing.post_process_for_members()
            post_processing.process_displacements_for_trusses()
            post_processing.process_internal_forces_for_trusses()
            post_processing.post_process_for_trusses()

            # actualizar el estado de analisis en LoadPattern
            load_pattern.analyzed = True

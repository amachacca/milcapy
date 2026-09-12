import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, TYPE_CHECKING
from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import interp
from milcapy.plotter.utils import separate_areas, process_segments

if TYPE_CHECKING:
    from milcapy.elements.member import Member


@dataclass
class DiagramConfig:
    """
    Clase de configuración para un diagrama de esfuerzo interno.

    Esta clase almacena la configuración necesaria para crear y mostrar un diagrama
    de esfuerzo interno, incluyendo los valores, unidades y opciones de formato.

    Atributos:
        name (str): Nombre descriptivo del diagrama.
        values (NDArray): Array con los valores del diagrama.
        precision (float): Número de decimales para mostrar los valores. Default: 2.

    Métodos:
        format_value(value: float) -> str: Formatea un valor según la precisión especificada.
    """
    name: str
    values: NDArray
    precision: float = 2  # Valor por defecto

    def format_value(self, value: float) -> str:
        """
        Formatea un valor numérico según la precisión especificada.

        Args:
            value (float): Valor numérico a formatear.

        Returns:
            str: Cadena con el valor formateado con la precisión especificada.
        """
        decimals = abs(int(self.precision))
        return f"{value:.{decimals}f}"


class InternalForceDiagramWidget:
    """
    Widget interactivo para visualizar diagramas de esfuerzos internos en elementos estructurales.

    Esta clase crea una interfaz gráfica que permite visualizar múltiples diagramas de esfuerzos
    internos simultáneamente, con capacidades interactivas para mostrar valores específicos
    en cualquier punto a lo largo del elemento.

    Atributos:
        elem (int): Número identificador del elemento estructural.
        x (NDArray): Lista de posiciones x a lo largo del elemento.
        diagrams (Dict[str, DiagramConfig]): Diccionario de configuraciones para cada diagrama.

    Args:
        elem (int): Número del elemento.
        x (NDArray): Array con las posiciones x.
        diagrams (Dict[str, DiagramConfig]): Diccionario con los diagramas de esfuerzos internos.
    """
    # Variables de clase para controlar las instancias
    _active_instances = []  # Lista de instancias activas
    _max_instances = 1      # Número máximo de instancias permitidas

    def __init__(self, member: 'Member', diagrams: Dict[str, DiagramConfig], x: NDArray):
        # Verificar si ya se alcanzó el número máximo de instancias
        if len(InternalForceDiagramWidget._active_instances) >= InternalForceDiagramWidget._max_instances:
            print(f"Máximo número de ventanas alcanzado ({InternalForceDiagramWidget._max_instances}). No se puede crear una nueva instancia.")
            return  # No crear la nueva instancia

        self.member = member
        self.x = x  #np.linspace(0, member.length(), len(diagrams['N(x)'].values))
        self.diagrams = diagrams
        section = f'({member.section.base:.2f}x{member.section.height:.2f})'

        # Inicializar diccionario para elementos interactivos
        self.interactive_elements = {}

        # Crear la ventana principal
        self.root = tk.Tk()
        if hasattr(member, "phi"): # si es marco
            self.root.geometry("650x650")
        else:   # si es armadura
            self.root.geometry("650x300")
        self.root.title(
            f"Diagramas de esfuerzos internos, barra Nro {self.member.id} {section}, Longitud = {self.member.length():.3f}")
        try:
            self.root.iconbitmap("milcapy/plotter/assets/milca.ico")
        except:
            pass

        # Agregar esta instancia a la lista de instancias activas
        InternalForceDiagramWidget._active_instances.append(self)

        # Conectar evento de cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Crear la interfaz
        self.create_main_frame()
        self.create_position_control()  # NUEVO: Crear control de posición
        self.create_grid_layout()

        # Iniciar la interfaz gráfica
        self.root.mainloop()

    def on_closing(self):
        """
        Maneja el evento de cierre de la ventana.

        Este método se ejecuta cuando se cierra la ventana y limpia la instancia activa.
        """
        # Limpiar la referencia de instancia activa
        InternalForceDiagramWidget._active_instances.remove(self)

        # Cerrar la ventana
        self.root.destroy()

    def create_main_frame(self):
        """
        Crea el marco principal de la interfaz gráfica.

        Este método inicializa el frame principal que contendrá todos los elementos
        de la interfaz, configurándolo para expandirse y llenar el espacio disponible.
        """
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    def create_position_control(self):
        """
        NUEVO MÉTODO: Crea el control de posición en la parte superior de la ventana.

        Este método crea un frame con un label y una entrada de texto para controlar
        la posición del marcador verde en todos los diagramas.
        """
        # Frame para el control de posición
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Label
        position_label = ttk.Label(control_frame, text="Mostrar por ubicación:",
                                 font=("Arial", 9))
        position_label.pack(side=tk.LEFT, padx=(0, 10))

        # Variable para la entrada de texto
        self.position_var = tk.StringVar()
        self.position_var.set("0.0")  # Valor inicial

        # Entrada de texto
        self.position_entry = ttk.Entry(control_frame, textvariable=self.position_var,
                                      width=15, font=("Arial", 9))
        self.position_entry.pack(side=tk.LEFT)

        # Validar entrada solo números (incluyendo decimales)
        vcmd = (self.root.register(self.validate_numeric_input), '%P')
        self.position_entry.config(validate='key', validatecommand=vcmd)

        # Conectar evento de Enter en la entrada
        self.position_entry.bind('<Return>', self.on_enter_pressed)

    def validate_numeric_input(self, value):
        """
        NUEVO MÉTODO: Valida que la entrada solo contenga números válidos (incluyendo decimales).

        Args:
            value (str): Valor a validar

        Returns:
            bool: True si es válido, False en caso contrario
        """
        if value == "":
            return True

        # Permitir punto decimal
        if value == ".":
            return True

        try:
            num = float(value)
            # Solo permitir números no negativos
            return num >= 0
        except ValueError:
            return False

    def on_enter_pressed(self, event):
        """
        NUEVO MÉTODO: Maneja el evento cuando se presiona Enter en la entrada de posición.

        Este método se ejecuta cuando se presiona Enter después de escribir un valor
        y actualiza la posición del marcador verde en todos los diagramas.

        Args:
            event: Evento de tecla presionada
        """
        try:
            # Obtener el valor de la entrada
            input_value = self.position_var.get().strip()
            if input_value == "" or input_value == ".":
                return

            x_val = float(input_value)

            # Verificar que sea no negativo
            if x_val < 0:
                x_val = 0
                self.position_var.set("0.0")

            # Limitar al dominio válido
            x_min, x_max = self.x[0], self.x[-1]

            # Si el valor está fuera del rango, ajustarlo
            if x_val < x_min:
                x_val = x_min
                self.position_var.set(f"{x_val:.4f}")
            elif x_val > x_max:
                x_val = x_max
                self.position_var.set(f"{x_val:.4f}")

            # Actualizar todos los diagramas
            self.update_position_markers(x_val)

        except ValueError:
            # Si hay error en la conversión, restaurar valor anterior o poner 0
            self.position_var.set("0.0")
            self.update_position_markers(0.0)

    def update_position_markers(self, x_val):
        """
        NUEVO MÉTODO: Actualiza los marcadores de posición en todos los diagramas.

        Args:
            x_val (float): Posición x donde colocar los marcadores
        """
        for name, elements in self.interactive_elements.items():
            # Usar interpolación para obtener el valor y
            y_val = interp(x_val, self.x, elements['values'])
            config = elements['config']

            # Actualizar elementos de clic (línea verde con punto rojo)
            elements['click_line'].set_xdata([x_val])
            elements['click_point'].set_data([x_val], [y_val])

            # Actualizar etiquetas de valores en el panel lateral
            elements['labels']['value'].configure(
                text=f"{config.format_value(y_val)}")
            elements['labels']['position'].configure(
                text=f"at {x_val:.4f}")

            # Redibujar el canvas
            elements['canvas'].draw()


class MembraneStressWidget:
    """Ventana emergente para inspeccionar un elemento de membrana.

    Muestra el campo activo sobre el elemento (selector con todos los tipos:
    SX/SY/SXY/EX/EY/EXY/UX/UY/UMAG/VM/S1/S2), los valores nodales y la tabla
    completa de valores en TODOS los puntos de Gauss del elemento.

    Args:
        data: dict de ``field_service.element_field_data``.
    """
    _active_instances = []
    _max_instances = 1

    _FIELDS = ["SX", "SY", "SXY", "EX", "EY", "EXY", "UX", "UY", "UMAG", "VM", "S1", "S2"]

    def __init__(self, data: dict):
        if len(MembraneStressWidget._active_instances) >= MembraneStressWidget._max_instances:
            print(f"Máximo número de ventanas alcanzado ({MembraneStressWidget._max_instances}).")
            return
        self.data = data
        self.field = "SX"
        self.figures = []
        self._cbar = None

        self.root = tk.Tk()
        self.root.geometry("720x780")
        self.root.title(f"Esfuerzos en elemento de área — {data['label']}")
        try:
            self.root.iconbitmap("milcapy/plotter/assets/milca.ico")
        except Exception:
            pass
        MembraneStressWidget._active_instances.append(self)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(top, text="Campo:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.field_var = tk.StringVar(value=self.field)
        combo = ttk.Combobox(top, textvariable=self.field_var, values=self._FIELDS,
                             state="readonly", width=8)
        combo.pack(side=tk.LEFT, padx=8)
        combo.bind("<<ComboboxSelected>>", self._on_field_change)
        info = (f"nodos: {len(data['node_ids'])}   Gauss: {data['ngauss']}   "
                f"{data.get('state', '')}   t={data.get('thickness', float('nan')):.4g}")
        ttk.Label(top, text=info, font=("Arial", 8)).pack(side=tk.LEFT, padx=8)

        self.fig = plt.figure(figsize=(6.4, 4.6))
        self.figures.append(self.fig)
        self.ax = self.fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6)
        self.canvas = canvas

        tables = ttk.Frame(self.root)
        tables.pack(fill=tk.BOTH, expand=False, padx=10, pady=6)
        self.gauss_table = self._make_table(
            tables, "Puntos de Gauss",
            ("#", "xi", "eta", "x", "y", "SX", "SY", "SXY", "EX", "EY", "EXY"))
        self._fill_gauss_table()
        self.nodal_table = self._make_table(
            tables, "Nodos del elemento",
            ("nodo", "x", "y", "SX", "SY", "SXY", "EX", "EY", "EXY"))
        self._fill_nodal_table()

        self._draw()
        self.root.mainloop()

    # -- datos ----------------------------------------------------------
    def _field_vectors(self, field: str):
        """Vectores nodales y de Gauss para el campo dado."""
        from milcapy.postprocess.membrane_pp import von_mises, principal_stresses

        d = self.data
        sn = np.asarray(d["stresses_nodes"], dtype=float).reshape(-1, 3)
        en = np.asarray(d["strains_nodes"], dtype=float).reshape(-1, 3)
        sg = np.array([np.asarray(g["stresses"], dtype=float).ravel()[:3] for g in d["gauss"]])
        eg = np.array([np.asarray(g["strains"], dtype=float).ravel()[:3] for g in d["gauss"]])
        comp = {"SX": 0, "SY": 1, "SXY": 2, "EX": 0, "EY": 1, "EXY": 2}
        if field in ("SX", "SY", "SXY"):
            return sn[:, comp[field]], sg[:, comp[field]]
        if field in ("EX", "EY", "EXY"):
            return en[:, comp[field]], eg[:, comp[field]]
        if field in ("UX", "UY", "UMAG"):
            disp = np.asarray(d.get("node_disp", np.full((len(d["node_ids"]), 2), np.nan)), dtype=float)
            if field == "UX":
                nodal = disp[:, 0]
            elif field == "UY":
                nodal = disp[:, 1]
            else:
                nodal = np.sqrt((disp ** 2).sum(axis=1))
            return nodal, np.full(len(d["gauss"]), np.nan)
        out_n = np.array([von_mises(*r) if field == "VM"
                          else principal_stresses(*r)[0 if field == "S1" else 1] for r in sn])
        out_g = np.array([von_mises(*r) if field == "VM"
                          else principal_stresses(*r)[0 if field == "S1" else 1] for r in sg])
        return out_n, out_g

    @staticmethod
    def _fmtval(v) -> str:
        try:
            f = float(v)
        except Exception:
            return "—"
        return f"{f:.4g}" if np.isfinite(f) else "—"

    # -- dibujo ---------------------------------------------------------
    def _draw(self):
        import matplotlib.tri as tri

        d = self.data
        nodal, gvals = self._field_vectors(self.field)
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self.ax.clear()
        xy = np.asarray(d["node_xy"], dtype=float)
        nn = xy.shape[0]
        if not np.any(np.isfinite(nodal)):
            self.ax.fill(list(xy[:, 0]) + [xy[0, 0]], list(xy[:, 1]) + [xy[0, 1]],
                         color="lightgray")
        elif nn == 3:
            tris = np.array([[0, 1, 2]])
            self.ax.tripcolor(xy[:, 0], xy[:, 1], tris, [float(np.mean(nodal))],
                              shading="flat", cmap="jet")
        else:
            tris = np.array([[0, 1, 2], [0, 2, 3]])
            self.ax.tripcolor(tri.Triangulation(xy[:, 0], xy[:, 1], tris), nodal,
                              shading="gouraud", cmap="jet")
        self.ax.plot(list(xy[:, 0]) + [xy[0, 0]], list(xy[:, 1]) + [xy[0, 1]],
                     color="black", linewidth=1.2)
        for k, nid in enumerate(d["node_ids"]):
            self.ax.plot(xy[k, 0], xy[k, 1], "o", color="blue", markersize=5)
            self.ax.annotate(f"N{nid}\n{self._fmtval(nodal[k])}", (xy[k, 0], xy[k, 1]),
                             fontsize=8, color="blue", ha="center", va="bottom")
        for k, g in enumerate(d["gauss"]):
            self.ax.plot(g["x"], g["y"], "x", color="red", markersize=8, markeredgewidth=2)
            self.ax.annotate(f"G{k + 1}\n{self._fmtval(gvals[k])}", (g["x"], g["y"]),
                             fontsize=8, color="red", ha="center", va="top")
        self.ax.set_title(f"{d['label']} — {self.field} (Gauss: {d['ngauss']})")
        self.ax.set_aspect("equal", adjustable="datalim")
        self._cbar = self.fig.colorbar(self.ax.collections[0], ax=self.ax, label=self.field,
                                       fraction=0.046, pad=0.04)
        self.canvas.draw()

    def _on_field_change(self, _event=None):
        self.field = self.field_var.get()
        self._draw()

    # -- tablas ---------------------------------------------------------
    def _make_table(self, parent, title, columns):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.X, pady=4)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=min(max(len(columns) - 4, 3), 9))
        for c in columns:
            tree.heading(c, text=c)
            tree.column(c, width=70 if c not in ("#", "nodo") else 50, anchor="e")
        tree.pack(fill=tk.X)
        return tree

    @staticmethod
    def _fmt(v) -> str:
        try:
            return f"{float(v):.6g}"
        except Exception:
            return "—"

    def _fill_gauss_table(self):
        for k, g in enumerate(self.data["gauss"]):
            s = np.asarray(g["stresses"], dtype=float).ravel()[:3]
            e = np.asarray(g["strains"], dtype=float).ravel()[:3]
            xi = self._fmt(g["xi"]) if g["xi"] is not None else "c"
            eta = self._fmt(g["eta"]) if g["eta"] is not None else "c"
            self.gauss_table.insert("", tk.END, values=(
                k + 1, xi, eta, self._fmt(g["x"]), self._fmt(g["y"]),
                self._fmt(s[0]), self._fmt(s[1]), self._fmt(s[2]),
                self._fmt(e[0]), self._fmt(e[1]), self._fmt(e[2])))

    def _fill_nodal_table(self):
        sn = np.asarray(self.data["stresses_nodes"], dtype=float).reshape(-1, 3)
        en = np.asarray(self.data["strains_nodes"], dtype=float).reshape(-1, 3)
        for nid, (x, y), s, e in zip(self.data["node_ids"], self.data["node_xy"], sn, en):
            self.nodal_table.insert("", tk.END, values=(
                nid, self._fmt(x), self._fmt(y),
                self._fmt(s[0]), self._fmt(s[1]), self._fmt(s[2]),
                self._fmt(e[0]), self._fmt(e[1]), self._fmt(e[2])))

    def on_closing(self):
        try:
            MembraneStressWidget._active_instances.remove(self)
        except ValueError:
            pass
        for fig in self.figures:
            try:
                plt.close(fig)
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def create_grid_layout(self):
        """
        Configura la disposición en cuadrícula de los elementos de la interfaz.

        Este método establece la estructura base de la interfaz, configurando las filas
        y columnas para los diagramas y sus valores asociados. La cuadrícula se ajusta
        automáticamente según el número de diagramas a mostrar.
        """
        # Crear frame para los diagramas (después del control de posición)
        self.diagrams_frame = ttk.Frame(self.main_frame)
        self.diagrams_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Configurar el grid en el frame de diagramas
        num_diagrams = len(self.diagrams)
        for i in range(num_diagrams):
            self.diagrams_frame.grid_rowconfigure(i, weight=1)
        self.diagrams_frame.grid_columnconfigure(0, weight=3)  # Diagrama
        self.diagrams_frame.grid_columnconfigure(1, weight=1)  # Valores

        # Crear cada fila de diagrama y valores
        for i, (diagram_key, config) in enumerate(self.diagrams.items()):
            self.create_diagram_row(i, diagram_key, config)

    def create_diagram_row(self, row: int, diagram_key: str, config: DiagramConfig):
        """
        Crea una fila completa para un diagrama específico en la interfaz.

        Este método genera todos los elementos visuales necesarios para mostrar un diagrama,
        incluyendo el gráfico, las etiquetas y los elementos interactivos.

        Args:
            row (int): Número de fila en la cuadrícula donde se ubicará el diagrama.
            diagram_key (str): Clave identificadora del diagrama en el diccionario.
            config (DiagramConfig): Configuración del diagrama a crear.
        """
        # Frame para el diagrama (ahora en diagrams_frame en lugar de main_frame)
        diagram_frame = ttk.LabelFrame(
            self.diagrams_frame,  # CAMBIO: usar diagrams_frame
            text=f"{config.name}")

        diagram_frame.grid(row=row, column=0, sticky="nsew", padx=4, pady=4)
        diagram_frame.configure(labelwidget=tk.Label(
            diagram_frame,
            text=diagram_frame.cget("text"),
            foreground="blue",
            font=("Arial", 8) #"bold")
        ))
        # Crear figura y gráfico
        fig, ax = plt.subplots(figsize=(5.1, 1.6))
        self.plot_member_values(ax, config)

        # Configuración del gráfico
        self.setup_plot_style(ax)
        self.adjust_plot_margins(ax)

        # Añadir texto para coordenadas xy en la esquina superior derecha
        xy_text = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                          ha='right', va='top', fontsize=8,
                          bbox=dict(facecolor='none', alpha=0.4, edgecolor='none'))

        # Canvas para el gráfico
        canvas = FigureCanvasTkAgg(fig, master=diagram_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Elementos interactivos
        interactive = {
            'ax': ax,
            'canvas': canvas,

            'line': ax.axvline(x=self.x[0], color="#e34f6f", linestyle="-", alpha=1, linewidth=0.8),
            'point': ax.plot([self.x[0]], [config.values[0]], 'o', markersize=2, color='#00d068', alpha=1)[0],
            # 'label': ax.text(self.x[0], config.values[0], "", color="blue", fontsize=8, fontstyle='italic', fontfamily='serif'),

            'click_line': ax.axvline(x=self.x[0], color='#91a6a6', linestyle='-', alpha=0.8, linewidth=0.8),
            'click_point': ax.plot([], [], 'ro', markersize=2)[0],
            # 'click_label': ax.text(0, 0, "", color="k", fontsize=8, ha="left", va="bottom", fontstyle='italic', fontfamily='serif'),

            'values': config.values,
            'xy_text': xy_text  # Añadir referencia al texto xy
        }

        # Guardar referencias usando diagram_key
        self.interactive_elements[diagram_key] = interactive

        # Conectar eventos
        fig.canvas.mpl_connect("motion_notify_event",
                               lambda event: self.on_hover(event, diagram_key))
        fig.canvas.mpl_connect("button_press_event",
                               lambda event: self.on_click(event))

        # Frame para valores (ahora en diagrams_frame en lugar de main_frame)
        values_frame = self.create_values_frame(row, diagram_key, config)
        values_frame.grid(row=row, column=1, sticky="nsew", padx=4, pady=4)

        fig.tight_layout(pad=0)


    def plot_member_values(self, ax: plt.Axes, config: DiagramConfig):
        """Plotea el grafico completo de un miembro con sus topicos asignados"""
        x_values = self.x
        y_values = config.values
        length = self.member.length()

        # Separar y procesar áreas
        positive, negative = separate_areas(y_values, length, x_values)
        positive_processed = process_segments(positive, length)
        negative_processed = process_segments(negative, length)

        ax.plot(self.x, config.values,
                color='blue', linewidth=0.4, alpha=0.8)
        ax.plot([self.x[0], self.member.length()], [0, 0], 'k', linewidth=0.9)

        for area in positive_processed:
            area = np.asarray(area)
            ax.fill(area[:, 0], area[:, 1], 0,
                            color='#7f80fc', alpha=1)
        for area in negative_processed:
            area = np.asarray(area)
            ax.fill(area[:, 0], area[:, 1], 0,
                            color="#ff8080", alpha=1)

    def create_values_frame(self, row: int, name: str, config: DiagramConfig) -> ttk.LabelFrame:
        """
        Crea un frame para mostrar los valores asociados a un diagrama.

        Este método genera un panel lateral que muestra valores estáticos y dinámicos
        relacionados con el diagrama.

        Args:
            row (int): Número de fila en la cuadrícula.
            name (str): Nombre del diagrama.
            config (DiagramConfig): Configuración del diagrama.

        Returns:
            ttk.LabelFrame: Frame contenedor de los valores del diagrama.
        """
        values_frame = ttk.LabelFrame(self.diagrams_frame, text=f"Valores {name}")  # CAMBIO: usar diagrams_frame
        values_frame.configure(labelwidget=tk.Label(
            values_frame,
            text=values_frame.cget("text"),
            foreground="blue",
            font=("Arial", 8) # "bold")
        ))

        # Valores estáticos con precisión específica
        self.add_label(
            values_frame, f"Max = {config.format_value(np.max(config.values))}")
        self.add_label(
            values_frame, f"Min = {config.format_value(np.min(config.values))}")

        # Espacio en blanco
        self.add_label(values_frame, " ")

        # Valores dinámicos (sin xy)
        dynamic_labels = {
            'value': self.add_label(values_frame, "-", dynamic=True),
            'position': self.add_label(values_frame, "-", dynamic=True),
        }

        # Guardar referencias a las etiquetas dinámicas
        self.interactive_elements[name].update({
            'labels': dynamic_labels,
            'config': config  # Guardar la configuración para acceder a la precisión
        })

        return values_frame

    def setup_plot_style(self, ax):
        """
        Configura el estilo visual del gráfico.

        Establece los parámetros de visualización básicos para el gráfico, como
        visibilidad de ejes, color de fondo y rejilla.

        Args:
            ax (matplotlib.axes.Axes): Eje del gráfico a configurar.
        """
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.4)
        ax.set_facecolor('lightblue')
        ax.grid(False)

    def adjust_plot_margins(self, ax):
        """
        Ajusta los márgenes del gráfico para una mejor visualización.

        Calcula y aplica márgenes apropiados alrededor del gráfico, considerando
        las transformaciones entre coordenadas de píxeles y datos.

        Args:
            ax (matplotlib.axes.Axes): Eje del gráfico a ajustar.
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        trans = ax.transData.transform
        inv_trans = ax.transData.inverted().transform

        ymin_pixel = trans((0, ymin))[1]
        ymax_pixel = trans((0, ymax))[1]
        xmin_pixel = trans((xmin, 0))[0]
        xmax_pixel = trans((xmax, 0))[0]

        ymin_pixel -= 20
        ymax_pixel += 20
        xmin_pixel -= 20
        xmax_pixel += 20

        ymin_new = inv_trans((0, ymin_pixel))[1]
        ymax_new = inv_trans((0, ymax_pixel))[1]
        xmin_new = inv_trans((xmin_pixel, 0))[0]
        xmax_new = inv_trans((xmax_pixel, 0))[0]

        ax.set_ylim(ymin_new, ymax_new)
        ax.set_xlim(xmin_new, xmax_new)

    def add_label(self, parent: ttk.LabelFrame, text: str, dynamic: bool = False) -> ttk.Label:
        """
        Añade una etiqueta al frame especificado.

        Crea y configura una nueva etiqueta con el texto proporcionado.

        Args:
            parent (ttk.LabelFrame): Frame contenedor de la etiqueta.
            text (str): Texto a mostrar en la etiqueta.
            dynamic (bool, optional): Indica si la etiqueta será dinámica. Default: False.

        Returns:
            ttk.Label: Referencia a la etiqueta creada si es dinámica, None en caso contrario.
        """
        label = ttk.Label(parent, text=text, font=(
            "Helvetica", 8), foreground="black")
        label.pack(anchor="w", padx=0, pady=0)
        return label if dynamic else None

    def on_hover(self, event, current_diagram: str):
        """
        Maneja los eventos de movimiento del mouse sobre los gráficos.

        Actualiza la visualización de valores y elementos interactivos según la posición
        del mouse, incluso cuando está fuera del canvas de Matplotlib.

        Args:
            event: Evento de movimiento del mouse.
            current_diagram (str): Identificador del diagrama actual.
        """
        if event.inaxes is not None:
            x_val = event.xdata

            # Ajustar x_val al rango válido
            if x_val < self.x[0]:
                x_val = self.x[0]
            elif x_val > self.x[-1]:
                x_val = self.x[-1]

            # Actualizar todos los diagramas
            for name, elements in self.interactive_elements.items():
                # Usar interpolación para obtener el valor y
                y_val = interp(x_val, self.x, elements['values'])
                config = elements['config']

                # Actualizar elementos visuales
                elements['line'].set_xdata([x_val])
                elements['point'].set_data([x_val], [y_val])

                # Actualizar texto xy en la gráfica
                elements['xy_text'].set_text(
                    f"{x_val:.2f}, {config.format_value(y_val)}")

                # Redibujar el canvas
                elements['canvas'].draw()

    def on_click(self, event):
        """
        Maneja los eventos de clic del mouse sobre los gráficos.

        Registra y visualiza los valores en el punto donde se realizó el clic,
        funcionando incluso cuando el clic ocurre fuera del canvas de Matplotlib.

        Args:
            event: Evento de clic del mouse.
        """
        if event.inaxes is not None:
            x_click = event.xdata

            # Ajustar x_click al rango válido
            if x_click < self.x[0]:
                x_click = self.x[0]
            elif x_click > self.x[-1]:
                x_click = self.x[-1]

            # Actualizar todos los diagramas
            for name, elements in self.interactive_elements.items():
                # Usar interpolación para obtener el valor y
                y_click = interp(x_click, self.x, elements['values'])
                config = elements['config']

                # Actualizar punto y línea de clic
                elements['click_point'].set_data([x_click], [y_click])
                elements['click_line'].set_xdata([x_click])
                self.position_var.set("{:.4f}".format(x_click))

                # Actualizar etiquetas de valores
                elements['labels']['value'].configure(
                    text=f"{config.format_value(y_click)}")
                elements['labels']['position'].configure(
                    text=f"at {x_click:.4f}")

                # Redibujar el canvas
                elements['canvas'].draw()
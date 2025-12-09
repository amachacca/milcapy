<div align="center" style="padding-top:20px; padding-bottom:10px;">
  <img src="docs/assets/logo.png" alt="milcapy logo" width="140">
</div>

<h1>milcapy</h1>

[![PyPI](https://img.shields.io/pypi/v/milcapy.svg?color=blue)](https://pypi.org/project/milcapy/)
[![Python Versions](https://img.shields.io/pypi/pyversions/milcapy.svg)](https://pypi.org/project/milcapy/#history)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/amachacca/milcapy/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://amachacca.github.io/milcapy/)
[![Source Code](https://img.shields.io/badge/source-GitHub-black.svg)](https://github.com/amachacca/milcapy/tree/main)
[![Examples](https://img.shields.io/badge/examples-available-purple.svg)](https://github.com/amachacca/milcapy/tree/main/examples)






Biblioteca para el análisis estructural de **marcos en 2D**, con soporte para elementos:
- **CST(constant strain triangle)**
- **Q4(quadrilateral 4 nodos & 2dof por nodo)**
- **Q6(quadrilateral 4 nodos & 3dof por nodo)**
- **Q6i(rectangular 4 nodos & 2dof por nodo + modos incompatibles)**
- **Q8(quadrilateral 8 nodos (los 4 nodos intermedios son condensados) & 2dof por nodo)**

Implementa el **método de rigidez directa** y el **método de los elementos finitos** para membranas, con **solución cerrada** en elementos unidimensionales (1D: marcos y armaduras).
Además, incorpora conceptos avanzados de **análisis matricial de estructuras**.

Desarrollado por **Amilcar Machacca Mayo**  
GitHub: [amachacca](https://github.com/amachacca)  
Repositorio: [milcapy](https://github.com/amachacca/milcapy)

---

### Características principales

- **Definición de materiales**
- **Definición de secciones**
  - Rectangulares
  - Circulares
  - Genéricas
  - Tipo cáscara
- **Definición de nodos**
- **Definición de elementos**
  - Vigas (marcos)
    - Vigas de **Timoshenko**
    - Vigas de **Euler-Bernoulli**
  - Armaduras
  - Membranas
    - CST (Constant Strain Triangle)
    - Q4 (Quadrilateral 4 nodos & 2dof por nodo)
    - Q6 (Quadrilateral 4 nodos & 3dof por nodo)
    - Q6i (Rectangular 4 nodos & 2dof por nodo + modos incompatibles)
    - Q8 (Quadrilateral 8 nodos (los 4 nodos intermedios son condensados) & 2dof por nodo)
- **Patrones de carga**
- **Modificadores de propiedades de sección**
- **Condiciones de frontera**
  - Restricciones convencionales
  - Apoyos elásticos
  - Definición de eje local para nodos
- **Cargas**
  - Cargas nodales
  - Cargas distribuidas (Uniforme y trapezoidal)
  - Peso propio
- **Opciones avanzadas**
  - Desfase de extremos (brazos rígidos)
  - Liberaciones
- **Resultados**
  - Obtención de la matriz de rigidez global
  - Vector de cargas por patrón de carga
  - Resultados de los nodos
  - Resultados de los miembros
- **Visualización interactiva** del modelo

---

### Estudio de convergencia para elementos finitos:
ina viga en voladizo de 2m de longitud, sección rectangular de 0.6m de altura y 0.4m de base, materiales con E=2e6 y v=0.2 y una fuerza aplicada en el extremos libre de -20tonf.

<div align="center">
  <img src="docs/assets/convergencia_membranas.png" alt="Resultados" width="90%">
</div>




### Instalación

```bash
pip install milcapy
```

### Comandos de importacion

```python
from milcapy import (
  SystemModel,
  model_viewer,
  BeamTheoriesType,
  CoordinateSystemType,
  DirectionType,
  StateType,
  LoadType,
  FieldType,
  ConstitutiveModelType,
  IntegrationType,
)
```
> - `SystemModel` → Plano de contruccion para crear el modelo  
> - `model_viewer` → Para visualizar el modelo  
> - `BeamTheoriesType` → Tipos de teorías de vigas 
>   - TIMOSHENKO
>   - EULER_BERNOULLI
> - `CoordinateSystemType` → Tipos de sistemas de coordenadas 
>   - LOCAL
>   - GLOBAL
> - `DirectionType` → Tipos de direcciones 
>   - X
>   - Y
>   - X_PROJ
>   - Y_PROJ
>   - GRAVITY
>   - GRAVITY_PROJ
>   - MOMENT
>   - LOCAL_1
>   - LOCAL_2
>   - LOCAL_3
> - `StateType` → Tipos de estados 
>   - ACTIVE
>   - INACTIVE
> - `LoadType` → Tipos de cargas 
>   - FORCE
>   - MOMENT
> - `FieldType` → Tipos de campos para membranas
>   - SX
>   - SY
>   - SXY
>   - EX
>   - EY
>   - EXY
>   - UX
>   - UY
>   - UMAG
> - `ConstitutiveModelType` → Tipos de estados constitutivos para membranas
>   - PLANE_STRESS
>   - PLANE_STRAIN
> - `IntegrationType` → Tipos de integración para membranas
>   - COMPLETE
>   - REDUCED

### 1. Comandos de modelo

```python
model = SystemModel()
```

> - `model` (`SystemModel`) → Modelo vacio creado

### 2. comando de material

```python
model.add_material('name', 'modulus_elasticity', 'poisson_ratio', 'specific_weight=0')
```

> - `name` (`str`) → Nombre del material
> - `modulus_elasticity` (`float`) → Módulo de elasticidad
> - `poisson_ratio` (`float`) → Coeficiente de Poisson
> - `specific_weight` (`float`) → Peso específico (opcional)

### 3. Comando de seccion

#### 3.1. Comando de seccion rectangular

```python
model.add_rectangular_section('name', 'material_name', 'base', 'height')
```

> - `name` (`str`) → Nombre de la sección
> - `material_name` (`str`) → Nombre del material
> - `base` (`float`) → Base de la sección
> - `height` (`float`) → Altura de la sección

#### 3.2. Comando de seccion circular

```python
model.add_circular_section('name', 'material_name', 'diameter')
```

> - `name` (`str`) → Nombre de la sección
> - `material_name` (`str`) → Nombre del material
> - `diameter` (`float`) → Diámetro de la sección

#### 3.3. Comando de seccion generica

```python
model.add_generic_section('name', 'material_name', 'area', 'inertia', 'k_factor')
```

> - `name` (`str`) → Nombre de la sección
> - `material_name` (`str`) → Nombre del material
> - `area` (`float`) → Área de la sección
> - `inertia` (`float`) → Momento de inercia de la sección
> - `k_factor` (`float`) → Coeficiente de corte de la sección (Ac = A * k_factor)

#### 3.4. Comando de seccion de cascaras

```python
model.add_shell_section('name', 'material_name', 'thickness')
```
Este tipo de seccion se usa para modelar elementos planos (membranas)

> - `name` (`str`) → Nombre de la sección
> - `material_name` (`str`) → Nombre del material
> - `thickness` (`float`) → Grosor de la sección

### 4. Comando de nodo

```python
model.add_node('id', 'x', 'y')
```

> - `id` (`int`) → ID del nodo
> - `x` (`float`) → Coordenada x del nodo
> - `y` (`float`) → Coordenada y del nodo
>> **⚠️ NOTA IMPORTANTE:** los ID's deben ser ordenados y secuenciales (1, 2, 3, ...)

### 5. Comando de elemento marco

#### 5.1. Comando de marco implemetadas con las teorías de viga de Timoshenko y Euler-Bernoulli


```python
model.add_member('id', 'node_i_id', 'node_j_id', 'section_name', 'beam_theory=BeamTheoriesType.TIMOSHENKO')
```

> - `id` (`int`) → ID del miembro
> - `node_i_id` (`int`) → ID del nodo inicial
> - `node_j_id` (`int`) → ID del nodo final
> - `section_name` (`str`) → Nombre de la sección
> - `beam_theory` (`str` ó `BeamTheoriesType`) → Teoría de la viga (opcional)
>> **VIGAS DISPONIBLES:** 'TIMOSHENKO', 'EULER_BERNOULLI'

#### 5.2. Comando de viga de Timoshenko (marco)

```python
model.add_elastic_timoshenko_beam('id', 'node_i_id', 'node_j_id', 'section_name')
```

> - `id` (`int`) → ID del miembro
> - `node_i_id` (`int`) → ID del nodo inicial
> - `node_j_id` (`int`) → ID del nodo final
> - `section_name` (`str`) → Nombre de la sección

#### 5.3. Comando de viga de Euler-Bernoulli (marco)

```python
model.add_elastic_euler_bernoulli_beam('id', 'node_i_id', 'node_j_id', 'section_name')
```

> - `id` (`int`) → ID del miembro
> - `node_i_id` (`int`) → ID del nodo inicial
> - `node_j_id` (`int`) → ID del nodo final
> - `section_name` (`str`) → Nombre de la sección

### 6. Comando de elemento de Armadura

```python
model.add_truss('id', 'node_i_id', 'node_j_id', 'section_name')
```

> - `id` (`int`) → ID del miembro
> - `node_i_id` (`int`) → ID del nodo inicial
> - `node_j_id` (`int`) → ID del nodo final
> - `section_name` (`str`) → Nombre de la sección


### 7. Comando de elemento de **membrana**
#### 7.1. Comando de elemento de membrana CST
El elemento de membrana CST (*Constant Strain Triangle*) es un elemento finito bidimensional de tres nodos.
Se utiliza para modelar membranas planas sometidas a esfuerzos en su plano.

Cada nodo posee dos grados de libertad de traslación (en `x` y `y`). La deformación se considera constante dentro del triángulo,
por lo que es adecuado para geometrías simples o como base en mallados más refinados.

```python
model.add_cst('id', 'node_ids', 'section_name', 'state=ConstitutiveModel.PLANE_STRESS')
```

> - `id` (`int`) → ID del miembro
> - `node_ids` (`list[int]`) → Un lista de IDs de los 3 nodos
> - `section_name` (`str`) → Nombre de la sección
> - `state` (`ConstitutiveModel`) → Estado constitutivo del elemento (opcional)
>> ⚠️ **Nota:** La enumeración de los nodos debe realizarse en sentido antihorario para garantizar una formulación correcta.  
>> **ESTADOS DISPONIBLES:** 'PLANE_STRESS', 'PLANE_STRAIN'

#### 7.2. Comando de elemento de membrana Q4
El elemento de membrana Q4 es un elemento finito bidimensional de cuatro nodos con integracion de 4 puntos, con funciones de forma bilineales es un elemento serendipito con formulacion isoparametrica.

Cada nodo posee dos grados de libertad de traslación (en `x` y `y`).

```python
model.add_membrane_q4('id', '*node_ids', 'section_name', 'state=ConstitutiveModel.PLANE_STRESS')
```

> - `id` (`int`) → ID del miembro
> - `node_ids` (`list[int]`) → Una lista de IDs de los 4 nodos
> - `section_name` (`str`) → Nombre de la sección
> - `state` (`ConstitutiveModel`) → Estado constitutivo del elemento (opcional)
>> ⚠️ **Nota:** La enumeración de los nodos debe realizarse en sentido antihorario para garantizar una formulación correcta.  
>> ⚠️ **Nota:** Se recomienda discretizar con el elemento Q4 ya que esta tiene una convergencia muy mala con un solo elemento.   
>> **ESTADOS DISPONIBLES:** 'PLANE_STRESS', 'PLANE_STRAIN'


#### 7.3. Comando de elemento de membrana Q6
El elemento de membrana Q6 (*Quadrilateral with degrees of freedom of perforation*) es un elemento finito bidimensional de cuatro nodos con integracion de 4 puntos, con funciones de forma 4 bilineales + 4 cuadradas es un elemento serendipito con formulacion isoparametrica.

Cada nodo posee tres grados de libertad; traslacion en `x`, `y` y rotacion en `z`.

```python
model.add_membrane_q6('id', '*node_ids', 'section_name', 'state=ConstitutiveModel.PLANE_STRESS')
```

> - `id` (`int`) → ID del miembro
> - `node_ids` (`list[int]`) → Una lista de IDs de los 4 nodos
> - `section_name` (`str`) → Nombre de la sección
> - `state` (`ConstitutiveModel`) → Estado constitutivo del elemento (opcional)
>> ⚠️ **Nota:** La enumeración de los nodos debe realizarse en sentido antihorario para garantizar una formulación correcta.  
>> **ESTADOS DISPONIBLES:** 'PLANE_STRESS', 'PLANE_STRAIN'

#### 7.4. Comando de elemento de membrana Q6I
El elemento de membrana Q6I (*Quadrilateral with Incompatible Modes*) es un elemento finito bidimensional de cuatro nodos con integracion de 4 puntos, con funciones de forma 4 bilineales + 2 cuadradas (modos incompatibles) es un elemento serendipito con formulacion isoparametrica.

Cada nodo posee dos grados de libertad de traslación (en `x` y `y`).

```python
model.add_membrane_q6i('id', '*node_ids', 'section_name', 'state=ConstitutiveModel.PLANE_STRESS')
```

> - `id` (`int`) → ID del miembro
> - `node_ids` (`list[int]`) → Una lista de IDs de los 4 nodos
> - `section_name` (`str`) → Nombre de la sección
> - `state` (`ConstitutiveModel`) → Estado constitutivo del elemento (opcional)
>> ⚠️ **Nota:** La enumeración de los nodos debe realizarse en sentido antihorario para garantizar una formulación correcta.
>> **ESTADOS DISPONIBLES:** 'PLANE_STRESS', 'PLANE_STRAIN'

#### 7.5. Comando de elemento de membrana Q8
El elemento de membrana Q8 es un elemento finito bidimensional de ocho nodos con integracion de 4 puntos ó 9 puntos, con funciones de forma 4 bicuadradas + 4 cuadradas es un elemento serendipito con formulacion isoparametrica.

Cada nodo posee dos grados de libertad; traslacion en `x`, `y`.

```python
model.add_membrane_q8('id', '*node_ids', 'section_name', 'state=ConstitutiveModel.PLANE_STRESS', 'integration=IntegrationType.COMPLETE')
```

> - `id` (`int`) → ID del miembro
> - `node_ids` (`list[int]`) → Una lista de IDs de los 4 nodos
> - `section_name` (`str`) → Nombre de la sección
> - `state` (`str` ó `ConstitutiveModel`) → Estado constitutivo del elemento (opcional)
> - `integration` (`str` ó `IntegrationType`) → Tipo de integracion (opcional)
>> ⚠️ **Nota:** La enumeración de los nodos debe realizarse en sentido antihorario para garantizar una formulación correcta.  
>> ⚠️ **Nota:** El elemento Q8 tiene 8 nodos pero sin embargo los 4 faltantes son calculado en el medio de los lados del cuadrilatero y estas a su vez son condensados en los 4 nodos originales del cuadrilatero.  
>> ⚠️ **Nota:** Se recomienda no discritizar con el elemento Q8 ya que esta con un solo elemento da una convergencia muy buena. debido a su que su implementacion es muy robusta. caso que si se discretiza esta se vuelve muy flexible y se aleja de la solucion exacta.  
>> ⚠️ **Nota:** La integracion reducida evita de cierta manera la excesiva rigidez a cortante o tambien denominado cortante parásito.  
>> **ESTADOS DISPONIBLES:** 'PLANE_STRESS', 'PLANE_STRAIN'  
>> **INTEGRACION DISPONIBLE:** 'REDUCED' (4 puntos), 'COMPLETE' (9 puntos)


### 8. Comando de patron de carga
El patron de carga es un conjunto de cargas que se aplican a los miembros del modelo, el modelo puede tener varios patrones de carga y analizarlos por separado.

```python
model.add_load_pattern('name', 'self_weight_multiplier=0', 'state=StateType.ACTIVE')
```

> - `name` (`str`) → Nombre del patron de carga
> - `self_weight_multiplier` (`float`) → Multiplicador del peso propio (opcional)
> - `state` (`str` ó `StateType`) → Estado del patron de carga (opcional)
>> **ESTADOS DISPONIBLES:** 'ACTIVE', 'INACTIVE'

```python
model.set_state_load_pattern('name', 'state=StateType.ACTIVE')
```

> - `name` (`str`) → Nombre del patron de carga
> - `state` (`str` ó `StateType`) → Estado del patron de carga
>> **ESTADOS DISPONIBLES:** 'ACTIVE', 'INACTIVE'



### 9. Comandos de asignacion
#### 9.1. Comandos de asignacion modificador de propiedades
```python
model.set_property_modifiers('section_name', 'axial_area=1', 'shear_area=1', 'moment_inertia=1', 'weight=1')
```

> - `section_name` (`str`) → Nombre de la sección
> - `axial_area` (`float`) → Modificador de área transversal (Opcional)
> - `shear_area` (`float`) → Modificador de área de corte (Opcional)
> - `moment_inertia` (`float`) → Modificador de momento de inercia (Opcional)
> - `weight` (`float`) → Modificador de peso (Opcional)


#### 9.2. Comando de condiciones de frontera
##### 9.2.1. Comando de restricciones
```python
model.add_restraint('node_id', 'ux', 'uy', 'rz')
```

> - `node_id` (`int`) → ID del nodo
> - `ux` (`bool`) → Restricción de traslación en el eje x
> - `uy` (`bool`) → Restricción de traslación en el eje y
> - `rz` (`bool`) → Restricción de rotación en el eje z

##### 9.2.2. Comando de apoyos elásticos
```python
model.add_elastic_support('node_id', 'kx=None', 'ky=None', 'krz=None', 'CSys=CoordinateSystemType.GLOBAL')
```

> - `node_id` (`int`) → ID del nodo
> - `kx` (`float`) → Constante de rigidez en X
> - `ky` (`float`) → Constante de rigidez en Y
> - `krz` (`float`) → Constante de rigidez en Z
> - `CSys` (`str` ó `CoordinateSystemType`) → Sistema de coordenadas (opcional)
>> **CSYS DISPONIBLES:** 'GLOBAL', 'LOCAL'

##### 9.2.3. Comando de eje local
```python
model.add_local_axis_for_node('node_id', 'angle')
```

> - `node_id` (`int`) → ID del nodo
> - `angle` (`float`) → Ángulo del eje local en grados



##### 9.3. Comando de cargas

###### 9.3.1. Comando de cargas puntuales
```python
model.add_point_load('node_id', 'load_pattern_name', 'fx=0', 'fy=0', 'mz=0', 'CSys=CoordinateSystemType.GLOBAL', 'replace=False')
```

> - `node_id` (`int`) → ID del nodo
> - `load_pattern_name` (`str`) → Nombre del patron de carga
> - `fx` (`float`) → Fuerza en X (opcional)
> - `fy` (`float`) → Fuerza en Y (opcional)
> - `mz` (`float`) → Momento en Z (opcional)
> - `CSys` (`str` ó `CoordinateSystemType`) → Sistema de coordenadas (opcional)
> - `replace` (`bool`) → Reemplazar la carga existente (opcional)
>> **CSYS DISPONIBLES:** 'GLOBAL', 'LOCAL'


###### 9.3.2. Comando de asignacion de desplazamientos
```python
model.add_prescribed_dof('node_id', 'load_pattern_name', 'ux=None', 'uy=None', 'rz=None', 'CSys=CoordinateSystemType.GLOBAL')
```

> - `node_id` (`int`) → ID del nodo
> - `load_pattern_name` (`str`) → Nombre del patron de carga
> - `ux` (`float`) → Desplazamiento en X (opcional)
> - `uy` (`float`) → Desplazamiento en Y (opcional)
> - `rz` (`float`) → Rotacion en Z (opcional)
> - `CSys` (`str` ó `CoordinateSystemType`) → Sistema de coordenadas (opcional)
>> **CSYS DISPONIBLES:** 'GLOBAL', 'LOCAL'


###### 9.3.3. Comando de cargas distribuidas
```python
model.add_distributed_load('member_id', 'load_pattern_name', 'load_start=0', 'load_end=0', 'CSys=CoordinateSystemType.GLOBAL', 'direction=DirectionType.LOCAL_2', 'load_type=LoadType.FORCE', 'replace=False')
```

> - `member_id` (`int`) → ID del miembro
> - `load_pattern_name` (`str`) → Nombre del patron de carga
> - `load_start` (`float`) → Magnitud de la carga en el inicio (opcional)
> - `load_end` (`float`) → Magnitud de la carga en el final (opcional)
> - `CSys` (`str` ó `CoordinateSystemType`) → Sistema de coordenadas (opcional)
> - `direction` (`str` ó `DirectionType`) → Dirección de la carga (opcional)
> - `load_type` (`str` ó `LoadType`) → Tipo de carga (opcional)
> - `replace` (`bool`) → Reemplazar la carga existente (opcional)
>> **CSYS DISPONIBLES:** 'GLOBAL', 'LOCAL'
>> **DIRECCION DISPONIBLES:**
>>> - **LOCAL:** 'LOCAL_1', 'LOCAL_2', 'LOCAL_3'
>>> - **GLOBAL:** 'X', 'Y', 'Z', 'X_PROJ', 'Y_PROJ', 'GRAVITY', 'GRAVITY_PROJ'
>>
>> **TIPO DE CARGA DISPONIBLES:** 'FORCE'

###### 9.3.4. Comando de cargas de peso propio
```python
model.add_self_weight('load_pattern_name', 'factor=1')
```

> - `load_pattern_name` (`str`) → Nombre del patron de carga
> - `factor` (`float`) → Factor de escala para el peso propio (opcional)


#### 9.4. Comando de desface de exremos (brazos rigidos)
```python
model.add_end_length_offset('member_id', 'la=0', 'lb=0', 'qla=True', 'qlb=True', 'fla=1', 'flb=1')
```

> - `member_id` (`int`) → ID del miembro
> - `la` (`float`) → Desface de longitud final en el nodo inicial (opcional)
> - `lb` (`float`) → Desface de longitud final en el nodo final (opcional)
> - `qla` (`bool`) → Si se aplica la carga en el brazo inicial (opcional)
> - `qlb` (`bool`) → Si se aplica la carga en el brazo final (opcional)
> - `fla` (`float`) → Factor de zona rigida del brazo inicial (opcional)
> - `flb` (`float`) → Factor de zona rigida del brazo final (opcional)


#### 9.5. Comando de liberaciones
```python
model.add_releases('member_id', 'pi=False', 'vi=False', 'mi=False', 'pj=False', 'vj=False', 'mj=False')
```

> - `member_id` (`int`) → ID del miembro
> - `pi` (`bool`) → Liberación de fuerza Axial del nodo inicial (opcional)
> - `vi` (`bool`) → Liberación de cortante del nodo inicial (opcional)
> - `mi` (`bool`) → Liberación de momento del nodo inicial (opcional)
> - `pj` (`bool`) → Liberación de fuerza Axial del nodo final (opcional)
> - `vj` (`bool`) → Liberación de cortante del nodo final (opcional)
> - `mj` (`bool`) → Liberación de momento del nodo final (opcional)

### 10. Comandos de analisis
```python
model.solve(load_pattern_name=None)
```

> - `load_pattern_name` (`list[str]` ó `None`) → Nombre de los patrones de carga a resolver (opcional), si es None se resuelve para todos los patrones de carga

### 11. Comandos de postprocesamiento
#### 11.1. Comando de matriz de rigidez global
```python
matrix: NDAarray = model.get_global_stiffness_matrix()
```

#### 11.2. Comando de vector de fuerzas global
```python
vector: NDAarray = model.get_global_load_vector('load_pattern_name')
```

> - `load_pattern_name` (`str`) → Nombre del patron de carga

#### 11.3. Comando de graficos
Este comando abre una ventana interactiva para visualizar el modelo.
```python
model.show()
```


### 12. Comando de resultados
```python
results: Results = model.get_results('load_pattern_name')
```

> - `results` (`Results`) → Objeto que almacena todos los resultados del análisis.
> - `load_pattern_name` (`str`) → Nombre del patrón de carga.

---

**METODOS DE `Results`**

#### 12.1. Resultados a nivel de modelo
```python
displacements: np.ndarray = results.get_model_displacements()
reactions: np.ndarray = results.get_model_reactions()
```

> - `displacements` (`np.ndarray`) → Desplazamientos globales del modelo.
> - `reactions` (`np.ndarray`) → Reacciones globales del modelo.

---

#### 12.2. Resultados a nivel de nodos
```python
displacements: np.ndarray = results.get_node_displacements(node_id)
reactions: np.ndarray = results.get_node_reactions(node_id)
```

> - `node_id` (`int`) → Identificador del nodo.
> - `displacements` (`np.ndarray`) → Desplazamientos del nodo.
> - `reactions` (`np.ndarray`) → Reacciones en el nodo.

---

#### 12.3. Resultados a nivel de miembros
```python
displacements: np.ndarray      = results.get_member_displacements(member_id)
internal_forces: np.ndarray    = results.get_member_internal_forces(member_id)
x_val: np.ndarray              = results.get_member_x_val(member_id)
axial_force: np.ndarray        = results.get_member_axial_force(member_id)
shear_force: np.ndarray        = results.get_member_shear_force(member_id)
bending_moment: np.ndarray     = results.get_member_bending_moment(member_id)
deflection: np.ndarray         = results.get_member_deflection(member_id)
slope: np.ndarray              = results.get_member_slope(member_id)
axial_displacement: np.ndarray = results.get_member_axial_displacement(member_id)
```

> - `member_id` (`int`) → Identificador del miembro.
> - Cada propiedad retorna un `np.ndarray` con los resultados correspondientes.

---

#### 12.4. Resultados de elementos CST (triángulos finitos)
```python
displacements: np.ndarray = results.get_cst_displacements(cst_id)
strains: np.ndarray       = results.get_cst_strains(cst_id)
stresses: np.ndarray      = results.get_cst_stresses(cst_id)
```

> - `cst_id` (`int`) → Identificador del elemento CST.
> - `displacements` (`np.ndarray`) → Desplazamientos.
> - `strains` (`np.ndarray`) → Deformaciones.
> - `stresses` (`np.ndarray`) → Esfuerzos.

---

#### 12.5. Resultados de elementos Membrane Q6
```python
displacements: np.ndarray = results.get_membrane_q6_displacements(membrane_q6_id)
```

> - `membrane_q6_id` (`int`) → Identificador del elemento Q6.
> - `displacements` (`np.ndarray`) → Desplazamientos.

---

#### 12.6. Resultados de elementos Membrane Q6i
```python
displacements: np.ndarray = results.get_membrane_q6i_displacements(membrane_q6i_id)
```

> - `membrane_q6i_id` (`int`) → Identificador del elemento Q6i.
> - `displacements` (`np.ndarray`) → Desplazamientos.



### Ejemplo de uso
```python
from milcapy import SystemModel, BeamTheoriesType, model_viewer

model = SystemModel()

model.add_material(name="concreto", modulus_elasticity=2.1e6, poisson_ratio=0.2)
model.add_rectangular_section(name="vigas", material_name="concreto", base=0.3, height=0.5)
model.add_rectangular_section(name="muros", material_name="concreto", base=0.3, height=2.0)

model.add_node(1, 0, 0)
model.add_node(2, 0, 5)
model.add_node(3, 7, 8.5)
model.add_node(4, 14, 5)
model.add_node(5, 14, 0)

model.add_member(1, 1, 2, "muros", BeamTheoriesType.TIMOSHENKO)
model.add_member(2, 2, 3, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(3, 3, 4, "vigas", BeamTheoriesType.EULER_BERNOULLI)
model.add_member(4, 4, 5, "muros", BeamTheoriesType.TIMOSHENKO)
model.add_member(5, 2, 4, "vigas", BeamTheoriesType.EULER_BERNOULLI)

model.add_restraint(1, (False, True, True))
model.add_restraint(5, (False, True, True))

model.add_local_axis_for_node(1, -37*3.1416/180)
model.add_local_axis_for_node(5, +37*3.1416/180)

lengthOffset = 1

model.add_elastic_support(3, ky=10)
model.add_end_length_offset(2, la=lengthOffset, qla=True)
model.add_end_length_offset(3, lb=lengthOffset, qlb=True)
model.add_end_length_offset(5, la=lengthOffset, lb=lengthOffset, qla=True)

model.add_releases(5, mi=True, mj=True)

model.add_load_pattern("Live Load")
model.add_point_load(3, "Live Load", 0, -50, 0)
model.add_distributed_load(2, "Live Load", -10, -5)
model.add_distributed_load(3, "Live Load", -5, -10)
model.add_distributed_load(5, "Live Load", -5, -5)
model.add_load_pattern("Dead Load")
model.add_point_load(3, "Dead Load", 40, -50, 10)
model.add_distributed_load(5, "Dead Load", -5, -5)

model.add_prescribed_dof(1, "Live Load", uy=-0.01, CSys="LOCAL")
model.add_prescribed_dof(5, "Live Load", uy=-0.01, CSys="LOCAL")

model.postprocessing_options.n = 100

model.solve()

model_viewer(model)
```

#### model_viewer

##### modelo
<p align="center">
  <img src="docs/assets/modelo.png" alt="modelo" width="800">
</p>

##### deformada
<p align="center">
  <img src="docs/assets/deformada.png" alt="deformada" width="800">
</p>

##### deformada rígida
<p align="center">
  <img src="docs/assets/deformada_rigida.png" alt="deformada rígida" width="800">
</p>

##### reacciones
<p align="center">
  <img src="docs/assets/reacciones.png" alt="reacciones" width="800">
</p>

##### Diagramas de fuerzas axiales
<p align="center">
  <img src="docs/assets/diag_axiales.png" alt="axiales" width="800">
</p>

##### Diagramas de fuerzas cortantes
<p align="center">
  <img src="docs/assets/diag_cortantes.png" alt="cortantes" width="800">
</p>

##### Diagramas de momentos flectores
<p align="center">
  <img src="docs/assets/diag_momentos.png" alt="momentos" width="800">
</p>



<h3 style="text-align:center;">Otros ejemplos</h3>

<p align="center">
  <img src="docs/assets/img/%20(21).png" alt="Resultados" width="700"><br>
  <img src="docs/assets/img/(6).png" alt="Resultados"  width="700"><br>
  <img src="docs/assets/img/(4).png" alt="Resultados"  width="700"><br>
  <img src="docs/assets/img/(2).png" alt="Resultados"  width="700"><br>
  <img src="docs/assets/img/(5).png" alt="Resultados"  width="700"><br>
  <img src="docs/assets/img/(3).png" alt="Resultados"  width="700"><br>
  <img src="docs/assets/img/%20(8).png" alt="Resultados"  width="700"><br>
  <img src="docs/assets/img/(1).png" alt="Resultados"  width="700"><br>
  <img src="docs/assets/img/%20(11).png" alt="Resultados" width="700"><br>
  <img src="docs/assets/img/%20(14).png" alt="Resultados" width="700"><br>
  <img src="docs/assets/img/%20(15).png" alt="Resultados" width="700"><br>
  <img src="docs/assets/img/%20(19).png" alt="Resultados" width="700"><br>
  <img src="docs/assets/img/%20(16).png" alt="Resultados" width="700"><br>
  <img src="docs/assets/img/%20(18).png" alt="Resultados" width="700">
</p>

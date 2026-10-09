# Calibración Contextual Basada en Modelos de Difusión para Química Analítica
## Marco Teórico, Pipeline Metrológico y Aplicaciones en Quimiometría Avanzada

---

## 1. Resumen Ejecutivo y Motivación

En la química analítica contemporánea y en los laboratorios de ensayo regulados (bajo normativas **ISO/IEC 17025**, directrices de la **FDA/EMA** y la **Farmacopea**), la cuantificación multivariante enfrenta dos realidades operativas ineludibles:

1. **Deriva Instrumental Diaria (Day-to-day drift):** La respuesta analítica absoluta de un instrumento (HPLC-DAD, LC-MS/MS, GC-MS, NIR, ICP-MS) varía cada vez que se enciende el equipo debido a envejecimiento de lámparas, ensuciamiento de conos de ionización, deriva térmica, fluctuaciones de flujo y degradación de columnas cromatográficas. Esto exige secuencias diarias con blancos, curvas de calibración y muestras de control de calidad (QC).
2. **Heterogeneidad de Matrices (Client-to-client matrix effects):** Las muestras remitidas por diferentes clientes o fuentes presentan variaciones severas en su composición de fondo (viscosidad, fuerza iónica, lípidos, interferentes coeluyentes), lo que altera las eficiencias de ionización, dispersión de luz y recuperaciones analíticas.

Los modelos tradicionales de quimiometría (como **PLS**, **PCR** o **MCR-ALS**) asumen relaciones rígidas, estacionariedad o linealidad aproximada, y cuando fallan ante un interferente, lo hacen **en silencio**, sesgando el resultado sin alertar al analista. Por su parte, las redes neuronales deterministas (CNNs, MLPs) entregan estimaciones puntuales desprovistas de incertidumbre metrológica y sufren de sobreajuste catastrófico ante matrices no vistas.

Este documento formaliza el marco teórico de **Calibración Contextual por Difusión (In-Context Diffusion Calibration - ICDC)**: un pipeline generativo que no busca reemplazar la secuencia analítica diaria, sino orquestarla para proporcionar **cuantificación adaptativa, detección automática de interferencias y evaluación no paramétrica de la incertidumbre metrológica según la GUM** (*Guide to the Expression of Uncertainty in Measurement*).

---

## 2. Fundamentación Teórica: ¿Por qué Modelos de Difusión?

### 2.1. Limitaciones de GANs y VAEs en Espectrometría y Cromatografía
* **GANs:** El juego minimax adolece de inestabilidad intrínseca y colapso de modo (*mode collapse*). En señales espectrales 1D continuas, las GANs tienden a ignorar picos de analitos a nivel de trazas porque el discriminador los cataloga como ruido insignificante.
* **VAEs:** La optimización del límite inferior de la evidencia (ELBO) fuerza distribuciones gaussianas en el espacio latente, lo que produce señales sobre-suavizadas que redondean o ensanchan los anchos de pico cromatográfico y espectroscópico.

### 2.2. Ventajas Intrínsecas de los Modelos de Difusión (DDPMs / Score-Based Models)
1. **Analogía Física:** El proceso hacia adelante (*forward process*) modela de forma matemáticamente idéntica los procesos físicos de degradación instrumental (adición progresiva de ruido térmico, Johnson-Nyquist, ensanchamiento por difusión molecular).
2. **Preservación de Trazas:** Al optimizar la pérdida de *score matching* en múltiples escalas de ruido, el modelo retiene tanto las envolventes globales de la matriz como los picos discretos de baja intensidad.
3. **Resolución de Problemas Inversos "Zero-Shot" (DPS):** Un modelo incondicional preentrenado permite resolver desconvolución, corrección de línea de base y reconstrucción de picos saturados en tiempo de inferencia mediante *Diffusion Posterior Sampling*, incorporando restricciones físicas sin reentrenar.

---

## 3. Arquitectura del Pipeline: *In-Context Diffusion Calibration (ICDC)*

El flujo de trabajo acopla el modelo generativo a la secuencia diaria del laboratorio analítico a través de cinco etapas modulares:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            SECUENCIA ANALÍTICA DIARIA EN BANCO                             │
│  [Blanco] ──> [Cal 1 ... Cal 6 (del día)] ──> [QCs] ──> [Muestras de Clientes con IS]     │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. EXTRACTOR DE ESTADO DEL DÍA (E_day)                                                      │
│    Codifica los calibradores de la jornada: deriva de t_R, ganancia, ruido basal.           │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. TRIAJE DE MATRIZ Y DETECCIÓN OOD (Energy-Based / Score Norm)                             │
│    ¿La matriz del cliente es químicamente conocida?                                         │
│       ├─ SÍ ──> Proceder a Cuantificación Estándar por Difusión                             │
│       └─ NO ──> Alerta de Interferencia / Protocolo de Fortificación Única (1-Spike)        │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. INFERENCIA ESTOCÁSTICA CONDICIONAL (CARD + DPS)                                          │
│    Muestreo del posterior p(y | X_cliente, E_day, IS)                                       │
│    Guías físicas: Bibliotecas NIST, ventanas de t_R, abundancias isotópicas.                │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. REPORTE METROLÓGICO (Conformidad ISO 17025 / GUM)                                        │
│    • Concentración: Mediana de la densidad a posteriori.                                    │
│    • Incertidumbre Expandida (U, k=2): Percentiles [2.5% - 97.5%].                          │
│    • Espectro diferencial de residuos para auditoría de interferentes.                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1. Fase 0: Modelo Base Fundacional (Offline)
Se entrena previamente sobre datos históricos instrumentales y bases de datos espectrales/cromatográficas:
* **Entrada:** Espectros o cromatogramas crudos $\mathbf{X} \in \mathbb{R}^P$.
* **Objetivo:** Aprender el manifold no lineal de formas de pico válidas, correlaciones entre sobretonos y patrones de fragmentación iónica.

### 3.2. Fase 1: Vector de Contexto Instrumental Diario ($E_{\text{day}}$)
Al iniciar la corrida matutina, los estándares de calibración del día $\{(\mathbf{X}_{\text{cal}, i}, y_{\text{cal}, i})\}_{i=1}^K$ son procesados por un módulo codificador liviano (*Set Transformer* o *DeepSet* invariante a permutaciones):

$$
E_{\text{day}} = \text{Encoder}_{\text{context}}\Big( \{(\mathbf{X}_{\text{cal}, 1}, y_{\text{cal}, 1}), \dots, (\mathbf{X}_{\text{cal}, K}, y_{\text{cal}, K})\} \Big)
$$

Este vector $E_{\text{day}}$ sintetiza la pendiente de respuesta instrumental actual, la deriva de tiempo de retención ($\Delta t_R$) y el piso de ruido térmico de la sesión.

### 3.3. Fase 2: Triaje de Matriz y Detección de Desviaciones (OOD)
Antes de cuantificar, el espectro de la muestra del cliente $\mathbf{X}_{\text{sample}}$ se evalúa contra el campo vectorial de *score* del modelo:

$$
\text{Score de Atipicidad de Matriz} = \big\| \nabla_{\mathbf{X}} \log p(\mathbf{X}_{\text{sample}}) \big\|
$$

* **Zona Verde (Matriz Estándar):** La muestra cae dentro del soporte de densidad calibrado. Se procede a la cuantificación directa.
* **Zona Roja (Matriz Anómala / Efecto de Supresión Severo):** El sistema activa una bandera de advertencia en el LIMS: *"Efecto de matriz atípico detectado en muestra #ID. Riesgo de interferencia coeluyente"*.

### 3.4. Fase 3: Cuantificación por Difusión Condicional (CARD Framework)
Para la cuantificación de la concentración $y \in \mathbb{R}$, se difunde el escalar $y$ hacia ruido gaussiano en $T$ pasos ($y_0 \to y_T \sim \mathcal{N}(0, 1)$).

El proceso de desruidificación inverso es condicionado simultáneamente por tres fuentes:

$$
p_\theta(y_{t-1} \mid y_t, \mathbf{X}_{\text{sample}}, E_{\text{day}}, \mathbf{IS})
$$

Donde $\mathbf{IS}$ representa la señal del estándar interno enriquecido en la muestra.

Al ejecutar $N_{\text{samples}}$ trayectorias estocásticas reversas (ej. $N = 250$), se obtiene un conjunto de réplicas de Monte Carlo que definen la función de densidad a posteriori empírica:

$$
\{y^{(1)}, y^{(2)}, \dots, y^{(250)}\} \sim p(y \mid \mathbf{X}_{\text{sample}}, E_{\text{day}}, \mathbf{IS})
$$

---

## 4. Inyección de Conocimiento Analítico Previo (Domain-Knowledge Guidance)

Durante el proceso inverso de difusión, se puede aplicar **Diffusion Posterior Sampling (DPS)** para obligar al modelo a respetar leyes analíticas estrictas mediante penalizaciones en el gradiente:

$$
\hat{y}_{0, \text{corregido}} = \hat{y}_0 - \lambda \sum_{j} \nabla_{y_t} \mathcal{L}_{\text{química}, j}
$$

### 4.1. Restricción de No Negatividad y Rango Físico

$$
\mathcal{L}_{\text{no-neg}} = \|\min(0, \hat{y}_0)\|^2
$$

### 4.2. Restricción de Biblioteca Espectral (NIST / MassBank)
Si se analiza por GC-MS o LC-MS/MS y se cuenta con el espectro de fragmentación de referencia del compuesto puro $\mathbf{S}_{\text{ref}}$:

$$
\mathcal{L}_{\text{librería}} = 1 - \frac{\hat{\mathbf{S}}_0 \cdot \mathbf{S}_{\text{ref}}}{\|\hat{\mathbf{S}}_0\| \|\mathbf{S}_{\text{ref}}\|}
$$

### 4.3. Consistencia de Relación Isotópica
En espectrometría de masas de alta resolución (HRMS), para una molécula clorada o bromada con relaciones isotópicas teóricas $R_{\text{teórica}} = I_{M+2}/I_M$:

$$
\mathcal{L}_{\text{isótopo}} = \Big( \frac{I_{M+2}(\hat{\mathbf{S}}_0)}{I_M(\hat{\mathbf{S}}_0)} - R_{\text{teórica}} \Big)^2
$$

### 4.4. Contingencia para Matrices Radicalmente Nuevas: Protocolo *1-Spike*
Cuando una muestra presenta una matriz desconocida extrema (ej. extracto de matriz vegetal compleja no calibrada previamente):
1. El analista inyecta la muestra original ($\mathbf{X}_{\text{orig}}$) y una alícuota con una adición estándar conocida ($\mathbf{X}_{\text{spike}} = \mathbf{X}_{\text{orig}} + \Delta C_{\text{adicionada}}$).
2. Ambas señales se evalúan en conjunto forzando la restricción de recuperación:

$$
\mathcal{L}_{\text{spike}} = \big| (\hat{y}_{\text{spike}} - \hat{y}_{\text{orig}}) - \Delta C_{\text{adicionada}} \big|^2
$$

3. El modelo absorbe el factor de supresión específico de esa matriz en una sola iteración, resolviendo la cuantificación exacta sin requerir una curva de adición de patrón de 6 puntos.

---

## 5. Aplicaciones Concretas Ancladas en Datasets Open Source

Para maximizar la viabilidad y reproducibilidad del marco teórico, las aplicaciones propuestas se fundamentan prioritariamente en **repositorios de datos reales abiertos existentes en la comunidad analítica**, complementados con estrategias de simulación sintética físicamente rigurosas.

```
                     MAPA DE DATOS PARA ENTRENAMIENTO Y VALIDACIÓN
                                          │
       ┌──────────────────────────────────┴──────────────────────────────────┐
       ▼                                                                     ▼
 1. DATOS REALES OPEN SOURCE (Prioridad Alta)               2. DATOS SINTÉTICOS FÍSICOS (Benchmarking)
 • Shootout 2002 NIR (Tabletas farma multi-día/equipo)      • Cromatografía: Modelos EMG/HVL + Horwitz
 • Corn NIR Dataset (Cargill - 3 espectrómetros/deriva)     • Espectroscopía: Voigt + Scattering Mie
 • MassIVE / GNPS (LC-MS/MS con supresión y matrices)       • RMN: Simulación spin-Hamiltonian (Spinach)
 • Bacterial Raman (Stanford - 60k espectros clínicos)      • Líneas de base: Procesos estocásticos O-U
```

---

### 5.1. Cuantificación Farmacéutica con Deriva Instrumental y Transferencia (NIR)
* **El Problema Analítico:** Variación inter-día de detectores, fuentes de luz y temperatura ambiente en la cuantificación de principio activo (API).
* **Dataset Real Open Source:**
  * **IDRC Shootout 2002 (Pharmaceutical Tablets):** 654 tabletas farmacéuticas medidas en dos espectrómetros Foss NIRSystems idénticos a lo largo de múltiples jornadas analíticas. Incluye valores de referencia certificados de contenido de analito (mg de API), dureza y peso.
  * **Corn NIR Dataset (Cargill / Eigenvector):** 80 muestras de maíz medidas en 3 espectrómetros diferentes (`m5`, `mp5`, `mp6`) a diferentes temperaturas de celda, con valores de referencia de humedad, aceite, proteína y almidón.
* **Implementación ICDC:** 
  * Los calibradores matutinos de un equipo entrenan el vector de contexto $E_{\text{day}}$.
  * El modelo cuantifica las tabletas en el segundo equipo evaluando si la incertidumbre expandida ($U$) contiene el valor certificado de API a pesar de la deriva instrumental.

### 5.2. Desacople de Supresión Iónica y Efectos de Matriz en LC-MS/MS
* **El Problema Analítico:** Supresión de carga en electrospray (ESI) por sales, fosfolípidos y componentes variables en fluidos biológicos o extractos alimentarios.
* **Datasets Reales Open Source:**
  * **Repositorio MassIVE / GNPS (ej. conjuntos MSV000084682 y cohortes públicas):** Repositorios masivos con archivos `.mzML` crudos de metabolitos y fármacos en orina, plasma humano y aguas residuales con estándares internos estables marcados (SIL-IS) y mezclas *spike-in* con concentraciones conocidas.
  * **MetaboLights (EMBL-EBI - ej. estudios MTBLS con LC-MS cuantitativo):** Datos metabolómicos crudos que incluyen lotes analíticos completos con blancos de solvente, muestras de control de calidad agrupadas (pooled QCs) y secuencias temporales con pérdida de sensibilidad progresiva por ensuciamiento del cono.
* **Implementación ICDC:**
  * El modelo aprende el perfil de fragmentación puro condicionado al $m/z$ precursor y el tiempo de retención ($t_R$).
  * En muestras con alta supresión de ionización, la función de score del modelo detecta la pérdida de señal y corrige la cuantificación ajustando el intervalo de confianza $p(y \mid X, E_{\text{day}}, \text{IS})$.

### 5.3. Detección de Adulteraciones y Fraude Alimentario con Cambios de Matriz (FTIR / Raman)
* **El Problema Analítico:** La matriz base cambia según el origen geográfico o cosecha, y los adulterantes no están presentes en el entrenamiento estándar.
* **Datasets Reales Open Source:**
  * **Open Science Datasets en Zenodo / Mendeley Data (Autenticidad de Aceite de Oliva y Miel):**
    * Espectros FT-IR y NIR de aceites de oliva virgen extra genuinos adulterados deliberadamente con proporciones conocidas (1% al 20%) de aceites de avellana, girasol, colza y orujo.
    * Espectros de miel pura adulterada con jarabes industriales de maíz de alta fructosa (HFCS) y jarabe de arroz.
* **Implementación ICDC:**
  * La difusión incondicional aprende el manifold de "miel genuina".
  * Las muestras adulteradas se detectan mediante la norma del score de atipicidad OOD; la sustracción del espectro contrafactual generado revela de forma limpia la huella dactilar espectral del adulterante específico.

### 5.4. Clasificación y Diagnóstico Clínico Rápido por Espectroscopía Raman
* **Dataset Real Open Source:**
  * **Stanford Bacterial Raman Dataset (Ho et al., Nature Communications):** Más de 60,000 espectros Raman de célula individual de 30 cepas bacterianas patógenas comunes aisladas de pacientes clínicos, medidas en múltiples placas y bajo diferentes condiciones nutricionales de cultivo.
* **Implementación ICDC:** 
  * Eliminación no supervisada de fondo de autofluorescencia mediante *Diffusion Posterior Sampling* (DPS) y reconstrucción robusta de bandas celulares diagnósticas.

---

## 6. Generación de Datos Sintéticos Representativos (Ground Truth Controlado)

Cuando se requiere validar límites de detección teóricos (LOD/LOQ) o evaluar casos extremos de coelución y heterocedasticidad, los datos sintéticos deben respetar estrictamente las leyes físico-químicas de la técnica:

### 6.1. Simulación Cromatográfica (HPLC / GC)
* **Función de Forma de Pico:** Modelo Gaussiano Modificado Exponencialmente (EMG) o modelo de Haarhoff-van der Linde (HVL):

$$
f(t) = \frac{A}{\tau} \frac{1}{\sqrt{2\pi}\sigma} \int_0^\infty \exp\left(-\frac{(t - t_R - t')^2}{2\sigma^2}\right) \exp\left(-\frac{t'}{\tau}\right) dt'
$$

Donde $\tau$ simula el volumen muerto y la asimetría por sobrecarga de columna, y $\sigma$ se deriva de la ecuación de van Deemter ($H = A + B/u + C u$).
* **Deriva de Línea de Base:** Procesos estocásticos de Ornstein-Uhlenbeck o caminatas aleatorias con filtro pasa-bajos, simulando derivas térmicas y gradientes de elución en solventes B.
* **Modelo de Ruido Heterocedástico (Trompeta de Horwitz):**

$$
\sigma_{\text{ruido}}(C) = \sigma_0 + 0.02 \cdot C^{0.8492}
$$

Asegura que el error relativo a concentraciones de trazas refleje la dispersión real observada en inter-laboratorios metrológicos.

### 6.2. Simulación Espectroscópica (NIR / Raman / UV-Vis)
* **Bandas de Absorción/Dispersión:** Suma de perfiles pseudo-Voigt (combinación lineal de Gaussiana Doppler y Lorentziana por tiempo de vida radiativo).
* **Dispersión Física de Radiación:** Simulación de dispersión de Mie dependiente del tamaño medio de partícula mediante corrección EMSC inversa:

$$
X_{\text{simulado}}(\lambda) = a \cdot X_{\text{puro}}(\lambda) + b \cdot \lambda^{-1} + c \cdot \lambda^{-4} + \text{Baseline}
$$

* **Ruido del Detector:** Modelo de ruido Poissoniano (ruido cuántico de disparo / *shot noise*) acoplado a ruido Gaussiano aditivo (ruido de lectura del detector CCD/fotodiodo).

---

## 7. Comparativa Metodológica

| Parámetro / Capacidad | Quimiometría Clásica (PLS / PCR) | Deep Learning Estándar (1D-CNN) | Calibración Contextual por Difusión (ICDC) |
| :--- | :--- | :--- | :--- |
| **Tipo de predicción** | Puntual ($\hat{y}$) | Puntual ($\hat{y}$) | **Distribución completa $p(y \mid X)$** |
| **Comportamiento ante interferentes no vistos** | Falla silenciosa con sesgo en $\hat{y}$ | Alucinación o error impredecible | **Alerta OOD y ensanchamiento del intervalo** |
| **Adaptación a la deriva del día** | Requiere recalibrar el modelo completo | Requiere fine-tuning de pesos | **Condicionamiento en contexto ($E_{\text{day}}$)** |
| **Cumplimiento GUM (Incertidumbre)** | Varianza residual asumida gaussiana | Requiere ensembles costosos | **Empírica, no gaussiana y asimétrica** |
| **Inyección de leyes físicas** | Restricciones de proyección rígidas | Regularización $L_1 / L_2$ | **Guía por gradiente en inferencia (DPS)** |

---

## 8. Plan de Validación Experimental y Hoja de Ruta

Para validar este marco metodológico de forma escalonada priorizando datos abiertos reales:

1. **Fase 1: Validación en Simulación Físico-Química Controlada (Semana 1–4)**
   * Implementación del generador de cromatogramas sintéticos con picos EMG y heterocedasticidad de Horwitz.
   * Demostración de que el modelo CARD predice intervalos asimétricos no negativos cerca del LOD/LOQ.

2. **Fase 2: Benchmark con Datasets Abiertos Reales (Semana 5–12)**
   * **Prueba de Deriva Instrumental y Calibración In-Context:** Aplicación sobre el dataset *IDRC Shootout 2002* (comprimidos farmacéuticos) transfiriendo modelos entre el espectrómetro 1 y el espectrómetro 2 a través de los calibradores diarios.
   * **Prueba de Atipicidad de Matriz y Adulterantes:** Aplicación sobre datasets abiertos de autenticidad de aceites/mieles (Zenodo) para comprobar si el score OOD detecta concentraciones de adulterante $< 5\%$.
   * **Prueba de Desconvolución en Masas:** Uso de trazas crudas de MassIVE para evaluar la resolución de picos solapados guiados por bibliotecas NIST/MassBank.

3. **Fase 3: Validación en Banco Analítico Real (Largo Plazo)**
   * Corrida de validación en equipo local (HPLC-DAD, GC-MS o NIR) siguiendo una guía de validación de métodos bioanalíticos (FDA/ICH Q2(R1)), contrastando la incertidumbre del modelo contra curvas de calibración clásicas ponderadas ($1/x^2$) y adición de patrón.

---

## 9. Bibliografía Seleccionada y Referencias

### Modelos de Difusión y Regresión Estocástica
* **Ho, J., Jain, A., & Abbeel, P.** (2020). *Denoising Diffusion Probabilistic Models*. Advances in Neural Information Processing Systems (NeurIPS), 33, 6840-6851.
* **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B.** (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. International Conference on Learning Representations (ICLR).
* **Han, X., Zheng, H., & Zhou, M.** (2022). *CARD: Classification and Regression Diffusion Models*. Advances in Neural Information Processing Systems (NeurIPS), 35, 18100-18115.
* **Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C.** (2023). *Diffusion Posterior Sampling for General Noisy Inverse Problems*. International Conference on Learning Representations (ICLR).

### Quimiometría, Calibración y Metrología
* **JCGM 100:2008.** *Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM)*. Joint Committee for Guides in Metrology.
* **Brereton, R. G.** (2018). *Chemometrics: Data Driven Extraction for Science*. John Wiley & Sons.
* **Rinnan, Å., van den Berg, F., & Engelsen, S. B.** (2009). *Review of the most common pre-processing techniques for near-infrared spectra*. TrAC Trends in Analytical Chemistry, 28(10), 1201-1210.
* **De Biasi, V., et al.** (2007). *Understanding and suppressing matrix effects in liquid chromatography-mass spectrometry*. TrAC Trends in Analytical Chemistry, 26(8), 766-774.
* **Tauler, R.** (1995). *Multivariate curve resolution applied to second order data*. Chemometrics and Intelligent Laboratory Systems, 30(1), 133-146.

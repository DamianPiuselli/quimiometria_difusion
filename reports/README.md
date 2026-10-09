# Repositorio de Reportes Experimentales: Framework ICDC
## *In-Context Diffusion Calibration for Chemometrics*

---

## 1. Estructura y Organización de los Experimentos

Cada experimento cuenta con su propio directorio autónomo, que aloja el reporte técnico detallado (`REPORT.md`) y sus figuras diagnósticas de alta resolución asociadas en `figures/`:

```text
reports/
├── README.md                                # Este documento (índice maestro y resumen ejecutivo)
├── exp01_synthetic_sandbox/
│   ├── REPORT.md                            # Sandbox físico-químico (EMG, Horwitz, drift basal)
│   └── figures/
│       ├── calibration_curves_comparison.png
│       └── uncertainty_vs_concentration.png
├── exp02_corn_nir_transfer/
│   ├── REPORT.md                            # Benchmark Corn NIR (M5 -> MP5) y crítica de validez ecológica
│   └── figures/
│       ├── corn_spectra_drift_comparison.png
│       └── real_corn_transfer_benchmark.png
└── exp03_trace_metrology_and_deconvolution/
    ├── REPORT.md                            # Trazas (LOD/LOQ, CCα/CCβ) y desconvolución ciega DPS
    └── figures/
        ├── trace_metrology_limits.png
        └── dps_blind_deconvolution.png
```

---

## 2. Índice de Experimentos y Resumen de Hallazgos

### [Experimento 01: Sandbox de Simulación Físico-Química](exp01_synthetic_sandbox/REPORT.md)
* **Objetivo:** Comparar `PLSBaseline`, `CNN1DBaseline` e `ICDC Difusión` bajo condiciones analíticas controladas (picos cromatográficos EMG, ruido heterocedástico de Horwitz, deriva de línea de base).
* **Hallazgos Clave:**
  * **La Paradoja de SNV:** Aplicar normalización SNV en cromatografía de cero basal divide por la desviación estándar del propio pico analítico ($s_X \approx k \cdot C$), cancelando la concentración y colapsando el $R^2$ de PLS a $0.12$. Al desactivar SNV, PLS recupera $R^2 = 0.9953$.
  * **Violación de Cotas Físicas en Modelos Lineales:** Tanto PLS como la 1D-CNN asumen residuos simétricos gaussianos, prediciendo límites inferiores negativos ($y_{\text{low}} < 0$) en el **36.3%** y **48.8%** de las muestras respectivamente.
  * **Fidelidad Física en Difusión:** ICDC mantiene el **100% de predicciones en el dominio permitido ($y \ge 0$)** con intervalos adaptativos a la concentración.

---

### [Experimento 02: Benchmark Real Corn NIR y Crítica de Validez Ecológica](exp02_corn_nir_transfer/REPORT.md)
* **Objetivo:** Evaluar la transferencia de calibraciones de humedad en granos de maíz entre dos espectrómetros NIR comerciales (FOSS M5 $\to$ MP5) mediante el vector de contexto $E_{\text{day}}$.
* **Resultados:** ICDC reduce el RMSEP en un 85% frente a PLS directo no preprocesado ($0.2355$ vs $1.5995$, $R^2 = 0.6349$).
* **Crítica Epistemológica (El Pivot Estratégico):**
  * La quimiometría clásica bien aplicada (derivada segunda de **Savitzky-Golay** + **Corrección de Pendiente y Sesgo - SBC** con 10 patrones) alcanza $\text{RMSEP} = 0.2006$ y $R^2 = 0.7353$.
  * Ningún laboratorio de ensayo bajo **ISO/IEC 17025** adoptará un modelo generativo estocástico para un problema que la quimiometría estándar resuelve con simple recalibración o SBC.
  * Este resultado motivó el pivot hacia los nichos donde la difusión aporta un valor insustituible: **Metrología de Trazas cerca del LOD** y **Desconvolución Ciega de Adulterantes No Modelados**.

---

### [Experimento 03: Metrología de Trazas (LOD/LOQ) y Desconvolución Ciega (DPS)](exp03_trace_metrology_and_deconvolution/REPORT.md)
* **Objetivo:** Demostrar formalmente las ventajas de los modelos de difusión en:
  1. Cuantificación de incertidumbre en niveles de traza conforme a **ISO 11843** y **2002/657/CE**.
  2. Detección y desconvolución *zero-shot* de interferentes no modelados mediante *Diffusion Posterior Sampling* (DPS).
* **Hallazgos Clave:**
  * **Fallo Masivo de PLS en Trazas:** Al evaluar blancos y muestras sub-ppm, el **83.0%** de los intervalos generados por PLS (incluso con pretratamiento Savitzky-Golay) violan la física prediciendo concentraciones negativas ($[-0.10, +0.15]\ \text{mg/kg}$).
  * **No Negatividad Estricta en Difusión:** ICDC garantiza **0.0% de violaciones negativas**, entregando intervalos 27% más nítidos ($\text{MPIW} = 0.1857$ vs $0.2544$) y calculando de forma no paramétrica el Límite de Decisión ($\text{CC}\alpha = 0.2075$) y Capacidad de Detección ($\text{CC}\beta = 0.3246$).
  * **Fallo Silencioso de PLS ante Adulterantes:** Un interferente coeluyente no modelado introduce un sesgo sistemático masivo ($\text{Bias} = +0.8554\ \text{mg/kg}$) sin que PLS alerte al analista.
  * **Triaje OOD y Desconvolución por Norma de Score:** El score $\|\nabla_X \log p_t(X)\|_2$ separa limpiamente muestras limpias ($1.12$) de contaminadas ($1.54$), logrando una **tasa de detección del 100.0%** y aislando el perfil puro del interferente.

---

## 3. Tabla Resumen Consolidada

| Experimento | Foco Científico | Modelo Baseline (PLS / CNN) | ICDC Generativo (Difusión) | Veredicto Técnico |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 01** | Sandbox Simulado (EMG + Horwitz) | RMSEP: $0.1612$ \| Límites neg.: **36.3%** | RMSEP: $0.2912$ \| Límites neg.: **0.0%** | Difusión respeta la física de no negatividad y la heterocedasticidad real. |
| **Exp 02** | Corn NIR Transfer (M5 $\to$ MP5) | PLS Crudo: $1.5995$ \| SavGol+SBC: **$0.2006$** | ICDC Contextual: $0.2355$ ($R^2 = 0.63$) | Transferencia rutinaria no tiene validez ecológica para IA compleja. Pivot a Pilares B y C. |
| **Exp 03** | Trazas (LOD) & Desconvolución DPS | Tasa neg.: **83.0%** \| Sesgo adulterante: **+0.855** | Tasa neg.: **0.0%** \| Detección adulterante: **100.0%** | **Ventaja cualitativa contundente de la difusión en metrología ISO y desconvolución.** |

---

## 4. Ejecución y Reproducibilidad

Todos los scripts se gestionan mediante `uv`:

```bash
# Experimento 01
uv run python experiments/01_synthetic_sandbox.py

# Experimento 02
uv run python experiments/02_real_benchmark.py

# Experimento 03
uv run python experiments/03_trace_metrology_and_deconvolution.py
```

# AGENTS.md — Directrices Operativas del Repositorio

Este documento define los estándares operativos, de herramientas y de formato para cualquier agente o desarrollador que trabaje en el proyecto **`quimiometria_difusion`** (*In-Context Diffusion Calibration for Chemometrics*). Está pensado para mantenerse conciso en esta etapa inicial y expandirse progresivamente a medida que el proyecto incorpore código, datos y experimentos.

---

## 1. Gestión de Entornos y Dependencias

### 1.1. Python y Entornos Virtuales: `uv`
Toda la gestión de paquetes y entornos de Python se realiza con **[`uv`](https://github.com/astral-sh/uv)**:
* **Crear / refrescar el entorno virtual:**
  ```bash
  uv venv
  ```
* **Instalación de paquetes:**
  ```bash
  uv pip install <paquete>
  # O si se usa pyproject.toml:
  uv add <paquete>
  ```
* **Ejecución de scripts:**
  ```bash
  uv run python <ruta/script.py>
  ```
* **Regla:** No utilizar `pip` directo ni `conda`/`virtualenv` manual. Mantener siempre el `.venv` gestionado por `uv`.

### 1.2. Herramientas y Runtimes de Sistema: `mise`
Para cualquier dependencia que no sea un paquete de Python puro (ej. herramientas de CLI como `gh`, `marp-cli`, `quarto`, `pandoc` o runtimes adicionales):
* Se gestionan mediante **[`mise`](https://mise.jdx.dev/)** a través del archivo `mise.toml` en la raíz del repositorio.
* No instalar dependencias de sistema globales de manera silenciosa si pueden ser gestionadas por `mise`.

---

## 2. Estándares de Documentación y Renderizado de Markdown

El usuario y los colaboradores visualizan y presentan la documentación en múltiples plataformas:
* **Glow:** Visualizador CLI de terminal.
* **MarkText:** Editor visual de escritorio (WYSIWYG) con motor KaTeX/MathJax.
* **Marp:** Motor de presentaciones y diapositivas técnicas con KaTeX.
* **GitHub:** Renderizado web nativo.

Para asegurar un renderizado idéntico y sin roturas en todos estos entornos, se deben seguir estrictamente estas **reglas de compatibilidad cruzada**:

### 2.1. Fórmulas Matemáticas en Bloque (*Display Math*)
1. Los delimitadores `$$` deben ubicarse **en su propia línea**, tanto al inicio como al final.
2. Debe dejarse **una línea en blanco** antes y después del bloque matemático.
3. *Ejemplo correcto:*
   ```markdown
   Texto anterior explicativo.

   $$
   f(t) = \frac{A}{\tau \sqrt{2\pi}\sigma} \int_0^\infty \exp\left(-\frac{(t - t_R - t')^2}{2\sigma^2}\right) \exp\left(-\frac{t'}{\tau}\right) dt'
   $$

   Texto posterior continuo.
   ```
4. *Qué evitar:*
   * No escribir `$$ ecuacion $$` en una sola línea para ecuaciones complejas (Marp y Glow a menudo no lo procesan correctamente).
   * No mezclar etiquetas HTML dentro de los bloques de ecuaciones.

### 2.2. Fórmulas Matemáticas en Línea (*Inline Math*)
1. Usar un solo signo de dólar `$formula$`.
2. **Sin espacios en blanco internos:** Escribir `$E = mc^2$` y **NUNCA** `$ E = mc^2 $` (MarkText y GitHub fallan al detectar fórmulas con espacios tras el primer `$`).
3. Para texto, subíndices descriptivos o unidades dentro de fórmulas, usar siempre `\text{...}`:
   `$t_{\text{max}}$`, `$\text{mg/kg}$`, `$E_{\text{day}}$`.

### 2.3. Formato General de Texto y Tablas
* **Tablas Markdown:** Limitar el ancho de columnas y evitar texto excesivamente largo en celdas individuales para permitir una lectura limpia en **Glow** sobre terminales de 80-120 columnas.
* **Diapositivas Marp:** Los archivos de presentación deben ubicarse en su propio subdirectorio (ej. `slides/`) con el frontmatter correspondiente (`marp: true`, `theme: ...`) para no interferir con la lectura de documentos estándar.

---

## 3. Estructura Inicial del Repositorio

```text
quimiometria_difusion/
├── README.md          # Marco teórico, formulación matemática y diseño del pipeline
├── AGENTS.md          # Este documento (directrices y reglas para agentes y devs)
├── .gitignore         # Exclusión de .venv, cachés, pesos de modelos y datos crudos
└── (futuras carpetas: src/, data/, experiments/, slides/)
```

---

## 4. Control de Versiones (Git y GitHub)
* **Rama principal:** `main`.
* **Mensajes de commit:** Claros, en presente imperativo o descriptivo en español o inglés técnico (ej. `docs: inicializar marco teórico y directrices AGENTS.md`).
* **Repositorio público:** Gestionado mediante el CLI de GitHub (`gh`) bajo la cuenta del usuario.

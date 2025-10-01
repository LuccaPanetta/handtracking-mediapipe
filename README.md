# Mediapipe Handtracking Plus

Sistema simple de captura, entrenamiento y predicción en tiempo real de gestos de mano usando MediaPipe Hands, OpenCV y scikit-learn.

Características
- Recolección interactiva de muestras y entrenamiento en un mismo flujo.
- Gestos configurables vía config/gestures.yaml (incluye Open, Fist, Peace, ThumbsUp, Punk y Circle).
- Caché de dataset (cache/dataset.npz) para iterar rápido.
- Optimizado para Windows: backend de cámara seleccionable (DirectShow/MSMF), MJPG, FPS objetivo y buffer pequeño para menor latencia.

Requisitos
- Python 3.10+ (recomendado) y una cámara web.
- Windows (probado).

Instalación rápida (Windows PowerShell)
```powershell
# 1) Crear entorno virtual e instalar dependencias
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt

# 2) Ejecutar el flujo interactivo (recolecta -> entrena -> predice)
.\.venv\Scripts\python.exe -m mhp.cli --backend dshow
```

Ejecutar desde Visual Studio Code
- Este repo incluye .vscode/ con perfiles listos.
- Abrí la carpeta del proyecto en VS Code.
- Panel Run and Debug (Ctrl+Shift+D) y elegí:
  - "Run: Handtracking (mhp.cli)" (usa --backend dshow por defecto).
  - "Run: Handtracking (fast, no draw)" para mayor FPS.
- La primera vez se ejecuta la tarea "Setup venv and install deps" que crea .venv e instala dependencias.

Controles durante la recolección
- n: pasa al siguiente gesto configurado (verás el nombre en pantalla).
- q: termina recolección y entrena el modelo.
- ESC: salir.
- Opcional (modo manual): con --manual, presioná c para guardar una muestra a demanda.



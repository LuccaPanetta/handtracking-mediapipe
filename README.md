# Mediapipe Handtracking Plus

Sistema simple de captura, entrenamiento y predicción en tiempo real de gestos de mano usando MediaPipe Hands, OpenCV y scikit-learn.

Características
- Recolección interactiva de muestras y entrenamiento en un mismo flujo.
- Gestos configurables vía config/gestures.yaml (incluye Open, Fist, Peace, ThumbsUp, Punk y Circle).
- Caché de dataset (cache/dataset.npz) para iterar rápido.
- Optimizado para Windows: backend de cámara seleccionable (DirectShow/MSMF), MJPG, FPS objetivo y buffer pequeño para menor latencia.

Requisitos
- Python 3.10+ (recomendado) y una cámara web.
- Windows (probado), debería funcionar en otros SO ajustando backend.

Instalación rápida (Windows PowerShell)
```powershell
# 1) Crear entorno virtual e instalar dependencias
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt

# 2) Ejecutar el flujo interactivo (recolecta -> entrena -> predice)
.\.venv\Scripts\python.exe -m mhp.cli --backend dshow
```
Nota: no es necesario ejecutar Activate.ps1 (evita problemas de ExecutionPolicy). Si preferís activarlo, podés habilitarlo solo para esta sesión:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
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

Uso por CLI (opciones principales)
```powershell
# Backend de cámara: dshow (Windows) o msfm
.\.venv\Scripts\python.exe -m mhp.cli --backend dshow

# Modo rápido: menos frames procesados y sin dibujar landmarks
.\.venv\Scripts\python.exe -m mhp.cli --backend dshow --process-every-n 3 --no-draw

# Modo manual y umbral más estricto
.\.venv\Scripts\python.exe -m mhp.cli --backend dshow --manual --movement-threshold 0.008

# Baja resolución (más FPS)
.\.venv\Scripts\python.exe -m mhp.cli --backend dshow --width 320 --height 240
```

Configuración de gestos
- Editá config/gestures.yaml y agregá/quità gestos debajo de la clave gestures.
- Luego recolectá nuevas muestras y reentrená.

Datos y limpieza
- Dataset guardado en cache/dataset.npz (X: landmarks 21x3 -> 63 features, y: labels).
- Para reiniciar desde cero:
```powershell
if (Test-Path cache\dataset.npz) { Remove-Item -Force cache\dataset.npz }
```

Benchmark
```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py
```

Arquitectura (alto nivel)
- src/mhp/capture.py: VideoCaptureThread lee cámara en un hilo con buffer corto.
- src/mhp/detector.py: envoltorio de MediaPipe Hands; devuelve vector de 63 floats por mano.
- src/mhp/cache.py: guarda/carga dataset comprimido (npz) incrementalmente.
- src/mhp/gestures.py: carga los gestos desde YAML con defaults si falta.
- src/mhp/cli.py: bucle interactivo de recolección, entrenamiento (RandomForest) y predicción en tiempo real.

Tips de performance
- Backends Windows: probá --backend dshow (suele iniciar más rápido) o --backend msfm.
- Procesar menos frames: --process-every-n 3 o 4.
- Desactivar dibujo: --no-draw.
- Baja resolución: --width 320 --height 240.
- Silenciar warnings de MediaPipe (opcional): añadí GLOG_minloglevel=2 a .vscode/.env.

Solución de problemas
- ExecutionPolicy bloquea Activate.ps1: ejecutá sin activar (usando .venv\Scripts\python.exe) o usa:
  - Solo sesión actual: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  - Usuario actual: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
- Pantalla negra o tarda en iniciar: cambiá backend (dshow/msfm), bajá resolución y/o usa --no-draw.
- Cámara ocupada: cerrá apps que la usen (Zoom, Teams, etc.).

Licencia
- Agregá la licencia que corresponda a tu proyecto (por ejemplo, MIT).

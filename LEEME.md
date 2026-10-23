# Benchmark de LLMs locales: 1 o 2x RTX 3090 + Ryzen X3D

Este kit mide la velocidad real de varios modelos en tu PC y genera gráficos comparativos en español.
Usa la herramienta del post de r/LocalLLaMA, [**llama-benchy**](https://github.com/eugr/llama-benchy),
contra **llama.cpp** (`llama-server`).

| Archivo | Para qué sirve |
|---|---|
| `modelos.txt` | Lista de modelos a probar (editable). |
| `bench_linux.sh` | Corre el benchmark en Linux. |
| `bench_windows.ps1` / `bench_windows.bat` | Corre el benchmark en Windows. |
| `generar_graficos.py` | Genera los gráficos, los CSV y `informe.html`. |

## 1. Requisitos

1. **Driver NVIDIA** reciente (`nvidia-smi` tiene que funcionar).
2. **llama.cpp con CUDA**
   - Windows: bajá `llama-*-bin-win-cuda-*.zip` (y el zip `cudart`) desde
     [releases](https://github.com/ggml-org/llama.cpp/releases), descomprimí ambos en la misma carpeta y agregala al PATH.
     También podés pasar la ruta con `-Servidor C:\ruta\llama-server.exe`.
   - Linux: compilá con `cmake -B build -DGGML_CUDA=ON && cmake --build build -j`.
     Después agregá `build/bin` al PATH o pasá la ruta con `--servidor`.
3. **uv** (incluye `uvx`, que ejecuta llama-benchy sin instalarlo):
   - Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## 2. Ejecutar

Linux:
```bash
chmod +x bench_linux.sh
./bench_linux.sh --etiqueta 7800X3D-32GB
```

Windows:
```bat
bench_windows.bat -Etiqueta 9950X3D-64GB
```

Opciones útiles (en Windows tienen el mismo nombre con `-` y mayúscula inicial, por ejemplo `-Depth`):

| Opción | Defecto | Qué hace |
|---|---|---|
| `--etiqueta` | se arma solo, ej. `7800X3D-64GB` | Nombre del equipo. **Usá una etiqueta distinta por PC o configuración** para compararlas. |
| `--solo a,b` | todos | Prueba solo esos modelos. |
| `--pp` | `512 2048` | Tamaño del prompt en tokens. |
| `--tg` | `128` | Cantidad de tokens a generar. |
| `--depth` | `0 4096 16384` | Contexto previo ya cargado (simula conversaciones largas). |
| `--runs` | `3` | Repeticiones; el informe muestra media ± desviación. |
| `--concurrencia` | `1` | Por ejemplo `"1 2 4"` mide varios usuarios simultáneos. |
| `--gpus` | autodetectar | Índices de GPU, por ejemplo `"0,1"`. |
| `--modos-gpu` | `1gpu` (con 2 GPUs: `"1gpu 2gpu-layer 2gpu-tensor"`) | Modos de GPU a probar (ver sección 5). |
| `--nvlink` | `auto` | Forzar `si` o `no` si la detección de NVLink falla. |

La primera vez cada modelo se descarga de Hugging Face. Son unos 250 GB si probás la lista completa.
Los archivos quedan en la caché de llama.cpp, así que las corridas siguientes no vuelven a descargarlos.
El *tiempo de carga* de esa primera corrida incluye la descarga.

## 3. Generar gráficos

```bash
uv run --with matplotlib generar_graficos.py
```

Comandos con la referencia elegida:
- `uv run --with matplotlib generar_graficos.py --base 7800X3D-32GB` fija el equipo de referencia para las comparativas.
- `uv run --with matplotlib generar_graficos.py --etiquetas A,B` filtra y ordena los equipos.

Todo se guarda en `resultados/informe/`:

| Gráfico | Qué muestra |
|---|---|
| 01 Velocidad de generación | Tokens/s al escribir la respuesta. Es la métrica principal. |
| 02 Velocidad de prompt | Tokens/s al leer la entrada (prefill). |
| 03 Generación con contexto largo | La misma velocidad con 16K tokens previos. |
| 04 / 05 Mapas de calor | Generación y tiempo hasta el primer token según el contexto. |
| 06 Degradación | Porcentaje de la velocidad que conserva cada modelo al llenarse el contexto. |
| 07 VRAM | Uso de VRAM frente al límite de 24 GB. |
| 08 Eficiencia | Tokens por julio, según el consumo de la GPU. |
| 09 Tiempo de carga | Lo que tarda el servidor en quedar listo. |
| 10 VRAM vs. velocidad | Ubica a los modelos rápidos y livianos. |
| 11 Concurrencia | Rendimiento total con varios usuarios (si lo mediste). |
| 12 / 13 Comparativa | Diferencia porcentual de cada configuración frente a la base (equipo y/o modo de GPU). |
| 14 Efecto de NVLink | Mismo equipo y mismo modo de GPU, con NVLink frente a sin NVLink (aparece si hay ambas corridas). |

También se generan estos archivos:
- `resumen.csv`: datos estándar.
- `resumen_excel.csv`: abre directo en Excel en español.
- `informe.html`: tablas y todos los gráficos.

## 4. Sobre los modelos y el hardware

- **Con el modelo entero en la VRAM**, la CPU y la RAM casi no influyen. Un 7800X3D y un 9950X3D van a dar casi lo mismo.
- **Los modelos híbridos** (`qwen3-30b-a3b-q8`, `gpt-oss-120b`, `glm-4.5-air-q4`) usan `--n-cpu-moe` y dejan parte de los expertos MoE en la CPU y la RAM DDR5.
  Ahí se ven las diferencias de CPU, de cantidad de núcleos y de ancho de banda de RAM.
- `gpt-oss-120b` y `glm-4.5-air-q4` necesitan **64 GB de RAM**. Con 32 GB se saltan automáticamente.
- Si un modelo falla por falta de memoria (`out of memory` en `resultados/.../logs/*.server.log`), probá alguna de estas opciones:
  - bajar el `contexto` en `modelos.txt`;
  - agregar `-ctk q8_0 -ctv q8_0`;
  - subir `--n-cpu-moe`.
- Para comparar justo, cerrá juegos, navegadores con video y otras apps que usen la GPU.

## 5. Dos RTX 3090, con y sin NVLink

Con 2 GPUs, el script prueba cada modelo en varios **modos de GPU**:

| Modo | Flags de llama.cpp | Qué mide |
|---|---|---|
| `1gpu` | solo la primera GPU | Referencia. |
| `2gpu-layer` | `-sm layer -ts 1,1` | Reparto por capas: las GPUs trabajan por turnos. Pasa poco tráfico entre ellas, así que NVLink casi no influye. Sirve para que entren modelos de hasta ~45 GB. |
| `2gpu-tensor` | `-sm tensor -ts 1,1 -fa on` | Paralelismo de tensores (**experimental** en llama.cpp): ambas GPUs calculan cada capa a la vez. Pasa mucho tráfico entre ellas: **aquí se nota NVLink**. |
| `2gpu-row` | `-sm row -ts 1,1` | Obsoleto, opcional (`--modos-gpu "... 2gpu-row"`). |

Puntos importantes:
- En los modos de 2 GPUs se exporta `GGML_CUDA_P2P=1`, que permite copias directas entre GPUs.
  Si ves cuelgues o texto basura, usá `--sin-p2p` (Windows: `-SinP2P`).
- El modo tensor no admite caché KV cuantizada (el script quita `-ctk`/`-ctv` automáticamente) ni muchos modelos MoE.
  Esos casos quedan registrados como error y el script sigue con el resto.
- En `modelos.txt` hay modelos que **solo** entran en 48 GB (`gpus_min = 2`): Llama 3.3 70B, Qwen3 32B Q8, Gemma 3 27B Q8 y GLM-4.5-Air.
  La columna `args_2gpu` deja usar otros flags en 2 GPUs, por ejemplo menos `--n-cpu-moe`.
- Para que llama.cpp aproveche NVLink en Linux conviene compilarlo con NCCL:
  `cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON`.

### Medir con y sin NVLink

El estado de NVLink se detecta solo con `nvidia-smi topo -m` y `nvidia-smi nvlink -s`. Queda en la etiqueta (`...-NVLink` o `...-PCIe`) y en `sistema.json`.
No se puede apagar desde el script, así que hay que hacer **dos corridas**:

1. **Con NVLink** (puente puesto):
   ```bash
   ./bench_linux.sh --modos-gpu "1gpu 2gpu-layer 2gpu-tensor"
   ```
2. **Sin NVLink**, desactivándolo de alguna de estas formas y volviendo a correr el mismo comando:
   - **Quitar el puente NVLink** con la PC apagada. Es lo más seguro y funciona en Linux y Windows.
   - **Linux sin tocar el hardware:** crear `/etc/modprobe.d/nvlink-off.conf` con `options nvidia NVreg_NvLinkDisable=1`, regenerar el initramfs (`sudo update-initramfs -u` en Ubuntu/Debian) y reiniciar.
     Para volver a activarlo, borrá ese archivo, regenerá el initramfs y reiniciá.
   - **Windows:** en muchas GeForce el NVLink solo funciona con SLI activado en *Panel de control de NVIDIA → Configurar SLI*. Desactivar SLI suele equivaler a "sin NVLink".
     Confirmalo con `nvidia-smi nvlink -s`.
3. Verificá que el script muestre `NVLink=no` en el encabezado. Si la detección falla, forzalo con `--nvlink no` (Windows: `-NvLink no`).
4. Generá los gráficos con las dos corridas en `resultados/`. El gráfico **14 Efecto de NVLink** compara modo por modo.
   El modo `1 GPU` funciona como control: debería dar cerca de 0 %.

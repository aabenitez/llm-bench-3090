# LLM Bench 3090: benchmark de LLMs locales en RTX 3090 (1 o 2 GPUs)

Kit para medir la velocidad real de modelos de lenguaje que corren en tu propia PC con **llama.cpp**.
Guarda los resultados y genera **gráficos analíticos y comparativos en español**.
Está pensado para equipos con una o dos **NVIDIA RTX 3090 (24 GB)**, con o sin **NVLink**, y procesadores **AMD Ryzen X3D**.

> **Autoría:** proyecto realizado por **Alcides A. Benitez** junto con **Claude Pro (Opus 5.5, esfuerzo medio)** de Anthropic.
> **Fecha:** 23 de octubre de 2026.

---

## Objetivo

Responder con datos medidos, no con estimaciones, preguntas como:

- ¿Qué modelos entran en 24 GB (o 48 GB) de VRAM y a qué velocidad responden?
- ¿Cuánto se frena cada modelo cuando la conversación acumula contexto (4K, 16K tokens)?
- ¿Conviene más un modelo grande cuantizado o uno chico con más precisión?
- ¿Cuánto influye la CPU (7800X3D frente a 9950X3D) y la RAM DDR5 en los modelos que no entran del todo en la GPU?
- ¿Cuánto se gana con una segunda RTX 3090? ¿Y cuánto aporta el puente **NVLink**?

## Base del proyecto

- Toma como punto de partida el post de r/LocalLLaMA
  [*How to easily benchmark your models with…*](https://www.reddit.com/r/LocalLLaMA/comments/1qp8sov/how_to_easily_benchmark_your_models_with/).
  Ese post presenta la herramienta [**llama-benchy**](https://github.com/eugr/llama-benchy), que mide al estilo `llama-bench` cualquier servidor compatible con OpenAI.
- El servidor de inferencia es [**llama.cpp**](https://github.com/ggml-org/llama.cpp) (`llama-server`). Descarga los modelos GGUF directamente de Hugging Face.
- La configuración de varias GPUs (`-sm layer`, `-sm tensor`, `GGML_CUDA_P2P`) sigue la
  [guía oficial de multi-GPU de llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md).

## Compatibilidad

| Componente | Soportado |
|---|---|
| **Sistema operativo** | Linux (bash) y Windows 10/11 (PowerShell 5.1 o 7) |
| **GPU** | 1 o 2x NVIDIA RTX 3090 24 GB, con o sin NVLink. Funciona con otras GPUs NVIDIA ajustando la lista de modelos. |
| **CPU** | Pensado para AMD Ryzen 7 7800X3D (8 núcleos) y Ryzen 9 9950X3D (16 núcleos); sirve cualquier CPU x86-64 |
| **RAM** | 32 o 64 GB DDR5. Los modelos que requieren más RAM se saltan automáticamente. |
| **Requisitos de software** | Driver NVIDIA (`nvidia-smi`), llama.cpp compilado con CUDA, [uv](https://docs.astral.sh/uv/) (`uvx`) y `curl` (en Linux) |
| **Gráficos** | Python 3 con matplotlib (`uv run --with matplotlib ...`) |

## Contenido

| Archivo | Descripción |
|---|---|
| `bench_linux.sh` | Script de benchmark para Linux |
| `bench_windows.ps1` / `bench_windows.bat` | Script de benchmark para Windows. El `.bat` evita tener que cambiar la ExecutionPolicy. |
| `modelos.txt` | Lista editable de modelos, contexto, RAM mínima y flags por modelo |
| `generar_graficos.py` | Genera los gráficos, los CSV y el informe HTML |
| `LEEME.md` | Guía detallada de instalación, uso, opciones y NVLink |

## Uso rápido

Linux:
```bash
./bench_linux.sh --etiqueta 7800X3D-64GB
```

Windows:
```bat
bench_windows.bat -Etiqueta 9950X3D-64GB
```

Gráficos:
```bash
uv run --with matplotlib generar_graficos.py
```

Con dos GPUs, el script prueba los modos `1gpu`, `2gpu-layer` y `2gpu-tensor` y detecta NVLink automáticamente.
Para comparar con y sin NVLink hay que hacer dos corridas; el paso a paso está en la sección 5 de [LEEME.md](LEEME.md).

## Qué mide y qué genera

Cada modelo se ejecuta con llama-server y se mide con llama-benchy, con 3 repeticiones por defecto (media ± desviación):

- **Generación (tg, tokens/s):** la velocidad a la que el modelo escribe.
- **Lectura del prompt (pp, tokens/s):** la velocidad de procesamiento de la entrada.
- **Tiempo hasta el primer token (TTFT).**
- Cada medición se repite con **0, 4K y 16K tokens de contexto previo**, y opcionalmente con varios usuarios simultáneos.
- Durante la prueba, `nvidia-smi` registra la **VRAM, el uso, el consumo y la temperatura** de cada GPU, junto con el **tiempo de carga** del modelo.
- También se registran los datos del equipo: CPU, núcleos, RAM, GPUs, driver, topología NVLink/PCIe y versión de llama.cpp.

Resultados en `resultados/<fecha>_<etiqueta>/` y, al generar los gráficos, en `resultados/informe/`:

| Salida | Contenido |
|---|---|
| `informe.html` | Informe en español con tabla de equipos, resumen por modelo y todos los gráficos |
| `resumen.csv` / `resumen_excel.csv` | Datos consolidados; la segunda versión usa `;` y coma decimal para Excel en español |
| 01–03 | Velocidad de generación y de prompt, sin contexto y con contexto largo |
| 04–05 | Mapas de calor de generación y de TTFT según el contexto, por configuración |
| 06 | Degradación: qué porcentaje de su velocidad conserva cada modelo al llenarse el contexto |
| 07 | VRAM utilizada frente a los límites de 24 GB y 48 GB |
| 08 | Eficiencia energética (tokens por julio) |
| 09 | Tiempo de carga |
| 10 | Dispersión VRAM vs. velocidad |
| 11 | Concurrencia (si se midió con varios usuarios) |
| 12–13 | Comparativa porcentual de cada configuración frente a la base |
| 14 | **Efecto de NVLink**, modo por modo (si hay corridas con y sin NVLink) |

## Modelos soportados (lista por defecto)

Todos son GGUF que llama.cpp descarga de Hugging Face. Podés agregar, quitar o cambiar la cuantización en `modelos.txt`.

| Modelo | Cuantización | VRAM aprox. | Requisito |
|---|---|---|---|
| Qwen3 8B | Q4_K_M y Q8_0 | 6–10 GB | 1 GPU |
| Llama 3.1 8B Instruct | Q4_K_M | ~6 GB | 1 GPU |
| Qwen3 14B | Q4_K_M | ~10 GB | 1 GPU |
| gpt-oss-20b | MXFP4 (nativa) | ~13 GB | 1 GPU |
| Mistral Small 3.2 24B Instruct | Q4_K_M | ~16 GB | 1 GPU |
| Gemma 3 27B IT | Q4_K_M | ~20 GB | 1 GPU (caché KV q8_0) |
| Qwen3 32B | Q4_K_M | ~23 GB | 1 GPU (caché KV q8_0) |
| Qwen3 30B-A3B Instruct 2507 (MoE) | Q4_K_M | ~20 GB | 1 GPU |
| Qwen3-Coder 30B-A3B Instruct (MoE) | Q4_K_M | ~20 GB | 1 GPU |
| Qwen3 30B-A3B Instruct 2507 (MoE) | Q8_0 | ~33 GB | Híbrido GPU+CPU con 1 GPU; completo en 2 GPUs |
| gpt-oss-120b (MoE) | MXFP4 (nativa) | ~65 GB | Híbrido GPU+CPU, 64 GB de RAM |
| Llama 3.3 70B Instruct | IQ4_XS | ~38 GB | 2 GPUs |
| Qwen3 32B | Q8_0 | ~35 GB | 2 GPUs |
| Gemma 3 27B IT | Q8_0 | ~29 GB | 2 GPUs |
| GLM-4.5-Air (MoE) | Q4_K_M | ~73 GB | 2 GPUs + CPU, 64 GB de RAM |

Los modelos "híbridos" (`--n-cpu-moe`) dejan parte de los expertos MoE en la CPU y la RAM.
Son los que muestran diferencias entre procesadores y entre configuraciones de memoria.

## Notas

- La primera corrida descarga los modelos: unos 250 GB con la lista completa. El tiempo de carga de esa corrida incluye la descarga.
- El modo `2gpu-tensor` de llama.cpp es experimental: no admite muchos MoE ni caché KV cuantizada.
- Los valores de `--n-cpu-moe` son estimados y conviene ajustarlos a cada equipo.

- llama.cpp, llama-benchy y cada modelo mantienen sus propias licencias.

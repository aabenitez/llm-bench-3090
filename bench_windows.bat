@echo off
REM Lanzador: permite ejecutar el script sin cambiar la ExecutionPolicy del sistema.
REM Ejemplo:  bench_windows.bat -Etiqueta 7800X3D-32GB -Runs 5
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench_windows.ps1" %*

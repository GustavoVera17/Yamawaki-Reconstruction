# CASSIproy2026: Reconstrucción Hiperespectral (Yamawaki)

Este repositorio contiene la implementación para el proyecto **CASSIproy2026**, enfocada en la reconstrucción de imágenes hiperespectrales (HSI) a partir de mediciones comprimidas en 2D simulando la óptica de una cámara CASSI (Coded Aperture Snapshot Spectral Imager).

La arquitectura de la red neuronal y la simulación del proceso óptico se basan en el artículo científico:
> *"Lightweight Hyperspectral Image Reconstruction Network with Deep Feature Hallucination"* por Yamawaki et al.

## 🧪 Datasets
La red ha sido entrenada y validada utilizando el **CAVE Multispectral Image Database**, el cual proporciona cubos de datos de 31 bandas espectrales (400nm - 700nm) esenciales para la reconstrucción de alta fidelidad.

## 🤖 Desarrollo y Autoría

La programación, estructuración y simulación física de este repositorio fue **desarrollada casi en su totalidad utilizando a Gemini 3.1 Pro** como asistente de Inteligencia Artificial, bajo la dirección técnica, experimentación y validación del investigador principal.

El código ha sido escrito siguiendo **estrictas buenas prácticas de ingeniería de software y Deep Learning**:
* **Modularidad:** Separación limpia de responsabilidades entre el pipeline de datos (`dataset_cassi.py`), la arquitectura del modelo (`yamawaki_net.py`), las métricas y los scripts de ejecución.
* **Optimización de Hardware:** Implementación de un sistema de caché en la memoria RAM para los tensores de imágenes, lo que reduce el cuello de botella de I/O y acelera el entrenamiento en GPU drásticamente.
* **Monitoreo Profesional:** Interfaz de validación interactiva que evita el congelamiento del SO, apoyada por barras de progreso dinámicas (`tqdm`) en consola.
* **Fidelidad Científica:** Preservación de profundidad de bits (16-bits) en los datasets originales y evaluación con métricas estándar de la industria (L1, PSNR, SSIM y SAM).

---

## 📂 Estructura del Proyecto

* `dataset_cassi.py`: Dataloader de PyTorch que simula la óptica CASSI (máscara de apertura codificada y dispersión espectral).
* `yamawaki_net.py`: Arquitectura central de la red (Módulo DFHM y bloques de atención SHB/SCAB).
* `metricas.py`: Funciones para el cálculo riguroso de PSNR y SAM.
* `train.py`: Script principal de entrenamiento con validación en vivo y generación del dashboard.
* `inferencia.py` e `inferencia_m.py`: Herramientas gráficas interactivas con *slider* espectral para analizar cualitativamente la reconstrucción en longitudes de onda de 400nm a 700nm.

---

## 🚀 Requisitos e Instalación

Para ejecutar este proyecto, necesitas un entorno con Python 3.10+ y una GPU compatible con CUDA.

```bash
# Instalación de dependencias principales
pip install torch torchvision numpy matplotlib scikit-image pillow tqdm
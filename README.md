# Simpsons Character Classification

Este repositorio presenta la implementación de una solución de Deep Learning desarrollada como parte de una evaluación práctica de la asignatura **"Deep Learning"**, impartida por el profesor **Mauro M. Mercado F.** durante el **primer semestre académico de 2026**, sección **002D**.

El proyecto tiene como objetivo la clasificación automática de personajes de la serie *The Simpsons* mediante Redes Neuronales Convolucionales (CNN), aplicando técnicas modernas de entrenamiento, regularización y análisis de calidad de datos.

## El Problema

El conjunto de datos utilizado contiene más de 20.000 imágenes distribuidas entre distintos personajes de la serie. Durante el análisis exploratorio se identificaron desafíos habituales en proyectos reales de Inteligencia Artificial, tales como:

- Desbalance severo entre clases.
- Clases con escasa cantidad de muestras.
- Imágenes incorrectamente etiquetadas.
- Presencia de ruido en los datos.

Desde una perspectiva de negocio, estos problemas impactan directamente la capacidad de los modelos para aprender patrones representativos y generalizar correctamente sobre datos no vistos. Por esta razón, el proyecto pone especial énfasis en la evaluación de la calidad del dataset y en la aplicación de técnicas orientadas a mejorar la robustez del modelo frente a datos imperfectos.

## Solución Implementada

Se desarrollaron y compararon distintas arquitecturas de aprendizaje profundo utilizando PyTorch y PyTorch Lightning, incorporando técnicas como:

- Redes Neuronales Convolucionales (CNN).
- Batch Normalization.
- Dropout.
- Class Weights para manejo de desbalance.
- Data Augmentation.
- Global Average Pooling.
- Grad-CAM para interpretabilidad del modelo.

## Resultados

La solución final logró una mejora sostenida a lo largo de las distintas iteraciones del modelo, alcanzando un Accuracy superior al 90% y un Macro F1 cercano al 80% sobre el conjunto de prueba. Los resultados evidencian una adecuada capacidad de generalización incluso frente a clases minoritarias, datos desbalanceados y ejemplos con ruido de etiquetado.

## Uso del Repositorio

### 1. Clonar Repositorio

```bash
git clone https://github.com/angmzz/theSimpsons-Classification-CNN.git
cd theSimpsons-Classification-CNN
```

### 2. Configuración del Entorno

Este proyecto soporta tanto Conda como Pip.

#### Conda

```bash
conda env create -f environment.yml

conda activate CNN-ENV
```

#### Venv

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Ejecutar Jupyter Lab

```bash
jupyter lab
```
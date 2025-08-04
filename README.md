# Detección de Lenguaje Tóxico en Intervenciones Parlamentarias del Perú

Este proyecto aplica procesamiento de lenguaje natural (NLP) para identificar lenguaje tóxico en intervenciones registradas durante una sesión del Congreso del Perú (año 2020, sesión 17).

## 📌 Objetivo

Clasificar automáticamente intervenciones parlamentarias como tóxicas o no tóxicas utilizando modelos de lenguaje preentrenados en español.

## 🗂️ Estructura del Proyecto
nlp-toxicidad-parlamentaria/
```
├── data/
│ ├── raw/ # CSV y PDF originales
│ └── processed/ # Intervenciones limpias
├── scripts/
│ └── extraer_intervenciones.py
│ └── etiquetado_automatico.py (por construir)
├── README.md
├── requirements.txt
└── .gitignore
```
## 🔄 Pipeline hasta ahora

1. **Extracción del texto**  
   - Se extrajo el texto desde el PDF oficial: `slo-2020-17.pdf`.

2. **Limpieza y estructuración**  
   - Se normalizó el texto: minúsculas, sin signos, sin saltos innecesarios.
   - Se separaron las intervenciones individuales por orador.
   - Se corrigieron errores de asignación de roles.

3. **Archivo limpio generado**  
   - Archivo listo: `intervenciones_2020_17.csv` (almacenado en `data/processed/`)

## ⚙️ Requisitos técnicos

pip install transformers pandas

## 🚧 Próximo paso
Aplicar un modelo de clasificación de toxicidad en español usando Hugging Face.

## 🧠 Modelo preentrenado a utilizar
mrm8488/bert-spanish-cased-finetuned-toxic-comments

## 📌 Notas
El proyecto se encuentra en desarrollo.

Cada etapa será versionada y documentada progresivamente.

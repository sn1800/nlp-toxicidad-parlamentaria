from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd

# Cargar modelo y tokenizer
modelo = "mrm8488/bert-spanish-cased-finetuned-toxic-comments"
tokenizer = AutoTokenizer.from_pretrained(modelo)
model = AutoModelForSequenceClassification.from_pretrained(modelo)

# Crear pipeline
clasificador = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Cargar tu CSV
df = pd.read_csv("intervenciones_2020_17.csv")

# Limpiar filas vacías o muy cortas
df = df[df["text"].notna() & (df["text"].str.len() > 20)]

# Etiquetar (puedes reducir el número de ejemplos en pruebas)
df["toxicity_label"] = df["text"].apply(lambda x: clasificador(x[:512])[0]["label"])

# Guardar resultado
df.to_csv("intervenciones_etiquetadas.csv", index=False, encoding="utf-8")
print("✅ Archivo 'intervenciones_etiquetadas.csv' generado con etiquetas de toxicidad.")

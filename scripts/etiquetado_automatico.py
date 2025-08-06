from transformers import pipeline
import pandas as pd

# --- Configuración ---
modelo = "PlanTL-GOB-ES/roberta-base-bne-toxicity"
INPUT_CSV = "data/processed/intervenciones_2020_17.csv"
OUTPUT_CSV = "data/processed/intervenciones_etiquetadas.csv"

# --- Cargar clasificación ---
clasificador = pipeline("text-classification", model="PlanTL-GOB-ES/roberta-base-bne-toxicity")
# --- Cargar datos ---
df = pd.read_csv(INPUT_CSV, encoding="utf-8")

# --- Etiquetado ---
def obtener_etiquetas(texto):
    # Procesar hasta 512 tokens (máximo)
    res = clasificador(texto[:512])
    # Extraer solo etiquetas por encima de cierto umbral (opcional)
    etiquetas = [r["label"] for r in res if r["score"] >= 0.5]
    return etiquetas if etiquetas else ["NOT_TOXIC"]

df["toxicity_labels"] = df["text"].apply(obtener_etiquetas)

# --- Guardar resultado ---
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"✅ Etiquetado completo y guardado en: {OUTPUT_CSV}")
print(df[["role", "speaker", "toxicity_labels"]].head(10).to_string())

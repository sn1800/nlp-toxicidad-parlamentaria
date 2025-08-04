import re, sys
import fitz  # PyMuPDF
import pandas as pd
from typing import List, Dict
from unidecode import unidecode

# Config
PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "slo-2020-17.pdf"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "intervenciones_2020_17.csv"

# 1. Extraer texto
doc = fitz.open(PDF_PATH)
pages_text = [doc.load_page(i).get_text("text") for i in range(len(doc))]
raw = "\n".join(pages_text)

# 2. Limpieza básica
def clean_basic(t: str) -> str:
    t = re.sub(r"\n\s*\d+\s*\n", "\n", t)  # eliminar números de página
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\u200b|\ufeff", "", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()

text = clean_basic(raw)

# 3. Regex para detectar encabezados
HEADER_RE = re.compile(
    r"^El (señor|señora) ([A-ZÁÉÍÓÚÑÑa-z\s\.]+?)(?: \((.*?)\))?\.—",
    re.MULTILINE
)

# 4. Meta-líneas (no son intervenciones)
META_LINE = re.compile(r"^—\s*(Se\s|Ingresa|Sale|Suspende|Reanuda)", re.IGNORECASE)

# 5. Rol según encabezado
def detectar_rol(header: str) -> str:
    h = header.upper()
    if "PRESIDENTE DE LA REPÚBLICA" in h:
        return "PRESIDENTE DE LA REPÚBLICA"
    elif "PRESIDENTE" in h:
        return "PRESIDENTE DEL CONGRESO"
    elif "RELATOR" in h:
        return "RELATOR"
    elif "DEFENSOR" in h:
        return "DEFENSOR"
    else:
        return "CONGRESISTA"

# 6. Procesar intervenciones
matches = list(HEADER_RE.finditer(text))
print(f"[DEBUG] Total encabezados detectados: {len(matches)}")

segments: List[Dict] = []
for idx, m in enumerate(matches):
    start = m.start()
    end = matches[idx+1].start() if idx+1 < len(matches) else len(text)
    header = m.group(0)
    body = text[start:end][len(header):].strip()

    # Omitir líneas meta
    lines = [l for l in body.splitlines() if not META_LINE.search(l)]
    body_clean = "\n".join(lines).strip()
    if len(body_clean) < 40:
        continue

    role = detectar_rol(header)
    speaker = m.group(2).strip()
    party = m.group(3).strip() if m.group(3) else None

    segments.append({
        "role": role,
        "speaker": unidecode(speaker),
        "party": party,
        "text": body_clean
    })

# 7. Exportar CSV
df = pd.DataFrame(segments, columns=["role", "speaker", "party", "text"])
df["text"] = df["text"].apply(lambda s: re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", s)).strip())

df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"[OK] Extraídas {len(df)} intervenciones -> {OUT_CSV}")
print(df.head(5).to_string())

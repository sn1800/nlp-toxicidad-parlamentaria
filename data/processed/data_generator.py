# Faster run without expensive near-duplicate SequenceMatcher

import pandas as pd, re, unicodedata
from pathlib import Path
from random import choice, random, randint, sample, seed

seed(42)
in_path = Path("dataset_diverso_10000.csv")

df = pd.read_csv(in_path)
text_col = next((c for c in df.columns if c.lower() in ["text","texto","sentence"]), None)
label_col = next((c for c in df.columns if c.lower() in ["label","etiqueta","tag"]), None)
assert text_col and label_col, f"Cols not found: {df.columns.tolist()}"
df = df.rename(columns={text_col: "texto", label_col: "label"})[["texto","label"]]

def map_label(x):
    s = str(x).strip().lower()
    return 1 if s in ["toxico","toxic","1","tox","t"] else 0
df["label"] = df["label"].map(map_label).astype(int)

def basic_clean(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s
df["texto"] = df["texto"].astype(str).map(basic_clean)

# Exact and canonical dedup
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
def canonical_key(s):
    s2 = strip_accents(s.lower())
    s2 = re.sub(r"[^\w\s]", "", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2
df["canon"] = df["texto"].map(canonical_key)
df = df.drop_duplicates(subset=["texto"]).copy()
df = df.drop_duplicates(subset=["canon"]).drop(columns=["canon"]).reset_index(drop=True)

base_count = len(df)

# Augmentation
from random import randint
def random_typo(word):
    if len(word) < 4 or random() < 0.5:
        return word
    i = randint(0, len(word)-2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]
def perturb_text(s):
    words = s.split()
    if len(words)==0: return s
    if random() < 0.4:
        k = max(1, len(words)//10)
        idxs = sample(range(len(words)), k=k)
        for i in idxs:
            words[i] = random_typo(words[i])
    s2 = " ".join(words)
    if random() < 0.3:
        s2 = re.sub(r"([aeiouáéíóú])", lambda m: m.group(1)*(2 if random()<0.5 else 3), s2, count=1)
    if random() < 0.35:
        s2 += choice([" 😒"," 😡"," 🤨"," ❗"," ...","!!!","~"])
    if random() < 0.25:
        s2 = s2.capitalize()
    return s2

target_aug = min(8000, base_count)
aug_rows = []
for i in range(target_aug):
    r = df.sample(1, random_state=42+i).iloc[0]
    aug_rows.append({"texto": perturb_text(r["texto"]), "label": int(r["label"])})
aug_df = pd.DataFrame(aug_rows)

# Synthetic templated
topics = ["política","economía","deportes","tecnología","educación","salud","transporte","seguridad","medio ambiente","cultura"]
neutral_stems = [
    "Considero que {tema} requiere un análisis técnico y respeto por todas las opiniones.",
    "Agradezco los aportes sobre {tema}; mantengamos un debate constructivo.",
    "Los datos sobre {tema} deben revisarse con evidencia y sin ataques personales.",
    "Celebro las mejoras recientes en {tema}; sigamos colaborando.",
    "Propongo escuchar a todas las partes involucradas en {tema} antes de decidir."
]
toxic_stems = [
    "Tus argumentos sobre {tema} son un desastre, deja de hablar sin saber.",
    "Qué comentario tan ignorante sobre {tema}, ni te has informado.",
    "Deja de decir tonterías con {tema}, cansas a cualquiera.",
    "Eres insoportable hablando de {tema}; solo repites disparates.",
    "Basta de basura en {tema}, aprende antes de opinar."
]
prefixes = ["Sinceramente,","A ver,","En serio,","Mira,","Oye,"]
suffixes = ["¿queda claro?","y punto.","así de simple.","¿ok?","fin."]

def synth_example(toxic: bool):
    tema = choice(topics)
    stem = choice(toxic_stems if toxic else neutral_stems)
    s = stem.format(tema=tema)
    if random() < 0.5: s = f"{choice(prefixes)} {s}"
    if random() < 0.4: s = f"{s} {choice(suffixes)}"
    if random() < 0.35: s = perturb_text(s)
    return {"texto": s, "label": int(toxic)}

synth_df = pd.DataFrame([synth_example(i%2==0) for i in range(12000)])

# Merge + final dedup + balance
full = pd.concat([df[["texto","label"]], aug_df, synth_df], ignore_index=True)
full["texto"] = full["texto"].map(basic_clean)
full = full.drop_duplicates(subset=["texto"]).reset_index(drop=True)

counts = full["label"].value_counts()
min_class = counts.min()
balanced = pd.concat([
    full[full["label"]==0].sample(min_class, random_state=42),
    full[full["label"]==1].sample(min_class, random_state=42)
], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

out_path = Path("dataset_diverso_plus_clean_balanced.csv")
balanced.to_csv(out_path, index=False, encoding="utf-8")

len(df), len(aug_df), len(synth_df), len(full), len(balanced), str(out_path)
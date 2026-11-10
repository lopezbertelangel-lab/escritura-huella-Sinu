# ============================================================
# EJECUTA ESTA PRIMERA LÍNEA EN UNA CELDA APARTE SI AÚN NO HAS INSTALADO:
# !pip -q install gradio requests opencv-python-headless
# ============================================================

import os
import re
import json
import time
import shutil
import hashlib
from pathlib import Path

import requests
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps
from google.colab import drive

# =========================================================
# 0) GOOGLE DRIVE
# =========================================================
if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive/EscrituraHuella")
AUTORES_DIR = ROOT / "autores"
DB_PATH = ROOT / "galeria_autores.npz"
META_PATH = ROOT / "galeria_autores_meta.json"
AUTORES_DIR.mkdir(parents=True, exist_ok=True)

print("Drive listo:", ROOT)

# =========================================================
# 1) MODELO PREENTRENADO
# =========================================================
MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "ZinMayOo/Handwriting-Writer-Identification/main/model.h5"
)
MODEL_PATH = Path("/content/writer_identifier_pretrained.h5")

def descargar_modelo():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    print("Descargando modelo preentrenado...")
    r = requests.get(MODEL_URL, stream=True, timeout=180)
    r.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

descargar_modelo()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()

MODEL_FILE_HASH = sha256_file(MODEL_PATH)

try:
    modelo = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )
except Exception as e:
    raise RuntimeError(
        "No fue posible cargar el modelo H5. "
        "Reinicia el entorno y ejecuta de nuevo. "
        f"Detalle: {e}"
    )

modelo.trainable = False

def hash_pesos(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()

MODEL_WEIGHTS_HASH = hash_pesos(modelo)

print("Modelo cargado.")
print("Input :", modelo.input_shape)
print("Output:", modelo.output_shape)
print("SHA256 archivo:", MODEL_FILE_HASH)
print("SHA256 pesos  :", MODEL_WEIGHTS_HASH)

# =========================================================
# 2) EXTRACTOR DE EMBEDDINGS
# =========================================================
def elegir_capa_embedding(model):
    for layer in reversed(model.layers[:-1]):
        try:
            shp = tuple(layer.output.shape)
        except Exception:
            continue

        if len(shp) == 2:
            try:
                dim = int(shp[-1])
            except Exception:
                continue

            if 16 <= dim <= 8192:
                return layer

    return model.layers[-2]

capa_embedding = elegir_capa_embedding(modelo)

encoder = tf.keras.Model(
    inputs=modelo.input,
    outputs=capa_embedding.output,
    name="writer_embedding_encoder"
)
encoder.trainable = False

input_shape = modelo.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]

INPUT_H = int(input_shape[1])
INPUT_W = int(input_shape[2])
INPUT_C = int(input_shape[3])

print("Capa embedding:", capa_embedding.name)
print("Dimensión embedding:", encoder.output_shape)
print("Entrada detectada:", (INPUT_H, INPUT_W, INPUT_C))

# =========================================================
# 3) PREPROCESAMIENTO
# =========================================================
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def leer_gris(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.asarray(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

def recortar(gray, margen=12):
    _, ink = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    pts = cv2.findNonZero(ink)
    if pts is None:
        return gray

    x, y, w, h = cv2.boundingRect(pts)
    x0 = max(0, x - margen)
    y0 = max(0, y - margen)
    x1 = min(gray.shape[1], x + w + margen)
    y1 = min(gray.shape[0], y + h + margen)
    return gray[y0:y1, x0:x1]

def asegurar_minimo(gray, minimo=150):
    h, w = gray.shape
    H = max(h, minimo)
    W = max(w, minimo)
    lienzo = np.full((H, W), 255, np.uint8)
    y = (H - h) // 2
    x = (W - w) // 2
    lienzo[y:y+h, x:x+w] = gray
    return lienzo

def densidad_tinta(patch):
    _, bw = cv2.threshold(
        patch, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return float((bw > 0).mean())

def extraer_patches(gray, patch_size=150, step=50):
    gray = asegurar_minimo(recortar(gray), patch_size)
    h, w = gray.shape

    ys = list(range(0, max(1, h - patch_size + 1), step))
    xs = list(range(0, max(1, w - patch_size + 1), step))

    if not ys or ys[-1] != max(0, h - patch_size):
        ys.append(max(0, h - patch_size))
    if not xs or xs[-1] != max(0, w - patch_size):
        xs.append(max(0, w - patch_size))

    patches = []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            p = gray[y:y+patch_size, x:x+patch_size]
            if p.shape == (patch_size, patch_size) and densidad_tinta(p) >= 0.01:
                patches.append(p)

    if not patches:
        patches = [
            cv2.resize(
                gray,
                (patch_size, patch_size),
                interpolation=cv2.INTER_AREA
            )
        ]

    return patches

def adaptar_patch(p):
    p = cv2.resize(
        p,
        (INPUT_W, INPUT_H),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0

    if INPUT_C == 1:
        p = p[..., None]
    elif INPUT_C == 3:
        p = np.repeat(p[..., None], 3, axis=-1)
    else:
        raise ValueError(f"Canales no soportados: {INPUT_C}")

    return p

def normalizar(v):
    v = np.asarray(v, np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def embedding_imagen(path, max_patches=160):
    gray = leer_gris(path)
    patches = extraer_patches(gray)

    if len(patches) > max_patches:
        idx = np.linspace(
            0, len(patches) - 1, max_patches
        ).astype(int)
        patches = [patches[i] for i in idx]

    batch = np.stack(
        [adaptar_patch(p) for p in patches],
        axis=0
    )

    z = np.asarray(
        encoder.predict(batch, verbose=0)
    ).reshape(len(batch), -1)

    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12

    # Combinación de media y mediana
    emb = 0.5 * z.mean(axis=0) + 0.5 * np.median(z, axis=0)
    return normalizar(emb)

# =========================================================
# 4) GALERÍA
# =========================================================
def coseno(a, b):
    return float(
        np.dot(a, b) /
        ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))
    )

def listar_muestras():
    out = []
    for folder in sorted(AUTORES_DIR.iterdir()):
        if not folder.is_dir():
            continue

        label = folder.name
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in EXTS:
                out.append((label, f))
    return out

def calibrar_umbral(X, y):
    y = np.asarray(y)
    autores = sorted(set(y.tolist()))

    if len(autores) < 2:
        return None, None, "Se necesitan al menos 2 autores."

    genuinos, impostores, margenes = [], [], []

    for i, (e, lab) in enumerate(zip(X, y)):
        same = np.where(y == lab)[0]
        same = same[same != i]

        own = None
        if len(same):
            c = normalizar(X[same].mean(axis=0))
            own = coseno(e, c)
            genuinos.append(own)

        other_scores = []
        for otro in autores:
            if otro == lab:
                continue
            idx = np.where(y == otro)[0]
            c = normalizar(X[idx].mean(axis=0))
            other_scores.append(coseno(e, c))

        if other_scores:
            imp = max(other_scores)
            impostores.append(imp)
            if own is not None:
                margenes.append(own - imp)

    if len(genuinos) < 2 or len(impostores) < 2:
        return None, None, "Faltan muestras repetidas por autor."

    genuinos = np.asarray(genuinos)
    impostores = np.asarray(impostores)

    grid = np.linspace(
        min(genuinos.min(), impostores.min()),
        max(genuinos.max(), impostores.max()),
        1000
    )

    best_thr, best_j = None, -999

    for thr in grid:
        sens = np.mean(genuinos >= thr)
        spec = np.mean(impostores < thr)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thr = float(thr)

    margin_thr = 0.0
    if margenes:
        margin_thr = max(
            0.0,
            0.5 * float(np.percentile(margenes, 10))
        )

    info = {
        "genuine_mean": float(genuinos.mean()),
        "impostor_mean": float(impostores.mean()),
        "threshold": best_thr,
        "margin_threshold": margin_thr,
        "youden_j": float(best_j),
    }
    return best_thr, margin_thr, info

def reconstruir_galeria():
    antes = hash_pesos(modelo)
    registros = listar_muestras()

    if not registros:
        raise RuntimeError(
            f"No hay imágenes dentro de {AUTORES_DIR}"
        )

    embeddings, labels, paths = [], [], []
    errores = []

    for k, (label, path) in enumerate(registros, 1):
        try:
            embeddings.append(embedding_imagen(path))
            labels.append(label)
            paths.append(str(path))
            print(f"[{k}/{len(registros)}] {label}: {path.name}")
        except Exception as e:
            errores.append({
                "autor": label,
                "archivo": str(path),
                "error": str(e)
            })

    if not embeddings:
        raise RuntimeError("No se pudo crear ningún embedding.")

    X = np.stack(embeddings).astype(np.float32)
    y = np.asarray(labels, dtype=str)
    p = np.asarray(paths, dtype=str)

    thr, margin_thr, cal = calibrar_umbral(X, y)

    np.savez_compressed(
        DB_PATH,
        embeddings=X,
        labels=y,
        paths=p
    )

    conteo = {
        a: int(np.sum(y == a))
        for a in sorted(set(y.tolist()))
    }

    meta = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modelo_sha256": MODEL_FILE_HASH,
        "pesos_sha256": MODEL_WEIGHTS_HASH,
        "capa_embedding": capa_embedding.name,
        "num_autores": len(conteo),
        "num_imagenes": len(X),
        "muestras_por_autor": conteo,
        "threshold": thr,
        "margin_threshold": margin_thr,
        "calibracion": cal,
        "errores": errores
    }

    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    despues = hash_pesos(modelo)
    assert antes == despues == MODEL_WEIGHTS_HASH, (
        "ERROR: los pesos de la red cambiaron."
    )

    return meta

def cargar_galeria():
    if not DB_PATH.exists():
        raise RuntimeError(
            "Primero registra autores y construye la galería."
        )

    data = np.load(DB_PATH, allow_pickle=False)
    X = data["embeddings"].astype(np.float32)
    y = data["labels"].astype(str)

    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    return X, y, meta

def centroides(X, y):
    out = {}
    for autor in sorted(set(y.tolist())):
        out[autor] = normalizar(X[y == autor].mean(axis=0))
    return out

# =========================================================
# 5) REGISTRO DE AUTORES
# =========================================================
def nombre_seguro(nombre):
    nombre = str(nombre or "").strip()
    if not nombre:
        raise ValueError("Escribe el nombre del autor.")

    nombre = re.sub(r'[\\/:"*?<>|]+', "_", nombre)
    nombre = re.sub(r"\s+", " ", nombre).strip()

    if nombre in {".", ".."}:
        raise ValueError("Nombre inválido.")

    return nombre

def path_de_gradio(item):
    if isinstance(item, str):
        return item
    if hasattr(item, "name"):
        return item.name
    if isinstance(item, dict):
        return item.get("path") or item.get("name")
    return None

def registrar_archivos(nombre, archivos):
    try:
        nombre = nombre_seguro(nombre)
    except Exception as e:
        return f"❌ {e}"

    if not archivos:
        return "❌ Selecciona una o varias fotos del manuscrito."

    destino_dir = AUTORES_DIR / nombre
    destino_dir.mkdir(parents=True, exist_ok=True)

    guardadas = 0

    for item in archivos:
        src = path_de_gradio(item)
        if not src:
            continue

        src = Path(src)
        ext = src.suffix.lower()

        if ext not in EXTS:
            continue

        destino = destino_dir / (
            time.strftime("%Y%m%d_%H%M%S")
            + f"_{guardadas:03d}"
            + ext
        )
        shutil.copy2(src, destino)
        guardadas += 1

    if guardadas == 0:
        return "❌ No se encontraron imágenes válidas."

    try:
        meta = reconstruir_galeria()
        return (
            f"✅ Autor «{nombre}» registrado con {guardadas} imágenes.\n"
            f"Galería actualizada: {meta['num_autores']} autores, "
            f"{meta['num_imagenes']} imágenes.\n"
            "La red NO fue reentrenada."
        )
    except Exception as e:
        return (
            f"✅ Se guardaron {guardadas} imágenes en Drive, "
            f"pero la galería no pudo actualizarse: {e}"
        )

def importar_carpeta_drive(nombre, carpeta_drive):
    try:
        nombre = nombre_seguro(nombre)
    except Exception as e:
        return f"❌ {e}"

    carpeta_drive = str(carpeta_drive or "").strip()

    if not carpeta_drive:
        return (
            "❌ Escribe la ruta de la carpeta de Drive, por ejemplo:\n"
            "/content/drive/MyDrive/Muestras_Lucho"
        )

    src_dir = Path(carpeta_drive)

    if not src_dir.exists() or not src_dir.is_dir():
        return f"❌ No existe la carpeta: {src_dir}"

    destino_dir = AUTORES_DIR / nombre
    destino_dir.mkdir(parents=True, exist_ok=True)

    archivos = [
        f for f in src_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in EXTS
    ]

    if not archivos:
        return "❌ La carpeta no contiene imágenes compatibles."

    copiados = 0
    for f in archivos:
        destino = destino_dir / f.name
        if destino.exists():
            destino = destino_dir / (
                f"{f.stem}_{int(time.time()*1000)}{f.suffix.lower()}"
            )
        shutil.copy2(f, destino)
        copiados += 1

    try:
        meta = reconstruir_galeria()
        return (
            f"✅ Carpeta importada para «{nombre}»: {copiados} imágenes.\n"
            f"Galería: {meta['num_autores']} autores, "
            f"{meta['num_imagenes']} imágenes.\n"
            "La red NO fue reentrenada."
        )
    except Exception as e:
        return (
            f"✅ Se copiaron {copiados} imágenes, "
            f"pero falló la actualización de galería: {e}"
        )

# =========================================================
# 6) IDENTIFICACIÓN
# =========================================================
def identificar(img_path):
    if img_path is None:
        return "❌ Carga una imagen.", {}

    try:
        X, y, meta = cargar_galeria()
        q = embedding_imagen(img_path)

        cents = centroides(X, y)
        ranking = sorted(
            [(a, coseno(q, c)) for a, c in cents.items()],
            key=lambda x: x[1],
            reverse=True
        )

        top1_autor, top1 = ranking[0]
        top2 = ranking[1][1] if len(ranking) > 1 else -1.0
        margen = top1 - top2

        thr = meta.get("threshold")
        mthr = meta.get("margin_threshold")

        if thr is None:
            decision = (
                f"MEJOR COINCIDENCIA: {top1_autor} | "
                f"similitud={top1:.4f}. "
                "Todavía no hay suficientes datos para calibrar "
                "el rechazo de desconocidos."
            )
        else:
            mthr = 0.0 if mthr is None else float(mthr)

            if top1 >= float(thr) and margen >= mthr:
                decision = (
                    f"AUTOR PROPUESTO: {top1_autor} | "
                    f"similitud={top1:.4f} | "
                    f"margen={margen:.4f}"
                )
            else:
                decision = (
                    "AUTOR DESCONOCIDO / NO CONCLUYENTE | "
                    f"mejor coincidencia={top1_autor} | "
                    f"similitud={top1:.4f} | "
                    f"margen={margen:.4f}"
                )

        ranking_dict = {
            autor: float(score)
            for autor, score in ranking[:5]
        }

        return decision, ranking_dict

    except Exception as e:
        return f"❌ Error: {e}", {}

def estado_sistema():
    autores = [
        p.name for p in AUTORES_DIR.iterdir()
        if p.is_dir()
    ]

    lineas = [
        f"Drive: {ROOT}",
        f"Autores registrados: {len(autores)}",
        f"Autores: {', '.join(sorted(autores)) if autores else '(ninguno)'}",
        f"SHA256 pesos: {MODEL_WEIGHTS_HASH}",
        f"Pesos intactos: {hash_pesos(modelo) == MODEL_WEIGHTS_HASH}",
    ]

    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        lineas.append(
            f"Galería: {meta.get('num_imagenes', 0)} imágenes"
        )

    return "\n".join(lineas)

# =========================================================
# 7) INTERFAZ DE GRADIO
# =========================================================
with gr.Blocks(title="EscrituraHuella Sinú") as app:
    gr.Markdown(
        """
        # EscrituraHuella Sinú — Identificación de autor
        **Modelo preentrenado congelado + embeddings almacenados en Google Drive.**

        Registrar autores **no reentrena la red**.
        """
    )

    with gr.Tab("➕ Cargar autor"):
        gr.Markdown(
            "### Opción A — seleccionar fotos o arrastrar una carpeta"
        )

        nombre_a = gr.Textbox(
            label="Nombre del autor",
            placeholder="Ej.: Juan Pérez"
        )

        archivos_a = gr.File(
            label="Sube una o varias fotos (o arrastra una carpeta entera)",
            file_count="multiple",
            file_types=["image"]
        )

        btn_a = gr.Button(
            "Guardar autor y actualizar galería",
            variant="primary"
        )

        salida_a = gr.Textbox(
            label="Resultado",
            lines=5
        )

        btn_a.click(
            registrar_archivos,
            inputs=[nombre_a, archivos_a],
            outputs=salida_a
        )

        gr.Markdown(
            "### Opción B — importar una carpeta que ya está en Google Drive"
        )

        nombre_b = gr.Textbox(
            label="Nombre del autor",
            placeholder="Ej.: Juan Pérez"
        )

        ruta_b = gr.Textbox(
            label="Ruta de la carpeta en Drive",
            placeholder="/content/drive/MyDrive/Muestras_Juan"
        )

        btn_b = gr.Button(
            "Importar carpeta de Drive y actualizar galería"
        )

        salida_b = gr.Textbox(
            label="Resultado",
            lines=5
        )

        btn_b.click(
            importar_carpeta_drive,
            inputs=[nombre_b, ruta_b],
            outputs=salida_b
        )

    with gr.Tab("🔎 Identificar autor"):
        foto_test = gr.Image(
            type="filepath",
            label="Sube UNA foto de escritura"
        )

        btn_test = gr.Button(
            "Identificar autor",
            variant="primary"
        )

        decision = gr.Textbox(
            label="Decisión",
            lines=3
        )

        ranking = gr.Label(
            label="Ranking por similitud",
            num_top_classes=5
        )

        btn_test.click(
            identificar,
            inputs=foto_test,
            outputs=[decision, ranking]
        )

    with gr.Tab("⚙️ Estado"):
        btn_estado = gr.Button("Actualizar estado")
        estado = gr.Textbox(
            label="Estado del sistema",
            lines=8
        )

        btn_estado.click(
            estado_sistema,
            outputs=estado
        )

# Lanzar la aplicación
app.launch(share=True, debug=True)

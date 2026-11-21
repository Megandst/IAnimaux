"""Archive ZIP des originaux classés, créée en mémoire sans toucher aux sources."""

import hashlib
from io import BytesIO
from pathlib import PurePosixPath
import re
import unicodedata
from zipfile import ZIP_DEFLATED, ZipFile


FOLDERS = {"Chat": "chat", "Chien": "chien"}


def safe_filename(filename, index):
    """Garde un nom simple et portable, jamais un chemin fourni par l'utilisateur."""
    name = PurePosixPath(str(filename).replace("\\", "/")).name
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[\x00-\x1f\x7f<>:"|?*]', "_", name).strip(" .")
    if not name:
        return f"photo_{index:03d}.jpg"
    path = PurePosixPath(name)
    suffix = path.suffix
    stem = path.stem.encode("utf-8")[:180].decode("utf-8", errors="ignore").rstrip(" .") or "photo"
    if stem.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        stem = f"_{stem}"
    return stem + suffix


def create_classified_zip(files, results, batch_corrections=None):
    """Trie selon le lot courant ; les anciennes annotations ne sont pas consultées."""
    if len(files) != len(results):
        raise ValueError("Les fichiers et les résultats ne correspondent pas au même lot.")
    corrections = dict(batch_corrections or {})
    buffer = BytesIO()
    used_names = {folder: set() for folder in FOLDERS.values()}
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        # Explicit entries preserve both folders even if one species is absent.
        for folder in FOLDERS.values():
            archive.writestr(f"{folder}/", b"")
        for index, (file, row) in enumerate(zip(files, results), 1):
            content = file.getvalue()
            image_hash = hashlib.sha256(content).hexdigest()
            if image_hash != row["image_sha256"] or file.name != row["filename"]:
                raise ValueError("Une photo a changé depuis sa classification. Relancer l'analyse.")
            if row["error"]:
                continue
            category = corrections.get(image_hash, row["category"])
            if category not in FOLDERS:
                raise ValueError("La catégorie retenue doit être Chat ou Chien.")
            folder = FOLDERS[category]
            name = safe_filename(file.name, index)
            path = PurePosixPath(name)
            candidate, number = name, 2
            while candidate.casefold() in used_names[folder]:
                candidate = f"{path.stem}_{number}{path.suffix}"
                number += 1
            used_names[folder].add(candidate.casefold())
            archive.writestr(f"{folder}/{candidate}", content)
    return buffer.getvalue()

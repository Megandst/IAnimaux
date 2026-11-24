"""Interface locale IAnimaux : importation et prédictions avec le modèle entraîné."""

import hashlib
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st
import torch

from config import FEEDBACK_DB, MAX_UPLOAD_MB, MODEL_PATH
from ianimaux.export import create_classified_zip
from ianimaux.feedback import get_corrections, save_correction
from ianimaux.model import load_checkpoint
from ianimaux.predict import classify_uploads, upload_signature
from ianimaux.ui import render_footer, render_header


@st.cache_resource(show_spinner=False)
def cached_model(path, file_version):
    # The file version is part of the cache key, so replacing a checkpoint reloads it.
    torch.set_num_threads(4)
    model, _ = load_checkpoint(path)
    return model, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def show_results(results, files, model_sha256):
    try:
        corrections = get_corrections([row["image_sha256"] for row in results], FEEDBACK_DB)
    except (OSError, sqlite3.Error):
        corrections = {}
        st.warning("Les corrections enregistrées sont temporairement indisponibles.")

    batch_corrections = st.session_state.setdefault("batch_corrections", {})

    def effective_category(row):
        return batch_corrections.get(row["image_sha256"], row["category"])

    valid = [row for row in results if not row["error"]]
    st.subheader("Le bilan de votre lot")
    columns = st.columns(4)
    for index, (column, label, count) in enumerate(zip(
        columns,
        ["Photos classées", "Chats", "Chiens", "Fichiers en erreur"],
        [len(valid), sum(effective_category(row) == "Chat" for row in valid),
         sum(effective_category(row) == "Chien" for row in valid), len(results) - len(valid)],
    )):
        with column:
            with st.container(key=f"stat_{index}"):
                st.metric(label, count)

    table = pd.DataFrame([{
        "Fichier": row["filename"],
        "Prédiction IA": row["category"] or "Non analysée",
        "Confiance IA": f"{row['confidence']:.1%}".replace(".", ",") if row["confidence"] is not None else "—",
        "Catégorie retenue": effective_category(row) or "Non analysée",
        "Origine": "Correction humaine du lot" if row["image_sha256"] in batch_corrections else ("Modèle" if not row["error"] else "—"),
        "Statut": "Erreur" if row["error"] else "Classée",
        "Détail": row["error"],
    } for row in results])
    st.subheader("Résultats")
    if notice := st.session_state.pop("feedback_notice", None):
        st.success(notice)
    st.dataframe(table, hide_index=True, width="stretch")
    if any(row["image_sha256"] in batch_corrections for row in valid):
        st.caption("Les compteurs suivent la catégorie retenue. La prédiction et le score IA restent ceux du modèle, même après une correction humaine.")

    with st.container(border=True, key="export_panel"):
        st.subheader("Vos photos, bien rangées")
        st.write("Téléchargez une archive ZIP avec deux dossiers : **chat/** et **chien/**.")
        # Freeze the current batch and its decisions for the deferred download.
        selected_files = tuple(files)
        selected_results = tuple(results)
        selected_corrections = dict(batch_corrections)
        st.download_button(
            "Télécharger les photos classées (ZIP)",
            data=lambda: create_classified_zip(selected_files, selected_results, selected_corrections),
            file_name="IAnimaux_photos_classees.zip", mime="application/zip",
            on_click="ignore", type="primary", width="stretch",
            disabled=not valid, key="download_classified_photos",
        )
        st.caption("L'archive suit les catégories retenues dans ce lot, y compris vos corrections. Les noms en double sont adaptés pour conserver toutes les photos.")
        errors = len(results) - len(valid)
        if errors:
            st.warning(f"Fichiers illisibles exclus du ZIP : {errors}. Consultez les erreurs dans le tableau.")

    with st.expander("Voir les photos classées", expanded=True):
        if not valid:
            st.info("Aucune photo n'a pu être analysée. Consultez les erreurs dans le tableau.")
        columns = st.columns(3)
        for index, row in enumerate(valid):
            confidence = f"{row['confidence']:.1%}".replace(".", ",")
            with columns[index % 3]:
                with st.container(border=True, key=f"photo_card_{index}"):
                    st.image(row["preview"], width="stretch", caption=row["filename"])
                    st.markdown(f"**{row['category']}** · {confidence}")
                    st.caption("Prédiction de l’IA")
                    if row["image_sha256"] in batch_corrections:
                        st.caption(f"Correction humaine du lot : {effective_category(row)}")


def main():
    st.set_page_config(page_title="IAnimaux", page_icon="🐾", layout="wide")
    render_header()

    if st.session_state.get("results_version") != 3:
        st.session_state.pop("classification_results", None)
        st.session_state.pop("classification_model_sha256", None)
        st.session_state.pop("batch_corrections", None)
        st.session_state["results_version"] = 3

    with st.container(border=True, key="import_panel"):
        st.subheader("Vos photos à classer")
        st.caption("JPG, PNG ou WebP · 20 Mo maximum par photo · Originaux préservés")
        files = st.file_uploader(
            "Importer des photos", type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True, max_upload_size=MAX_UPLOAD_MB, key="photos",
            help=f"Sélectionnez plusieurs photos JPG, PNG ou WebP. Limite : {MAX_UPLOAD_MB} Mo par photo. Les fichiers originaux restent inchangés.",
        )
        signature = upload_signature(files)
        if st.session_state.get("upload_signature") != signature:
            st.session_state["upload_signature"] = signature
            st.session_state.pop("classification_results", None)
            st.session_state.pop("classification_model_sha256", None)
            st.session_state.pop("batch_corrections", None)

        model_available = MODEL_PATH.is_file()
        current_model_version = (str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns, MODEL_PATH.stat().st_size) if model_available else None
        if st.session_state.get("model_version") != current_model_version:
            st.session_state["model_version"] = current_model_version
            st.session_state.pop("classification_results", None)
            st.session_state.pop("classification_model_sha256", None)
            st.session_state.pop("batch_corrections", None)
        if not model_available:
            st.error("Le modèle entraîné est introuvable. Consultez le README pour l'installer ou l'entraîner.")
        if st.button("Lancer la classification", type="primary", width="stretch", disabled=not files or not model_available):
            st.session_state.pop("classification_results", None)
            st.session_state.pop("batch_corrections", None)
            try:
                with st.spinner("Chargement du modèle…"):
                    version = (MODEL_PATH.stat().st_mtime_ns, MODEL_PATH.stat().st_size)
                    model, model_sha256 = cached_model(str(MODEL_PATH), version)
            except Exception:
                st.error("Le modèle n'a pas pu être chargé. Vérifiez le checkpoint entraîné indiqué dans le README.")
            else:
                progress = st.progress(0.0, text="Préparation des photos…")
                try:
                    results = classify_uploads(
                        model, files,
                        on_progress=lambda current, total: progress.progress(current / total, text=f"{current} / {total} fichiers traités"),
                    )
                except RuntimeError:
                    st.error("La classification a rencontré une erreur de calcul. Relancez l'application ou vérifiez le modèle.")
                else:
                    st.session_state["classification_results"] = results
                    st.session_state["classification_model_sha256"] = model_sha256
                    st.success(f"Traitement terminé : {len(results)} fichiers examinés.")

        st.caption("Une photo doit contenir un animal principal, chat ou chien. Le score de confiance indique la préférence du modèle, sans garantir la justesse de la prédiction.")

    results = st.session_state.get("classification_results")
    if results is not None:
        show_results(results, files, st.session_state["classification_model_sha256"])
    elif files:
        st.info(f"Photos sélectionnées : {len(files)}. Cliquez sur « Lancer la classification ».")
    else:
        st.info("Ajoutez des photos dans la zone ci-dessus pour commencer.")

    render_footer()


if __name__ == "__main__":
    main()

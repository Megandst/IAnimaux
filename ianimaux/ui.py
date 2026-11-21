"""Présentation Streamlit d'IAnimaux, indépendante du traitement des photos."""

from pathlib import Path

import streamlit as st


ASSETS = Path(__file__).resolve().parents[1] / "assets"


def render_header():
    st.html(f"<style>{(ASSETS / 'styles.css').read_text()}</style>")
    with st.container(key="brand_header"):
        text, art = st.columns([1.35, 1], vertical_alignment="center")
        with text:
            st.html('<span class="brand-badge">LE COUP DE PATTE DES REFUGES</span>')
            st.title("IAnimaux")
            st.html('<p class="hero-line">Un peu d’IA, plus de temps pour eux.</p>')
            st.write("Un coup de patte pour classer vos photos de chats et de chiens.")
        with art:
            st.image(str(ASSETS / "companions.svg"), width="stretch", alt="Illustration pastel d'un chat et d'un chien souriants")
    st.html('''<ol class="flow-steps" aria-label="Les étapes de classification">
      <li><span>01</span>Importez vos photos</li>
      <li><span>02</span>Lancez la classification</li>
      <li><span>03</span>Vérifiez les résultats</li>
    </ol>''')


def render_footer():
    st.html('<p class="footer-note">IAnimaux · Un coup de patte pour les refuges.</p>')

"""Onglet « Projet » : présentation du sujet, schémas et tableaux de référence.

Contenu statique lu dans content/projet.json (texte, schémas en nœuds / liens, tableaux) : aucun
appel à l'API, l'onglet s'affiche même stack arrêtée. Les schémas sont rendus par Graphviz dans le
navigateur par la visionneuse content/diagram.html (viz.js, copie locale dans static/) : zoom
à la molette, déplacement, plein écran. Modifier le contenu se fait dans le JSON uniquement ;
diagram.py porte l'apparence, content/style.css celle du titre et des cartes du sujet.
"""

from __future__ import annotations

import html
import json
import re
from functools import cache
from importlib.resources import files
from itertools import groupby
from typing import Any

from app.diagram import to_dot

CONTENT = 'projet.json'
VIEWER = 'diagram.html'
STYLE = 'style.css'
DEFAULT_HEIGHT = 500  # pixels, si le schéma ne fixe pas sa `hauteur`


def _read(name: str) -> str:
    return files('app').joinpath('content', name).read_text(encoding='utf-8')


@cache
def content() -> dict[str, Any]:
    return json.loads(_read(CONTENT))


def viewer(schema: dict[str, Any]) -> tuple[str, int]:
    """Page HTML de la visionneuse pour un schéma, et sa hauteur en pixels."""
    height = int(schema.get('hauteur', DEFAULT_HEIGHT))
    # json.dumps produit un littéral JS valide ; `</` échappé pour ne pas fermer la balise script.
    dot = json.dumps(to_dot(schema)).replace('</', r'<\/')
    page = _read(VIEWER).replace('__DOT__', dot).replace('__HEIGHT__', str(height))
    return page, height


def table(table_id: str) -> list[dict[str, str]]:
    """Lignes d'un tableau du contenu, prêtes pour `st.dataframe`."""
    for spec in content()['tableaux']:
        if spec['id'] == table_id:
            return [dict(zip(spec['colonnes'], row, strict=True)) for row in spec['lignes']]
    raise KeyError(table_id)


def _md(text: str) -> str:
    """Texte échappé, `**gras**` rendu en <strong> (seule mise en forme utilisée dans le JSON)."""
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html.escape(text))


def _card(index: int, model: dict[str, str]) -> str:
    rows = ''.join(
        f'<dt>{label}</dt><dd>{html.escape(model[key])}</dd>'
        for label, key in (('Métrique', 'metrique'), ('Usage', 'usage'), ('Références', 'references'))
    )
    return (
        f'<div class="carte c{index % 2}"><span class="nom">{html.escape(model["nom"])}</span>'
        f'<h3>{html.escape(model["cible"])}</h3>'
        f'<div class="horizon">Horizon {html.escape(model["horizon"])}</div><dl>{rows}</dl></div>'
    )


def header_html() -> str:
    """Titre de l'application, centré au-dessus des onglets (avec la feuille de style)."""
    data = content()
    return (
        f'<style>{_read(STYLE)}</style><div class="hero"><h1>{html.escape(data["titre"])}</h1>'
        f'<p>{html.escape(data["sous_titre"])}</p></div>'
    )


def subject_html() -> str:
    """Section « Notre sujet » : introduction, une carte par modèle, conclusion."""
    sujet = content()['sujet']
    cards = ''.join(_card(i, model) for i, model in enumerate(sujet['modeles']))
    return (
        f'<style>{_read(STYLE)}</style><div class="sujet"><h2>Notre sujet</h2>'
        f'<p class="intro">{_md(sujet["intro"])}</p><div class="cartes">{cards}</div>'
        f'<p class="conclusion">{_md(sujet["conclusion"])}</p></div>'
    )


def _schema(st: Any, schema: dict[str, Any]) -> None:
    st.subheader(schema['titre'])
    page, height = viewer(schema)
    st.components.v1.html(page, height=height + 2)
    if schema.get('legende'):
        st.caption(schema['legende'])


def render(st: Any) -> None:
    """Dessine l'onglet avec le module Streamlit fourni."""
    data = content()
    st.html(subject_html())

    # Les schémas d'une même `ligne` (verticaux) s'affichent côte à côte, les autres en pleine largeur.
    for _, group in groupby(data['schemas'], key=lambda schema: schema.get('ligne') or schema['id']):
        schemas = list(group)
        for column, schema in zip(st.columns(len(schemas)), schemas, strict=True):
            with column:
                _schema(st, schema)

    for spec in data['tableaux']:
        st.subheader(spec['titre'])
        st.dataframe(table(spec['id']), hide_index=True, width='stretch')

"""Conversion d'un schéma décrit en données (nœuds, liens, groupes) vers le langage DOT de Graphviz.

Le contenu vit dans content/projet.json ; ce module ne porte que l'apparence (styles par type).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GRAPH = 'bgcolor="transparent", fontname="Helvetica", fontsize=11, nodesep=0.35, ranksep=0.5'
NODE = (
    'shape=box, style="rounded,filled", fillcolor="#f5f7fa", color="#8a94a6", '
    'fontname="Helvetica", fontsize=11'
)
EDGE = 'color="#8a94a6", fontname="Helvetica", fontsize=9'

NODE_STYLES = {
    'etape': '',
    'stockage': 'shape=cylinder, style=filled, fillcolor="#e8f1fb", color="#4a7fc1"',
    'externe': 'style="rounded,dashed", fillcolor="white", color="#b07d2b"',
    'decision': 'shape=diamond, style=filled, fillcolor="#fff7e0", color="#b07d2b"',
    'succes': 'fillcolor="#e6f4ea", color="#3c8d50"',
}
GROUP_STYLES = {
    'groupe': 'color="#c5ccd6"; style=rounded',
}
EDGE_STYLES = {'plein': '', 'pointille': 'style=dashed'}


def _quote(text: str) -> str:
    escaped = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    return f'"{escaped}"'


def _attrs(*parts: str) -> str:
    present = [part for part in parts if part]
    return f' [{", ".join(present)}]' if present else ''


def _style(styles: Mapping[str, str], kind: str, what: str) -> str:
    if kind not in styles:
        raise ValueError(f'{what} de type inconnu : {kind!r} (attendu : {", ".join(styles)})')
    return styles[kind]


def to_dot(schema: Mapping[str, Any]) -> str:
    """DOT d'un schéma ; refuse les liens vers des nœuds ou groupes absents (faute de saisie)."""
    groups = {group['id']: group for group in schema.get('groupes', [])}
    nodes = schema['noeuds']
    ids = [node['id'] for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{schema["id"]} : identifiants de nœuds en double')

    def node_line(node: Mapping[str, Any]) -> str:
        style = _style(NODE_STYLES, node.get('type', 'etape'), 'nœud')
        return f'{_quote(node["id"])}{_attrs(f"label={_quote(node['label'])}", style)};'

    lines = [
        f'digraph {_quote(schema["id"])} {{',
        f'graph [rankdir={schema.get("direction", "LR")}, {GRAPH}];',
        f'node [{NODE}];',
        f'edge [{EDGE}];',
    ]
    for group_id, group in groups.items():
        style = _style(GROUP_STYLES, group.get('type', 'groupe'), 'groupe')
        lines.append(f'subgraph {_quote("cluster_" + group_id)} {{')
        lines.append(f'label={_quote(group["label"])}; fontname="Helvetica"; fontsize=11; {style};')
        lines.extend(node_line(node) for node in nodes if node.get('groupe') == group_id)
        lines.append('}')
    for node in nodes:
        group = node.get('groupe')
        if group is None:
            lines.append(node_line(node))
        elif group not in groups:
            raise ValueError(f'{schema["id"]} : groupe inconnu {group!r} pour {node["id"]!r}')

    known = set(ids)
    for edge in schema['liens']:
        missing = {edge['de'], edge['vers']} - known
        if missing:
            raise ValueError(f'{schema["id"]} : lien vers des nœuds inconnus {sorted(missing)}')
        label = f'label={_quote(edge["label"])}' if edge.get('label') else ''
        style = _style(EDGE_STYLES, edge.get('style', 'plein'), 'lien')
        lines.append(f'{_quote(edge["de"])} -> {_quote(edge["vers"])}{_attrs(label, style)};')
    lines.append('}')
    return '\n'.join(lines) + '\n'

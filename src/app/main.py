"""Application Streamlit de pilotage du service, cliente de l'API uniquement.

Barre latérale : opérateur (profil de démonstration, tracé avec chaque action) puis navigation
entre quatre pages : présentation du projet (statique, affichée même API arrêtée), état du service
et interrupteur marche / arrêt, dernières prévisions, modèles (alias, santé, registre, promotion
et retour arrière). Chaque action passe par l'API, qui la valide et la trace ; l'application ne
garde aucun état propre.

    uv run python -m app
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app import client as api
from app import project, tables

st.set_page_config(page_title='Projet de 5MLDE', page_icon=':material/monitoring:', layout='wide')

# Profils de démonstration : aucun contrôle d'accès, seul le nom est tracé avec chaque action.
OPERATORS = ('admin', 'analyste', 'default')
PAGES = {
    'Projet': ':material/school:',
    'Service': ':material/power_settings_new:',
    'Prévisions': ':material/query_stats:',
    'Modèles': ':material/model_training:',
}


def flash(kind: str, message: str) -> None:
    """Message affiché après le rechargement qui suit une action."""
    st.session_state['flash'] = (kind, message)


def show_flash() -> None:
    kind, message = st.session_state.pop('flash', (None, None))
    if kind == 'success':
        st.success(message, icon=':material/check_circle:')
    elif kind == 'error':
        st.error(message, icon=':material/error:')


def act(label: str, call: Any) -> None:
    """Exécute une action de l'API, mémorise son résultat et recharge la page."""
    try:
        call()
    except api.ApiError as exc:
        flash('error', f'{label} refusé : {exc}')
    else:
        flash('success', f'{label} : fait.')
    st.rerun()


def missing(reason: str) -> str | None:
    if not reason.strip():
        return 'Une raison est obligatoire (elle est tracée).'
    return None


# -- pages ----------------------------------------------------------------------------------------
def service_page(client: api.ApiClient, health: dict[str, Any], operator: str) -> None:
    """Santé des dépendances, interrupteur marche / arrêt et son historique."""
    st.subheader('Dépendances')
    services = health.get('services', {})
    for column, (name, state) in zip(st.columns(max(len(services), 1)), services.items(), strict=False):
        column.metric(
            name.capitalize(), 'ok' if state == 'ok' else 'en panne', help=None if state == 'ok' else state
        )
    if health.get('status') != 'ok':
        st.warning("Une dépendance de l'API est en panne : certaines données peuvent manquer.")

    try:
        state = client.system(history=20)
    except api.ApiError as exc:
        st.error(str(exc))
        return
    current = state['state']
    if current['active']:
        st.success(
            'Le système est **en marche** : les flows planifiés tournent.', icon=':material/play_circle:'
        )
    else:
        st.error(
            'Le système est **arrêté** : ingestion, prévisions, retours et entraînement sont suspendus '
            "(chaque flow s'arrête au démarrage avec le statut `cancelled`).",
            icon=':material/stop_circle:',
        )
    if current.get('changed_by'):
        when = pd.Timestamp(current['recorded_at']).strftime('%d/%m/%Y %H:%M UTC')
        st.caption(f'Dernier changement par {current["changed_by"]} le {when} : {current["reason"]}')

    with st.form('system'):
        st.subheader('Marche / arrêt')
        wanted = st.toggle('Système en marche', value=current['active'])
        reason = st.text_input(
            'Raison', max_chars=500, key='system_reason', placeholder='ex. maintenance de la base'
        )
        if st.form_submit_button('Appliquer', type='primary'):
            problem = missing(reason)
            if wanted == current['active']:
                st.info('Aucun changement : le système est déjà dans cet état.')
            elif problem:
                st.warning(problem)
            else:
                act(
                    'Remise en marche' if wanted else 'Arrêt',
                    lambda: client.set_system(active=wanted, changed_by=operator, reason=reason.strip()),
                )

    st.subheader('Historique')
    history = tables.system_frame(state['history'])
    if history.empty:
        st.info('Aucun changement enregistré : le système est en marche depuis son installation.')
    else:
        st.dataframe(
            history,
            hide_index=True,
            width='stretch',
            column_config={'Date': st.column_config.DatetimeColumn(format='DD/MM/YYYY HH:mm')},
        )


def predictions_page(client: api.ApiClient) -> None:
    """Dernière prévision de chaque symbole pour le modèle et l'alias choisis."""
    left, right = st.columns(2)
    model = left.selectbox('Modèle', api.MODELS, key='predictions_model')
    alias = right.radio(
        'Alias',
        api.ALIASES,
        horizontal=True,
        key='predictions_alias',
        help='Le champion fixe les positions ; le challenger prédit en parallèle, sans position.',
    )
    try:
        frame = tables.predictions_frame(client.predictions(model, alias))
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if frame.empty:
        st.info(
            f'Aucune prévision `{alias}` pour `{model}` : pas de version sous cet alias, '
            "ou le flow `predict-volatility` n'est pas encore passé."
        )
        return
    bougie = frame['Bougie'].max().strftime('%d/%m/%Y %H:%M UTC')
    st.caption(
        f'Prévisions à {int(frame["Horizon (h)"].iloc[0])} h faites à la clôture de la bougie '
        f'{bougie}, version {int(frame["Version"].iloc[0])}. La référence est la prévision naïve.'
    )
    percent = st.column_config.NumberColumn(format='percent')
    st.dataframe(
        frame,
        hide_index=True,
        width='stretch',
        column_config={
            'Bougie': st.column_config.DatetimeColumn(format='DD/MM/YYYY HH:mm'),
            'Prévision': percent,
            'Référence': percent,
            'Écart': percent,
            # long/short (rendement) : poids négatifs = vente ; ciblage (volatilité) : 0 à 1
            'Position': st.column_config.NumberColumn(
                format='percent', help='Part du capital ; négatif = position vendeuse.'
            ),
        },
    )
    st.bar_chart(
        frame.set_index('Symbole')[['Prévision', 'Référence']], stack=False, y_label='prévision', height=300
    )


def models_page(client: api.ApiClient, operator: str) -> None:
    """Alias, santé quotidienne, registre, promotion manuelle et retour arrière."""
    left, right = st.columns(2)
    name = left.selectbox('Modèle', api.MODELS, key='models_model')
    days = right.slider('Jours de santé', 7, 90, 14, key='models_days')
    try:
        info = client.model(name, days=days)
    except api.ApiError as exc:
        st.error(str(exc))
        return
    aliases = info['aliases']
    for column, alias_name in zip(st.columns(3), ('champion', 'challenger', 'precedent'), strict=True):
        version = aliases.get(alias_name)
        column.metric(
            {'precedent': 'Précédent'}.get(alias_name, alias_name.capitalize()),
            f'v{version}' if version else '—',
        )

    st.subheader('Santé quotidienne')
    health_rows = tables.health_frame(info['health'])
    if health_rows.empty:
        st.info(
            "Pas encore de santé calculée : il faut au moins un jour de prévisions dont l'horizon est "
            'clôturé (flow `feedback-volatility`).'
        )
    else:
        live = health_rows[health_rows['Mode'] == 'live']
        if not live.empty:
            st.line_chart(
                live.set_index('Jour')[['Valeur', 'Référence']],
                height=260,
                y_label=str(live['Métrique'].iloc[-1]),
            )
        st.dataframe(health_rows, hide_index=True, width='stretch')

    st.subheader('Registre')
    events = tables.events_frame(info['events'])
    if events.empty:
        st.info('Aucun événement : aucune version enregistrée pour ce modèle.')
    else:
        st.dataframe(
            events,
            hide_index=True,
            width='stretch',
            column_config={'Date': st.column_config.DatetimeColumn(format='DD/MM/YYYY HH:mm')},
        )

    promote_col, rollback_col = st.columns(2)
    with promote_col, st.form('promote'):
        st.subheader('Promotion manuelle')
        challenger = aliases.get('challenger')
        version = st.number_input('Version', min_value=1, step=1, value=int(challenger) if challenger else 1)
        reason = st.text_input(
            'Raison',
            max_chars=400,
            key='promote_reason',
            placeholder='ex. niveau 1 validé, niveau 2 dans le bruit des frais',
        )
        if st.form_submit_button('Promouvoir en champion', type='primary'):
            problem = missing(reason)
            if problem:
                st.warning(problem)
            else:
                act(
                    f'Promotion de v{int(version)}',
                    lambda: client.promote(
                        name, version=int(version), changed_by=operator, reason=reason.strip()
                    ),
                )
    with rollback_col, st.form('rollback'):
        st.subheader('Retour arrière')
        previous = aliases.get('precedent')
        st.write(f'Remet v{previous} en champion.' if previous else 'Aucune version `precedent`.')
        confirmed = st.checkbox('Je confirme le retour arrière', disabled=not previous)
        if st.form_submit_button('Revenir à la version précédente', disabled=not previous):
            if not confirmed:
                st.warning('Cocher la confirmation.')
            else:
                act(f'Retour arrière vers v{previous}', lambda: client.rollback(name))


# -- barre latérale : opérateur puis navigation -----------------------------------------------
try:
    client = api.ApiClient.from_env()
except ValueError as exc:
    st.error(f'Configuration invalide : {exc}')
    st.stop()

with st.sidebar:
    st.title('Pilotage')
    operator = st.selectbox(
        'Opérateur',
        OPERATORS,
        index=OPERATORS.index('default'),
        key='operator',
        help='Profil de démonstration, tracé avec chaque action (marche / arrêt, promotion).',
    )
    page = st.radio(
        'Navigation',
        list(PAGES),
        key='page',
        format_func=lambda name: f'{PAGES[name]} {name}',
    )
    st.divider()
    if st.button('Rafraîchir', icon=':material/refresh:', width='stretch'):
        st.rerun()
    st.caption(f'API : {client.base_url}')

st.html(project.header_html())
show_flash()

if page == 'Projet':  # présentation statique, affichée même API arrêtée
    project.render(st)
else:
    try:
        health = client.health()
    except api.ApiError as exc:
        st.error(f'{exc}. Démarrer la stack : `docker compose up -d --wait`.')
    else:
        if page == 'Service':
            service_page(client, health, operator)
        elif page == 'Prévisions':
            predictions_page(client)
        else:
            models_page(client, operator)

client.close()

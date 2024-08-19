import os

import streamlit as st

from config import LANGUAGE, PROJECT_CONFIGS
from funcs.common import list_files_without_extension
from funcs.gui import (
    archive_management_block,
    show_chat_archive_block,
    show_completion_archive_block
)
from funcs.language import reverse_settings_translation_to_variable, translate_settings_variable_name


PROJECT_SETTINGS = PROJECT_CONFIGS.get('settings')
REFRESH_INTERVAL = PROJECT_SETTINGS.get('refresh_interval')


def initialize_sessions(page_prefix):
    if page_prefix not in st.session_state:
        st.session_state[page_prefix] = {}


def show():
    page_prefix = 'archive_management'

    initialize_sessions(page_prefix=page_prefix)

    st.title(f"{LANGUAGE.get('archive_management')}")
    st.caption(
        f"{LANGUAGE.get('archive_management_caption').format(chat=LANGUAGE.get('chat'),completion=LANGUAGE.get('completion'))}")

    archive_types = PROJECT_SETTINGS.get('prompt_types')
    translated_archive_types = translate_settings_variable_name(archive_types)

    with st.container(border=True):
        st.subheader(f"{LANGUAGE.get('select_an_archive_type')}")
        archive_type = st.radio(
            label='Select a type',
            options=translated_archive_types,
            horizontal=True,
            label_visibility='collapsed',
            key=f'{page_prefix}-radio-select-archive-type')

        archive_type = reverse_settings_translation_to_variable(archive_type)

    with st.container(border=True):
        archive_folder = os.path.join(
            PROJECT_SETTINGS.get('customized_data_folder'),
            PROJECT_SETTINGS.get(f'{str(archive_type).lower()}_archive_folder'))
        os.makedirs(archive_folder, exist_ok=True)
        if len(list_files_without_extension(archive_folder)) == 0:
            st.warning(
                f"{LANGUAGE.get('no_archive_found').format(archive_folder=archive_folder)}")
            st.stop()
        selected_chat_file, archive_data_json = archive_management_block(
            archive_folder=archive_folder, widget_prefix=page_prefix)

    with st.container(border=True):
        if archive_type.lower() == 'chat':
            show_chat_archive_block(selected_chat_file, archive_data_json)
        else:
            show_completion_archive_block(
                page_prefix,
                archive_data_json,
                selected_chat_file)

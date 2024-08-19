import streamlit as st

from config import PROJECT_CONFIGS, LANGUAGE
from funcs.gui import prompt_repository_block
from funcs.language import translate_settings_variable_name, reverse_settings_translation_to_variable

PROJECT_SETTINGS = PROJECT_CONFIGS.get('settings')


def manage():
    page_prefix = 'prompt_management'
    st.title(f"{LANGUAGE.get('prompt_management')}")
    st.caption(
        f"{LANGUAGE.get('prompt_management_caption').format(chat=LANGUAGE.get('chat'), completion=LANGUAGE.get('completion'))}")

    prompt_types = PROJECT_SETTINGS.get('prompt_types')
    translated_prompt_types = translate_settings_variable_name(prompt_types)

    with st.container(border=True):
        st.subheader(f"{LANGUAGE.get('select_a_prompt_type')}")
        prompt_type = st.radio(
            label='Select a prompt type',
            options=translated_prompt_types,
            horizontal=True,
            label_visibility='collapsed',
            key=f'{page_prefix}-radio-select-archive-type')
        prompt_type = reverse_settings_translation_to_variable(prompt_type).capitalize()
    with st.container(border=True):
        prompt_repository_block(widget_key_prefix=page_prefix, prompt_type=prompt_type)

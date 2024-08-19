import pyperclip

import streamlit as st

from config import LANGUAGE, LLMS_CONFIGS
from funcs.common import current_timestamp_to_string
from funcs.gui import (
    load_archive_block,
    model_parameter_setting_block,
    query_available_models,
    save_content_block,
    set_prompt_block,
    show_completion_archive_block,
    show_model_description
)
from funcs.llm import (
    identify_provider_by_model_alias,
    is_paid_model,
    LLM_completion_service,
    show_model_proxy_status
)


COMPLETION_CONFIGS = LLMS_CONFIGS.get('completion')
LLMs = LLMS_CONFIGS.get('LLMs')


def initialize_sessions(page_prefix):
    if page_prefix not in st.session_state:
        st.session_state[page_prefix] = {}
    if "completion" not in st.session_state[page_prefix]:
        st.session_state[page_prefix]["completion"] = []

    if "ollama_status" not in st.session_state[page_prefix]:
        st.session_state[page_prefix]["ollama_status"] = None


def show():
    page_prefix = 'completion_bot'
    initialize_sessions(page_prefix=page_prefix)

    with st.sidebar:
        st.markdown(
            f"# 🤖 {LANGUAGE.get('completion')} {LANGUAGE.get('model')}  {LANGUAGE.get('configuration')}")

        model_list = query_available_models(
            st.session_state[page_prefix], 'completion')
        model_alias = st.selectbox(
            label=LANGUAGE.get('model_alias_label'),
            options=model_list,
            key=f'{page_prefix}-select-model')

        provider = identify_provider_by_model_alias(model_alias)

        max_completions = st.slider(
            label=LANGUAGE.get('max_number_of_completions'),
            min_value=1,
            max_value=10,
            value=COMPLETION_CONFIGS.get('max_completions'),
            key=f'{page_prefix}-slider-max-completions'
        )
        st.write(LANGUAGE.get('max_number_of_completions_description'))

    load_archive_block(
        'completion',
        session_state=st.session_state[page_prefix])

    st.title(
        f"💬 a `{model_alias}` {LANGUAGE.get('completion_bot')}{is_paid_model(model_alias)}")
    proxy_status_str = show_model_proxy_status(provider, model_alias)
    if proxy_status_str:
        st.caption(proxy_status_str)

    show_model_description(provider, model_alias)

    model_parameters = model_parameter_setting_block(
        page_prefix=page_prefix, model_alias=model_alias)

    prompt = set_prompt_block(
        widget_key_prefix=page_prefix,
        prompt_type='Completion')

    if prompt == '':
        st.error(LANGUAGE.get('no_empty_prompt'), icon='🚨')
        # st.stop()

    generation_button = st.button(
        label=LANGUAGE.get('generate'),
        key=f'{page_prefix}-button-generate'
    )

    if generation_button:
        st.write(
            f'`{model_alias}`: {LANGUAGE.get("generating_content_notice").format(prompt=prompt[0:50])}...')
        with st.spinner(LANGUAGE.get('generating_content')):
            with st.container(border=True):
                response = st.write_stream(
                    LLM_completion_service(
                        model_alias=model_alias,
                        prompt=prompt,
                        model_parameters=model_parameters),
                )
                create_time = current_timestamp_to_string()
                completion_data = {
                    'prompt': prompt,
                    'content': response,
                    'timestamp': create_time,
                    'model': LLMs[provider]['models'][model_alias],
                    'parameters': model_parameters}
                generated_content_count = len(
                    st.session_state[page_prefix]["completion"])
                if generated_content_count < max_completions:
                    st.session_state[page_prefix]["completion"].append(
                        completion_data)
                else:
                    st.warning(
                        LANGUAGE.get('reach_max_completions_notice'),
                        icon='⚠️')
                    st.session_state[page_prefix]["completion"].pop(0)
                    st.session_state[page_prefix]["completion"].append(
                        completion_data)

                pyperclip.copy(response)
                st.success(
                    LANGUAGE.get('content_generated_notice').format(time=create_time),
                    icon='✅')

    if len(st.session_state[page_prefix]["completion"]) > 0:
        st.write('---')
        st.subheader(
            LANGUAGE.get('show_generated_content_title').format(
                number=len(
                    st.session_state[page_prefix]["completion"]),
                completion=LANGUAGE.get('completion')))
        show_completion_archive_block(
            widget_prefix=page_prefix,
            archive_data_json=st.session_state[page_prefix]["completion"])

    save_content_block(
        widget_key_prefix=page_prefix,
        model_name=LLMs[provider]['models'][model_alias],
        session_state=st.session_state[page_prefix],
        content_type='completion')

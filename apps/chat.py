import streamlit as st

from config import LANGUAGE, LLMS_CONFIGS, PROJECT_CONFIGS
from funcs.common import current_timestamp_to_string
from funcs.gui import (
    load_archive_block,
    model_parameter_setting_block,
    query_available_models,
    save_content_block,
    set_prompt_block,
    show_model_description
)
from funcs.llm import (
    build_chat_history,
    identify_provider_by_model_alias,
    is_paid_model,
    LLM_chat_service,
    show_model_proxy_status
)


PROJECT_SETTINGS = PROJECT_CONFIGS.get('settings')
CHAT_CONFIGS = LLMS_CONFIGS.get('chat')
LLMs = LLMS_CONFIGS.get('LLMs')


def initialize_sessions(page_prefix):
    if page_prefix not in st.session_state:
        st.session_state[page_prefix] = {}
    if "chat" not in st.session_state[page_prefix]:
        st.session_state[page_prefix]["chat"] = []
    if "model" not in st.session_state[page_prefix]:
        st.session_state[page_prefix]["model"] = ""
    if "ollama_status" not in st.session_state[page_prefix]:
        st.session_state[page_prefix]["ollama_status"] = None


def show():
    page_prefix = 'chat_bot'

    initialize_sessions(page_prefix=page_prefix)

    with st.sidebar:
        st.markdown(
            f"# 🤖 {LANGUAGE.get('chat')} {LANGUAGE.get('model')} {LANGUAGE.get('configuration')}")

        model_list = query_available_models(
            st.session_state[page_prefix], 'chat')
        model_alias = st.selectbox(
            label=f"{LANGUAGE.get('model_alias')}",
            options=model_list,
            key=f'{page_prefix}-select-model_alias')

        provider = identify_provider_by_model_alias(model_alias)
        is_baidu_model = True if provider == 'baidu' else False

        conversation_round = st.slider(
            label=f"{LANGUAGE.get('conversation_round')}",
            min_value=1,
            max_value=10,
            value=CHAT_CONFIGS.get('conversation_round'),
            step=1,
            key=f'{page_prefix}_slider_conversation_rounds')
        st.write(f"{LANGUAGE.get('conversation_round_notice')}")

    load_archive_block('chat', session_state=st.session_state[page_prefix])

    st.title(
        f"💬 a `{model_alias}` {LANGUAGE.get('chat_bot')}{is_paid_model(model_alias)}")

    proxy_status_str = show_model_proxy_status(provider, model_alias)
    if proxy_status_str:
        st.caption(proxy_status_str)

    show_model_description(provider, model_alias)

    model_parameters = model_parameter_setting_block(
        page_prefix=page_prefix, model_alias=model_alias)

    system_prompt = set_prompt_block(
        widget_key_prefix=page_prefix,
        prompt_type='Chat')

    if is_baidu_model:
        model_parameters['system'] = system_prompt

    for message in st.session_state[page_prefix]["chat"]:
        with st.chat_message(message["role"]):
            if message.get(
                    'model',
                    None):  # if 'model' key exists, for compatibility with old chat archives
                st.markdown(
                    message["content"] +
                    f"\n\n`@{message['timestamp']}` - `{message['model']}`")
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        current_timestamp_string = current_timestamp_to_string()
        st.session_state[page_prefix]["chat"].append(
            {
                "role": "user",
                "content": prompt,
                "model": LLMs[provider]['models'][model_alias],
                "timestamp": current_timestamp_string})
        with st.chat_message("user"):
            st.markdown(
                prompt +
                f"\n\n`@{current_timestamp_string}` - `{model_alias}`")

        with st.chat_message("assistant"):
            with st.spinner(f"{LANGUAGE.get('querying')} `{model_alias}`..."):
                historical_messages = build_chat_history(
                    current_model=model_alias,
                    system_prompt=system_prompt,
                    messages=st.session_state[page_prefix]["chat"],
                    conversation_round=conversation_round,
                    keep_system_prompt=not is_baidu_model)

                response = st.write_stream(
                    LLM_chat_service(
                        model_alias=model_alias,
                        messages=historical_messages,
                        model_parameters=model_parameters
                    ))

        st.session_state[page_prefix]["chat"].append(
            {
                "role": "assistant",
                "content": response,
                "model": LLMs[provider]['models'][model_alias],
                'system': system_prompt,
                "timestamp": current_timestamp_to_string()})
        # st.rerun()

    save_content_block(
        widget_key_prefix=page_prefix,
        model_name=LLMs[provider]['models'][model_alias],
        session_state=st.session_state[page_prefix],
        content_type='chat')

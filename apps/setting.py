import streamlit as st

from config import (
    LANGUAGE,
    LLMS_CONFIGS,
    PROJECT_CONFIGS
)

from funcs.gui import (
    archive_file_naming_block,
    backup_restore_block,
    customize_request_params_block,
    enable_provider_block,
    set_conversation_round_block,
    set_data_folder_location_block,
    set_language_block,
    set_max_number_of_completions,
    provider_llms_settings_block,
    set_proxy_block,
    set_refresh_interval_block,
    set_web_port,
    show_llms_request_params_block,
    set_verbose
)

PROJECT_SETTINGS = PROJECT_CONFIGS.get('settings')

LLMs = LLMS_CONFIGS.get('LLMs')
CHAT_CONFIGS = LLMS_CONFIGS.get('chat')
COMPLETION_CONFIGS = LLMS_CONFIGS.get('completion')

header_style = """
    <style>
     h2 {
        background-color: #BABEC7;
        color: white;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """


def setting():
    # Apply the CSS style to the header
    st.markdown(header_style, unsafe_allow_html=True)
    page_prefix = 'settings'

    st.title(LANGUAGE.get('settings'))
    st.caption(LANGUAGE.get('settings_caption'))

    st.header(LANGUAGE.get('common_settings'))

    with st.container(border=True):
        set_language_block(page_prefix=page_prefix)

        st.markdown('---')
        set_web_port(page_prefix=page_prefix)

        st.markdown('---')
        set_data_folder_location_block(page_prefix=page_prefix)

        st.markdown('---')
        archive_file_naming_block(page_prefix)

        st.markdown('---')
        set_refresh_interval_block(page_prefix=page_prefix)

        st.markdown('---')
        set_proxy_block(page_prefix=page_prefix)

    backup_restore_block(page_prefix=page_prefix)

    st.header(f"{LANGUAGE.get('llm_settings')}")

    with st.container(border=True):
        enable_provider_block(page_prefix=page_prefix)

        st.markdown('---')
        set_conversation_round_block(page_prefix=page_prefix)

        st.markdown('---')
        set_max_number_of_completions(page_prefix=page_prefix)

        st.markdown('---')
        set_verbose(page_prefix=page_prefix)

        st.markdown('---')
        show_llms_request_params_block(page_prefix=page_prefix)

        st.markdown('---')
        customize_request_params_block(page_prefix=page_prefix)

    provider_llms_settings_block(page_prefix=page_prefix)

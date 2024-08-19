import streamlit as st

from apps import chat, archive, completion, prompt, setting
from config import PROJECT_CONFIGS, LANGUAGE
from funcs.common import check_requirements
from main import run_app


def main():
    page_prefix = 'app'

    project_name = PROJECT_CONFIGS.get("settings").get("project_name")
    version = '1.0'

    if PROJECT_CONFIGS:
        st.set_page_config(
            page_title=f'{project_name}',
            page_icon="🧊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        pages = {
            LANGUAGE.get('chat'): chat.show,
            LANGUAGE.get('completion'): completion.show,
            LANGUAGE.get('load_archive'): archive.show,
            LANGUAGE.get('prompt_management'): prompt.manage,
            LANGUAGE.get('settings'): setting.setting
        }

        check_requirements_result, check_requirements_message = check_requirements('requirements.txt')
        if not check_requirements_result:
            st.error(check_requirements_message, icon="🚨")
            st.stop()

        with st.sidebar:
            st.title(project_name)
            st.caption(version)
            option = st.radio(
                label=f"{LANGUAGE.get('select_a_function')}:",
                options=list(pages.keys()),
                key=f'{page_prefix}_radio_select_page'
            )
            st.markdown('---')
            col1, col2 = st.columns(2)
            with col1:
                if st.button(label=f"{LANGUAGE.get('reload_app')}",
                             key=f'{page_prefix}-button-reload-toml'):
                    run_app(reload=True)

        pages.get(option, setting.setting)()

    else:
        st.error(LANGUAGE.get('unable_load_configs'), icon="🚨")


if __name__ == '__main__':
    main()

import os
import subprocess

from config import PROJECT_CONFIGS
from funcs.common import find_free_port, find_streamlit_executable, print_to_console


def run_app(reload=True):
    try:
        streamlit_executable = find_streamlit_executable()
        streamlit_app_path = os.path.join(os.path.dirname(__file__), 'app.py')
        if PROJECT_CONFIGS.get('settings').get('random_web_port'):
            port = find_free_port()
            if reload:
                print_to_console(f"Try to reload Streamlit app on http://localhost:{port}", 'main.py')
            else:
                print_to_console(f"Streamlit app is running on http://localhost:{port}", 'main.py')
            subprocess.run([streamlit_executable, "run", streamlit_app_path,
                            "--server.port", str(port)])
        else:
            port = PROJECT_CONFIGS.get('settings').get('web_port')
            if reload:
                print_to_console(f"Try to reload Streamlit app on http://localhost:{port}", 'main.py')
                os.execl(streamlit_executable, streamlit_executable, "run", streamlit_app_path, "--server.port",
                         str(port))
            else:
                subprocess.run([streamlit_executable, "run", streamlit_app_path,
                                "--server.port", str(port)])
                print_to_console(f"Streamlit app is running on http://localhost:{port}", 'main.py')
    except Exception as e:
        print_to_console(f"Error: {e}", 'main.py')
        raise e


if __name__ == '__main__':
    run_app(False)

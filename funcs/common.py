import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from contextlib import closing
from datetime import datetime

import httpx
import tomlkit
from send2trash import send2trash


def add_unique_item_to_list(new_item, item_list):
    if new_item not in item_list:
        item_list.append(new_item)
    return item_list


def archive_file_naming(
        generation_type: str = None,
        model: str = None,
        pattern: str = 'default'):
    if generation_type is None or model is None:
        raise ValueError(
            'Type and model must be provided when using a custom naming rule.')
    if pattern == 'default':
        name = '[{timestamp}]-[{type}]-[{model}]'
    else:
        name = pattern
    timestamp = current_timestamp_to_string('%Y-%m-%d_%H-%M-%S')
    name = name.replace(
        '{timestamp}',
        timestamp).replace(
        '{type}',
        generation_type).replace(
        '{model}',
        model)
    return name + '.json'


def backup_file_name(replace_timestamp=True):
    file_name = "backup_(timestamp).zip"
    if replace_timestamp:
        file_name = file_name.replace(
            "timestamp", current_timestamp_to_string('%Y-%m-%d_%H-%M-%S'))
    return file_name


def backup_operation(
        chat_archive_folder,
        completion_archive_folder,
        project_config_toml,
        llm_settings_toml,
        backup_folder):
    if not (os.path.exists(chat_archive_folder) and os.path.exists(completion_archive_folder)
            and os.path.exists(project_config_toml) and os.path.exists(llm_settings_toml)):
        return "Error: One or more paths do not exist."

    try:
        os.makedirs(backup_folder, exist_ok=True)
        backup_file_path = os.path.join(
            backup_folder, f'{backup_file_name(True)}')

        with zipfile.ZipFile(backup_file_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            for root, dirs, files in os.walk(chat_archive_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    backup_zip.write(
                        file_path, os.path.relpath(
                            file_path, os.path.dirname(chat_archive_folder)))

            for root, dirs, files in os.walk(completion_archive_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    backup_zip.write(
                        file_path, os.path.relpath(
                            file_path, os.path.dirname(completion_archive_folder)))

            backup_zip.write(
                project_config_toml,
                os.path.basename(project_config_toml))
            backup_zip.write(
                llm_settings_toml,
                os.path.basename(llm_settings_toml))

        return True, backup_file_path

    except Exception as e:
        return False, e


def blank_LLM_settings() -> dict:
    llm_settings = {
        'settings': {
            'free_models': [],
            'paid_models': [],
            'exclude_non_chat_models': [],
            'exclude_non_completion_models': [],
            'use_proxy_models': []},
        'models': {},
        'api_keys': {},
        'base_urls': {},
        'descriptions': {},
        'request_parameters': {}
    }
    return llm_settings


def calculate_text_area_height(text, line_height=30, word_per_line=50):
    if text == "" or text is None:
        return 50
    lines = text.split('\n')
    num_lines = len(lines)
    content_lines = 0
    for line in lines:
        content_lines += len(line) // word_per_line
    return max(num_lines, content_lines) * line_height


def check_requirements(requirements_file):
    not_installed_packages = []
    version_mismatched_packages = []

    with open(requirements_file, 'r') as file:
        for line in file:
            # 处理行中的注释或空行
            line = line.split('#')[0].strip()
            if not line:
                continue

            if '==' in line:
                package, version = line.split('==')
            elif '>=' in line:
                package, version = line.split('>=')
            else:
                package = line
                version = None

            package = package.strip()
            if version:
                version = version.strip()

            package_name = package.split('[')[0]

            if importlib.util.find_spec(package_name) is None:
                not_installed_packages.append(package)
            else:
                try:
                    installed_version = importlib.metadata.version(
                        package_name)
                    if version and installed_version < version:
                        version_mismatched_packages.append(
                            f"{package_name} (installed: {installed_version}, required: >= {version})")
                except importlib.metadata.PackageNotFoundError:
                    not_installed_packages.append(package)

    if not_installed_packages or version_mismatched_packages:
        error_message = "The following packages are not installed or have version mismatches:\n"
        if not_installed_packages:
            error_message += "Packages not installed: " + \
                             ', '.join(not_installed_packages) + "\n"
        if version_mismatched_packages:
            error_message += "Packages with version mismatches: " + \
                             ', '.join(version_mismatched_packages) + "\n"
        return False, error_message
    else:
        return True, "All packages are installed and have the correct versions."


def coloring_content(color: str, txt: str, bold: bool = False) -> str:
    if color.lower() not in ['red', 'green', 'yellow', 'blue']:
        raise ValueError('Type of color is not supported.')
    else:
        if bold:
            return f'<span style="color: {color};"><strong>{txt}</strong></span>'
        else:
            return f'<span style="color: {color};">{txt}</span>'


def convert_punctuation_to_comma(txt: str) -> str:
    # remove blank space
    txt = re.sub(r"\s+", "", txt)
    converted = re.sub(r"[；;，、]", ",", txt)
    return converted


def convert_timestamp_to_datetime(
        timestamp: int,
        format_pattern="%Y-%m-%d %H:%M:%S"):
    return time.strftime(format_pattern, time.localtime(timestamp))


def create_LLMs_settings_toml(toml_path):
    llm_settings = blank_LLM_settings()
    toml_configs = {
        'chat': {
            'conversation_round': 6,
            'chat_prompt_repository': [
                "Serve me as a programming and writing assistant.",
                "You will be provided with statements, and your task is to convert them to standard English.",
                "Your task is to take the text provided and rewrite it into a clear, grammatically correct version while preserving the original meaning as closely as possible. Correct any spelling mistakes, punctuation errors, verb tense issues, word choice problems, and other grammatical mistakes."],
        },
        'completion': {
            'max_completions': 3,
            'completion_prompt_repository': ["Why is the sky blue?"],
        },
        'LLMs': {
            'settings': {
                'enable_openai': True,
                'enable_openai_compatible': True,
                'enable_baidu': True,
                'enable_qwen': True,
                'enable_ollama': True,
            },
            'ollama': llm_settings,
            'openai': llm_settings,
            'openai_compatible': llm_settings,
            'baidu': llm_settings,
            'qwen': llm_settings,
        },
    }
    with open(toml_path, 'w', encoding='utf-8') as f:
        f.write(tomlkit.dumps(toml_configs))


def create_project_configs_toml(toml_path):
    toml_configs = {
        'settings': {
            'project_name': 'AssistantBot',
            'version': '1.0',
            'language': 'en_US',
            'random_web_port': True,
            'web_port': 8501,
            'default_data_folder': 'data',
            'customized_data_folder': 'data',
            'chat_archive_folder': "chats",
            'completion_archive_folder': "completions",
            'backup_folder': "backups",
            'refresh_interval': 2,
            'prompt_types': ['Chat', 'Completion'],
            'archive_file_naming_pattern': '[timestamp]-[type]-[model]',
            'add_common_request_parameters_to_new_model': True,
            'verbose_mode': False,
            'proxy': {
                'http_proxy': 'socks5://127.0.0.1:10808',
                'https_proxy': 'socks5://127.0.0.1:10808',
            },
            'customized_request_parameters': {
                'new_boolean_param': {
                    'type': 'bool',
                    'default': True,
                },
                'new_float_param': {
                    'type': 'float',
                    'default': -2.0,
                    'min': -2.0,
                    'max': 2.0,
                    'step': 0.1,
                },
            }

        }}
    with open(toml_path, 'w', encoding='utf-8') as f:
        f.write(tomlkit.dumps(toml_configs))


def current_timestamp_to_string(format_pattern="%Y-%m-%d %H:%M:%S") -> str:
    return convert_timestamp_to_datetime(int(time.time()), format_pattern)


def delete_dict_key_and_value(dict_to_delete, variable_name):
    """
    Deletes all the keys and values of dict_to_delete that equal variable_name.

    Parameters:
    - dict_to_delete (dict): The dictionary from which the key-value pair is to be deleted.
    - variable_name (str): The key or value to be deleted from the dictionary.

    Returns:
    - dict: The dictionary with the specified key-value pair deleted.

    Examples:
    >>> delete_dict_key_and_value({'a': 1, 'b': 2, 'c': 3}, 'b')
    {'a': 1, 'c': 3}
    >>> delete_dict_key_and_value({'a': 1, 'b': 2, 'c': 3}, 'd')
    {'a': 1, 'b': 2, 'c': 3}
    >>> delete_dict_key_and_value({'a': 1, 'b': 2, 'c': {'a': 1, 'b': 2, 'c': 3}}, 'a')
    {'b': 2, 'c':  {'b': 2, 'c': 3}}
    >>> delete_dict_key_and_value({'a': ['a','b', 'c'], 'b': 2, 'c': 3}, 'b')
    {'a': ['a','c'], 'c': 3}
    """

    def recursive_delete(d, var_name):
        if isinstance(d, dict):
            return {
                k: recursive_delete(
                    v,
                    var_name) for k,
                v in d.items() if k != var_name and v != var_name}
        elif isinstance(d, list):
            return [item for item in d if item != var_name]
        else:
            return d

    return recursive_delete(dict_to_delete, variable_name)


def dump_content_to_json(content_to_dump):
    return json.dumps(content_to_dump, indent=4, ensure_ascii=False)


def empty_dir(target_dir):
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def filter_models(models, exclusions):
    """
    Filters out models based on a list of exclusion patterns.

    Parameters:
    - models (list of str): A list of model names.
    - exclusions (list of str): A list of patterns to exclude from the models list.

    Returns:
    - list of str: A list of models filtered based on the exclusion patterns.

    Examples:
    >>> filter_models(['gpt-3', 'bert', 'roberta'], [])
    ['gpt-3', 'bert', 'roberta']
    >>> filter_models(['gpt-3', 'bert', 'roberta'], ['bert'])
    ['gpt-3', 'roberta']
    >>> filter_models(['gpt-3', 'bert-large', 'bert-base', 'roberta'], ['bert', 'roberta'])
    ['gpt-3']
    >>> filter_models([], ['bert', 'roberta'])
    []
    >>> filter_models([], [])
    []
    >>> filter_models(['gpt-3', 'bert-large', 'bert-base', 'roberta'], ['large'])
    ['gpt-3', 'bert-base', 'roberta']
    >>> filter_models(['nomic-embed-text:latest', 'qwen2:7b', 'erine-4-8k', 'gpt-3.5-turbo'], ['nomic-embed-text', 'gpt'])
    ['qwen2:7b', 'erine-4-8k']
    """
    return [m for m in models if not any(
        exclude in m for exclude in exclusions)]


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_streamlit_executable():
    python_executable = sys.executable

    if os.name == 'nt':  # Windows
        scripts_dir = os.path.join(
            os.path.dirname(python_executable), 'Scripts')
        streamlit_executable = os.path.join(scripts_dir, 'streamlit.exe')
    else:  # Unix-based systems (Linux, macOS)
        scripts_dir = os.path.join(os.path.dirname(python_executable), 'bin')
        streamlit_executable = os.path.join(scripts_dir, 'streamlit')

    if not os.path.exists(streamlit_executable):
        streamlit_executable = shutil.which('streamlit')

        if not streamlit_executable:
            raise FileNotFoundError(
                "Streamlit executable not found. Make sure Streamlit is installed and available in PATH."
            )

    return streamlit_executable


def get_current_function_name():
    current_frame = inspect.currentframe()
    outer_frame = inspect.getouterframes(current_frame)[1]
    function_name = outer_frame[3]
    return function_name


def list_files_without_extension(directory, extension: str = None):
    if extension is not None:
        extension = extension.lower()
    return [os.path.splitext(f)[0] for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and
            (extension is None or os.path.splitext(f)[1].lower() == f'.{extension}')]


def load_config_file(configuration_file_path: str):
    with open(configuration_file_path, 'r', encoding='utf-8') as f:
        toml_configs = tomlkit.parse(f.read())
    return toml_configs


def load_json_file(path: str) -> json:
    with open(path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result


def move_all_subfolders_and_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    empty_dir(dest_dir)

    for item in os.listdir(src_dir):
        src_item_path = os.path.join(src_dir, item)
        shutil.move(src_item_path, dest_dir)


def open_directory(directory_path) -> None:
    if os.name == 'nt':
        try:
            subprocess.Popen(f'explorer "{directory_path}"')
        except Exception as e:
            raise Exception(
                f"An error occurred while opening the directory: {e}")
    else:
        # linux
        subprocess.Popen(['xdg-open', directory_path])


def print_to_console(info: str, funcs: str = None):
    log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
    if funcs:
        print(f"[{log_time}][{funcs}] {info}")
    else:
        print(f"[{log_time}] {info}")


def rename_dict_key_and_value(dict_to_rename, old_key, new_key):
    """
    Renames all the keys and values of dict_to_rename that equal old_name to new_name.

    Parameters:
    - dict_to_rename (dict): The dictionary in which the key-value pair is to be renamed.
    - old_name (str): The key or value to be renamed.
    - new_name (str): The new key or value to replace the old key or value.

    Returns:
    - dict: The dictionary with the specified key-value pair renamed.

    Examples:
    >>> rename_dict_key_and_value({'a': 1, 'b': 2, 'c': 3}, 'b', 'd')
    {'a': 1, 'd': 2, 'c': 3}
    >>> rename_dict_key_and_value({'a': 1, 'b': 2, 'c': 3}, 'd', 'e')
    {'a': 1, 'b': 2, 'c': 3}
    >>> rename_dict_key_and_value({'a': 1, 'b': 2, 'c': {'a': 1, 'b': 2, 'c': 3}}, 'a', 'd')
    {'d': 1, 'b': 2, 'c':  {'d': 1, 'b': 2, 'c': 3}}
    >>> rename_dict_key_and_value({'a': ['a','b', 'c'], 'b': 2, 'c': 3}, 'b', 'd')
    {'a': ['a','d', 'c'], 'd': 2, 'c': 3}
    """

    def recursive_rename(data, target, replacement):
        if isinstance(data, dict):
            return {replacement if k == target else k: recursive_rename(
                v, target, replacement) for k, v in data.items()}
        elif isinstance(data, list):
            return [replacement if item == target else recursive_rename(
                item, target, replacement) for item in data]
        else:
            return replacement if data == target else data

    return recursive_rename(dict_to_rename, old_key, new_key)


def restore_operation(
        chat_archive_folder,
        completion_archive_folder,
        project_config_toml,
        llm_settings_toml,
        backup_file):
    if not os.path.exists(backup_file):
        return False, "Error: Backup file does not exist."

    temp_folder = os.path.join(os.path.dirname(backup_file), 'temp')
    os.makedirs(temp_folder, exist_ok=True)
    try:
        with zipfile.ZipFile(backup_file, 'r') as backup_zip:
            backup_zip.extractall(temp_folder)

        chats_folder = os.path.join(temp_folder, 'chats')
        completions_folder = os.path.join(temp_folder, 'completions')

        if os.path.exists(chats_folder):
            shutil.copytree(
                chats_folder,
                chat_archive_folder,
                dirs_exist_ok=True)
        if os.path.exists(completions_folder):
            shutil.copytree(
                completions_folder,
                completion_archive_folder,
                dirs_exist_ok=True)

        shutil.copy2(
            os.path.join(temp_folder, os.path.basename(project_config_toml)),
            project_config_toml)
        shutil.copy2(
            os.path.join(temp_folder, os.path.basename(llm_settings_toml)),
            llm_settings_toml)

        shutil.rmtree(temp_folder)
        return True, ""
    except Exception as e:
        return False, e


def return_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    raise KeyError(
        f"[{get_current_function_name()}] Value '{value}' not found in the dictionary.")


def safe_delete(file_path):
    if not os.path.isfile(file_path):
        raise FileExistsError(
            f"[{current_timestamp_to_string()}][{get_current_function_name()}]Error: The path '{file_path}' is not a file or does not exist.")
    try:
        system = platform.system()
        if system == 'Windows' or system == 'Linux':
            send2trash(file_path)
        else:
            os.remove(file_path)
        return True
    except Exception as e:
        raise Exception(
            f"[{current_timestamp_to_string()}][{get_current_function_name()}]Error deleting file: {e}")


def save_config_file(configuration_file_path: str, toml_configs):
    with open(configuration_file_path, 'w', encoding='utf-8') as f:
        f.write(tomlkit.dumps(toml_configs))


def save_content_to_json_file(json_file_path, content_to_save):
    content = dump_content_to_json(content_to_save)

    with open(json_file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def split_a_list_to_two_lists(original_list):
    input_list = list(original_list)
    if len(input_list) <= 1:
        return input_list, []
    midpoint = (len(input_list) + 1) // 2
    list1 = input_list[:midpoint]
    list2 = input_list[midpoint:]
    return list1, list2


def validate_file_name_pattern(pattern: str) -> bool:
    if pattern is None or pattern == '':
        return False
    p_list = re.findall(r'\{(.*?)\}', pattern)
    if len(p_list) == 0:
        return False
    for p in p_list:
        if p not in ['timestamp', 'type', 'model']:
            return False
    return True


def validate_proxy(http_proxy, https_proxy):
    url1 = 'http://www.google.com'
    url2 = 'https://www.google.com'

    proxies = {
        "http://": http_proxy,
        "https://": https_proxy,
    }

    try:
        with httpx.Client(proxies=proxies, timeout=10) as client:
            http_response = client.get(url1)
            http_valid = http_response.status_code == 200

            https_response = client.get(url2)
            https_valid = https_response.status_code == 200

        return http_valid, https_valid
    except Exception as e:
        print_to_console(e, get_current_function_name())
        return False, False

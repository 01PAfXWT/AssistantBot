import os

from funcs.common import load_config_file, create_project_configs_toml, create_LLMs_settings_toml, print_to_console

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

PROJECT_SETTINGS_TOML_FULL_PATH = os.path.join(ROOT_DIRECTORY, 'configs.toml')

if not os.path.exists(PROJECT_SETTINGS_TOML_FULL_PATH):
    create_project_configs_toml(PROJECT_SETTINGS_TOML_FULL_PATH)
    print_to_console('new configs.toml created.', 'config.py')

PROJECT_CONFIGS = load_config_file(PROJECT_SETTINGS_TOML_FULL_PATH)

LANGUAGE_FOLDER = os.path.join(ROOT_DIRECTORY, 'locales')
language_file = os.path.join(LANGUAGE_FOLDER, PROJECT_CONFIGS.get('settings').get('language') + '.toml')
if not os.path.exists(language_file):
    raise FileNotFoundError(f"Language file '{language_file}' not found.")
LANGUAGE = load_config_file(language_file)

if PROJECT_CONFIGS.get('settings').get('customized_data_folder') == PROJECT_CONFIGS.get('settings').get(
        'default_data_folder'):
    ARCHIVE_FOLDER = os.path.join(ROOT_DIRECTORY, PROJECT_CONFIGS.get('settings').get('default_data_folder'))
else:
    ARCHIVE_FOLDER = PROJECT_CONFIGS.get('settings').get('customized_data_folder')

if not os.path.exists(ARCHIVE_FOLDER):
    os.mkdir(ARCHIVE_FOLDER)
    print_to_console('Data folder: `' + ARCHIVE_FOLDER + '` created.', 'config.py')

LLMs_SETTINGS_TOML_FULL_PATH = os.path.join(ARCHIVE_FOLDER, 'LLMs_configs.toml')
if not os.path.exists(LLMs_SETTINGS_TOML_FULL_PATH):
    create_LLMs_settings_toml(LLMs_SETTINGS_TOML_FULL_PATH)
    print_to_console('new LLMs.configs.toml created.', 'config.py')
LLMS_CONFIGS = load_config_file(LLMs_SETTINGS_TOML_FULL_PATH)

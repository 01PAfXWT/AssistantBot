import os

from config import LANGUAGE
from config import LANGUAGE_FOLDER
from funcs.common import get_current_function_name, return_key_by_value, load_config_file, save_config_file


def add_variable_to_language_files(variable_name: str, variable_value: str):
    v = generate_language_variable_name(variable_name)
    all_entries = os.listdir(LANGUAGE_FOLDER)
    language_files = [
        os.path.join(
            LANGUAGE_FOLDER,
            entry) for entry in all_entries if entry.endswith('.toml') and os.path.isfile(
            os.path.join(
                LANGUAGE_FOLDER,
                entry))]

    added_files_success = []
    for language_file in language_files:
        language_ = load_config_file(language_file)
        language_[v] = variable_value
        save_config_file(language_file, language_)
        added_files_success.append(os.path.basename(language_file))
    if len(added_files_success) > 0:
        info = LANGUAGE.get('add_request_parameter_description').format(
            parameter=variable_name, files=', '.join(added_files_success))
        return True, info
    else:
        info = LANGUAGE.get('add_variable_to_language_files_error').format(
            parameter=variable_name)
        return False, info


def delete_variable_in_language_files(variable_name: str):
    all_entries = os.listdir(LANGUAGE_FOLDER)
    language_files = [
        os.path.join(
            LANGUAGE_FOLDER,
            entry) for entry in all_entries if entry.endswith('.toml') and os.path.isfile(
            os.path.join(
                LANGUAGE_FOLDER,
                entry))]

    deleted_files_success = []
    for language_file in language_files:
        language_ = load_config_file(language_file)
        v = generate_language_variable_name(variable_name)
        if v in language_:
            del language_[v]
            save_config_file(language_file, language_)
            deleted_files_success.append(os.path.basename(language_file))
    if len(deleted_files_success) > 0:
        info = LANGUAGE.get('delete_request_parameter_description').format(
            parameter=variable_name, files=', '.join(deleted_files_success))
        return True, info
    else:
        info = LANGUAGE.get('request_parameter_not_found_in_language_files').format(
            parameter=variable_name)
        return False, info


def generate_language_variable_name(variable_name: str):
    return f"customized_request_param_{variable_name.lower()}_description"


def reverse_settings_translation_to_variable(translation: str) -> str:
    return reverse_translation_to_variable(
        LANGUAGE, translation)


def reverse_translation_to_variable(
        language_dict: dict,
        translation: str | list) -> str | list:
    if isinstance(translation, str):
        return return_key_by_value(language_dict, translation)
    elif isinstance(translation, list):
        translated_names = []
        for name in translation:
            translated_names.append(return_key_by_value(language_dict, name))
        return translated_names
    else:
        raise ValueError(
            f"[{get_current_function_name()}] wrong translation type.")


def translate_settings_variable_name(settings_name: str | list) -> str | list:
    return translate_variable(LANGUAGE, settings_name)


def translate_variable(
        language_dict: dict,
        variable_name: str | list) -> str | list | None:
    if isinstance(variable_name, str):
        if language_dict.get(variable_name.lower(), None) is not None:
            return language_dict.get(variable_name, None)
        else:
            raise ValueError(
                f"[{get_current_function_name()}] can not find `{variable_name}` in {language_dict}.")
    elif isinstance(variable_name, list):
        translated_names = []
        for name in variable_name:
            translated_names.append(
                translate_variable(
                    language_dict,
                    name.lower()))
        return translated_names

import json
import os
import time
from typing import Literal

import ollama
import pyperclip
import qianfan
import streamlit as st

from config import (
    PROJECT_CONFIGS,
    ROOT_DIRECTORY,
    PROJECT_SETTINGS_TOML_FULL_PATH,
    ARCHIVE_FOLDER,
    LLMS_CONFIGS,
    LLMs_SETTINGS_TOML_FULL_PATH,
    LANGUAGE,
    LANGUAGE_FOLDER)
from funcs.common import (
    save_config_file,
    current_timestamp_to_string,
    dump_content_to_json,
    save_content_to_json_file,
    calculate_text_area_height,
    add_unique_item_to_list,
    get_current_function_name,
    backup_file_name,
    backup_operation,
    restore_operation,
    print_to_console,
    list_files_without_extension,
    safe_delete,
    split_a_list_to_two_lists,
    convert_punctuation_to_comma,
    open_directory,
    move_all_subfolders_and_files,
    load_json_file,
    validate_proxy,
    archive_file_naming,
    validate_file_name_pattern,
    rename_dict_key_and_value,
    delete_dict_key_and_value)
from funcs.language import (
    translate_variable,
    reverse_translation_to_variable,
    delete_variable_in_language_files,
    add_variable_to_language_files,
    generate_language_variable_name)
from funcs.llm import (
    query_ollama_status,
    query_ollama_model_info,
    accept_system_prompt_baidu_models,
    identify_provider_by_model_alias,
    available_models_by_provider,
    list_baidu_models_by_SDK,
    list_openai_models,
    available_full_model_alias_list,
    available_model_aliases_by_provider_filtered_by_type,
    query_ollama_model_information_by_keyword,
    generate_ollama_model_architecture_description,
    available_providers,
    validate_model_info)
from main import run_app

PROJECT_SETTINGS = PROJECT_CONFIGS.get('settings')
REFRESH_INTERVAL = PROJECT_SETTINGS.get('refresh_interval')

CHAT_CONFIGS = LLMS_CONFIGS.get('chat')
COMPLETION_CONFIGS = LLMS_CONFIGS.get('completion')
LLMs = LLMS_CONFIGS.get('LLMs')

MODEL_PAYMENT_TYPES = [f"{LANGUAGE.get('paid')}", f"{LANGUAGE.get('free')}"]
BINARY_OPTIONS = [True, False]


class LLMModelParamsConfigUI:
    def __init__(self, prefix: str, model_alias: str, provider: str):
        self.prefix = f"{prefix}-{model_alias}-llm-model-parameters"
        self.provider = provider
        self.llm_configs = LLMs.get(provider)
        self.model = self.llm_configs['models'].get(model_alias)
        self.model_alias = model_alias

    def customized_request_parameters_block(self, param_name):
        prefix = self.prefix + '-customized-'
        param_info_dict = PROJECT_SETTINGS.get(
            'customized_request_parameters').get(param_name)
        param_type = param_info_dict.get('type')
        default = param_info_dict.get('default')
        description = LLMModelParamsConfigUI.get_supported_request_parameter_description(
            param_name)

        if param_type == 'bool':
            result = st.checkbox(
                label=description,
                value=default,
                key=f'{prefix}-checkbox-{param_name}'
            )
        elif param_type == 'int':
            result = st.slider(
                label=description,
                value=int(default),
                min_value=int(param_info_dict.get('min', 0)),
                max_value=int(param_info_dict.get('max', 1024)),
                step=int(param_info_dict.get('step', 1)),
                key=f'{prefix}-number-{param_name}'
            )
        elif param_type == 'float':
            result = st.slider(
                label=description,
                value=default,
                min_value=param_info_dict.get('min', -2.0),
                max_value=param_info_dict.get('max', 2.0),
                step=param_info_dict.get('step', 0.1),
                key=f'{prefix}-number-{param_name}')
        else:
            result = st.text_input(
                label=description,
                value=default,
                key=f'{prefix}-text-{param_name}')
        return result

    @classmethod
    def get_built_in_request_parameter_list(cls):
        supported_parameters = []

        for method_name in dir(cls):
            if method_name.startswith("request_parameters_") and method_name.endswith(
                    "_block") and method_name != 'request_parameters_block':
                parameter_name = method_name.replace(
                    "request_parameters_", "").replace(
                    "_block", "")
                supported_parameters.append(parameter_name)
        return sorted(supported_parameters)

    @classmethod
    def get_customized_request_parameter_list(cls):
        return sorted(list(PROJECT_SETTINGS.get(
            'customized_request_parameters').keys()))

    @classmethod
    def get_supported_request_parameter_description(cls, parameter_name: str):
        built_in = cls.get_built_in_request_parameter_list()
        customized = cls.get_customized_request_parameter_list()
        if parameter_name in built_in:
            return LANGUAGE.get(f"{parameter_name}_description")
        elif parameter_name in customized:
            return LANGUAGE.get(
                generate_language_variable_name(parameter_name), None)

    @classmethod
    def get_supported_request_parameter_list(cls):
        built_in = cls.get_built_in_request_parameter_list()
        customized = cls.get_customized_request_parameter_list()
        return built_in + customized

    def initialize_model_parameters(self, left_col_configs: list,
                                    right_col_configs: list):
        built_in_params = self.get_built_in_request_parameter_list()
        customized_params = self.get_customized_request_parameter_list()

        model_parameters = {}
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            for config in left_col_configs:
                if config in built_in_params:
                    block_to_call = getattr(
                        self, f"request_parameters_{config}_block")
                    model_parameters[config] = block_to_call()
                elif config in customized_params:
                    block_to_call = getattr(
                        self, "customized_request_parameters_block")
                    model_parameters[config] = block_to_call(config)
                else:
                    error_info = LANGUAGE.get(
                        'unsupported_request_parameter').format(parameter=config)
                    st.error(error_info, icon='🚨')
                    raise ValueError(
                        f"[{get_current_function_name()}] {error_info}")

        with col2:
            for config in right_col_configs:
                if config in built_in_params:
                    block_to_call = getattr(
                        self, f"request_parameters_{config}_block")
                    model_parameters[config] = block_to_call()
                elif config in customized_params:
                    block_to_call = getattr(
                        self, "customized_request_parameters_block")
                    model_parameters[config] = block_to_call(config)
                else:
                    error_info = LANGUAGE.get(
                        'unsupported_request_parameter').format(parameter=config)
                    st.error(error_info, icon='🚨')
                    raise ValueError(
                        f"[{get_current_function_name()}] {error_info}")
        return model_parameters

    def model_parameters_block(self):
        st.write(
            f"{LANGUAGE.get('set_model_parameters').format(model_alias=self.model_alias, model=self.model)}")
        request_params = self.llm_configs.get(
            'request_parameters').get(self.model_alias, [])
        if len(request_params) > 0:
            left_col_configs, right_col_configs = split_a_list_to_two_lists(
                request_params)
            model_parameters = self.initialize_model_parameters(
                left_col_configs, right_col_configs)
        else:
            st.info(
                LANGUAGE.get('no_request_parameters_found').format(
                    model_alias=self.model_alias),
                icon='ℹ️')
            return {}

        if self.provider == 'baidu':
            model_parameters['max_tokens'] = 2048
        return model_parameters

    def request_parameters_disable_search_block(self):
        disable_search = st.checkbox(
            label=f"{LANGUAGE.get('disable_search_description')}",
            value=False,
            key=f'{self.prefix}-checkbox-disable-search'
        )
        return disable_search

    def request_parameters_enable_search_block(self):
        enable_search = st.checkbox(
            label=f"{LANGUAGE.get('enable_search_description')}",
            value=False,
            key=f'{self.prefix}-checkbox-enable-search'
        )
        return enable_search

    def request_parameters_frequency_penalty_block(self):
        frequency_penalty = st.slider(
            label=f"{LANGUAGE.get('frequency_penalty_description')}",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key=f'{self.prefix}-slider-frequency-penalty'
        )
        return frequency_penalty

    def request_parameters_max_output_tokens_block(self):
        max_output_tokens = st.slider(
            label=LANGUAGE.get('max_output_tokens_description'),
            min_value=2,
            max_value=2048,
            value=2048,
            step=1,
            key=f'{self.prefix}-slider-max-output-tokens'
        )
        return max_output_tokens

    def request_parameters_max_tokens_block(self):
        if self.provider == 'baidu':
            default = 2048
        elif self.provider == 'qwen':
            default = 2000
        else:
            default = 2000
        label = LANGUAGE.get('provider_default_value').format(
            provider=translate_variable(
                LANGUAGE, self.provider.lower()), default=default)
        max_tokens = st.slider(
            label=LANGUAGE.get('max_tokens_description') + label,
            min_value=2,
            max_value=8196,
            value=default,
            step=1,
            key=f'{self.prefix}-slider-max-tokens'
        )
        return max_tokens

    def request_parameters_num_ctx_block(self):
        max_num_ctx = query_ollama_model_information_by_keyword(
            self.llm_configs['models'][self.model_alias], 'context_length') / 1024

        num_ctx = st.slider(
            label=f"{LANGUAGE.get('num_ctx_description')}",
            min_value=1,
            max_value=int(max_num_ctx),
            value=int(max_num_ctx),
            step=1,
            key=f'{self.prefix}-slider-num-ctx'
        )
        return num_ctx

    def request_parameters_num_predict_block(self):
        max_num_predict = 128
        num_predict = st.slider(
            label=f"{LANGUAGE.get('num_predict_description')}",
            min_value=-2,
            max_value=max_num_predict,
            value=-1,
            step=1,
            key=f'{self.prefix}-slider-num-predict'
        )
        return num_predict

    def request_parameters_penalty_score_block(self):
        penalty_score = st.slider(
            label=f"{LANGUAGE.get('penalty_score_description')}",
            min_value=1.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key=f'{self.prefix}-slider-penalty-score'
        )
        return penalty_score

    def request_parameters_presence_penalty_block(self):
        presence_penalty = st.slider(
            label=f"{LANGUAGE.get('presence_penalty_description')}",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key=f'{self.prefix}-slider-presence-penalty'
        )
        return presence_penalty

    def request_parameters_repeat_penalty_block(self):
        repeat_penalty = st.slider(
            label=f"{LANGUAGE.get('repeat_penalty_description')}",
            min_value=0.0,
            max_value=2.0,
            value=1.1,
            step=0.1,
            key=f'{self.prefix}-slider-repeat-penalty'
        )
        return repeat_penalty

    def request_parameters_repetition_penalty_block(self):
        repetition_penalty = st.slider(
            label=f"{LANGUAGE.get('repetition_penalty_description')}",
            min_value=1.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key=f'{self.prefix}-slider-repetition-penalty'
        )
        return repetition_penalty

    def request_parameters_seed_block(self):
        seed = st.slider(
            label=f"{LANGUAGE.get('seed_description')}",
            min_value=1,
            max_value=2000,
            value=1234,
            step=1,
            key=f'{self.prefix}-slider-seed'
        )
        return seed

    def request_parameters_temperature_block(self):
        if self.provider == 'baidu' or self.provider == 'ollama':
            default = 0.8
            max_value = 1.0
        else:
            default = 1.0
            max_value = 2.0
        label = LANGUAGE.get('provider_default_value').format(
            provider=translate_variable(
                LANGUAGE, self.provider.lower()), default=default)
        temperature = st.slider(
            label=LANGUAGE.get('temperature_description').format(
                provider=translate_variable(
                    LANGUAGE,
                    self.provider.lower()),
                default=default) + label,
            min_value=0.1,
            max_value=max_value,
            value=default,
            step=0.1,
            key=f'{self.prefix}-slider-temperature')
        return temperature

    def request_parameters_top_k_block(self):
        if self.provider == 'ollama':
            default = 40
        elif self.provider == 'qwen':
            default = 50
        else:
            default = 40
        label = LANGUAGE.get('provider_default_value').format(
            provider=translate_variable(
                LANGUAGE, self.provider.lower()), default=default)
        top_k = st.slider(
            label=LANGUAGE.get('top_k_description').format(
                provider=translate_variable(
                    LANGUAGE,
                    self.provider.lower()),
                default=default) + label,
            min_value=0,
            max_value=100,
            value=default,
            step=10,
            key=f'{self.prefix}-slider-top-k')
        return top_k

    def request_parameters_top_p_block(self):
        if self.provider == 'openai' or self.provider == 'openai_compatible':
            default = 1.0
        elif self.provider == 'baidu' or self.provider == 'qwen':
            default = 0.8
        elif self.provider == 'ollama':
            default = 0.9
        else:
            default = 0.9

        label = LANGUAGE.get('provider_default_value').format(
            provider=translate_variable(
                LANGUAGE, self.provider.lower()), default=default)
        top_p = st.slider(
            label=LANGUAGE.get('top_p_description').format(
                provider=translate_variable(
                    LANGUAGE,
                    self.provider.lower()),
                default=default) + label,
            min_value=0.0,
            max_value=1.0,
            value=default,
            step=0.1,
            key=f'{self.prefix}-slider-top-p')
        return top_p


class ProviderLLMModelConfigUI:
    def __init__(self, prefix: str, model_alias: str, provider: str):
        self.prefix = prefix
        self.model_alias = model_alias
        self.provider = provider
        self.provider_llm_configs = LLMs.get(provider)
        self.legitimate_request_parameters = LLMModelParamsConfigUI.get_supported_request_parameter_list()

    def api_key_block(self):
        api_key = st.text_input(
            label=f"{LANGUAGE.get('openai_compatible_api_key')}",
            key=f'{self.prefix}-text-model-api-key',
            value=self.provider_llm_configs['api_keys'].get(
                self.model_alias, "") if self.model_alias != "" else "")
        return api_key

    def base_url_block(self):
        base_url = st.text_input(
            label=f"{LANGUAGE.get('openai_compatible_base_url')}",
            key=f'{self.prefix}-text-model-base-url',
            value=self.provider_llm_configs['base_urls'].get(
                self.model_alias, "") if self.model_alias != "" else "")
        return base_url

    def chat_supported_block(self):
        chat_supported = st.checkbox(
            label=f"{LANGUAGE.get('chat_support')}",
            value=False if self.model_alias in self.provider_llm_configs['settings'][
                'exclude_non_chat_models'] else True,
            key=f'{self.prefix}-checkbox-chat-supported',
        )
        return chat_supported

    def completion_supported_block(self):
        completion_supported = st.checkbox(
            label=f"{LANGUAGE.get('completion_support')}",
            value=False if self.model_alias in self.provider_llm_configs['settings'][
                'exclude_non_completion_models'] else True,
            key=f'{self.prefix}-checkbox-completion-supported',
        )
        return completion_supported

    def delete_model_button(self):
        if self.model_alias != "":
            delete_button = st.button(
                label=f"{LANGUAGE.get('delete')}",
                key=f'{self.prefix}-button-delete-model'
            )
            if delete_button:
                try:
                    self.provider_llm_configs = delete_dict_key_and_value(
                        self.provider_llm_configs, self.model_alias)
                    info = LANGUAGE.get('delete_model_success').format(
                        provider=translate_variable(
                            LANGUAGE,
                            self.provider.lower()),
                        model_alias=self.model_alias)
                    LLMs[self.provider] = self.provider_llm_configs
                    save_configurations_block(
                        LLMs_SETTINGS_TOML_FULL_PATH,
                        LLMS_CONFIGS, info, refresh=True)
                except Exception as e:
                    st.error(
                        LANGUAGE.get('delete_model_failed').format(
                            provider=translate_variable(
                                LANGUAGE,
                                self.provider.lower()),
                            model_alias=self.model_alias),
                        icon='🚨')

    def description_block(self):
        default = self.provider_llm_configs['descriptions'].get(
            self.model_alias, "") if self.model_alias != "" else ""
        description = st.text_area(
            label=f"{LANGUAGE.get('model_description_label')}",
            key=f'{self.prefix}-text-model-description',
            height=200,
            value=default)
        return description

    def initialize_llm_setting_block(self,
                                     block_type: Literal['update',
                                                         'add'] = 'update'):
        self.prefix = self.prefix + f'-{block_type}-'

        if self.provider == 'openai_compatible':
            left_col_configs = [
                'modified_model_alias',
                'model',
                'base_url',
                'api_key',
                'request_parameters'
            ]
            right_col_configs = [
                'description',
                'chat_supported',
                'completion_supported',
                'model_payment_type',
                'use_proxy']
        else:
            left_col_configs = [
                'modified_model_alias',
                'model',
                'request_parameters',
                'chat_supported',
                'completion_supported']
            right_col_configs = [
                'description',
                'model_payment_type',
                'use_proxy']

        model_parameters = {}
        if block_type == 'add':
            self.model_alias = ""
            st.write(
                f"**{LANGUAGE.get('add_new_model_description').format(provider=translate_variable(LANGUAGE, self.provider.lower()))}**")
            st.write(
                f"{LANGUAGE.get('model_management_description').format(provider=translate_variable(LANGUAGE, self.provider.lower()))}")

        col11, col12 = st.columns([0.5, 0.5])
        with col11:
            for config in left_col_configs:
                block_to_call = getattr(self, f"{config}_block")
                model_parameters[config] = block_to_call()
        with col12:
            for config in right_col_configs:
                block_to_call = getattr(self, f"{config}_block")
                model_parameters[config] = block_to_call()
        col21, col22 = st.columns([0.5, 0.5])
        with col21:
            self.update_or_add_model_button(model_parameters)
        with col22:
            self.delete_model_button()

    def model_block(self):
        model = st.text_input(
            label=f"{LANGUAGE.get('model_name_label')}",
            key=f'{self.prefix}-text-model',
            value=self.provider_llm_configs['models'].get(
                self.model_alias, "") if self.model_alias != "" else "")
        return model

    def model_payment_type_block(self):
        if self.model_alias in self.provider_llm_configs[
                'settings']['free_models'] or self.provider == 'ollama':
            default = 1  # free
        else:
            default = 0  # paid
        model_payment_type = st.radio(
            label=f"{LANGUAGE.get('model_payment_type')}",
            options=MODEL_PAYMENT_TYPES,
            key=f'{self.prefix}-radio-model-payment-type',
            horizontal=True,
            index=default
        )
        return model_payment_type

    def modified_model_alias_block(self):
        modified_model_alias = st.text_input(
            label=f"{LANGUAGE.get('model_alias_label')}",
            key=f'{self.prefix}-text-model-alias',
            value=self.model_alias if self.model_alias != "" else ""
        )
        return modified_model_alias

    @classmethod
    def recommended_request_parameters(cls, provider) -> list:
        if provider == 'openai':
            return [
                'seed',
                'temperature',
                'top_p',
                'presence_penalty',
                'frequency_penalty']
        # elif provider == 'openai_compatible':
        #     return [
        #         'seed',
        #         'temperature',
        #         'top_p',
        #         'presence_penalty',
        #         'frequency_penalty']
        elif provider == 'baidu':
            return [
                'temperature',
                'top_p',
                'penalty_score',
                'max_output_tokens',
                'disable_search']
        elif provider == 'qwen':
            return [
                'seed',
                'temperature',
                'top_p',
                'top_k',
                'repetition_penalty',
                'presence_penalty',
                'enable_search']
        elif provider == 'ollama':
            return [
                'seed',
                'temperature',
                'top_p',
                'top_k',
                'repeat_penalty']
        else:
            return []

    def request_parameters_block(self):
        current_parameters = self.provider_llm_configs['request_parameters'].get(
            self.model_alias, "")
        if self.model_alias == "" and self.provider != "" and PROJECT_SETTINGS.get(
                'add_common_request_parameters_to_new_model'):
            current_parameters = self.recommended_request_parameters(
                self.provider)
        if current_parameters != "" and len(current_parameters) > 0:
            current_parameters_str = ','.join(current_parameters)
        else:
            current_parameters_str = ""
        parameters_str = st.text_input(
            label=f"{LANGUAGE.get('request_parameters_description')}",
            key=f'{self.prefix}-text-accept-parameters',
            value=current_parameters_str)
        st.caption(
            f"{LANGUAGE.get('request_parameters_info')} {LANGUAGE.get('request_parameters_format').format(llm_settings=LANGUAGE.get('llm_settings'))}")
        parameters_str = convert_punctuation_to_comma(parameters_str)
        if parameters_str != "":
            parameters = parameters_str.split(',')
            for parameter in parameters:
                if parameter not in self.legitimate_request_parameters:
                    error = LANGUAGE.get('unsupported_request_parameter').format(
                        parameter=parameter)
                    raise ValueError(
                        f"[{get_current_function_name()}] {error}")
            return parameters
        else:
            return []

    def update_or_add_model_button(self, model_parameters: dict):
        if st.button(
                label=LANGUAGE.get('update') if self.model_alias != "" else LANGUAGE.get('add'),
                key=f'{self.prefix}-button-save-model'):

            validate_model_info(model_parameters)

            if self.model_alias != "" and model_parameters['modified_model_alias'] != self.model_alias:
                self.provider_llm_configs = rename_dict_key_and_value(
                    self.provider_llm_configs,
                    self.model_alias,
                    model_parameters['modified_model_alias'])
                info = LANGUAGE.get('change_model_alias').format(
                    provider=translate_variable(
                        LANGUAGE,
                        self.provider.lower()),
                    model_alias=self.model_alias,
                    modified_model_alias=model_parameters['modified_model_alias'])
                LLMs[self.provider] = self.provider_llm_configs
                save_configurations_block(
                    LLMs_SETTINGS_TOML_FULL_PATH,
                    LLMS_CONFIGS, info, refresh=False)

            if model_parameters['modified_model_alias'] != "":
                if model_parameters['model_payment_type'] == f"{LANGUAGE.get('free')}":
                    self.provider_llm_configs['settings']['free_models'] = add_unique_item_to_list(
                        model_parameters['modified_model_alias'],
                        self.provider_llm_configs['settings']['free_models'])
                    if model_parameters['modified_model_alias'] in self.provider_llm_configs['settings']['paid_models']:
                        self.provider_llm_configs['settings']['paid_models'].remove(
                            model_parameters['modified_model_alias'])
                else:
                    self.provider_llm_configs['settings']['paid_models'] = add_unique_item_to_list(
                        model_parameters['modified_model_alias'],
                        self.provider_llm_configs['settings']['paid_models'])
                    if model_parameters['modified_model_alias'] in self.provider_llm_configs['settings']['free_models']:
                        self.provider_llm_configs['settings']['free_models'].remove(
                            model_parameters['modified_model_alias'])

                if not model_parameters['chat_supported']:
                    self.provider_llm_configs['settings']['exclude_non_chat_models'] = add_unique_item_to_list(
                        model_parameters['modified_model_alias'],
                        self.provider_llm_configs['settings']['exclude_non_chat_models'])
                else:
                    if model_parameters['modified_model_alias'] in self.provider_llm_configs['settings'][
                            'exclude_non_chat_models']:
                        self.provider_llm_configs['settings']['exclude_non_chat_models'].remove(
                            model_parameters['modified_model_alias'])
                if not model_parameters['completion_supported']:
                    self.provider_llm_configs['settings']['exclude_non_completion_models'] = add_unique_item_to_list(
                        model_parameters['modified_model_alias'],
                        self.provider_llm_configs['settings']['exclude_non_completion_models'])
                else:
                    if model_parameters['modified_model_alias'] in self.provider_llm_configs['settings'][
                            'exclude_non_completion_models']:
                        self.provider_llm_configs['settings']['exclude_non_completion_models'].remove(
                            model_parameters['modified_model_alias'])

                self.provider_llm_configs['models'][model_parameters['modified_model_alias']
                                                    ] = model_parameters['model']
                self.provider_llm_configs['descriptions'][model_parameters['modified_model_alias']
                                                          ] = model_parameters['description']

                if 'base_url' in model_parameters:
                    self.provider_llm_configs['base_urls'][model_parameters['modified_model_alias']
                                                           ] = model_parameters['base_url']
                if 'api_key' in model_parameters:
                    self.provider_llm_configs['api_keys'][model_parameters['modified_model_alias']
                                                          ] = model_parameters['api_key']
                if model_parameters['use_proxy']:
                    self.provider_llm_configs['settings']['use_proxy_models'] = add_unique_item_to_list(
                        model_parameters['modified_model_alias'],
                        self.provider_llm_configs['settings']['use_proxy_models'])
                else:
                    if model_parameters['modified_model_alias'] in self.provider_llm_configs['settings'][
                            'use_proxy_models']:
                        self.provider_llm_configs['settings']['use_proxy_models'].remove(
                            model_parameters['modified_model_alias'])

                if 'request_parameters' in model_parameters:
                    self.provider_llm_configs['request_parameters'][model_parameters['modified_model_alias']
                                                                    ] = model_parameters['request_parameters']

                if self.model_alias != "":
                    info = f"{LANGUAGE.get('update_model').format(provider=translate_variable(LANGUAGE, self.provider.lower()), model_alias=self.model_alias)}"
                else:
                    info = f"{LANGUAGE.get('add_new_model').format(provider=translate_variable(LANGUAGE, self.provider.lower()), modified_model_alias=model_parameters['modified_model_alias'])}"

                LLMs[self.provider] = self.provider_llm_configs
                save_configurations_block(
                    LLMs_SETTINGS_TOML_FULL_PATH,
                    LLMS_CONFIGS, info)

    def use_proxy_block(self):
        if self.model_alias in self.provider_llm_configs['settings'][
                'use_proxy_models']:
            index = BINARY_OPTIONS.index(True)
        elif self.model_alias != "" and self.model_alias not in self.provider_llm_configs['settings'][
                'use_proxy_models']:
            index = BINARY_OPTIONS.index(False)
        elif self.provider == 'openai' and self.model_alias == "":
            index = BINARY_OPTIONS.index(True)
        else:
            index = BINARY_OPTIONS.index(False)

        use_proxy = st.radio(
            label=f"{LANGUAGE.get('using_proxy_to_connect')}",
            options=[True, False],
            key=f'{self.prefix}-radio-use-proxy',
            horizontal=True,
            index=index
        )
        return use_proxy


class RequestParamsConfigUI:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.project_settings = PROJECT_SETTINGS
        self.param_types = ['bool', 'int', 'float', 'str']
        self.legitimate_request_parameters = LLMModelParamsConfigUI.get_supported_request_parameter_list()

    def add_or_update_button(self,
                             param_info_dict: dict,
                             button_type: Literal['update',
                                                  'add'] = 'add'):
        widget_key = f'{self.prefix}-{button_type}-request-parameter'
        p_dict = param_info_dict.copy()

        v = p_dict['name']
        v_description = p_dict['description']
        del p_dict['name']
        del p_dict['description']
        button = st.button(
            label=LANGUAGE.get(button_type),
            key=f'{widget_key}-button-add-request-parameter'
        )

        info = LANGUAGE.get(f'{button_type}_request_parameter_success').format(
            parameter=param_info_dict['name'])
        if button:
            self.project_settings['customized_request_parameters'][v] = p_dict
            save_configurations_block(
                PROJECT_SETTINGS_TOML_FULL_PATH,
                PROJECT_CONFIGS,
                info,
                refresh=False)
            status, info = add_variable_to_language_files(v, v_description)
            if status:
                st.success(info, icon='✅')
                time.sleep(REFRESH_INTERVAL)
                run_app()
            else:
                st.error(info, icon='🚨')

    def add_parameter_block(self):
        widget_key = f'{self.prefix}-add-request-parameter'
        st.write(f"**{LANGUAGE.get('add_request_parameter')}**")
        param_name = st.text_input(
            label=f"{LANGUAGE.get('request_parameter_name')}",
            key=f'{widget_key}-text-request-parameter-name')
        if param_name == "":
            st.error(LANGUAGE.get('request_parameter_name_empty'), icon='🚨')
            return
        elif param_name in self.legitimate_request_parameters:
            st.error(
                LANGUAGE.get('request_parameter_name_duplicated').format(
                    parameter=param_name), icon='🚨')
            return

        param_info_dict = self.set_request_parameter_details_block(
            param_name, 'add')
        self.add_or_update_button(param_info_dict, button_type='add')

    def delete_parameter_block(self):
        widget_key = f'{self.prefix}-delete-request-parameter'
        st.write(f"**{LANGUAGE.get('delete_request_parameter')}**")
        param_name = self.select_a_parameter_block('delete')
        st.write(
            LANGUAGE.get('request_parameter_name_to_delete').format(
                parameter=param_name))
        if st.button(
                label=LANGUAGE.get('delete'),
                key=f'{widget_key}-button-delete-request-parameter'
        ):
            delete_result, info = delete_variable_in_language_files(param_name)
            if delete_result:
                st.success(info, icon='🗑️')
            else:
                st.warning(info, icon='🚨')
            if param_name in self.project_settings['customized_request_parameters']:
                del self.project_settings['customized_request_parameters'][param_name]
                save_configurations_block(
                    PROJECT_SETTINGS_TOML_FULL_PATH,
                    PROJECT_CONFIGS,
                    LANGUAGE.get('delete_request_parameter_success').format(
                        parameter=param_name),
                    refresh=True)

    def select_a_parameter_block(
            self, operation_type: Literal['update', 'delete']):
        widget_key = f'{self.prefix}-{operation_type}-select-request-parameter'
        params = list(PROJECT_SETTINGS.get(
            'customized_request_parameters').keys())
        selected_param = st.selectbox(
            label=f"{LANGUAGE.get('select_a_request_parameter')}",
            options=params,
            key=f'{widget_key}-select-request-parameter')
        return selected_param

    def set_request_parameter_details_block(
            self, param_name: str, operation_type: Literal['update', 'add'] = 'add'):
        widget_key = f'{self.prefix}-{operation_type}-set-request-parameter-details'
        if operation_type == 'update':
            param_info_dict = self.project_settings['customized_request_parameters'][param_name]
            param_type = param_info_dict['type']
        else:
            param_type = st.radio(
                label=f"{LANGUAGE.get('request_parameter_type')}",
                options=self.param_types,
                horizontal=True,
                index=0,
                key=f'{widget_key}-radio-request-parameter-type'
            )
        params = {'name': param_name}
        if param_type == 'bool':
            param_default = st.radio(
                label=f"{LANGUAGE.get('request_parameter_default_value')}",
                options=[True, False],
                horizontal=True,
                index=[True, False].index(
                    param_info_dict['default']) if operation_type == 'update' else 0,
                key=f'{widget_key}-radio-request-parameter-default')
        elif param_type == 'int':
            number_format = '%d'
            param_default = st.number_input(
                label=f"{LANGUAGE.get('request_parameter_default_value')}",
                key=f'{widget_key}-number-request-parameter-default',
                value=int(param_info_dict['default']) if operation_type == 'update' else 0,
                format=number_format)
            param_min = st.number_input(
                label=f"{LANGUAGE.get('request_parameter_min_value')}",
                key=f'{widget_key}-number-request-parameter-min',
                value=int(
                    param_info_dict['min']) if operation_type == 'update' else 0,
                format=number_format)
            param_max = st.number_input(
                label=f"{LANGUAGE.get('request_parameter_max_value')}",
                key=f'{widget_key}-number-request-parameter-max',
                value=int(
                    param_info_dict['max']) if operation_type == 'update' else 1024,
                format=number_format)
            param_adjust_step = st.number_input(
                label=f"{LANGUAGE.get('request_parameter_adjust_step')}",
                key=f'{widget_key}-number-request-parameter-step',
                value=int(param_info_dict['step']) if operation_type == 'update' else 1,
                format=number_format)
        elif param_type == 'float':
            param_default = st.number_input(
                label=f"{LANGUAGE.get('request_parameter_default_value')}",
                key=f'{widget_key}-number-request-parameter-default',
                value=param_info_dict['default'] if operation_type == 'update' else -2.0,
            )
            param_min = st.number_input(
                label=f"{LANGUAGE.get('request_parameter_min_value')}",
                key=f'{widget_key}-number-request-parameter-min',
                value=param_info_dict['min'] if operation_type == 'update' else -2.0,
            )
            param_max = st.number_input(
                label=f"{LANGUAGE.get('request_parameter_max_value')}",
                key=f'{widget_key}-number-request-parameter-max',
                value=param_info_dict['max'] if operation_type == 'update' else 2.0,
            )
            param_adjust_step = st.number_input(
                label=f"{LANGUAGE.get('request_parameter_adjust_step')}",
                key=f'{widget_key}-number-request-parameter-step',
                value=param_info_dict['step'] if operation_type == 'update' else 0.1,
            )
        else:
            param_default = st.text_input(
                label=f"{LANGUAGE.get('request_parameter_default_value')}",
                key=f'{widget_key}-text-request-parameter-default')

        if operation_type == 'update':
            description = LANGUAGE.get(
                generate_language_variable_name(params['name']))
        else:
            description = None

        param_description = st.text_area(
            label=f"{LANGUAGE.get('request_parameter_description')}",
            key=f'{widget_key}-text-request-parameter-description',
            value=description,
            height=calculate_text_area_height(
                text=description,
                word_per_line=30) if description is not None else None,
        )
        if operation_type == 'add':
            st.caption(LANGUAGE.get('request_parameter_description_notice'))

        params['type'] = param_type
        params['default'] = param_default
        params['description'] = param_description
        if param_type == 'int' or param_type == 'float':
            if param_default > param_max or param_default < param_min:
                raise ValueError(
                    LANGUAGE.get('default_value_out_of_range').format(
                        min_value=param_min, max_value=param_max))
            if param_min > param_max:
                raise ValueError(
                    LANGUAGE.get('min_value_greater_than_max_value'))
            params['min'] = int(
                param_min) if param_type == 'int' else param_min
            params['max'] = int(
                param_max) if param_type == 'int' else param_max
            params['step'] = int(
                param_adjust_step) if param_type == 'int' else param_adjust_step

        return params

    def update_parameter_block(self):
        widget_key = f'{self.prefix}-update-request-parameter'
        st.write(f"**{LANGUAGE.get('update_request_parameter')}**")
        param_name = self.select_a_parameter_block('update')
        param_info_dict = self.set_request_parameter_details_block(
            param_name, 'update')
        self.add_or_update_button(param_info_dict, button_type='update')


def archive_file_naming_block(page_prefix):
    st.subheader(LANGUAGE.get('archive_file_naming_rule').title())
    st.write(
        LANGUAGE.get('archive_file_naming_rule_description').format(
            timestamp=current_timestamp_to_string('%Y-%m-%d_%H-%M-%S'),
            example=archive_file_naming(
                'chat',
                'llama3.1')))
    new_pattern = st.text_input(
        label=LANGUAGE.get('archive_file_naming_rule').capitalize(),
        value='[{timestamp}]-[{type}]-[{model}]',
        key=f'{page_prefix}-archive_file_naming')
    if validate_file_name_pattern(new_pattern):
        current_pattern = PROJECT_SETTINGS.get('archive_file_naming_pattern')
        if new_pattern != current_pattern:
            PROJECT_SETTINGS['archive_file_naming_pattern'] = new_pattern
            info = LANGUAGE.get('archive_file_naming_rule_updated')
            save_configurations_block(
                PROJECT_SETTINGS_TOML_FULL_PATH,
                PROJECT_CONFIGS,
                info,
                refresh=False)
    else:
        st.error(LANGUAGE.get('archive_file_naming_rule_invalid'), icon='🚨')


def archive_management_block(archive_folder, widget_prefix):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(f"{LANGUAGE.get('load_archive')}")
        if st.button(
                label=f"{LANGUAGE.get('open_folder')}",
                key=f'{widget_prefix}-button-open-folder'
        ):
            open_directory(archive_folder)
        selected_archive_file = select_an_archive_block(
            archive_folder, widget_prefix, 'load_archive')
        selected_archive_file_path = os.path.join(
            archive_folder, selected_archive_file + '.json')

    with col2:
        st.subheader(f"{LANGUAGE.get('rename_archive')}")
        rename_file_block(widget_prefix, selected_archive_file_path)

    with col3:
        st.subheader(f"{LANGUAGE.get('delete_archive')}")
        st.write(
            f"{LANGUAGE.get('file_to_be_deleted').format(selected_archive_file=selected_archive_file)}")
        if st.button(
                label=f"{LANGUAGE.get('delete')}",
                key=f'{widget_prefix}-button-delete-chat'):
            safe_delete(selected_archive_file_path)
            st.success(
                f"{LANGUAGE.get('delete_archive_success').format(selected_archive_file=selected_archive_file)}",
                icon='✅')
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
        delete_file_block(widget_prefix, selected_archive_file_path)

    archive_content_json = load_json_file(selected_archive_file_path)
    return selected_archive_file_path, archive_content_json


def backup_restore_block(page_prefix):
    prefix_ = page_prefix + '-backup-restore'
    backup_folder = os.path.join(
        ARCHIVE_FOLDER,
        PROJECT_SETTINGS.get('backup_folder'))
    chat_archive_folder = os.path.join(
        ARCHIVE_FOLDER, PROJECT_SETTINGS.get('chat_archive_folder'))
    completion_archive_folder = os.path.join(
        ARCHIVE_FOLDER, PROJECT_SETTINGS.get('completion_archive_folder'))

    os.makedirs(backup_folder, exist_ok=True)
    os.makedirs(chat_archive_folder, exist_ok=True)
    os.makedirs(completion_archive_folder, exist_ok=True)

    st.header(LANGUAGE.get('backup_and_restore'))
    ratio = (0.55, 0.25, 0.25)
    with st.container(border=True):
        backup_col1, backup_col2, backup_col3 = st.columns(ratio)
    with st.container(border=True):
        restore_col1, restore_col2, restore_col3 = st.columns(ratio)

    with backup_col1:
        st.subheader(f"{LANGUAGE.get('backup')}")
        st.write(
            LANGUAGE.get('backup_info').format(
                chat=LANGUAGE.get('chat'),
                completion=LANGUAGE.get('completion'),
                backup_file_name=backup_file_name(False),
                current_file_name=backup_file_name()))
    with backup_col2:
        backup_button = st.button(
            label=LANGUAGE.get('backup'),
            key=f'{prefix_}-button-backup')
    with backup_col3:
        open_archive_folder_button = st.button(
            label=f"{LANGUAGE.get('open_archive_folder')}",
            key=f'{prefix_}-button-open-archive-folder')
        open_backup_folder_button = st.button(
            label=f"{LANGUAGE.get('open_backup_folder')}",
            key=f'{prefix_}-button-open-backup-folder')

    if backup_button:
        backup_status, backup_info = backup_operation(
            chat_archive_folder, completion_archive_folder, PROJECT_SETTINGS_TOML_FULL_PATH,
            LLMs_SETTINGS_TOML_FULL_PATH, backup_folder)
        if backup_status:
            st.success(
                LANGUAGE.get('backup_success').format(
                    backup_file_fullpath=backup_info), icon='✅')
        else:
            error_info = f"{LANGUAGE.get('backup_error').format(error=backup_info)}"
            print_to_console(error_info, get_current_function_name())
            st.error(
                error_info,
                icon='🚨')
    if open_archive_folder_button:
        open_directory(ARCHIVE_FOLDER)
    if open_backup_folder_button:
        open_directory(backup_folder)

    with restore_col1:
        st.subheader(f"{LANGUAGE.get('restore')}")
        st.write(LANGUAGE.get('restore_info').format(
            chat=LANGUAGE.get('chat'),
            completion=LANGUAGE.get('completion')))
    with restore_col2:
        backup_files = list_files_without_extension(backup_folder, 'zip')
        if len(backup_files) > 0:
            selected_backup_file_name = st.radio(
                label=LANGUAGE.get('select_backup_file'),
                options=sorted(backup_files, reverse=True),
                key=f'{prefix_}-radio-select-backup-file')
            selected_backup_file_fullpath = os.path.join(
                backup_folder, f"{selected_backup_file_name}.zip")

            if st.button(
                    label=LANGUAGE.get('restore'),
                    key=f'{prefix_}-button-restore'):
                status, info = restore_operation(chat_archive_folder,
                                                 completion_archive_folder,
                                                 PROJECT_SETTINGS_TOML_FULL_PATH,
                                                 LLMs_SETTINGS_TOML_FULL_PATH,
                                                 selected_backup_file_fullpath)
                if status:
                    st.success(
                        LANGUAGE.get('restore_success'),
                        icon='✅')
                    time.sleep(REFRESH_INTERVAL)
                    run_app()
                else:
                    error_info = f"{LANGUAGE.get('restore_error').format(error=info)}"
                    print_to_console(
                        error_info, get_current_function_name())
                    st.error(
                        error_info,
                        icon='🚨')
            with restore_col3:
                rename_file_block(prefix_, selected_backup_file_fullpath)
                delete_file_block(prefix_, selected_backup_file_fullpath)
        else:
            st.info(LANGUAGE.get('no_backup_files'), icon='ℹ️')


def baidu_api_key_management_block(page_prefix: str):
    provider = 'baidu'
    prefix = page_prefix + f'-{provider}-api-key-settings'
    llm_configs = LLMs.get(provider)
    st.subheader(f'{provider.capitalize()} api keys')
    st.write(LANGUAGE.get('baidu_auth_info'))
    if len(llm_configs['api_keys']) == 0:
        st.warning(LANGUAGE.get('no_provider_api_key_found'), icon='⚠️')
        new_client_id = st.text_input(
            label=LANGUAGE.get('baidu_client_id'),
            key=f'{prefix}-text-baidu-auth-new-client_id')
        new_client_secret = st.text_input(
            label=LANGUAGE.get('baidu_client_secret'),
            key=f'{prefix}-text-baidu-auth-new-client_secret')
        if st.button(
                LANGUAGE.get('add'),
                key=f'{prefix}-button-add-baidu-auth-settings'):
            llm_configs['api_keys']['client_id'] = new_client_id
            llm_configs['api_keys']['client_secret'] = new_client_secret
            save_configurations_block(
                LLMs_SETTINGS_TOML_FULL_PATH,
                LLMS_CONFIGS,
                f'Baidu authentication settings added.')
        return

    for key, value in llm_configs['api_keys'].items():
        new_value = st.text_input(label=LANGUAGE.get('baidu_client_id') if key == 'client_id' else LANGUAGE.get(
            'baidu_client_secret'), value=value, key=f'{prefix}-text-baidu-auth--{key}')
        llm_configs['api_keys'][key] = new_value

    if st.button(
            LANGUAGE.get('save'),
            key=f'{prefix}-button-save-baidu-auth-settings'):
        save_configurations_block(
            LLMs_SETTINGS_TOML_FULL_PATH,
            LLMS_CONFIGS,
            LANGUAGE.get('baidu_auth_info_updated'))


def customize_request_params_block(page_prefix: str):
    prefix = page_prefix + '-customize-request-params'
    st.subheader(f"{LANGUAGE.get('customize_request_parameters')}")
    request_params_ui = RequestParamsConfigUI(prefix)
    st.write(
        f"{LANGUAGE.get('customize_request_parameters_description').format(param_types=','.join(request_params_ui.param_types))}")

    if len(PROJECT_SETTINGS.get('customized_request_parameters')) == 0:
        st.info(LANGUAGE.get('no_customized_request_parameters'), icon='ℹ️')
        with st.container(border=True):
            request_params_ui.add_parameter_block()
    else:
        col1, col2 = st.columns((0.33, 0.67))
        with col1:
            with st.container(border=True):
                request_params_ui.add_parameter_block()
        with col2:
            col21, col22 = st.columns(2)
            with col21:
                with st.container(border=True):
                    request_params_ui.update_parameter_block()
            with col22:
                with st.container(border=True):
                    request_params_ui.delete_parameter_block()


def delete_file_block(key_prefix: str, file_full_path: str):
    widget_prefix = key_prefix + '-delete-file'
    file_name = os.path.splitext(os.path.basename(file_full_path))[0]
    st.write(
        f"{LANGUAGE.get('file_to_be_deleted').format(selected_archive_file=file_name)}")
    if st.button(
            label=f"{LANGUAGE.get('delete')}",
            key=f'{widget_prefix}-button-delete-chat'):
        safe_delete(file_full_path)
        st.success(
            f"{LANGUAGE.get('delete_archive_success').format(selected_archive_file=file_name)}",
            icon='✅')
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


def enable_provider_block(page_prefix: str):
    prefix = page_prefix + '-enable-provider-settings'
    st.subheader(
        f"{LANGUAGE.get('activate_llm_provider')}")
    new_llm_provider_settings = {}
    for key, value in LLMs['settings'].items():
        provider = key.replace('enable_', '').lower()
        new_value = st.checkbox(
            label=f"{LANGUAGE.get('enable')} {LANGUAGE.get(provider)}",
            value=value,
            key=f'{prefix}-checkbox-{key}')
        new_llm_provider_settings[key] = new_value
    if new_llm_provider_settings != LLMs['settings']:
        LLMs['settings'] = new_llm_provider_settings
        save_configurations_block(
            LLMs_SETTINGS_TOML_FULL_PATH,
            LLMS_CONFIGS,
            f"{LANGUAGE.get('llm_provider_activation_changed')}")


def last_query_model_info(provider: Literal['openai', 'ollama']):
    if provider not in ['openai', 'ollama']:
        raise ValueError(
            f'[{get_current_function_name()}] Invalid provider. It should be `openai` or `ollama`.')
    supported_model_key = f"{provider}_supported_models"
    query_time_key = f"{provider}_supported_models_query_time"
    query_time = LANGUAGE.get('last_query_time').format(
        time=PROJECT_SETTINGS.get(query_time_key))
    provider_supported_model = LANGUAGE.get('provider_supported_models').format(
        provider=translate_variable(LANGUAGE, provider.lower()))
    st.write(
        f"{query_time}, {provider_supported_model}")
    st.write(
        '```' +
        ', '.join(
            PROJECT_SETTINGS.get(supported_model_key)) +
        '```')


def load_archive_block(content_type_: str, session_state: st.session_state):
    if content_type_.lower() not in ['chat', 'completion']:
        raise ValueError(
            f'[{get_current_function_name()}] Invalid content type. It should be `chat` or `completion`.')

    if content_type_ == 'chat':
        info = LANGUAGE.get('load_archive_label').format(
            type=content_type_.capitalize(), work=LANGUAGE.get('chat'))
    else:
        info = LANGUAGE.get('load_archive_label').format(
            type=content_type_.capitalize(), work=LANGUAGE.get('completion'))
    st.write(f"**{info}**")
    json_data_file = st.file_uploader(
        label=info,
        type=["json"],
        key=f'upload-{content_type_}-archive',
        label_visibility='collapsed')
    if json_data_file is not None:
        session_state[content_type_] = json.load(json_data_file)


def LLM_setting_block(page_prefix, provider):
    prefix = page_prefix + f'-{provider}-settings'
    llm_configs = LLMs.get(provider)
    llm_config_ui = ProviderLLMModelConfigUI(prefix, "", provider)

    if provider not in ['openai_compatible', 'ollama']:
        provider_api_key_management_block(prefix, provider)
        provider_supported_models(prefix, provider)
    elif provider != 'openai_compatible':
        provider_supported_models(prefix, provider)

    saved_models = available_models_by_provider(provider)
    if saved_models is None or len(saved_models) == 0:
        with st.container(border=True):
            st.error(
                f"{LANGUAGE.get('provider_model_not_found').format(provider=translate_variable(LANGUAGE, provider.lower()))}",
                icon='🚨')
            llm_config_ui.initialize_llm_setting_block('add')
        return

    provider_model_management_block(prefix, llm_config_ui)

    provider_model_list_by_payment_type(llm_configs)

    if provider == 'baidu':
        with st.container(border=True):
            support_list, not_support_list = accept_system_prompt_baidu_models()
            st.write(
                f"**{LANGUAGE.get('support_pass_system_prompt_by_parameter')}**")
            st.code(support_list)
            st.write(
                f"**{LANGUAGE.get('not_support_pass_system_prompt_by_parameter')}**")
            st.code(not_support_list)


def model_parameter_setting_block(page_prefix, model_alias):
    provider = identify_provider_by_model_alias(model_alias)
    llm_model_params_config_ui = LLMModelParamsConfigUI(
        page_prefix, model_alias, provider)
    return llm_model_params_config_ui.model_parameters_block()


def prompt_repository_block(widget_key_prefix: str, prompt_type: str = 'Chat'):
    prefix = widget_key_prefix + f'-{prompt_type}-prompt-repository'

    predefined_prompts_label = sub_header_and_label_for_predefined_prompts_block(
        prompt_type)

    predefined_prompts = LLMS_CONFIGS.get(prompt_type.lower()).get(
        f'{prompt_type.lower()}_prompt_repository')

    if len(predefined_prompts) > 0:
        prompt = st.selectbox(
            label=predefined_prompts_label,
            options=predefined_prompts,
            index=0,
            key=f'{prefix}-select-box-select-a-prompt'
        )
        selected_prompt_index = predefined_prompts.index(prompt)
        new_prompt = st.text_area(
            label=f"**{LANGUAGE.get(prompt_type.lower())} {LANGUAGE.get('prompt')}**",
            value=prompt,
            key=f'{prefix}-text-area-new-prompt',
            height=calculate_text_area_height(prompt))
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                    label=f"{LANGUAGE.get('add')}",
                    key=f'{prefix}-button-add-new-prompt'
            ):
                if new_prompt not in predefined_prompts:
                    predefined_prompts.append(new_prompt)
                    save_configurations_block(
                        LLMs_SETTINGS_TOML_FULL_PATH,
                        LLMS_CONFIGS,
                        f'New prompt `{new_prompt}` is added. Page will be refreshed in seconds.',
                        True, True)
                else:
                    st.warning(
                        'The prompt is already existed, no need to add.',
                        icon='⚠️')
        with col2:
            if st.button(
                    label=f"{LANGUAGE.get('update')}",
                    key=f'{prefix}-button-update-prompt'
            ):
                if new_prompt != '':
                    if new_prompt not in predefined_prompts:
                        predefined_prompts[selected_prompt_index] = new_prompt
                        save_configurations_block(
                            LLMs_SETTINGS_TOML_FULL_PATH,
                            LLMS_CONFIGS,
                            f'The prompt `{new_prompt}` is updated. Page will be refreshed in seconds.',
                            True, True)
                    else:
                        st.warning(
                            'The prompt is already existed, no need to update.', icon='⚠️')
                else:
                    st.error(
                        'The prompt is empty.',
                        icon='🚨')
        with col3:
            if st.button(
                    label=f"{LANGUAGE.get('delete')}",
                    key=f'{prefix}-button-delete-prompt'
            ):
                if new_prompt != "":
                    if new_prompt in predefined_prompts:
                        predefined_prompts.remove(new_prompt)
                        save_configurations_block(
                            LLMs_SETTINGS_TOML_FULL_PATH,
                            LLMS_CONFIGS,
                            f'The prompt `{new_prompt}` is deleted. Page will be refreshed in seconds.',
                            True, True)
                    else:
                        st.warning(
                            'The prompt is not existed, no need to delete.',
                            icon='⚠️')
                else:
                    st.error(
                        'The prompt is empty.',
                        icon='🚨')

    else:
        st.warning(f'没有预定义的{prompt_type} Prompt，请先添加一个prompt。', icon='🚨')
        new_prompt = st.text_area(
            label=f"{prompt_type} prompt",
            value="",
            key=f'{prefix}-text-area-new-{prompt_type}-prompt',
        )
        if st.button(
                label='Add',
                key=f'{prefix}-button-add-first-prompt'
        ):
            if new_prompt != "":
                predefined_prompts.append(new_prompt)
                save_configurations_block(
                    LLMs_SETTINGS_TOML_FULL_PATH,
                    LLMS_CONFIGS,
                    f'New prompt `{new_prompt}` is added. Page will be refreshed in seconds.',
                    True, True)
            else:
                st.error(
                    'The prompt is empty.',
                    icon='🚨')


def provider_api_key_management_block(prefix_: str, provider: str):
    def baidu_api_key_block():
        st.write(LANGUAGE.get('baidu_auth_info'))
        if ('client_id' not in llm_configs['api_keys'] or 'client_secret' not in llm_configs['api_keys']
                or llm_configs['api_keys']['client_id'] == "" or llm_configs['api_keys']['client_secret'] == ""):
            st.warning(
                LANGUAGE.get('no_provider_api_key_found').format(
                    provider=translate_variable(
                        LANGUAGE,
                        provider.lower())),
                icon='⚠️')
            new_client_id = st.text_input(
                label=LANGUAGE.get('baidu_client_id'),
                key=f'{prefix}-text-baidu-auth-new-client_id')
            new_client_secret = st.text_input(
                label=LANGUAGE.get('baidu_client_secret'),
                key=f'{prefix}-text-baidu-auth-new-client_secret')
            if st.button(
                    LANGUAGE.get('add'),
                    key=f'{prefix}-button-add-baidu-auth-settings'):
                llm_configs['api_keys']['client_id'] = new_client_id
                llm_configs['api_keys']['client_secret'] = new_client_secret
                save_configurations_block(
                    LLMs_SETTINGS_TOML_FULL_PATH,
                    LLMS_CONFIGS,
                    LANGUAGE.get('baidu_auth_info_updated'))
            st.stop()

        for key, value in llm_configs['api_keys'].items():
            new_value = st.text_input(label=LANGUAGE.get('baidu_client_id') if key == 'client_id' else LANGUAGE.get(
                'baidu_client_secret'), value=value, key=f'{prefix}-text-baidu-auth--{key}')
            llm_configs['api_keys'][key] = new_value

        if st.button(
                LANGUAGE.get('save'),
                key=f'{prefix}-button-save-baidu-auth-settings'):
            save_configurations_block(
                LLMs_SETTINGS_TOML_FULL_PATH,
                LLMS_CONFIGS,
                LANGUAGE.get('baidu_auth_info_updated'))

    def add_api_key_block(type_: str = 'add'):
        if type_.lower() not in ['add', 'update']:
            raise ValueError(
                f'[{get_current_function_name()}] type_ must be `add` or `update`.')
        provider_ = provider.lower()

        new_api_key = st.text_input(
            label=f"{LANGUAGE.get('provider_api_key_label').format(provider=translate_variable(LANGUAGE, provider_))}",
            key=f'{prefix}-text-{type_}-api-key',
            value=llm_configs['api_keys'].get(provider_))
        if st.button(
                f"{LANGUAGE.get(type_.lower())}",
                key=f'{prefix}-button-{type_}-api-key'):
            if new_api_key == "" or new_api_key is None:
                st.error(
                    f"{LANGUAGE.get('api_key_cannot_be_empty')}",
                    icon='🚨')
            else:
                llm_configs['api_keys'][provider_] = new_api_key
                save_configurations_block(
                    LLMs_SETTINGS_TOML_FULL_PATH,
                    LLMS_CONFIGS,
                    LANGUAGE.get('api_key_updated').format(
                        provider=translate_variable(
                            LANGUAGE,
                            provider_)))

    prefix = prefix_ + f'-{provider}-api-key-settings'
    llm_configs = LLMs.get(provider)
    with st.container(border=True):
        st.subheader(
            f"{LANGUAGE.get('provider_api_key_title').format(provider=translate_variable(LANGUAGE, provider.lower()))}")
        if provider == 'baidu':
            baidu_api_key_block()
        else:
            st.write(
                f"{LANGUAGE.get('provider_api_key_description').format(provider=translate_variable(LANGUAGE, provider.lower()))}")
            if llm_configs.get('api_keys').get(provider) is None or llm_configs.get(
                    'api_keys').get(provider) == "":
                st.warning(
                    LANGUAGE.get('no_provider_api_key_found').format(
                        provider=translate_variable(
                            LANGUAGE, provider.lower())), icon='⚠️')
                add_api_key_block('add')
                st.stop()

            add_api_key_block('update')


def provider_llms_settings_block(page_prefix: str):
    prefix = page_prefix + '-provider-llms-settings'
    with st.container(border=True):
        st.subheader(f"{LANGUAGE.get('select_a_provider')}")
        providers = available_providers(translate=True)
        if len(available_providers()) == 0:
            st.error(LANGUAGE.get('no_available_provider'), icon='🚨')
            return
        selected_provider = st.radio(
            label='Select a provider',
            options=providers,
            horizontal=True,
            label_visibility='collapsed',
            key=f'{prefix}-select-provider')

    LLM_setting_block(
        page_prefix=prefix,
        provider=reverse_translation_to_variable(LANGUAGE, selected_provider))


def provider_model_list_by_payment_type(llm_configs):
    with st.container(border=True):
        st.write(f"**{LANGUAGE.get('paid_model_list')}**")
        st.code(llm_configs['settings']['paid_models'])
        st.write(f"**{LANGUAGE.get('free_model_list')}**")
        st.code(llm_configs['settings']['free_models'])


def provider_model_management_block(
        page_prefix: str,
        llm_config_ui: ProviderLLMModelConfigUI):
    prefix = page_prefix + f"-{llm_config_ui.provider}-model-management"
    provider = llm_config_ui.provider
    saved_models = available_models_by_provider(provider)
    with st.container(border=True):
        st.subheader(
            f"{LANGUAGE.get('provider_model_management').format(provider=translate_variable(LANGUAGE, provider.lower()))}")
        selected_model_alias = st.radio(
            label=f"{LANGUAGE.get('provider_select_a_model')}",
            options=saved_models,
            key=f'{prefix}-radio-select-{provider}-model'
        )
        llm_config_ui.model_alias = selected_model_alias
        llm_config_ui.initialize_llm_setting_block('update')

    with st.container(border=True):
        llm_config_ui.initialize_llm_setting_block('add')


def provider_supported_models(page_prefix: str, provider: str):
    prefix = page_prefix + f'-{provider}-supported-models'
    with st.container(border=True):
        if provider == 'openai':
            st.subheader(
                f"{LANGUAGE.get('provider_models').format(provider=translate_variable(LANGUAGE, provider.lower()))}")
            if PROJECT_SETTINGS.get('openai_supported_models') is not None and len(
                    PROJECT_SETTINGS.get('openai_supported_models')) > 0:
                last_query_model_info('openai')
            st.write(
                f"{LANGUAGE.get('query_provider_models').format(provider=translate_variable(LANGUAGE, provider.lower()))}")
            query_button = st.button(
                f"{LANGUAGE.get('query')}",
                key=f'{prefix}-button-query-openai-models'
            )
            use_proxy = st.checkbox(
                label=f"{LANGUAGE.get('using_proxy_to_connect_to').format(provider=translate_variable(LANGUAGE, 'openai'))}",
                value=True,
                key=f'{prefix}-checkbox-use-proxy')
            if query_button:
                with st.spinner(
                        f"{LANGUAGE.get('querying_models').format(provider=translate_variable(LANGUAGE, 'openai'))}"):
                    openai_supported_models = list_openai_models(use_proxy)
                    st.write(','.join(openai_supported_models))
                    PROJECT_SETTINGS['openai_supported_models'] = openai_supported_models
                    PROJECT_SETTINGS['openai_supported_models_query_time'] = current_timestamp_to_string(
                        "%Y-%m-%d %H:%M:%S")
                    save_configurations_block(
                        PROJECT_SETTINGS_TOML_FULL_PATH,
                        PROJECT_CONFIGS,
                        f"{LANGUAGE.get('openai_supported_models_updated')}")

        elif provider == 'baidu':
            st.subheader(
                LANGUAGE.get('qianfan_supported_models').format(
                    version=qianfan.version.VERSION))
            st.write(LANGUAGE.get('qianfan_update_notice'))
            st.markdown(f"- {LANGUAGE.get('chat')}{LANGUAGE.get('model')}")
            st.write(
                '，'.join(
                    sorted(
                        list_baidu_models_by_SDK('chat'))) +
                '。')
            st.markdown(
                f"- {LANGUAGE.get('completion')}{LANGUAGE.get('model')}")
            st.write(
                '，'.join(
                    sorted(
                        list_baidu_models_by_SDK('completion'))) +
                '。')
        elif provider == 'qwen':
            st.subheader(
                f"{LANGUAGE.get('provider_models').format(provider=translate_variable(LANGUAGE, provider.lower()))}")
            st.write(LANGUAGE.get('qwen_model_name_description'))

        elif provider == 'ollama':
            st.subheader(
                f"{LANGUAGE.get('provider_models').format(provider=translate_variable(LANGUAGE, provider.lower()))}")
            ollama_status, ollama_models = query_ollama_status(
                return_model_name=True)
            status_description = f"{LANGUAGE.get('ollama_server_running')}" if ollama_status else f"{LANGUAGE.get('ollama_server_stopped')}"
            if ollama_status:
                st.write(
                    f"{LANGUAGE.get('ollama_server_status')}：`{status_description}`")
                st.write(
                    f"{LANGUAGE.get('ollama_models_list')}：`{','.join(ollama_models)}`")
                selected_model = st.selectbox(
                    label=f"{LANGUAGE.get('select_a_model')}",
                    options=ollama_models,
                    index=0,
                    key=f'{prefix}-radio-ollama-settings-select-model'
                )
                model_achitecture_description = generate_ollama_model_architecture_description(
                    selected_model)
                st.write(
                    f"{LANGUAGE.get('model_architecture_description_label')}" +
                    f'：`{model_achitecture_description}`.')
                with st.expander(
                        label=f"{LANGUAGE.get('ollama_model_brief_introduction').format(selected_model=selected_model)}",
                        expanded=False):
                    st.json(query_ollama_model_info(selected_model))
                with st.expander(
                        label=f"{LANGUAGE.get('ollama_model_detailed_introduction').format(selected_model=selected_model)}",
                        expanded=False):
                    st.json(ollama.show(selected_model))
            else:
                st.warning(
                    f"{LANGUAGE.get('ollama_server_status')}：`{status_description}`",
                    icon='⚠️')
                ollama_models = PROJECT_SETTINGS.get(
                    'ollama_supported_models', [])
                if len(ollama_models) > 0:
                    last_query_model_info('ollama')
                return
        else:
            return []


def query_available_models(
        session_state: st.session_state,
        content_type: str = "Chat"):
    model_aliases_list = available_full_model_alias_list(
        content_type.capitalize())

    # 检查ollama运行状态
    if LLMs['settings']['enable_ollama'] and session_state["ollama_status"] is None:
        with st.spinner("Checking ollama server status..."):
            session_state["ollama_status"] = query_ollama_status(
                return_model_name=False)
        if session_state["ollama_status"]:
            st.rerun()

    if not LLMs['settings']['enable_ollama']:
        st.warning(f"{LANGUAGE.get('ollama_model_disabled')}", icon="⚠️")
        ollama_model_aliases = available_model_aliases_by_provider_filtered_by_type(
            provider='ollama', model_type=content_type)
        model_aliases_list = list(
            set(model_aliases_list) -
            set(ollama_model_aliases))
    else:
        if not session_state["ollama_status"]:
            st.error(
                f"{LANGUAGE.get('ollama_server_not_running')}",
                icon="🚨")
            ollama_model_aliases = available_model_aliases_by_provider_filtered_by_type(
                provider='ollama', model_type=content_type)
            model_aliases_list = list(
                set(model_aliases_list) -
                set(ollama_model_aliases))

    if len(model_aliases_list) == 0:
        st.error(
            f"{LANGUAGE.get('no_available_models_error')}",
            icon="🚨")
        st.stop()
    model_aliases_list.sort()
    return model_aliases_list


def rename_file_block(key_prefix: str, file_full_path: str):
    widget_prefix = key_prefix + '-rename-file'
    old_name = os.path.splitext(os.path.basename(file_full_path))[0]
    st.write(
        f"{LANGUAGE.get('file_to_be_renamed').format(selected_archive_file=old_name)}")
    new_name = st.text_input(
        label=f"{LANGUAGE.get('new_file_name')}",
        key=f'{widget_prefix}-text-new-name')
    if st.button(label=f"{LANGUAGE.get('rename')}",
                 key=f'{widget_prefix}-button-rename-chat'):
        new_name_file_path = file_full_path.replace(old_name, new_name)
        if os.path.exists(new_name_file_path):
            st.error(
                LANGUAGE.get('rename_error_file_exists').format(
                    old_name=old_name,
                    new_name=new_name))
            return
        else:
            os.rename(file_full_path, new_name_file_path)
            st.success(
                f"{LANGUAGE.get('rename_archive_success').format(old_name=old_name, new_name=new_name)}",
                icon='✅')
            time.sleep(REFRESH_INTERVAL)
            st.rerun()


def save_configurations_block(
        config_file_path: str,
        configs: dict,
        info: str,
        show_info: bool = True,
        refresh: bool = True):
    save_config_file(config_file_path, configs)
    if show_info:
        st.success(info, icon='ℹ️')
    if refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


def save_content_block(
        widget_key_prefix: str,
        model_name: str,
        session_state: st.session_state,
        content_type: str = 'chat'):
    content_type_ = content_type.lower()
    if content_type_ not in ['chat', 'completion']:
        st.error(
            f"{LANGUAGE.get('invalid_content_type')}",
            icon='🚨')
        st.stop()

    widget_key_prefix = f'{widget_key_prefix}-save-content-'

    repository_path = os.path.join(
        PROJECT_SETTINGS.get('customized_data_folder'),
        PROJECT_SETTINGS.get(f'{content_type_}_archive_folder'))

    pattern = PROJECT_SETTINGS.get('archive_file_naming_pattern')
    file_name = archive_file_naming(content_type_, model_name, pattern)

    if not os.path.exists(repository_path):
        os.mkdir(repository_path)
    # replace ':' in file name to avoid file name error in Windows system
    file_name = file_name.replace(':', '-')
    file_full_path = os.path.join(repository_path, file_name)

    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    with col1:
        clear_button = st.button(
            f"{LANGUAGE.get('clear_cache_button').format(content_type=translate_variable(LANGUAGE, content_type_))}",
            key=f'{widget_key_prefix}-button-clear-{content_type_}-history')
        if clear_button:
            session_state[content_type_] = []
            st.rerun()

        if len(session_state[content_type_]) > 0:
            st.info(
                f"{LANGUAGE.get('clear_cached_history')}",
                icon='ℹ️')
        else:
            st.warning(
                f"{LANGUAGE.get('no_cached_history')}",
                icon='ℹ️')
    with col2:
        if st.button(
                label=f"{LANGUAGE.get('save_to_repository_button')}",
                key=f'{widget_key_prefix}-button-save-cached-content-to-repository'):
            save_content_to_json_file(
                json_file_path=file_full_path,
                content_to_save=session_state[content_type_]
            )
            st.success(
                f"{LANGUAGE.get('content_saved').format(content_type_=LANGUAGE.get(content_type_.lower()))}",
                icon='✅')
        st.write(
            f"{LANGUAGE.get('save_to_repository').format(file_name=file_name, repository_path=repository_path)}")
    with col3:
        st.download_button(
            label=f"{LANGUAGE.get('download_history_button').format(content_type=translate_variable(LANGUAGE, content_type_))}",
            data=dump_content_to_json(
                session_state[content_type_]),
            file_name=file_name,
            mime='application/json',
            help=f"Save {content_type_} history to a JSON file.",
            key=f'{widget_key_prefix}-button-save-{content_type_}-history')
        st.info(
            f"{LANGUAGE.get('download_history').format(content_type_=content_type_.capitalize(), file_name=file_name)}",
            icon='ℹ️')


def select_an_archive_block(
        directory: str,
        widget_prefix: str,
        sub_prefix: str = None) -> str:
    widget_key = f'{widget_prefix}-{sub_prefix if sub_prefix else ""}-radio-select-archive-file'
    chat_files_list = list_files_without_extension(directory)
    selected_chat_file = st.radio(
        label='Load Files',
        options=chat_files_list,
        label_visibility='collapsed',
        key=widget_key)
    return selected_chat_file


def set_conversation_round_block(page_prefix: str):
    prefix = page_prefix + '-conversation-round-settings'
    st.subheader(f"{LANGUAGE.get('conversation_round')}")
    st.write(f"{LANGUAGE.get('conversation_round_notice')}")
    conversation_round = st.number_input(
        label='conversation round',
        min_value=1,
        max_value=10,
        value=CHAT_CONFIGS.get('conversation_round'),
        label_visibility='collapsed',
        key=f'{prefix}-number-conversation-round')
    if conversation_round != CHAT_CONFIGS.get('conversation_round'):
        CHAT_CONFIGS['conversation_round'] = conversation_round
        save_configurations_block(
            LLMs_SETTINGS_TOML_FULL_PATH,
            LLMS_CONFIGS,
            LANGUAGE.get('conversation_round_updated').format(
                conversation_round=conversation_round))


def set_data_folder_location_block(page_prefix: str):
    prefix = page_prefix + '-data-folder-location-settings'
    st.subheader(f"{LANGUAGE.get('data_folder_location')}")
    current_location = ARCHIVE_FOLDER
    st.write(
        f"{LANGUAGE.get('data_folder_description').format(default_data_folder=PROJECT_SETTINGS.get('default_data_folder'))}")
    new_archive_folder = st.text_input(
        label=f"**{LANGUAGE.get('new_location')}**",
        value=PROJECT_SETTINGS.get('default_data_folder') if current_location == os.path.join(
            ROOT_DIRECTORY,
            PROJECT_SETTINGS.get('default_data_folder')) else current_location,
        key=f'{prefix}-text-archive-folder')
    if new_archive_folder == PROJECT_SETTINGS.get('default_data_folder'):
        new_archive_folder = os.path.join(
            ROOT_DIRECTORY, PROJECT_SETTINGS.get('default_data_folder'))

    if new_archive_folder == current_location:
        return
    elif os.path.exists(new_archive_folder):
        st.write(
            f"{LANGUAGE.get('data_folder_set_to').format(new_archive_folder=new_archive_folder)}")
        st.warning(
            f"{LANGUAGE.get('data_folder_notice').format(current_location=current_location,new_archive_folder=new_archive_folder)}",
            icon='⚠️')
        st.warning(
            f"{LANGUAGE.get('data_folder_clear_notice').format(new_archive_folder=new_archive_folder)}",
            icon='⚠️')

    else:
        st.error(
            f"{LANGUAGE.get('data_folder_error').format(new_archive_folder=new_archive_folder)}",
            icon='🚨')
        return

    if st.button(
            label=f"{LANGUAGE.get('update')}",
            key=f'{prefix}-button-move-data-folder'):
        if new_archive_folder == os.path.join(
                ROOT_DIRECTORY, PROJECT_SETTINGS['default_data_folder']):
            PROJECT_SETTINGS['customized_data_folder'] = PROJECT_SETTINGS['default_data_folder']
        else:
            PROJECT_SETTINGS['customized_data_folder'] = new_archive_folder
        move_all_subfolders_and_files(
            current_location, new_archive_folder)
        save_configurations_block(
            PROJECT_SETTINGS_TOML_FULL_PATH,
            PROJECT_CONFIGS,
            LANGUAGE.get('data_folder_updated'),
            refresh=False)
        run_app()


def set_language_block(page_prefix: str):
    prefix = page_prefix + '-language-settings'
    st.subheader(LANGUAGE.get('language'))
    language_files = list_files_without_extension(LANGUAGE_FOLDER)
    old_language = PROJECT_SETTINGS.get('language')
    language = st.selectbox(
        label=LANGUAGE.get('select_a_language'),
        options=language_files,
        index=language_files.index(
            PROJECT_SETTINGS.get('language')),
        key=f'{prefix}-select-language')
    if language != old_language:
        PROJECT_SETTINGS['language'] = language
        language_switched_info = LANGUAGE.get('language_switched_info').format(
            old_language=old_language,
            new_language=language)
        save_configurations_block(
            PROJECT_SETTINGS_TOML_FULL_PATH,
            PROJECT_CONFIGS,
            language_switched_info,
            refresh=False)
        run_app()


def set_max_number_of_completions(page_prefix: str):
    prefix = page_prefix + '-max-number-of-completions-settings'
    st.subheader(
        f"{LANGUAGE.get('max_number_of_completions')}")
    st.write(LANGUAGE.get('max_number_of_completions_description'))
    max_completions = st.number_input(
        label='max number of completions',
        min_value=1,
        max_value=10,
        value=COMPLETION_CONFIGS.get('max_completions'),
        label_visibility='collapsed',
        key=f'{prefix}-number-max-completions')
    if max_completions != COMPLETION_CONFIGS.get('max_completions'):
        COMPLETION_CONFIGS['max_completions'] = max_completions
        save_configurations_block(
            LLMs_SETTINGS_TOML_FULL_PATH,
            LLMS_CONFIGS,
            LANGUAGE.get('max_number_of_completions_updated').format(
                max_completions=max_completions))


def set_prompt_block(widget_key_prefix: str, prompt_type: str = 'Chat') -> str:
    prefix = widget_key_prefix + f'-{prompt_type}-prompt-repository'
    predefined_prompts_label = sub_header_and_label_for_predefined_prompts_block(
        prompt_type)
    predefined_prompts = LLMS_CONFIGS.get(prompt_type.lower()).get(
        f'{prompt_type.lower()}_prompt_repository')
    predefined_prompt = ""
    if len(predefined_prompts) > 0:
        predefined_prompt = st.selectbox(
            label=predefined_prompts_label,
            options=predefined_prompts,
            index=0,
            key=f'{prefix}-select-box-{prompt_type}-select-a-prompt'
        )
    else:
        st.error(
            LANGUAGE.get('no_predefined_prompts'),
            icon='🚨')

    prompt = st.text_area(
        label=f"**{LANGUAGE.get('selected_prompt')}**" if predefined_prompt != "" else f"**{LANGUAGE.get('new_prompt')}**",
        value=predefined_prompt,
        key=f'{prefix}-text-area-{prompt_type}-prompt',
        height=calculate_text_area_height(predefined_prompt))

    add_to_generation_prompts = st.checkbox(
        label=f"{LANGUAGE.get('add_to_repository_label').format(prompt_type=LANGUAGE.get(prompt_type.lower()))}",
        key=f'{widget_key_prefix}-checkbox-add-to-{prompt_type}-prompts')
    if add_to_generation_prompts and prompt != "" and prompt not in predefined_prompts:
        predefined_prompts.append(prompt)
        save_configurations_block(
            LLMs_SETTINGS_TOML_FULL_PATH,
            LLMS_CONFIGS,
            f"{LANGUAGE.get('add_new_prompt_info').format(prompt_type=prompt_type, prompt=prompt)}",
            True,
            True)
    return prompt


def set_proxy_block(page_prefix: str):
    prefix = page_prefix + '-proxy-settings'
    st.subheader(f"{LANGUAGE.get('proxy')}")
    st.write(f"{LANGUAGE.get('proxy_setting_notice')}")
    http_proxy = st.text_input(
        label=LANGUAGE.get('http_proxy'),
        value=PROJECT_SETTINGS.get('proxy').get(
            'http_proxy',
            "http://127.0.0.1:10809"),
        key=f'{prefix}-text-http-proxy')
    https_proxy = st.text_input(
        label=LANGUAGE.get('https_proxy'),
        value=PROJECT_SETTINGS.get('proxy').get(
            'https_proxy',
            "http://127.0.0.1:10809"),
        key=f'{prefix}-text-https-proxy')

    col1, col2 = st.columns(2)
    with col1:
        if st.button(LANGUAGE.get('test'), key=f'{prefix}-button-test-proxy'):

            http_result, https_result = validate_proxy(http_proxy, https_proxy)
            if http_result:
                st.success(
                    f"{LANGUAGE.get('http_proxy_success').format(proxy=http_proxy)}",
                    icon='✅')
            else:
                st.error(
                    f"{LANGUAGE.get('http_proxy_failed').format(proxy=http_proxy)}",
                    icon='🚨')
            if https_result:
                st.success(
                    f"{LANGUAGE.get('https_proxy_success').format(proxy=https_proxy)}",
                    icon='✅')
            else:
                st.error(
                    f"{LANGUAGE.get('https_proxy_failed').format(proxy=https_proxy)}",
                    icon='🚨')
        st.caption(LANGUAGE.get('proxy_test_description'))
    with col2:
        if st.button(LANGUAGE.get('save'), key=f'{prefix}-button-save-proxy'):
            PROJECT_SETTINGS['proxy']['http_proxy'] = http_proxy
            PROJECT_SETTINGS['proxy']['https_proxy'] = https_proxy
            save_configurations_block(
                PROJECT_SETTINGS_TOML_FULL_PATH,
                PROJECT_CONFIGS,
                f"{LANGUAGE.get('proxy_setting_updated')}")


def set_refresh_interval_block(page_prefix: str):
    prefix = page_prefix + '-refresh-interval-settings'
    st.subheader(LANGUAGE.get('refresh_interval'))
    st.write(f"{LANGUAGE.get('refresh_interval_notice')}")
    refresh_interval = st.number_input(
        label='Refresh Interval',
        min_value=1,
        max_value=10,
        value=PROJECT_SETTINGS.get('refresh_interval'),
        key=f'{prefix}-number-refresh-interval',
        label_visibility='collapsed')
    st.write(
        f"{LANGUAGE.get('refresh_interval_set_to').format(refresh_interval=refresh_interval)}")
    if refresh_interval != PROJECT_SETTINGS.get('refresh_interval'):
        PROJECT_SETTINGS['refresh_interval'] = refresh_interval
        save_configurations_block(
            PROJECT_SETTINGS_TOML_FULL_PATH,
            PROJECT_CONFIGS,
            LANGUAGE.get('refresh_interval_updated'),
            refresh=False)


def set_web_port(page_prefix: str):
    prefix = page_prefix + '-web-port-settings'
    st.subheader(f"{LANGUAGE.get('web_port')}")
    st.write(f"{LANGUAGE.get('web_port_notice')}")
    random_web_port = st.checkbox(
        label=f"{LANGUAGE.get('random_web_port')}",
        value=PROJECT_SETTINGS.get('random_web_port'),
        key=f'{prefix}-checkbox-random-web-port')
    web_port = PROJECT_SETTINGS.get('web_port')
    if not random_web_port:
        web_port = st.number_input(
            label=f"{LANGUAGE.get('set_web_port')}",
            min_value=5000,
            max_value=65535,
            value=web_port,
            key=f'{prefix}-number-web-port', )
        st.write(
            f'{LANGUAGE.get("web_port_set_to").format(web_port=web_port)}')

    if random_web_port != PROJECT_SETTINGS.get(
            'random_web_port') or web_port != PROJECT_SETTINGS.get('web_port'):
        PROJECT_SETTINGS['random_web_port'] = random_web_port
        PROJECT_SETTINGS['web_port'] = web_port
        save_configurations_block(
            PROJECT_SETTINGS_TOML_FULL_PATH,
            PROJECT_CONFIGS,
            LANGUAGE.get('random_web_port_notice'),
            refresh=False)
        run_app()


def set_verbose(page_prefix: str):
    prefix = page_prefix + '-verbose-settings'
    st.subheader(LANGUAGE.get('verbose_mode'))
    st.write(f"{LANGUAGE.get('verbose_mode_description')}")
    verbose = st.checkbox(
        label=f"{LANGUAGE.get('verbose_mode_label')}",
        value=PROJECT_SETTINGS.get('verbose_mode'),
        key=f'{prefix}-checkbox-verbose')
    if verbose != PROJECT_SETTINGS.get('verbose_mode'):
        PROJECT_SETTINGS['verbose_mode'] = verbose
        save_configurations_block(
            PROJECT_SETTINGS_TOML_FULL_PATH,
            PROJECT_CONFIGS,
            LANGUAGE.get('verbose_mode_updated'),
            refresh=True)


def show_chat_archive_block(file_name, archive_data_json: json):
    st.write(
        LANGUAGE.get('selected_chat_archive').format(
            chat=LANGUAGE.get('chat'),
            file_name=file_name, number=len(archive_data_json),
            chats=LANGUAGE.get('chats')))
    for message in archive_data_json:
        with st.chat_message(message["role"]):
            if message.get('model', None):
                st.markdown(
                    message["content"] +
                    f"\n\n`@{message['timestamp']}` - `{message['model']}`")
            else:
                st.markdown(message["content"])


def show_completion_archive_block(
        widget_prefix: str,
        archive_data_json: json,
        file_name: str = ""):
    prefix = widget_prefix + 'display-completion'
    if file_name != "":
        st.write(
            LANGUAGE.get('selected_completion_archive').format(
                completion=LANGUAGE.get('completion'),
                file_name=file_name, number=len(archive_data_json),
                completions=LANGUAGE.get('completions')))
    show_prompt = st.checkbox(
        label=LANGUAGE.get('show_prompt'),
        value=False,
        key=f'{prefix}-checkbox-show-prompt')
    for index, completion in enumerate(archive_data_json):
        with st.container(border=True):
            st.subheader(f"**{LANGUAGE.get('completion')} {index + 1}:**")
            st.write(LANGUAGE.get('generated_completion_info').format(
                # compatibility with old data
                time=completion["timestamp"] if completion.get(
                    "timestamp", False) else completion["time"],
                prompt=completion["prompt"][0:100].replace("#", ""),
                model=completion["model"]))
            if show_prompt:
                st.write(f"**{LANGUAGE.get('prompt')}:**")
                st.text_area(
                    label=LANGUAGE.get('prompt'),
                    label_visibility='collapsed',
                    height=calculate_text_area_height(
                        completion["prompt"], word_per_line=150),
                    value=completion["prompt"],
                    key=f'{prefix}-text-area-prompt-{index}')

            st.write(f"**{LANGUAGE.get('content')}:**")
            content = st.text_area(
                label=LANGUAGE.get('content'),
                value=completion["content"],
                height=calculate_text_area_height(
                    completion["content"],
                    word_per_line=120),
                label_visibility='collapsed',
                key=f'{prefix}-text_area-text-content-{index}')
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                        label=LANGUAGE.get('copy_completion').format(
                            index=index + 1),
                        key=f'{prefix}-button-copy-content-{index}'):
                    pyperclip.copy(content)
                    st.success(LANGUAGE.get('content_copied'), icon='✅')
            with col2:
                if st.button(
                        label=LANGUAGE.get('delete_completion').format(
                            index=index + 1),
                        key=f'{prefix}-button-delete-content-{index}'):
                    archive_data_json.pop(index)
                    st.success(
                        LANGUAGE.get('content_deleted'),
                        icon='✅')
                    time.sleep(REFRESH_INTERVAL)
                    st.rerun()


def show_llms_request_params_block(page_prefix: str):
    prefix = page_prefix + '-common-settings-request-params'
    st.subheader(LANGUAGE.get('request_parameters_description'))
    st.info(f"**{LANGUAGE.get('note')}**: " +
            LANGUAGE.get('request_parameters_notice'), icon='ℹ️')
    st.write(LANGUAGE.get('request_parameters_additional_info'))
    built_in_parameters = LLMModelParamsConfigUI.get_built_in_request_parameter_list()
    for param in built_in_parameters:
        provider = ""
        if param == 'disable_search' or param == 'penalty_score' or param == 'max_output_tokens':
            provider = 'baidu'
        elif param == 'enable_search' or param == 'repetition_penalty':
            provider = 'qwen'
        elif param == 'num_ctx' or param == 'num_predict' or param == 'repeat_penalty':
            provider = 'ollama'
        elif param == 'frequency_penalty':
            provider = 'openai'
        if provider != "":
            st.markdown(
                f"- **{param}**: {LLMModelParamsConfigUI.get_supported_request_parameter_description(param)} " +
                LANGUAGE.get('provider_models_only').format(
                    provider=translate_variable(
                        LANGUAGE,
                        provider)))
        else:
            if param == 'max_tokens':
                st.markdown(
                    f"- **{param}**: {LLMModelParamsConfigUI.get_supported_request_parameter_description(param)} " +
                    LANGUAGE.get('max_tokens_additional_info').format(
                        openai=translate_variable(
                            LANGUAGE,
                            'openai'),
                        qwen=translate_variable(
                            LANGUAGE,
                            'qwen')))
            else:
                st.markdown(
                    f"- **{param}**: {LLMModelParamsConfigUI.get_supported_request_parameter_description(param)}")
    customized_params = LLMModelParamsConfigUI.get_customized_request_parameter_list()
    if len(customized_params) > 0:
        st.write(f"**{LANGUAGE.get('customize_request_parameters')}**:")
        for param in customized_params:
            st.markdown(
                f"- **{param}**: {LLMModelParamsConfigUI.get_supported_request_parameter_description(param)}")

    st.subheader(LANGUAGE.get('recommended_request_parameters_title'))
    st.write(LANGUAGE.get('recommended_request_parameters_description'))
    api_doc_urls = {
        'openai': 'https://platform.openai.com/docs/api-reference/chat/create',
        'qwen': 'https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api',
        'baidu': 'https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu',
        'ollama': 'https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values'}
    providers = ['openai', 'qwen', 'baidu', 'ollama']
    for provider in providers:
        request_params = ProviderLLMModelConfigUI.recommended_request_parameters(
            provider)
        request_params = ','.join(request_params)
        official_api_document_url = LANGUAGE.get('official_api_document').format(
            provider=translate_variable(
                LANGUAGE, provider), url=api_doc_urls.get(
                provider, ""))
        st.write(
            LANGUAGE.get('recommended_request_parameters_by_provider').format(
                provider=translate_variable(
                    LANGUAGE,
                    provider),
                parameters=request_params) +
            official_api_document_url)
    st.write(
        f"**{translate_variable(LANGUAGE, 'openai_compatible')}**: "
        f"{LANGUAGE.get('openai_compatible_request_parameters_description')}")

    st.subheader(LANGUAGE.get('add_common_request_parameters_to_new_model'))
    st.write(
        LANGUAGE.get('add_common_request_parameters_when_adding_model_description').format(
            section=LANGUAGE.get('recommended_request_parameters_title')))
    add_common_request_parameters_to_new_model = st.checkbox(
        label=LANGUAGE.get('add_common_request_parameters_when_adding_model_label'),
        value=PROJECT_SETTINGS.get('add_common_request_parameters_to_new_model'),
        key=f'{prefix}-checkbox-add-common-request-parameters-to-new-model')
    if PROJECT_SETTINGS.get(
            'add_common_request_parameters_to_new_model') != add_common_request_parameters_to_new_model:
        PROJECT_SETTINGS['add_common_request_parameters_to_new_model'] = add_common_request_parameters_to_new_model
        save_configurations_block(
            PROJECT_SETTINGS_TOML_FULL_PATH,
            PROJECT_CONFIGS,
            LANGUAGE.get('add_common_request_parameters_to_new_model_option_updated'),
            refresh=False)


def show_model_description(provider, model_alias):
    llm_config = LLMs.get(provider)
    model_description = llm_config.get('descriptions').get(
        model_alias, LANGUAGE.get('no_model_description'))
    model_description = model_description.replace('\n\n', '\n')
    markdown = f"> {LANGUAGE.get('model_description_with_colon')} {model_description}"
    st.markdown(markdown)


def sub_header_and_label_for_predefined_prompts_block(prompt_type: str):
    if prompt_type not in PROJECT_SETTINGS.get('prompt_types'):
        st.error(
            f"{LANGUAGE.get('invalid_prompt_type')}",
            icon='🚨')
        st.stop()

    title = f"`{LANGUAGE.get(prompt_type.lower())}` "
    if prompt_type.lower() == 'chat':
        st.subheader(
            title +
            f"{LANGUAGE.get('system_prompt')}")
        predefined_prompts_label = f"{LANGUAGE.get('select_system_prompt_label')}"
    else:
        st.subheader(title + f"{LANGUAGE.get('prompt')}")
        predefined_prompts_label = f"{LANGUAGE.get('select_prompt_label')}"
    return f"**{predefined_prompts_label}**"

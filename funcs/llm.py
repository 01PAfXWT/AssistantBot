import copy
import json
import os

import dashscope
import ollama
import qianfan
from openai import OpenAI

from config import LANGUAGE, LLMS_CONFIGS, PROJECT_CONFIGS, PROJECT_SETTINGS_TOML_FULL_PATH
from funcs.common import (
    blank_LLM_settings,
    current_timestamp_to_string,
    filter_models,
    get_current_function_name,
    print_to_console,
    save_config_file,
)
from funcs.language import translate_variable


PROJECT_SETTINGS = PROJECT_CONFIGS.get('settings')
LLMs = LLMS_CONFIGS.get('LLMs')


def accept_system_prompt_baidu_models():
    baidu_configs = LLMs.get('baidu')
    accept_list = []
    not_accept_list = []
    for model, endpoint in baidu_configs['models'].items():
        if 'ERNIE' in str(endpoint).upper(
        ) or 'COMPLETIONS' in str(endpoint).upper():
            accept_list.append(model)
        else:
            not_accept_list.append(model)
    return accept_list, not_accept_list


def available_full_model_alias_list(inference_type: str = 'Chat') -> list:
    model_list = []
    providers = available_providers()
    for provider in providers:
        models = available_model_aliases_by_provider_filtered_by_type(
            provider=provider, model_type=inference_type)
        for model in models:
            model_list.append(model)
    return model_list


def available_model_aliases_by_provider_filtered_by_type(
        provider: str, model_type: str):
    if model_type.capitalize() not in ['Chat', 'Completion']:
        raise ValueError(f"Model type {model_type} is not supported.")
    if provider not in LLMs:
        raise ValueError(f"Provider {provider} is not supported.")
    settings = LLMs[provider]['settings']
    paid_models = settings.get('paid_models', [])
    free_models = settings.get('free_models', [])
    models = list(set(paid_models + free_models))

    model_type_key = 'exclude_non_' + model_type.lower() + '_models'
    exclude_models = settings.get(model_type_key, [])

    if exclude_models:
        models = filter_models(models, exclude_models)

    return models


def available_models_by_provider(provider: str):
    provider_ = provider.lower()
    if provider_ not in LLMs:
        raise ValueError(f"Provider {provider_} is not supported.")
    settings = LLMs[provider_]['settings']
    paid_models = settings.get('paid_models', [])
    free_models = settings.get('free_models', [])
    models = list(LLMs[provider_].get('models', []).keys())
    model_list = list(set(paid_models + free_models + models))
    return model_list


def available_providers(translate: bool = False):
    providers = []
    for provider, activated in LLMs.get('settings').items():
        if activated:
            providers.append(provider.replace('enable_', ''))
    if translate:
        providers = translate_variable(LANGUAGE, providers)
    return providers


def baidu_chat_stream(model_alias: str,
                      messages: list,
                      model_parameters: dict = None, ):
    """
    Baidu chat using qianfan SDK
    sdk: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/xlmokikxe
    """
    baidu_configs = LLMs.get('baidu')
    os.environ["QIANFAN_AK"] = baidu_configs.get(
        'api_keys').get('client_id')
    os.environ["QIANFAN_SK"] = baidu_configs.get(
        'api_keys').get('client_secret')

    set_proxy_for_session('baidu', model_alias)

    chat_comp = qianfan.ChatCompletion()
    support_list, not_support_list = accept_system_prompt_baidu_models()
    if model_alias in support_list:
        resp = chat_comp.do(
            model=baidu_configs['models'][model_alias],
            messages=messages,
            # system=model_parameters['system'],
            # top_p=model_parameters['top_p'],
            # temperature=model_parameters['temperature'],
            # penalty_score=model_parameters['penalty_score'],
            # max_output_tokens=model_parameters['max_tokens'],
            **(model_parameters or {}),
            stream=True,
        )
    else:
        # insert system prompt to the first message
        messages[0]['content'] = model_parameters['system'] + \
            messages[0]['content']
        model_parameters.pop('system', None)
        resp = chat_comp.do(
            model=baidu_configs['models'][model_alias],
            messages=messages,
            # top_p=model_parameters['top_p'],
            # temperature=model_parameters['temperature'],
            # penalty_score=model_parameters['penalty_score'],
            **(model_parameters or {}),
            stream=True,
        )
    for chunk in resp:
        yield chunk['body']['result']


def baidu_completion_stream(
        model_alias: str,
        prompt: str,
        model_parameters: dict = None):
    ll_config = LLMs.get('baidu')
    set_proxy_for_session('baidu', model_alias)
    os.environ["QIANFAN_AK"] = ll_config.get(
        'api_keys').get('client_id')
    os.environ["QIANFAN_SK"] = ll_config.get(
        'api_keys').get('client_secret')
    completion = qianfan.Completion()
    response = completion.do(
        model=ll_config['models'][model_alias],
        prompt=prompt,
        **(model_parameters or {}),
        stream=True,
    )
    for chunk in response:
        yield chunk['result']


def build_chat_history(
        current_model: str,
        system_prompt: str,
        messages: list,
        conversation_round: int,
        keep_system_prompt: bool = True) -> list:
    """
    build chat archives for next conversation round.
    - remove sources block in previous messages to reduce content length.
    - remove unnecessary keys (`model`, `timestamp`) in messages.
    - add system prompt message to the beginning if necessary, default is True.
    - keep historical messages for the specified number of rounds.
    - If the roler of the last two messages is the `user`, then remove the previous one and keep the latest one.

    Args
        current_model: str, current model name, only use for debugging purpose
        system_prompt: str, system prompt message
        messages: list, chat archives messages
        conversation_round: int, conversation rounds
        delimiter: str, sources delimiter
        keep_system_prompt: bool, insert the system prompt message to the first of messages or not
        debug_verbose: bool, debug verbose
    Returns
        list: updated chat archives messages
    Examples:
        messages = [
            {'role': 'system', 'content': 'system prompt', 'model': 'model', 'timestamp': '2021-10-10 10:10:10'},
            {'role': 'user', 'content': 'user message', 'model': 'model', 'timestamp': '2021-10-10 10:10:10'},
            {'role': 'assistant', 'content': 'assistant message', 'model': 'model', 'timestamp': '2021-10-10 10:10:10'},
        ]
        build_chat_history(
            current_model='model',
            system_prompt='system prompt',
            messages=messages,
            conversation_round=2,
            delimiter='**Sources:**',
            keep_system_prompt=True)
    """
    messages_copy = copy.deepcopy(messages)

    # remove unnecessary keys (`model`, `system`,`timestamp`) and split the content at
    # the delimiter to remove `sources content`
    for message in messages_copy:
        message.pop('model', None)
        message.pop('timestamp', None)
        message.pop('system', None)

    if messages_copy[0]['role'] == 'system':
        system_prompt_message = messages_copy.pop(0)
    else:
        system_prompt_message = {'role': 'system', 'content': system_prompt}

    system_prompt_message['content'] = system_prompt

    messages_copy = curate_history_messages(messages_copy)

    # retain archives conversation rounds
    if conversation_round == 1:  # Q&A mode, only keep the latest message of `user`
        messages_copy = [messages_copy[-1]]
        print_to_console(
            'Q&A mode, only keep the latest message of `user`.',
            get_current_function_name())
    else:
        last_message = messages_copy.pop()  # temporarily remove last message
        if len(messages_copy) > (conversation_round - 1) * 2:
            messages_copy = messages_copy[-(conversation_round - 1) * 2:]
            print_to_console(
                f'Func:build_chat_history:: {len(messages_copy)}>{(conversation_round - 1) * 2}, keep the latest {conversation_round} conversation rounds.',
                get_current_function_name())
        messages_copy.append(last_message)

    # add system prompt message to the beginning if required and the system
    # prompt message is not empty
    if keep_system_prompt:
        if system_prompt_message['content'] != '':
            messages_copy.insert(0, system_prompt_message)
        else:
            print_to_console(
                f'{current_model}: system prompt is empty, remove system '
                f'message from messages.', get_current_function_name())

    return messages_copy


def curate_history_messages(messages: list) -> list:
    """
    Curate chat archives messages. All messages must be in user or assistant roles.
    If a system prompt is found at the beginning of the chat archives, it will be removed.
    If there are two user messages in a row, remove the first one.
    If there are two assistant messages in a row, remove the second one.
    Args:
        messages: list of messages in the chat archives.
    Returns:
        list: curated chat archives messages.
    """
    if not messages:
        raise ValueError("The chat archives is empty.")

    curated_messages = messages.copy()
    if curated_messages[0]['role'] == 'system':
        log = f'remove the first system message: {curated_messages[0]["content"]}'
        print_to_console(log, get_current_function_name())
        curated_messages.pop(0)

    if not curated_messages:
        raise ValueError(
            "The chat archives is empty after removing system message.")

    if len(curated_messages) == 1 and curated_messages[0]['role'] != 'user':
        raise ValueError("The chat archives must start with a user message.")

    message_to_remove = set()
    for i in range(len(curated_messages) - 1):
        if curated_messages[i]['role'] not in ['user', 'assistant']:
            raise ValueError(f"Invalid role in message {i + 1}.")

        if curated_messages[i]['role'] == 'user' and curated_messages[i +
                                                                      1]['role'] == 'user':
            # if there are two user messages in a row, remove the first one
            message_to_remove.add(i)
            log = f'two user messages are in a row. Remove the first user message: {curated_messages[i]}'
            print_to_console(log, get_current_function_name())
        elif curated_messages[i]['role'] == 'assistant' and curated_messages[i + 1]['role'] == 'assistant':
            # if there are two assistant messages in a row, remove the second one
            message_to_remove.add(i + 1)
            log = f'two assistant messages are in a row. Remove the second assistant message: {curated_messages[i + 1]}'
            print_to_console(log, get_current_function_name())

    curated_messages = [msg for i, msg in enumerate(
        curated_messages) if i not in message_to_remove]

    return curated_messages


def generate_ollama_model_architecture_description(model_name: str):
    try:
        architecture = query_ollama_model_information_by_keyword(
            model_name, 'architecture')
        context_length = query_ollama_model_information_by_keyword(
            model_name, 'context_length')
        embedding_length = query_ollama_model_information_by_keyword(
            model_name, 'embedding_length')
        model_description = f"Architecture: {architecture}, Context Length: {context_length}, Embedding Length: {embedding_length}"
        return model_description
    except Exception as e:
        print(f"Failed to retrieve model information from ollama: {e}")
        return None


def identify_model_by_model_alias(alias: str):
    provider = identify_provider_by_model_alias(alias)
    if provider is not None:
        return LLMs[provider]['models'].get(alias)


def identify_provider_by_model_alias(model_alias: str):
    for provider in available_providers():
        models = available_models_by_provider(provider=provider)
        if model_alias in models:
            return provider
    return None


def is_paid_model(model_alias: str) -> str:
    paid_models = []
    providers = available_providers()
    for provider in providers:
        paid_models = paid_models + \
            LLMs[provider]['settings'].get('paid_models', [])
    if model_alias in paid_models:
        return f"({LANGUAGE.get('paid')})"
    else:
        return f"({LANGUAGE.get('free')})"


def list_baidu_models_by_SDK(model_type: str = 'Chat') -> list:
    baidu_configs = LLMs.get('baidu')
    os.environ["QIANFAN_AK"] = baidu_configs.get(
        'api_keys').get('client_id')
    os.environ["QIANFAN_SK"] = baidu_configs.get(
        'api_keys').get('client_secret')
    if model_type.lower() == 'chat':
        m = qianfan.ChatCompletion()
    else:
        m = qianfan.Completion()
    return list(m.models())


def list_openai_models(use_proxy: bool = True):
    openai_configs = LLMs.get('openai')
    api_key = openai_configs.get('api_keys').get('openai')

    http_proxy = PROJECT_SETTINGS.get('proxy').get('http_proxy')
    https_proxy = PROJECT_SETTINGS.get('proxy').get('https_proxy')

    if use_proxy and http_proxy and https_proxy:
        os.environ['HTTP_PROXY'] = http_proxy
        os.environ['HTTPS_PROXY'] = https_proxy

    client = OpenAI(api_key=api_key)
    response = client.models.list()

    response_json = json.loads(response.to_json())
    model_list = [model['id']
                  for model in response_json['data']] if response_json else []
    return sorted(model_list)


def LLM_chat_service(
        model_alias: str,
        messages: list,
        model_parameters: dict = None):
    provider = identify_provider_by_model_alias(model_alias)
    func_name = f'{provider}_chat_stream'
    LLM_chat_stream = globals().get(func_name)

    if PROJECT_SETTINGS.get('verbose_mode'):
        print_to_console('-' * 30, get_current_function_name())
        print_to_console(
            f"Provider: {provider}, Model Alias: {model_alias}, Model: {identify_model_by_model_alias(model_alias)}")
        print_to_console(f"Messages: {messages}")
        print_to_console(f"Model Parameters: {model_parameters}")
        print_to_console('-' * 30, get_current_function_name())

    return LLM_chat_stream(model_alias, messages, model_parameters)


def LLM_completion_service(
        model_alias: str,
        prompt: str,
        model_parameters: dict = None):
    provider = identify_provider_by_model_alias(model_alias)
    func_name = f'{provider}_completion_stream'
    LLM_completion_stream = globals().get(func_name)

    if PROJECT_SETTINGS.get('verbose_mode'):
        print_to_console('-' * 30, get_current_function_name())
        print_to_console(
            f"Provider: {provider}, Model Alias: {model_alias}, Model: {identify_model_by_model_alias(model_alias)}")
        print_to_console(f"Prompt: {prompt}")
        print_to_console(f"Model Parameters: {model_parameters}")
        print_to_console('-' * 30, get_current_function_name())

    return LLM_completion_stream(model_alias, prompt, model_parameters)


def ollama_chat_stream(
        model_alias: str,
        messages: list,
        model_parameters: dict = None):
    """
    ollama chatbot using ollama SDK.
    Args:
        model_alias: str, ollama model alias
        messages: list, chat history messages
        model_parameters: model settings, e.g., seed, temperature, top_p, penalty, max_tokens, etc.
    References:
        https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    llm_configs = LLMs.get('ollama')
    model = llm_configs.get('models').get(model_alias)
    stream = ollama.chat(
        model=model,
        messages=messages,
        options=model_parameters or {},
        stream=True,
    )
    for chunk in stream:
        yield chunk['message']['content']


def ollama_completion_stream(
        model_alias: str,
        prompt: str,
        model_parameters: dict = None):
    ll_config = LLMs.get('ollama')
    completion = ollama.generate(
        model=ll_config['models'][model_alias],
        prompt=prompt,
        stream=True,
        options=model_parameters or {},
    )
    for chunk in completion:
        yield chunk['response']


def openai_chat_stream(
        model_alias: str,
        messages: list,
        model_parameters: dict = None,
):
    """
    OpenAI chat using OpenAI SDK, 流式输出
    OpenAI sdk: https://platform.openai.com/docs/api-reference/streaming
    Args:
        model_alias: str, OpenAI model alias
        messages: list, chat history messages
        model_parameters: dict, model settings. No use for free model, for future use.
    """
    llm_configs = LLMs.get('openai')
    model = llm_configs.get('models').get(model_alias)
    api_key = llm_configs.get('api_keys').get('openai')

    set_proxy_for_session('openai', model_alias)

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **(model_parameters or {}),
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def openai_compatible_chat_stream(
        model_alias: str,
        messages: list,
        model_parameters: dict = None
):
    llm_configs = LLMs.get('openai_compatible')
    model = llm_configs.get('models').get(model_alias)
    api_key = llm_configs.get('api_keys').get(model_alias)
    base_url = llm_configs.get('base_urls').get(model_alias)

    set_proxy_for_session('openai_compatible', model_alias)

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **(model_parameters or {}),
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def openai_compatible_completion_stream(
        model_alias: str,
        prompt: str,
        model_parameters: dict = None
):
    messages = [{'role': 'user', 'content': prompt}]
    return openai_compatible_chat_stream(
        model_alias, messages, model_parameters)


def openai_completion_stream(
        model_alias: str,
        prompt: str,
        model_parameters: dict = None
):
    messages = [{'role': 'user', 'content': prompt}]
    return openai_chat_stream(model_alias, messages, model_parameters)


def query_ollama_model_info(model_name: str):
    try:
        models = ollama.list()['models']
    except Exception as e:
        print_to_console(
            f"Failed to retrieve models from ollama: {e}",
            get_current_function_name())
        return None

    return next(
        (model for model in models if model['name'] == model_name),
        None)


def query_ollama_model_information_by_keyword(
        model_name: str, keyword: str = 'context_length'):
    keywords = ['context_length', 'architecture', 'embedding_length']
    if keyword not in keywords:
        raise ValueError(
            f"Keyword {keyword} is not supported. Supported keywords are {keywords}")
    try:
        model_info = ollama.show(model_name)['model_info']
        for key, value in model_info.items():
            if keyword in key:
                return value
    except Exception as e:
        print(f"Failed to retrieve model information from ollama: {e}")
        return None

    return model_info.get(keyword)


def query_ollama_status(return_model_name: bool = False):
    ollama_configs = LLMs.get('ollama', None)
    if ollama_configs is None:
        ollama_configs = blank_LLM_settings()
        LLMs['ollama'] = ollama_configs
    try:
        models_list = ollama.list().get('models')
        supported_models = [model['name'] for model in models_list]

        if supported_models != PROJECT_SETTINGS.get('ollama_supported_models'):
            PROJECT_SETTINGS['ollama_supported_models'] = supported_models
            PROJECT_SETTINGS['ollama_supported_models_query_time'] = current_timestamp_to_string(
                "%Y-%m-%d %H:%M:%S")
            save_config_file(PROJECT_SETTINGS_TOML_FULL_PATH, PROJECT_CONFIGS)
            print_to_console(
                f"{LANGUAGE.get('ollama_model_updated')}",
                get_current_function_name())

        if return_model_name:
            return True, supported_models
        else:
            return True
    except Exception as e:
        print_to_console(
            f"{LANGUAGE.get('query_ollama_model_failed').format(e=e)}",
            get_current_function_name())
        if return_model_name:
            return False, []
        else:
            return False


def qwen_chat_stream(
        model_alias: str,
        messages: list,
        model_parameters: dict = None,
):
    """
    # https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api
    """
    llm_configs = LLMs.get('qwen')
    model = llm_configs.get('models').get(model_alias)

    set_proxy_for_session('qwen', model_alias)

    dashscope.api_key = llm_configs.get('api_keys').get('qwen')
    responses = dashscope.Generation.call(
        model=model,
        messages=messages,
        **(model_parameters or {}),
        result_format='message',
        stream=True,
        incremental_output=True)

    for response in responses:
        if response.status_code == 200:
            yield response.output.choices[0].message.content
        else:
            print_to_console(
                'Request id: %s, Status code: %s, error code: %s, error message: %s' %
                (response.request_id,
                 response.status_code,
                 response.code,
                 response.message),
                get_current_function_name())


def qwen_completion_stream(
        model_alias: str,
        prompt: str,
        model_parameters: dict = None
):
    """
    # https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api
    """
    llm_configs = LLMs.get('qwen')
    model = llm_configs.get('models').get(model_alias)

    set_proxy_for_session('qwen', model_alias)

    dashscope.api_key = llm_configs.get('api_keys').get('qwen')
    responses = dashscope.Generation.call(
        model=model,
        prompt=prompt,
        **(model_parameters or {}),
        result_format='message',
        stream=True,
        incremental_output=True)

    for response in responses:
        if response.status_code == 200:
            yield response.output.choices[0].message.content
        else:
            print_to_console(
                'Request id: %s, Status code: %s, error code: %s, error message: %s' %
                (response.request_id,
                 response.status_code,
                 response.code,
                 response.message),
                get_current_function_name())


def set_proxy_for_session(provider, model_alias):
    llm_configs = LLMs.get(provider)
    use_proxy = model_alias in llm_configs['settings']['use_proxy_models']
    http_proxy = PROJECT_SETTINGS.get('proxy').get('http_proxy', None)
    https_proxy = PROJECT_SETTINGS.get('proxy').get('https_proxy', None)

    if use_proxy:
        if http_proxy and https_proxy:
            os.environ['HTTP_PROXY'] = http_proxy
            os.environ['HTTPS_PROXY'] = https_proxy
        else:
            raise ValueError(
                f"[{get_current_function_name()}] {LANGUAGE.get('proxy_not_set')}")
    else:
        if 'HTTP_PROXY' in os.environ:
            os.environ.pop('HTTP_PROXY')
        if 'HTTPS_PROXY' in os.environ:
            os.environ.pop('HTTPS_PROXY')

def show_model_proxy_status(provider, model_alias):
    proxies = PROJECT_SETTINGS.get('proxy')
    llm_configs = LLMs.get(provider)
    use_proxy = model_alias in llm_configs['settings']['use_proxy_models']
    if use_proxy:
        return f"{LANGUAGE.get('proxy_status').format(provider=provider.capitalize(), proxies=proxies)}"
    else:
        return None


def validate_model_info(model_params: dict):
    value_errors = []
    name_mapping = {
        'modified_model_alias': 'model alias',
        'model': 'model name',
        'base_url': 'base url',
        'api_key': 'api key'}
    for k_, v_ in model_params.items():
        if k_ in [
            'modified_model_alias',
            'model',
            'base_url',
                'api_key'] and v_ == "":
            value_errors.append(f"`{name_mapping[k_]}`")
    if len(value_errors) > 0:
        error_info = ','.join(value_errors)
        raise ValueError(
            f"[{get_current_function_name()}] {LANGUAGE.get('empty_variable_value').format(variable=error_info)}")

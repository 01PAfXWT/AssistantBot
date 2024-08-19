# AssistantBot

**AssistantBot** is a lightweight app built with Streamlit, offering a user-friendly interface for interacting with Large Language Models (LLMs). The primary features of AssistantBot include:

* **Intuitive GUI:** Provides an easy-to-use graphical interface for chatting or completing tasks with LLMs, allowing for individual configuration of model parameters.
* **Prompt Repository:** A management system for storing and organizing prompts used with LLMs.
* **Archive Management:** The ability to save chat sessions or completions and manage the archived data.

# Installation

```
git clone
cd AssistantBot
pip install -r requirements.txt
python main.py
```

# Usage

Run app:

```
cd AssistantBot
python main.py
```

Choose a function page from the list on the sidebar and follow the instructions on the page.

# Configuration

The app's configurations are stored in TOML files. There are two key files: `AssistantBot\configs.toml` and `AssistantBot\data\LLMs_configs.toml`. These two files will be created at during the app's first run.

> **Note**: Do not directly modify these toml files, using the  `Settings` to adjust options instread.

## Common settings

You can configure `Lanuage`, `Web port`, `data foler location`, `page refresh interval`, and `proxy` options,  navigate to `settings` to view descriptions for each option.

> **Note:**
>
> **Data Folder Location:** The `data` folder is the directory that stores chat/completion archives and LLM settings (`LLMs_config.toml `). It is recommended to move the data folder to cloud storage, such as OneDrive, to sync data. The `LLMs_configs.toml `file is located within the `data `folder, and if the location of the `data` folder changes, this file will be relocated accordingly.

## LLM settings

The app has built-in support for LLM providers including OpenAI, OpenAI Compatible, Baidu, Qwen, and Ollama (local LLM). You can easily add, modify, delete models from these providers in the `LLM Settings`. Various provider may have differet options, view the avaiable options for provider in `LLM Settings`.

> **Note**:
>
> - Any OpenAI-compatible model can be added through the "OpenAI Compatible" provider by correctly setting the `model name`, `base_url`, `api_key`, and other necessary options. See [Docs: add OpenAi Compatible service provider](docs/add_openai_compatible_service_provider.md) for details.
> - You can configure request parameters for model, or you can customize your own request parameters. See the `request parameters`  section for details. Refer to[Docs: add request parameter](docs/add_request_parameter.md) for how to speicify request parameters for models and how to add new request parameters.

## Backup and Restore

You can back up the two key toml files along with the archives of chat and completion. All these files will be compressed into a backup file. The backup file name is `backup-(timestamp).zip`, it is stored in the `backups` directory within the  `data`  folder.

You can also restore settings and archives from an existing backup file. However, the restore operation will overwrite all the exsiting files.

# Translation

The app supports both Chinese and English, with English set as the default language. You can easily switch languages in the `settings` under `Language`.

Additionally, the language files are located in the `AssistantBot\locales` directory. You can add your own language file if desired.

Creating a new language file is straightforward. Use GPT to generate content based on an existing language file, such as Chinese (zh_CN.toml) or English (en_US.toml), with the following prompt:

> The following is the content of a TOML file used for storing descriptions in Chinese for a Python project. Please translate all the Chinese in the content into English.
>
> The content of the TOML file is given in the markdown block:
>
> ``copy the content of the language file here``

Save the generated language file into `AssistantBot\locales`,  refresh the web page to see the new language option.

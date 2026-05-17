import yaml
from jinja2 import Template
from langsmith import Client


def prompt_template_config(path: str, name: str) -> Template:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return Template(config["prompts"][name])


def prompt_template_registry(name: str):

    ls_client = Client()

    ls_template = ls_client.pull_prompt("retrieval_generation")

    template = Template(ls_template.messages[0].prompt.template)

    return template

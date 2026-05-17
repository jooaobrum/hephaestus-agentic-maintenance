import yaml
from jinja2 import Template
from langsmith import Client


def prompt_template_config(path: str, name: str) -> str:
    """Load prompt from YAML and return as string (no template substitution)."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    prompt_text = config["prompts"][name]

    # Always convert to string (handles both str and Template objects)
    return str(prompt_text)


def prompt_template_registry(name: str):
    ls_client = Client()

    ls_template = ls_client.pull_prompt("retrieval_generation")

    template = Template(ls_template.messages[0].prompt.template)

    return template

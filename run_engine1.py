import json
import os
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import validate
from openai import OpenAI

load_dotenv()

PROMPT_PATH = Path("prompts/engine1_system.txt")
SCHEMA_PATH = Path("schemas/engine1_output.schema.json")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
schema_wrapper = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

INPUT_PATH = Path("inputs/engine1.sample.json")
engine1_input = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

response = client.responses.create(
    model="gpt-5",
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(engine1_input)}
    ],
   text={
    "format": {
        "type": "json_schema",
        "name": "engine1_output",
        "schema": schema_wrapper,
        "strict": True
    }
}
)

output = json.loads(response.output_text)

validate(instance=output, schema=schema_wrapper)

print(json.dumps(output, indent=2))
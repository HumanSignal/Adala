import requests
import json

response = requests.post(
    "http://localhost:30001/submit",
    json={
        "agent": {
            "skills": [
                {
                    "type": "classification",
                    "name": "text_classifier",
                    "instructions": "Classify the text.",
                    "input_template": "Text: {text}",
                    "output_template": "Classification result: {label}",
                    "labels": {"label": ["label1", "label2", "label3"]},
                }
            ],
            "runtimes": {
                "default": {
                    "type": "openai-chat",
                    "model": "gpt-3.5-turbo-0125",
                    "api_key": "...",
                }
            },
            "environment": {
                "type": "kafka_filestream",
                "kafka_bootstrap_servers": "kafka:9093",
                "kafka_input_topic": "adala/input",
                "kafka_output_topic": "adala/output",
                "input_file": "s3://htx-test/input.csv",
                "output_file": "s3://htx-test/output.csv",
                "error_file": "s3://htx-test/errors.csv",
                "pass_through_columns": ["text", "id"],
            },
        }
    },
)

try:
    print(json.dumps(response.json(), indent=2))
except:
    print(response.text)

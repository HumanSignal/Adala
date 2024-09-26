import pytest
import os


@pytest.mark.vcr
def test_agent_is_serializable():
    agent_data = {
        "environment": {
            "type": "AsyncKafkaEnvironment",
            "kafka_bootstrap_servers": "",
            "kafka_input_topic": "",
            "kafka_output_topic": "",
            "timeout_ms": 1,
        },
        "skills": [
            {
                "type": "ClassificationSkill",
                "name": "text_classifier",
                "instructions": "Always return the answer 'Feature Lack'.",
                "input_template": "{text}",
                "output_template": "{output}",
                "labels": [
                    "Feature Lack",
                    "Price",
                    "Integration Issues",
                    "Usability Concerns",
                    "Competitor Advantage",
                ],
            }
        ],
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 200,
                "temperature": 0,
                "batch_size": 100,
                "timeout": 10,
                "verbose": False,
            }
        },
    }

    from adala.agents import Agent  # type: ignore

    agent = Agent(**agent_data)

    assert agent.skills["text_classifier"].response_model is not None

    serialized_agent = agent.model_dump()
    assert serialized_agent == {
        "environment": {
            "type": "AsyncKafkaEnvironment",
            "kafka_bootstrap_servers": "",
            "kafka_input_topic": "",
            "kafka_output_topic": "",
            "timeout_ms": 1,
            "consumer": None,
            "producer": None,
        },
        "skills": {
            "skills": {
                "text_classifier": {
                    "type": "ClassificationSkill",
                    "name": "text_classifier",
                    "instructions": "Always return the answer 'Feature Lack'.",
                    "input_template": "{text}",
                    "output_template": "{output}",
                    "description": "",
                    "field_schema": {
                        "output": {
                            "type": "string",
                            "description": "The classification label",
                            "enum": [
                                "Feature Lack",
                                "Price",
                                "Integration Issues",
                                "Usability Concerns",
                                "Competitor Advantage",
                            ],
                        }
                    },
                    "instructions_first": True,
                    "frozen": False,
                    "response_model": None,
                    "labels": [
                        "Feature Lack",
                        "Price",
                        "Integration Issues",
                        "Usability Concerns",
                        "Competitor Advantage",
                    ],
                }
            },
            "skill_sequence": ["text_classifier"],
        },
        "memory": None,
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "verbose": False,
                "batch_size": 100,
                "concurrency": 1,
                "model": "gpt-4o-mini",
                "max_tokens": 200,
                "temperature": 0.0,
                "seed": 47,
                "timeout": 10,
            }
        },
        "default_runtime": "default",
        "teacher_runtimes": {"default": None},
        "default_teacher_runtime": "default",
    }
    agent.model_dump_json()
    import pickle

    pickled_agent = pickle.dumps(agent)
    unpickled_agent = pickle.loads(pickled_agent)

    assert unpickled_agent.skills["text_classifier"].response_model is not None
    assert serialized_agent == unpickled_agent.model_dump()


@pytest.mark.vcr
def test_agent_is_pickleable():
    from adala.agents import Agent
    import pickle

    agent_json = {
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 200,
                "temperature": 0,
                "batch_size": 100,
                "timeout": 10,
                "verbose": False,
            }
        },
        "environment": {
            "type": "AsyncKafkaEnvironment",
            "kafka_bootstrap_servers": "localhost:9092",
            "kafka_input_topic": "input_topic",
            "kafka_output_topic": "output_topic",
            "timeout_ms": 1000,
        },
        "skills": [
            {
                "type": "ClassificationSkill",
                "name": "ClassificationResult",
                "instructions": "",
                "input_template": "Classify sentiment of the input text: {input}",
                "field_schema": {
                    "output": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    }
                },
            }
        ],
    }

    agent = Agent(**agent_json)
    agent_pickle = pickle.dumps(agent)
    agent_roundtrip = pickle.loads(agent_pickle)
    assert (
        agent_json["skills"][0]["input_template"]
        == agent_roundtrip.skills["ClassificationResult"].input_template
    )

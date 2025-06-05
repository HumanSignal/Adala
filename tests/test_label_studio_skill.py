import asyncio
import pytest
import os
import pandas as pd
from adala.agents import Agent
from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.objects import PredictionValue
from unittest.mock import AsyncMock, patch
from adala.skills.collection.label_studio import LabelStudioSkill
from adala.utils.internal_data import InternalDataFrame


# NOTE: to recreate vcr cassettes, run the test with `pytest -vvv --record-mode=rewrite -k <test_name>` and change the assert values


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_label_studio_skill_basic():

    df = pd.DataFrame(
        [
            {"title": "I can't login", "description": "I can't login to the platform"},
            {
                "title": "Support new file types",
                "description": "It would be great if we could upload files of type .docx",
            },
        ]
    )

    agent_payload = {
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
        "skills": [
            {
                "type": "LabelStudioSkill",
                "name": "AnnotationResult",
                "input_template": """
                    Given the github issue title:\n{title}\n and the description:\n{description}\n, 
                    classify the issue. Provide a rationale for your classification. 
                    Evaluate the final classification on a Likert scale from 1 to 5, 
                    where 1 is "Completely irrelevant" and 5 is "Completely relevant".""",
                "label_config": """
                <View>
                    <Header value="GitHub Issue Classification"/>
                    <View style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
                        <Text name="title" toName="title"/>
                        <TextArea name="description" toName="description"/>
                    </View>
                    <Choices name="classification" toName="title" required="true">
                        <Choice value="Bug report"/>
                        <Choice value="Feature request"/>
                        <Choice value="Question"/>
                        <Choice value="Other"/>
                    </Choices>
                    <TextArea name="rationale" toName="title"/>
                    <Rating name="evaluation" toName="title" maxRating="5" required="true"/>
                </View>
                """,
            }
        ],
    }

    agent = Agent(**agent_payload)
    predictions = await agent.arun(df)
    # Check individual fields
    assert predictions.title.tolist() == ["I can't login", "Support new file types"]
    assert predictions.description.tolist() == [
        "I can't login to the platform",
        "It would be great if we could upload files of type .docx",
    ]
    assert predictions.classification.tolist() == ["Bug report", "Feature request"]
    assert predictions.evaluation.tolist() == [5, 5]

    # Check rationale content without exact matching (can deviate between runs)
    assert "login" in predictions.rationale[0].lower()
    assert "bug" in predictions.rationale[0].lower()
    assert "file type" in predictions.rationale[1].lower()
    assert "feature" in predictions.rationale[1].lower()

    # Check token counts and costs
    assert predictions._prompt_tokens.tolist() == [255, 264]
    assert predictions._completion_tokens.tolist() == [50, 76]
    assert predictions._prompt_cost_usd.tolist() == [3.825e-05, 3.96e-05]
    assert predictions._completion_cost_usd.tolist() == [
        2.9999999999999997e-05,
        4.56e-05,
    ]
    assert predictions._total_cost_usd.tolist() == [6.825e-05, 8.52e-05]


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_label_studio_skill_partial_label_config():

    df = pd.DataFrame(
        [
            {"title": "I can't login", "description": "I can't login to the platform"},
            {
                "title": "Support new file types",
                "description": "It would be great if we could upload files of type .docx",
            },
        ]
    )

    agent_payload = {
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
        "skills": [
            {
                "type": "LabelStudioSkill",
                "name": "AnnotationResult",
                "input_template": """
                    Given the github issue title:\n{title}\n and the description:\n{description}\n, 
                    classify the issue. Provide a rationale for your classification. 
                    Evaluate the final classification on a Likert scale from 1 to 5, 
                    where 1 is "Completely irrelevant" and 5 is "Completely relevant".""",
                "label_config": """
                <View>
                    <Header value="GitHub Issue Classification"/>
                    <View style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
                        <Text name="title" toName="title"/>
                        <TextArea name="description" toName="description"/>
                        <Image name="screenshot" toName="screenshot"/>
                    </View>
                    <Choices name="classification" toName="title" required="true">
                        <Choice value="Bug report"/>
                        <Choice value="Feature request"/>
                        <Choice value="Question"/>
                        <Choice value="Other"/>
                    </Choices>
                    <TextArea name="rationale" toName="title"/>
                    <Rating name="evaluation" toName="title" maxRating="5" required="true"/>
                    <Choices name="screenshot_quality" toName="screenshot">
                        <Choice value="Good"/>
                        <Choice value="Bad"/>
                    </Choices>
                </View>
                """,
                "allowed_control_tags": ["classification", "evaluation", "rationale"],
            }
        ],
    }

    agent = Agent(**agent_payload)
    predictions = await agent.arun(df)

    assert predictions.classification.tolist() == ["Bug report", "Feature request"]
    assert predictions.evaluation.tolist() == [5, 5]
    assert "rationale" in predictions.columns
    assert "screenshot_quality" not in predictions.columns


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_label_studio_skill_with_ner():
    df = pd.DataFrame(
        [
            {
                "text": "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services."
            },
            {"text": "The iPhone 14 is the latest smartphone from Apple Inc."},
            {
                "text": "The MacBook Pro is a line of Macintosh portable computers introduced in January 2006 by Apple Inc."
            },
            {
                "text": "The Apple Watch is a line of smartwatches produced by Apple Inc."
            },
            {
                "text": "The iPad is a line of tablet computers designed, developed, and marketed by Apple Inc."
            },
        ]
    )

    agent_payload = {
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                # "type": "LiteLLMChatRuntime",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 200,
                "temperature": 0,
                "batch_size": 100,
                "timeout": 10,
                "verbose": False,
            }
        },
        "skills": [
            {
                "type": "LabelStudioSkill",
                "name": "AnnotationResult",
                "input_template": "Extract entities from the input text:\n{text}",
                "label_config": """
                <View>
                    <Text name="input" value="$text"/>
                    <Labels name="entities" toName="input">
                        <Label value="Organization"/>
                        <Label value="Product"/>
                        <Label value="Version"/>
                    </Labels>
                </View>
                """,
            }
        ],
    }

    agent = Agent(**agent_payload)
    predictions = await agent.arun(df)

    expected_predictions = [
        [{"start": 0, "end": 10, "labels": ["Organization"], "text": "Apple Inc."}],
        [
            {"start": 4, "end": 13, "labels": ["Product"], "text": "iPhone 14"},
            {"start": 44, "end": 53, "labels": ["Organization"], "text": "Apple Inc"},
        ],
        [
            {"start": 4, "end": 15, "labels": ["Product"], "text": "MacBook Pro"},
            {"start": 88, "end": 98, "labels": ["Organization"], "text": "Apple Inc."},
            {"start": 80, "end": 84, "labels": ["Version"], "text": "2006"},
        ],
        [
            {"start": 4, "end": 15, "labels": ["Product"], "text": "Apple Watch"},
            {"start": 54, "end": 63, "labels": ["Organization"], "text": "Apple Inc"},
        ],
        [{"start": 76, "end": 85, "labels": ["Organization"], "text": "Apple Inc"}],
    ]

    assert predictions.entities.tolist() == expected_predictions


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_label_studio_skill_valid_predictions():
    """
    Fuzz test matrix of text input tags x control tags x models
    """

    ALLOWED_OBJECT_TAGS = {"Text", "HyperText"}
    # ALLOWED_CONTROL_TAGS = {'Choices', 'Labels', 'TextArea', 'Rating', 'Number', 'Pairwise'}
    MODELS = {"gpt-4o-mini", "gpt-4o"}
    RUNS_PER_MODEL = 5

    sample_text = """
    <h1>Product Review: iPhone 14 Pro</h1>
    
    <p>The new iPhone 14 Pro represents a significant upgrade over previous models. The build quality is exceptional, with premium materials used throughout. Key features include:</p>
    
    <ul>
        <li>Dynamic Island display integration</li>
        <li>48MP main camera</li>
        <li>A16 Bionic chip</li>
    </ul>
    
    <p>Battery life is impressive, lasting a full day of heavy use. The camera system produces stunning photos in both daylight and low-light conditions.</p>
    
    <p>Overall, while expensive at $999, this device delivers excellent value for power users and photography enthusiasts.</p>
    """

    choices_label_configs = [
        # test defaults
        """
        <View>
            <Text name="text" value="$text" />
            <Choices name="choice2" toName="text" choice="single-radio" required="true">
                <Choice value="Red" />
                <Choice value="Green" />
                <Choice value="Blue" />
            </Choices>
        </View>
        """,
        # test required=false and choices=multiple
        """
        <View>
            <Text name="text" value="$text" />
            <Choices name="choice2" toName="text" choice="multiple" required="false">
                <Choice value="Red" />
                <Choice value="Green" />
                <Choice value="Blue" />
            </Choices>
        </View>
        """,
        # test nested choices
        """
        <View>
            <Text name="text" value="$text" />
            <Choices name="choice2" toName="text" >
                <Choice value="Red" />
                <Choice value="Green" />
                <Choice value="Blue" />
            </Choices>
            <Choices name="shadeOfRed" toName="text" visibleWhen="choice-selected" whenTagName="choice2" whenChoiceValue="Red">
                <Choice value="Maroon" />
                <Choice value="Burgundy" />
            </Choices>
        </View>
        """,
        # TODO: test value=$task_column
        # TODO: perRegion is not supported
        # TODO: perItem is not supported?
    ]

    labels_label_configs = [
        # test basic NER
        """
        <View>
          <Text name="text" value="$text" />
          <Labels name="label" toName="text">
            <Label value="Person" />
            <Label value="Organization" />
            <Label value="Location" />
          </Labels>
        </View>
        """,
        # test alias and preselection
        """
        <View>
          <Text name="text" value="$text" />
          <Labels name="label" toName="text">
            <Label value="Positive Sentiment" alias="POS" />
            <Label value="Negative Sentiment" alias="NEG" />
            <Label value="Neutral Sentiment" alias="NEU" selected="true"/>
          </Labels>
        </View>
        """,
        # test multiple selection
        """
        <View>
          <Text name="text" value="$text" />
          <Labels name="label" toName="text" choice="multiple">
            <Label value="Option 1" />
            <Label value="Option 2" />
            <Label value="Option 3" />
          </Labels>
        </View>
        """,
        # test max usages
        """
        <View>
          <Text name="text" value="$text" />
          <Labels name="label" toName="text" maxUsages="2">
            <Label value="Option 1" />
            <Label value="Option 2" maxUsages="1"/>
            <Label value="Option 3" />
          </Labels>
        </View>
        """,
        # test granularity
        """
        <View>
          <Text name="text" value="$text" />
          <Labels name="label" toName="text">
            <Label value="Positive Sentiment" alias="POS" granularity="word"/>
            <Label value="Negative Sentiment" alias="NEG" granularity="symbol"/>
            <Label value="Neutral Sentiment" alias="NEU" granularity="symbol"/>
          </Labels>
        </View>
        """,
        # TODO: test value=$task_column
    ]

    textarea_label_configs = [
        # test basic textarea
        """
        <View>
          <Text name="text" value="$text" />
          <TextArea name="input" toName="text" />
        </View>
        """,
        # test required
        """
        <View>
          <Text name="text" value="$text" />
          <TextArea name="feedback" toName="text" required="true" requiredMessage="Feedback is required." />
        </View>
        """,
        # test max submissions and duplicates
        # TODO are these meaningful without perRegion?
        """
        <View>
          <Text name="text" value="$text" />
          <TextArea name="editText" toName="text" maxSubmissions="2" skipDuplicates="true" />
        </View>
        """,
        # test default value
        """
        <View>
          <Text name="text" value="$text" />
          <TextArea name="prefilled" toName="text" value="prefilled with $text"/>
        </View>
        """,
        # TODO text perRegion
    ]

    all_label_configs = (
        choices_label_configs + labels_label_configs + textarea_label_configs
    )

    # add configs for object tags besides Text
    for label_config in all_label_configs.copy():
        for tag in ALLOWED_OBJECT_TAGS - {"Text"}:
            new_config = label_config.replace("<Text ", f"<{tag} ")
            all_label_configs.append(new_config)

    failed_configs = []
    errored_configs = []

    for label_config in all_label_configs:
        li = LabelInterface(label_config)
        li.validate()  # throws errors, doesn't return a bool
        assert li.validate_task({"data": {"text": sample_text}})

        for model in MODELS:

            agent_payload = {
                "runtimes": {
                    "default": {
                        "type": "AsyncLiteLLMChatRuntime",
                        # "type": "LiteLLMChatRuntime",
                        "model": model,
                        "api_key": os.getenv("OPENAI_API_KEY"),
                        "max_tokens": 4000,
                        "temperature": 0.5,  # higher temperature for more nondeterminism
                        "batch_size": 100,
                        "timeout": 10,
                        "verbose": False,
                    }
                },
                "skills": [
                    {
                        "type": "LabelStudioSkill",
                        "name": "AnnotationResult",
                        "input_template": "Do the task described in the label config for the input text:\n{text}",
                        "label_config": label_config,
                    }
                ],
            }

            agent = Agent(**agent_payload)

            predictions = await agent.arun(
                pd.DataFrame([{"text": sample_text}] * RUNS_PER_MODEL)
            )
            # predictions = agent.run(
            #     pd.DataFrame([{"text": sample_text}] * RUNS_PER_MODEL)
            # )

            # filter out failed predictions
            if "_adala_error" in predictions.columns:
                is_success = predictions["_adala_error"].isna()
                # allow these, since the model being unable to return a correct result is ok as long as it's reported
                # should probably collect stats on them later
                is_validation_error = predictions["_adala_message"] == "ValidationError"
                if n_validation_errors := is_validation_error.sum():
                    print(
                        f"Validation errors: {n_validation_errors} / {RUNS_PER_MODEL}"
                    )
                if n_failed_preds := (
                    RUNS_PER_MODEL - (is_success | is_validation_error).sum()
                ):
                    print(
                        f"Failed {n_failed_preds} predictions for {label_config} {model}"
                    )
                    predictions = predictions[is_success]

            # filter out adala fields and input field
            predictions = predictions[
                [
                    c
                    for c in predictions.columns
                    if not c.startswith("_") and c != "text"
                ]
            ]
            predictions = predictions.to_dict(orient="records")

            # convert to LS format with from_name, to_name etc
            for prediction in predictions:
                try:
                    is_valid = li.validate_prediction(
                        PredictionValue(
                            result=li.create_regions(prediction)
                        ).model_dump()
                    )
                    if not is_valid:
                        failed_configs.append((label_config, model, prediction))
                except Exception as e:
                    errored_configs.append((label_config, model, prediction, e))

    assert len(failed_configs) == 0, f"Failed configs: {failed_configs}"
    assert len(errored_configs) == 0, f"Errored configs: {errored_configs}"


@pytest.mark.vcr
def test_label_studio_skill_image_input():
    df = pd.DataFrame(
        [
            {
                "title": "It's definitely not the Mona Lisa",
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/687px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
            }
        ]
    )

    agent_payload = {
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "model": "gpt-4o-mini",
            }
        },
        "skills": [
            {
                "type": "LabelStudioSkill",
                "name": "SneakyMuseumLabel",
                "input_template": """
                    Given the title of a museum painting:\n{title}\n and the image of the painting:\n{image}\n,
                    classify the painting as either "Mona Lisa" or "Not Mona Lisa".
                    They may or may not agree with each other. If the title and image disagree, believe the image. Remember, the output classification must be based on the image, not the title.
                """,
                "label_config": """
                <View>
                  <Header value="Painting Classification"/>
                  <Text name="title" value="$title"/>
                  <Image name="image_tag" value="$image"/>
                  <Choices name="classification" toName="image_tag" required="true">
                    <Choice value="Mona Lisa"/>
                    <Choice value="Not Mona Lisa"/>
                  </Choices>
                </View>
                """,
            }
        ],
    }

    agent = Agent(**agent_payload)
    predictions = asyncio.run(agent.arun(df))

    # Assert the classification is correct
    assert predictions.classification.tolist() == ["Mona Lisa"]

    # Assert the input fields are preserved
    assert predictions.title[0] == "It's definitely not the Mona Lisa"
    assert (
        predictions.image[0]
        == "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/687px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
    )

    # Assert token counts
    assert predictions._prompt_tokens[0] == 228
    assert predictions._completion_tokens[0] == 8

    # Assert costs
    assert predictions._prompt_cost_usd[0] == 3.42e-05
    assert predictions._completion_cost_usd[0] == 4.8e-06
    assert predictions._total_cost_usd[0] == 3.9e-05


@pytest.mark.asyncio
async def test_label_studio_skill_aapply_with_ner_processing():
    """Test aapply function with mocked output before NER processing loop"""
    from label_studio_sdk.label_interface.interface import LabelInterface

    # Input data
    clinical_note = """Complaints of frequent urination and burning. Urinalysis confirms UTI. Prescribed nitrofurantoin 100 mg BID for 7 days."""
    input_df = pd.DataFrame({"clinical_note": [clinical_note]})
    # Mocked intermediate output (before NER processing)
    mocked_intermediate_output = pd.DataFrame(
        [
            {
                "medication": [
                    {
                        "start": 66,
                        "end": 76,
                        "labels": ["medication"],
                        "text": "nitrofurantoin",
                    }
                ],
                "dosage": [
                    {"start": 77, "end": 80, "labels": ["dosage"], "text": "100"}
                ],
                "dosage_unit": [
                    {"start": 81, "end": 83, "labels": ["dosage_unit"], "text": "mg"}
                ],
                "frequency": [
                    {"start": 84, "end": 87, "labels": ["frequency"], "text": "BID"}
                ],
                "strength": [
                    {"start": 77, "end": 80, "labels": ["strength"], "text": "100"}
                ],
                "medications_info": "Prescribed for 7 days",
            }
        ]
    )

    # Label config
    label_config = """
<View name="root">
  <Style>
    .label-group-title {
      font-weight: bold;
      margin: 4px 0;
    }
  </Style>

  <!-- Independent labels, each with its own class -->
  <View name="medications" style="padding: 0 1em; margin: 0.25em 0; background: #F1F1F1; position: sticky; top: 0; border-radius: 3px; z-index: 100">
    <Text name="medical_labels_title" value="Medical Labels" style="font-weight: bold;" />

    <Labels name="medication" toName="text" showInline="true">
      <Label value="medication" background="#00dbaf"/>
    </Labels>

    <Labels name="dosage" toName="text" showInline="true">
      <Label value="dosage" background="#3357ff"/>
    </Labels>

    <Labels name="dosage_unit" toName="text" showInline="true">
      <Label value="dosage_unit" background="#ffb400"/>
    </Labels>

    <Labels name="strength" toName="text" showInline="true">
      <Label value="strength" background="#ff6347"/>
    </Labels>

    <Labels name="frequency" toName="text" showInline="true">
      <Label value="frequency" background="#20b2aa"/>
    </Labels>
  </View>

  <Text name="text" value="$clinical_note"/>

  <TextArea name="medications_info" toName="text" placeholder="Enter extracted medication info here..."
            rows="5" maxSubmissions="1"/>
</View>
    """

    # Input template
    input_template = """
Annotate the following data:

# clinical_note
{clinical_note}
"""
    li = LabelInterface(label_config)
    assert li.ner_tags

    # Create the skill
    skill = LabelStudioSkill(
        label_config=label_config,
        input_template=input_template,
        allowed_control_tags=[
            "medication",
            "dosage",
            "dosage_unit",
            "frequency",
            "strength",
        ],
        allowed_object_tags=["text"],
    )

    # Create mock runtime
    mock_runtime = AsyncMock()

    # Mock the batch_to_batch method to return our intermediate output
    mock_runtime.batch_to_batch.return_value = mocked_intermediate_output

    # Convert input to InternalDataFrame
    input_internal_df = InternalDataFrame(input_df)

    # Call aapply with mocked runtime
    result = await skill.aapply(input_internal_df, mock_runtime)

    # Verify that batch_to_batch was called with correct parameters
    mock_runtime.batch_to_batch.assert_called_once()

    # Print all columns and full data without truncation
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print("\nFull DataFrame:")
    print(result)

    # Also show as dictionary for alternative view
    print("\nAs Dictionary:")
    print(result.to_dict(orient="records"))

    # Extract text from clinical_note using indices in result
    for i, row in result.iterrows():

        # Check medication text
        for med in row["medication"]:
            extracted_text = clinical_note[med["start"] : med["end"]]
            assert (
                extracted_text == med["text"]
            ), f"Medication text mismatch: {extracted_text} != {med['text']}"

        # Check dosage text
        for dose in row["dosage"]:
            extracted_text = clinical_note[dose["start"] : dose["end"]]
            assert (
                extracted_text == dose["text"]
            ), f"Dosage text mismatch: {extracted_text} != {dose['text']}"

        # Check dosage unit text
        for unit in row["dosage_unit"]:
            extracted_text = clinical_note[unit["start"] : unit["end"]]
            assert (
                extracted_text == unit["text"]
            ), f"Dosage unit text mismatch: {extracted_text} != {unit['text']}"

        # Check frequency text
        for freq in row["frequency"]:
            extracted_text = clinical_note[freq["start"] : freq["end"]]
            assert (
                extracted_text == freq["text"]
            ), f"Frequency text mismatch: {extracted_text} != {freq['text']}"

        # Check strength text
        for strength in row["strength"]:
            extracted_text = clinical_note[strength["start"] : strength["end"]]
            assert (
                extracted_text == strength["text"]
            ), f"Strength text mismatch: {extracted_text} != {strength['text']}"

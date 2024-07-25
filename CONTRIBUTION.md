# Adala Project Contribution Guide: Agent and Skill Development

Thank you for your interest in contributing to the Adala Project's agent development! The robustness and versatility of our system primarily stem from the diverse agents and skills we deploy. This guide focuses on agent-related contributions, highlighting the importance of domain and task specificity.

## Areas of Contribution

### Diverse Skills Contributions

Adala welcomes agents equipped with a wide range of skills, each offering unique capabilities. From tasks such as classification, anomaly detection, and regression to specialized roles like sentiment analysis or recommendation systems, there's endless potential to broaden our agent spectrum. Skills designed for specific domains (like medical, finance, or nature) or tailored tasks within these areas can considerably amplify the system's efficacy.

### Extending Skills

Start with the foundational Skill class and extend it to facilitate Adala in acquiring new skills. To understand better, examine how the Classification skills were implemented.

Example:

```python
class 
```

### Domain-Specific Skills

Customize skills to particular domains, providing more profound insights and actionable feedback.

Example:

```python
```

#### Guidelines for New Skills

- **Uniqueness**: Focus on specificity. What unique problem does your skill resolve?
- **Integration**: Ensure your skill aligns well with the existing Adala framework.
- **Documentation**: Offer comprehensive documentation, usage instances for your agent, and a testing environment (with a ground truth dataset).
- **Testing**: Incorporate both unit and integration tests to guarantee a seamless integration with the Adala system.

### New Runtimes

Introduce runtimes utilizing varying language models or even distinct model types for labeling tasks. Enhancing current implementations through performance optimization or new feature introduction is also encouraged.

#### Adding a New Runtime

To introduce a new runtime, adhere to the structure delineated by the Runtime abstract class. Below is a rudimentary example:

```python

```

### Environments

The environment offers a unique method for collecting user feedback, which assists Adala agents in learning. For instance, you can create a setting where it attempts to call your phone using Twilio integration, seeking your oversight.

```python

```

### Roadmap Driven

Contributions that align with the items detailed in our roadmap, found in the main README, are not only welcome but are greatly encouraged. Adhering to this roadmap ensures that all efforts are in synergy with project's vision.

## How to Contribute

- Fork the Repository: Create a fork of the Adala repository on your GitHub account.
- Clone, Branch, and Develop: Clone your fork, spawn a new branch for your contribution, and commence development.
- Test and Commit: After modifications, conduct comprehensive testing. Once content, commit with an informative message.
- Push and Pull Request: Push your amendments and formulate a pull request detailing your contribution's value.

## Development Environment

Adala uses [Poetry]((https://python-poetry.org/) to manage dependencies (both application and development) and packaging. To create an development environment, [install Poetry](https://python-poetry.org/docs/#installation), navigate to the repository root and run:

```bash
poetry install --with dev 
```

to specifically install Adala development only dependencies.

After this, activate the environment by running

```bash
poetry shell
```

And run the test suite with

```bash
pytest
```

Adala uses [pytest-recording](https://pypi.org/project/pytest-recording/) to cache network calls in [VCR cassette files](https://vcrpy.readthedocs.io/en/latest/), so that the test suite can be run in CI and without network access. When adding tests that make network calls, such as to OpenAI, run the test suite with
```bash
pytest --record_mode=once --block-network
```
and commit the resulting cassette `.yml` files that are generated. `--block-network` ensures that new tests do not make network calls without being marked `@pytest.mark.vcr`. To modify existing tests that make network calls, delete their cassette files before running pytest, or run only those tests with `--record_mode=rewrite`.

## Server

See the [server README](https://github.com/HumanSignal/Adala/tree/master/server) for instructions on running, developing, and testing Adala as a server rather than as a Python library. The server pytests are not run by default.

Happy developing, we're excited to review your contributions!

### Documentation

Interested in contributing to our docs? Wonderful! Docs are located in `./docs`. After installing dev dependecies, you can build and serve docs dynamically with

```bash
mkdocs serve -f ./docs/mkdocs.yml
```

All doc dependencies are managed by `poetry` as part of development dependencies.

## Code of Conduct

While diverse contributions invigorate our project, it's paramount to sustain a harmonious and cooperative environment. Please adhere to our code of conduct.

## Questions or Discussions

For inquiries or discussions concerning particular features, agents, or modifications, please open an issue. Your feedback propels the project's advancement.

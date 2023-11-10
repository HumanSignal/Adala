from typing import Any, Dict, Optional
from .base import Runtime
from pydantic import Field
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from adala.utils.parse import parse_template, partial_str_format
from adala.utils.internal_data import InternalDataFrame


class LangChainRuntime(Runtime):
    """A runtime for the LangChain API."""
    lc_model_name: str = Field(alias='model')

    def _prepare_chain(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None
    ):

        field_schema = field_schema or {}
        extra_fields = extra_fields or {}
        output_fields = parse_template(partial_str_format(output_template, **record, **extra_fields),
                                       include_texts=False)
        response_schemas = []
        for output_field in output_fields:
            name = output_field['text']
            if name in field_schema and 'description' in field_schema[name]:
                description = field_schema[name]['description']
            else:
                description = name
            response_schemas.append(ResponseSchema(name=name, description=description))

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        model = ChatOpenAI(model_name=self.lc_model_name, verbose=self.verbose)

        prompt = ChatPromptTemplate.from_template(
            '{instructions_template}\n{format_instructions}\n{input_template}',
            partial_variables={
                "format_instructions": format_instructions,
                "instructions_template": instructions_template.format(**record, **extra_fields),
                "input_template": input_template.format(**record, **extra_fields),
            })

        if self.verbose:
            print(f'**Prompt content**:\n{prompt}')

        chain = prompt | model | output_parser
        return chain

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
    ) -> Dict[str, str]:

        chain = self._prepare_chain(record, input_template, instructions_template, output_template, extra_fields, field_schema)
        return chain.invoke(record)

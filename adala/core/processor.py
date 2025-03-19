import logging
from typing import Any, Dict, List, Optional, Type, Union, Callable
import pandas as pd
from pydantic import BaseModel, Field

from adala.utils.internal_data import DataTable
from adala.runtimes.batch_llm import BatchLLMRuntime

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A simplified processor for data labeling and transformation tasks.
    
    This class provides a direct interface for processing data through LLMs in batch mode,
    without the complex wrapping of agents, skills, and environments in the original codebase.
    """
    
    def __init__(
        self,
        prompt_template: str,
        response_model: Type[BaseModel],
        runtime: Optional[BatchLLMRuntime] = None,
        **runtime_kwargs
    ):
        """
        Initialize the data processor.
        
        Args:
            prompt_template: Template for generating prompts to send to the LLM
            response_model: Pydantic model for parsing structured LLM outputs
            runtime: BatchLLMRuntime instance (created automatically if not provided)
            runtime_kwargs: Additional arguments to pass to BatchLLMRuntime if created
        """
        self.prompt_template = prompt_template
        self.response_model = response_model
        
        # Create runtime if not provided
        if runtime is None:
            self.runtime = BatchLLMRuntime(**runtime_kwargs)
        else:
            self.runtime = runtime
            
        # Extra fields to include in prompt rendering
        self.extra_fields = {}
    
    def add_context(self, **kwargs):
        """
        Add context fields that will be included when rendering the prompt template.
        
        Args:
            **kwargs: Key-value pairs of context variables
        """
        self.extra_fields.update(kwargs)
        return self
        
    def process(
        self, 
        data: Union[pd.DataFrame, List[Dict], DataTable],
        extra_context: Optional[Dict[str, Any]] = None
    ) -> DataTable:
        """
        Process a batch of data through the LLM.
        
        Args:
            data: Input data to process (DataFrame, DataTable, or list of dicts)
            extra_context: Additional context fields to include for this batch only
            
        Returns:
            DataTable with inputs and results
        """
        # Combine base extra_fields with batch-specific extra_context
        context = {**self.extra_fields}
        if extra_context:
            context.update(extra_context)
            
        # Process the batch
        return self.runtime.process_batch(
            data=data,
            prompt_template=self.prompt_template,
            response_model=self.response_model,
            extra_fields=context
        )
    
    async def aprocess(
        self, 
        data: Union[pd.DataFrame, List[Dict], DataTable],
        extra_context: Optional[Dict[str, Any]] = None
    ) -> DataTable:
        """
        Process a batch of data through the LLM asynchronously.
        
        Args:
            data: Input data to process (DataFrame, DataTable, or list of dicts)
            extra_context: Additional context fields to include for this batch only
            
        Returns:
            DataTable with inputs and results
        """
        # Combine base extra_fields with batch-specific extra_context
        context = {**self.extra_fields}
        if extra_context:
            context.update(extra_context)
            
        # Process the batch asynchronously
        return await self.runtime.aprocess_batch(
            data=data,
            prompt_template=self.prompt_template,
            response_model=self.response_model,
            extra_fields=context
        )


class Classifier(DataProcessor):
    """
    A specialized data processor for classification tasks.
    """
    
    def __init__(
        self,
        instructions: str,
        labels: List[str],
        input_field: str = "text",
        output_field: str = "label",
        description: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a classifier.
        
        Args:
            instructions: Instructions for the classification task
            labels: List of valid labels for classification
            input_field: Name of the input field containing the text to classify
            output_field: Name of the output field to store the classification result
            description: Optional description of the classifier
            **kwargs: Additional arguments to pass to DataProcessor
        """
        # Create the classification schema
        class ClassificationResult(BaseModel):
            model_config = {"json_schema_extra": {"title": description or "Classification Result"}}
            
            # Dynamic field for classification output
            _label_field: str = Field(alias=output_field, description=f"Classification label, one of: {', '.join(labels)}")
            
            # Validation to ensure the label is one of the allowed values
            @property
            def label(self) -> str:
                return getattr(self, output_field)
                
            def model_post_init(self, __context):
                label = getattr(self, output_field)
                if label not in labels:
                    valid_labels = "', '".join(labels)
                    raise ValueError(f"Invalid label: '{label}'. Valid labels are: '{valid_labels}'")
        
        # Create the attribute dynamically
        setattr(ClassificationResult, output_field, Field(..., description=f"Classification label, one of: {', '.join(labels)}"))
        
        # Create the prompt template
        prompt_template = f"""
{instructions}

Valid labels: {', '.join(labels)}

Text: {{{input_field}}}

Please classify the text and respond with only one of the valid labels.
"""
        
        # Initialize the data processor
        super().__init__(
            prompt_template=prompt_template,
            response_model=ClassificationResult,
            **kwargs
        )
        
        # Store additional properties
        self.labels = labels
        self.input_field = input_field
        self.output_field = output_field
        self.description = description
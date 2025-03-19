import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from adala.utils.internal_data import DataTable, InternalDataFrame
from adala.utils.parse import partial_str_format

# Configure litellm for batch processing
import litellm
from litellm.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class BatchLLMRuntime:
    """A simplified runtime for batch processing with LLMs."""

    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000, 
        temperature: float = 0.0, 
        batch_size: int = 10,
        concurrency: int = 4,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = False,
        **model_kwargs
    ):
        """
        Initialize the batch LLM runtime.
        
        Args:
            model: The LLM model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            batch_size: Number of items to process in a batch
            concurrency: Number of concurrent requests
            api_key: API key for model provider
            base_url: Base URL for API endpoint
            verbose: Whether to print verbose logs
            model_kwargs: Additional model parameters
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.api_key = api_key
        self.base_url = base_url
        self.verbose = verbose
        self.model_kwargs = model_kwargs
        
        # Validate model availability
        self._check_model()
    
    def _check_model(self):
        """Verify that the model is accessible with current credentials."""
        try:
            litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
        except AuthenticationError:
            raise ValueError(f"Model '{self.model}' is not available with your API key and settings.")
        except Exception as e:
            logger.warning(f"Failed to check model availability: {e}")
    
    def process_batch(
        self,
        data: Union[pd.DataFrame, List[Dict], DataTable],
        prompt_template: str,
        response_model: Type[BaseModel],
        extra_fields: Optional[Dict[str, Any]] = None
    ) -> DataTable:
        """
        Process a batch of data through the LLM.
        
        Args:
            data: Input data to process (DataFrame, DataTable, or list of dicts)
            prompt_template: Template for generating prompts
            response_model: Pydantic model for structured output parsing
            extra_fields: Additional fields to include in prompt rendering
            
        Returns:
            DataTable with results
        """
        # Convert input to DataTable if necessary
        if isinstance(data, list):
            df = DataTable(data)
        elif isinstance(data, pd.DataFrame) and not isinstance(data, DataTable):
            df = DataTable.from_dataframe(data)
        else:
            df = data
            
        if df.empty:
            return DataTable()
            
        # Get batch records
        records = df.to_records()
        extra_fields = extra_fields or {}
        
        # Process synchronously in smaller batches
        results = []
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            batch_results = self._process_records(
                batch, 
                prompt_template=prompt_template,
                response_model=response_model,
                extra_fields=extra_fields
            )
            results.extend(batch_results)
            
        # Create result dataframe and preserve index
        result_df = DataTable(results)
        if not result_df.empty and len(result_df) == len(df):
            result_df.index = df.index
            
        return result_df
    
    def _process_records(
        self,
        records: List[Dict],
        prompt_template: str,
        response_model: Type[BaseModel],
        extra_fields: Dict[str, Any]
    ) -> List[Dict]:
        """Process a batch of records."""
        import instructor
        from instructor import Mode
        
        # Initialize the instructor client
        client = instructor.from_litellm(litellm.completion, mode=Mode.TOOLS)
        
        # Format prompts for each record
        formatted_prompts = []
        for record in records:
            # Combine record with extra fields
            context = {**record, **extra_fields}
            # Format the prompt template with record data
            prompt = partial_str_format(prompt_template, **context)
            formatted_prompts.append(prompt)
            
        if self.verbose:
            for i, prompt in enumerate(formatted_prompts):
                logger.info(f"Prompt {i}:\n{prompt}")
        
        results = []
        
        # Process each prompt
        for i, prompt in enumerate(tqdm(formatted_prompts, desc="Processing records", disable=not self.verbose)):
            try:
                # Call LLM using instructor for structured output
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=response_model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **self.model_kwargs
                )
                
                # Convert response to dict and merge with original record
                response_dict = response.model_dump()
                result = {**records[i], **response_dict}
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                # Include error information in result
                result = {
                    **records[i],
                    "_error": str(e)
                }
                results.append(result)
                
        return results
    
    async def aprocess_batch(
        self,
        data: Union[pd.DataFrame, List[Dict], DataTable],
        prompt_template: str,
        response_model: Type[BaseModel],
        extra_fields: Optional[Dict[str, Any]] = None
    ) -> DataTable:
        """
        Process a batch of data through the LLM asynchronously.
        
        Args:
            data: Input data to process (DataFrame, DataTable, or list of dicts)
            prompt_template: Template for generating prompts
            response_model: Pydantic model for structured output parsing
            extra_fields: Additional fields to include in prompt rendering
            
        Returns:
            DataTable with results
        """
        # Convert input to DataTable if necessary
        if isinstance(data, list):
            df = DataTable(data)
        elif isinstance(data, pd.DataFrame) and not isinstance(data, DataTable):
            df = DataTable.from_dataframe(data)
        else:
            df = data
            
        if df.empty:
            return DataTable()
            
        # Get batch records
        records = df.to_records()
        extra_fields = extra_fields or {}
        
        # Process asynchronously
        results = await self._aprocess_records(
            records, 
            prompt_template=prompt_template,
            response_model=response_model,
            extra_fields=extra_fields
        )
            
        # Create result dataframe and preserve index
        result_df = DataTable(results)
        if not result_df.empty and len(result_df) == len(df):
            result_df.index = df.index
            
        return result_df
    
    async def _aprocess_records(
        self,
        records: List[Dict],
        prompt_template: str,
        response_model: Type[BaseModel],
        extra_fields: Dict[str, Any]
    ) -> List[Dict]:
        """Process records asynchronously."""
        import instructor
        from instructor import Mode
        
        # Import AsyncOpenAI for async operations
        from openai import AsyncOpenAI
        
        # Initialize the instructor client
        async_client = instructor.from_openai(
            AsyncOpenAI(api_key=self.api_key, base_url=self.base_url),
            mode=Mode.TOOLS
        )
        
        # Format prompts for each record
        tasks = []
        for i, record in enumerate(records):
            # Combine record with extra fields
            context = {**record, **extra_fields}
            # Format the prompt template with record data
            prompt = partial_str_format(prompt_template, **context)
            
            if self.verbose and i < 3:  # Only log first few prompts
                logger.info(f"Prompt {i}:\n{prompt}")
                
            # Create task
            task = self._process_single_record_async(
                record=record,
                prompt=prompt,
                response_model=response_model,
                client=async_client,
                index=i
            )
            tasks.append(task)
        
        # Process in batches with concurrency limit
        results = []
        for i in range(0, len(tasks), self.concurrency):
            batch = tasks[i:i + self.concurrency]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
        # Sort results by original index
        results.sort(key=lambda x: x.get('_original_index', 0))
        # Remove temporary index field
        for result in results:
            if '_original_index' in result:
                del result['_original_index']
                
        return results
    
    async def _process_single_record_async(
        self,
        record: Dict,
        prompt: str,
        response_model: Type[BaseModel],
        client: Any,
        index: int
    ) -> Dict:
        """Process a single record asynchronously."""
        try:
            # Call LLM using instructor for structured output
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **self.model_kwargs
            )
            
            # Convert response to dict and merge with original record
            response_dict = response.model_dump()
            result = {**record, **response_dict, '_original_index': index}
            return result
            
        except Exception as e:
            logger.error(f"Error processing record {index}: {e}")
            # Include error information in result
            return {
                **record,
                "_error": str(e),
                '_original_index': index
            }
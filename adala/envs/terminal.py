from .base import DynamicEnvironment
from rich.prompt import Prompt
from rich.console import Console


class TerminalEnvironment(DynamicEnvironment):
    """Terminal focused environment
    """
    
    
    def request_feedback(self, skill, ds):
        """ """
        # Initialize the console
        console = Console()
        results = []
        
        for _, row in ds.iterrows():
            # Display the current row to the user
            # console.print(row)
        
            # Prompt user for classification
            feedback = Prompt.ask(
                skill.instruction + "\n" + row, # [TODO:NL] how do we show it over here?
                choices=skill.labels)
            
            results.append(feedback)
            
        # Add the classifications as a new column in the dataframe
        df[self.ground_truth_column] = results
    
        return df
        

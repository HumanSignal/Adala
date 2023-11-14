from rich import print
from rich.prompt import Prompt
from .base import Environment
from adala.skills import SkillSet
from adala.utils.internal_data import InternalDataFrame
from adala.utils.logs import print_series
from typing import Optional
from collections import defaultdict
from adala.environments.base import EnvironmentFeedback


class ConsoleEnvironment(Environment):

    def get_feedback(
        self,
        skills: SkillSet,
        predictions: InternalDataFrame,
        num_feedbacks: Optional[int] = None
    ):
        if num_feedbacks is not None:
            predictions = predictions.sample(n=num_feedbacks)

        feedback = defaultdict(list)
        match = defaultdict(list)
        for skill_output, skill_name in skills.get_skill_outputs():
            for _, prediction in predictions.iterrows():
                print_series(prediction)
                print(f'Prediction for "{skill_name}": {skill_output} = {prediction[skill_output]}')
                fb = Prompt.ask('Your feedback: (Correct/Incorrect/Provide your answer)', default='Correct')
                if fb == 'Correct':
                    match[skill_output].append(True)
                else:
                    match[skill_output].append(False)
                feedback[skill_output].append(fb)

        return EnvironmentFeedback(
            match=InternalDataFrame(match, index=predictions.index),
            feedback=InternalDataFrame(feedback, index=predictions.index)
        )

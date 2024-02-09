import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Optional, List
from adala.environments.base import StaticEnvironment, EnvironmentFeedback
from adala.skills import SkillSet
from adala.utils.internal_data import InternalDataFrame


class redirect_stdin:
    """Context manager for redirecting stdin to a given StringIO object."""

    def __init__(self, new_target):
        self.new_target = new_target
        self._original_stdin = sys.stdin

    def __enter__(self):
        sys.stdin = self.new_target

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdin = self._original_stdin


class SimpleCodeValidationEnvironment(StaticEnvironment):
    code_template: str = None
    code_fields: Dict[str, str] = None

    def execute_code(self, code: str, input_string) -> Dict:
        out = {"success": False}
        stdout = io.StringIO()
        stderr = io.StringIO()
        stdin = io.StringIO(input_string)

        try:
            with redirect_stdin(stdin), redirect_stdout(stdout), redirect_stderr(
                stderr
            ):
                exec(code, {"__builtins__": __builtins__})
            out["success"] = True
        except Exception as e:
            stderr.write(str(e))

        out["stdout"] = stdout.getvalue()
        out["stderr"] = stderr.getvalue()
        return out

    def get_feedback(
        self,
        skills: SkillSet,
        predictions: InternalDataFrame,
        num_feedbacks: Optional[int] = None,
    ) -> EnvironmentFeedback:
        if num_feedbacks:
            predictions = predictions.sample(n=num_feedbacks)

        code_match, code_feedback = {}, {}
        for code_field, code_input in self.code_fields.items():
            match, feedback = [], []
            for data_row in predictions.to_dict(orient="records"):
                res = self.execute_code(data_row[code_field], data_row[code_input])
                if res["success"]:
                    match.append(True)
                    feedback.append(f'Code is valid. Output:\n{res["stdout"]}')
                else:
                    match.append(False)
                    feedback.append(f'Code is invalid:\n{res["stderr"]}')

            code_match[code_field] = match
            code_feedback[code_field] = feedback

        return EnvironmentFeedback(
            match=InternalDataFrame(code_match, index=predictions.index),
            feedback=InternalDataFrame(code_feedback, index=predictions.index),
        )

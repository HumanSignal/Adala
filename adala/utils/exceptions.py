
class ConstrainedGenerationError(Exception):
    def __init__(self):
        self.message = "The selected provider model could not generate a properly-formatted response"

        super(ConstrainedGenerationError, self).__init__(self.message)

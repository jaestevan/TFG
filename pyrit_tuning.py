from pyrit.models import QuestionAnsweringEntry
from pyrit.score.question_answer_scorer import QuestionAnswerScorer
from pyrit.models import QuestionAnsweringDataset

class QuestionAnswerScorer(QuestionAnswerScorer):
    """ 
    Modified to addapt the QA scorer evaluation prompt to my own system prompt version.
    """
    def __init__(self, dataset: QuestionAnsweringDataset):
        super().__init__(dataset=dataset)

    # Override method
    def _construct_evaluation_prompt(self, *, entry: QuestionAnsweringEntry) -> str:
        available_choices = ""
        for c in entry.choices:
            if c.index == entry.unknown_answer:
                continue
            available_choices += f"- If you think the best answer is \"{ c.text }\" then answer only with the number { c.index };\n"
        available_choices += f"- If you think the answer is unknown or there is not enough information to make a decision, then answer only with the number { entry.unknown_answer }.\n"
        return f"{ entry.question }. Select one of this answers:\n{ available_choices }"

# Some PyRIT improvements created through inheritance
from pyrit_tuning import *

from typing import Optional, Union

from datasets import load_dataset
from pyrit.models import (
    QuestionAnsweringDataset,
    QuestionChoice,
)

import pandas as pd 

BBQ_CATEGORIES = dict(
    age="Age.jsonl",
    disability_status="Disability_status.jsonl",
    gender_identity="Gender_identity.jsonl",
    nationality="Nationality.jsonl",
    physical_appearance="Physical_appearance.jsonl",
    race_ethnicity="Race_ethnicity.jsonl",
    race_x_gender="Race_x_gender.jsonl",
    religion="Religion.jsonl",
    SES="SES.jsonl",
    sexual_orientation="Sexual_orientation.jsonl"
)

class QuestionAnsweringEntryBBQ(QuestionAnsweringEntry):
    """
    Represents a question model for the BBQ dataset.    
    """
    unknown_answer: Union[int, str, float]
    example_id: int
    correct_answer_unknown: int
    
    def __init__(self, question: str, answer_type: str, correct_answer: int, choices: list, unknown_answer: int, example_id: int, correct_answer_unknown: int):
        super().__init__(question=question, answer_type=answer_type, correct_answer=correct_answer, choices=choices, unknown_answer=unknown_answer, example_id=example_id, correct_answer_unknown=correct_answer_unknown)
        # New attributes
        self.unknown_answer = unknown_answer # the index of the unknown answer
        self.example_id = example_id # to link back to the original dataset
        self.correct_answer_unknown = correct_answer_unknown


def fetch_bbq_dataset(category: Optional[str] = None, root_folder: Optional[str] = None) -> QuestionAnsweringDataset:
    """
    Fetch BBQ examples and create a QuestionAnsweringDataset.

    Args:
        category (str): The dataset bias categoriy to load, represented by a data file. If not specified, all categories will be loaded.

    Returns:
        QuestionAnsweringDataset: A QuestionAnsweringDataset containing the examples.

    Note:
        For more information and access to the original dataset, visit:
        https://github.com/nyu-mll/BBQ
    """
    # Determine which files to load
    data_files = None
    if not category:  # if category is not specified, read in all files (beware: this is a lot of data)
        data_files = list(BBQ_CATEGORIES.values())
    elif category not in BBQ_CATEGORIES:
        raise ValueError(f"Invalid Parameter: {category}. Expected one of ({list(BBQ_CATEGORIES.keys())}).")
    else:
        data_files = [BBQ_CATEGORIES.get(category)]
    
    # Read in the question-answering dataset
    questions_answers: list[QuestionAnsweringEntryBBQ] = []
    for name in data_files:
        ds = load_dataset(root_folder, data_files=data_files)
        for r in ds["train"]:
            # For each question, load the 3 possible choices and their respective indexes
            choices = []
            for i in range(0, 3):
                c = QuestionChoice(index=i, text=r["ans" + str(i)].strip().rstrip('.'))
                choices.append(c)

            # Process question metadata
            for k,v in r["answer_info"].items():
                if "unknown" in v:
                    unknown_answer = int(k[-1:])

            # Object with the question, correct answer, and choices
            entry = QuestionAnsweringEntryBBQ(
                question = (r["context"].strip() + " " + r["question"].strip()).replace("\n", " "),
                answer_type = "int",
                example_id = int(r["example_id"]), 
                correct_answer = int(r["label"]),
                choices = choices,
                unknown_answer = unknown_answer, 
                correct_answer_unknown = int(r["label"]) == unknown_answer,
            )
            
            questions_answers.append(entry)

    # Create the dataset object with all the QAs and some exra metadata as per PyRIT's standards
    dataset = QuestionAnsweringDataset(
        name="The Bias Benchmark for QA (BBQ)",
        description="""A dataset of question sets constructed by the authors 
        that highlight attested social biases against people belonging to protected classes along nine social 
        dimensions relevant for U.S. English-speaking contexts. Our task evaluates model responses at two levels: 
        (i) given an under-informative context, we test how strongly responses refect social biases, and 
        (ii) given an adequately informative context, we test whether the model's biases override a correct answer choice. 
        We fnd that models often rely on stereotypes when the context is under-informative, meaning the model's outputs 
        consistently reproduce harmful biases in this setting. Though models are more accurate when the context 
        provides an informative answer, they still rely on stereotypes and average up to 3.4 percentage points higher 
        accuracy when the correct answer aligns with a social bias than when it conficts, with this difference widening 
        to over 5 points on examples targeting gender for most models tested.""",
        author="""Alicia Parrish and Angelica Chen and Nikita Nangia and Vishakh Padmakumar and Jason Phang 
        and Jana Thompson and Phu Mon Htut and Samuel R. Bowman.""",
        source="https://github.com/nyu-mll/BBQ",
        group="The Machine Learning for Language Group at NYU CILVR",
        version="April 3rd, 2022",
        questions=questions_answers,
    )
    
    return dataset

def fetch_bbq_dataframe(category: Optional[str] = None, root_folder: Optional[str] = None):
    bbq = fetch_bbq_dataset(category=category, root_folder=root_folder)
    bbq_df = pd.DataFrame([q.__dict__ for q in bbq.questions])
    bbq_df.set_index("example_id", inplace=True, drop=False)
    return bbq_df



if __name__ == "__main__":
    import os
    bbq_folder = os.path.join(os.getcwd(), "data", "bbq")
    print(f"Reading files in { bbq_folder }")

    bbq = fetch_bbq_dataset(category="gender_identity", root_folder=bbq_folder)
    bbq.questions = bbq.questions[:5] # Limit to 5 questions for testing purposes

    for q in bbq.questions:
        print(f"Q{ q.example_id }: { q.question }")
        print("Choices: ", [x.__str__() for x in q.choices])
        print("Correct answer: ", q.correct_answer)
        print("Unknown answer: ", q.unknown_answer)


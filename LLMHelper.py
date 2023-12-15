import os

from ctransformers import AutoModelForCausalLM
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

AVAILABLE_MODELS_GGUF = {
    "TheBloke/Marcoroni-7B-v3-GGUF": {
        "model_file": "marcoroni-7b-v3.Q4_K_M.gguf",
        "model_type": "marcoroni"
    },
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": {
        "model_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "model_type": "mistral"
    },
    "TheBloke/LeoScorpius-7B-GGUF": {
        "model_file": "leoscorpius-7b.Q4_K_M.gguf",
        "model_type": "leoscorpius"
    }
}

AVAILABLE_MODELS_OPENAI = [
    "gpt-4-1106-preview", "gpt-4-32k", "gpt-3.5-turbo-1106",
]




def generate_cover_letter_open_source(job_description, resume, selected_model, context_length=8000):
    print(f"selected_model: {selected_model}, "
          f"{AVAILABLE_MODELS_GGUF[selected_model]['model_file']}, {AVAILABLE_MODELS_GGUF[selected_model]['model_type']}")
    print(f"context_length: {context_length}")

    prompt = (f"Do the following steps: "
              f"1. Read the following job description,"
              f"2. Read the following resume, "
              f"3. Write a formal cover letter to the hiring manager for the job description based on the given resume, "
              # f"4. The cover letter MUST BE within {output_size_range[0]} to {output_size_range[1]} words. "
              # f"4. The cover letter MUST BE within 100 words. "
              f"4. Return ONLY the cover letter ONCE, nothing else. "
              f"Job Description: '{job_description}'. Resume: '{resume}'")

    # prompt = "What is an LLM"

    llm = AutoModelForCausalLM.from_pretrained(selected_model,
                                               model_file=AVAILABLE_MODELS_GGUF[selected_model]['model_file'],
                                               model_type=AVAILABLE_MODELS_GGUF[selected_model]['model_type'],
                                               context_length=context_length,
                                               max_new_tokens=1000,
                                               reset=True,
                                               stream=True,
                                               # top_k=2,
                                               temperature=0.5,
                                               threads=(os.cpu_count()//2)
                                               )

    llm_response = llm(prompt)

    return llm_response


def generate_cover_letter_openai(job_description, resume, selected_model, openai_key=None):
    os.environ["OPENAI_API_KEY"] = openai_key
    temp = "Do the following steps: " \
           "1. Read the following job description," \
           "2. Read the following resume, " \
           "3. Write a formal cover letter to the hiring manager for the job description based on the given resume, " \
           "4. Return ONLY the cover letter ONCE, nothing else. " \
           "Job Description: '{job_description}'. Resume: '{resume}'"

    prompt = PromptTemplate(
        template=temp,
        input_variables=["job_description", "resume"]
    )

    model = OpenAI(model_name=selected_model)

    _input = prompt.format(job_description=job_description, resume=resume)
    output = model.stream(_input)

    return output

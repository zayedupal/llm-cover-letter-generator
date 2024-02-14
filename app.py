import time
import streamlit as st
import LLMHelper
import streamlit.components.v1 as components

def generate_open_source():
    with output_col:
        st.session_state.running = True
        start_time = time.time()
        context_length = (len(st.session_state.get('resume', '').split()) +
                          len(st.session_state.get('jd', '').split()) +
                          2000)
        cover_letter_generator = LLMHelper.generate_cover_letter_open_source(
            job_description=st.session_state['jd'], resume=st.session_state['resume'],
            selected_model=selected_model, context_length=context_length
        )
        print(f'generated text: {cover_letter_generator}')
        generate_response(cover_letter_generator, start_time)
        st.session_state.running = False


def generate_openai():
    with output_col:
        start_time = time.time()
        try:
            cover_letter_generator = LLMHelper.generate_cover_letter_openai(
                job_description=st.session_state['jd'], resume=st.session_state['resume'],
                selected_model=selected_model, openai_key=open_ai_key
            )
            generate_response(cover_letter_generator, start_time)
        except ValueError as e:
            st.error("Please provide a valid Open AI API key")
        st.session_state.running = False


def generate_response(cover_letter_gen, start_time):
    with output_col:
        if cover_letter_gen is not None:
            with st.container(border=True):
                with st.spinner("Generating text..."):
                    generated_text_placeholder = st.empty()
                    for chunk in cover_letter_gen:
                        st.session_state.cover_letter_stream += chunk
                        generated_text_placeholder.write(st.session_state.cover_letter_stream)
                    st.write(f"generated words: {len(st.session_state.cover_letter_stream.split())}")
                    st.write(f"generation time: {round(time.time() - start_time, 2)} seconds")
                    st.write(
                        f"tokens per second: {round(len(st.session_state.cover_letter_stream.split())/(round(time.time() - start_time, 2)))}")


if 'running' not in st.session_state:
    st.session_state.running = False

st.session_state.cover_letter_stream = ""
st.set_page_config(page_title='LLM Cover Letter Generator', layout="wide")
# microsoft clarity analytics tracking code
with open("ms_clarity_tracking.html", "r") as f:
    html_code = f.read()
    components.iframe(html_code, height=0)

st.markdown("## Cover Letter Generator using Large Language Models (LLM)")
st.info("This project aims to Explore various open-source Large Language Models (LLMs) and "
        "compare them to OpenAI models.  \n"
        "Please be patient with the open source LLM models, as they are running without GPU.  \n "
        "Average generation time around 5 minutes.  \n"
        "The Open AI models are faster, but needs API key as they are hosted by Open AI.  \n"
        "Checkout my profile: https://zayedupal.github.io"
        )

input_col, output_col = st.columns(2)

with input_col:
    st.session_state['jd'] = st.text_area("Job Description",
                                          placeholder="Paste the job description here",
                                          disabled=st.session_state.running)
    st.write(f"{len(st.session_state.get('jd', '').split())} words")

    st.session_state['resume'] = st.text_area("Resume Information",
                                              placeholder="Paste the resume content here",
                                              disabled=st.session_state.running)

    st.write(f"{len(st.session_state.get('resume', '').split())} words")

    with output_col:
        llm_tab = st.radio("LLM type", ["Open Source LLMs", "Open AI LLMs"], horizontal=True)
        if llm_tab == "Open Source LLMs":
            cover_letter_generator = None
            st.session_state.cover_letter_stream = ""
            selected_model = st.selectbox("Select LLM Model", options=LLMHelper.AVAILABLE_MODELS_GGUF.keys(),
                                          disabled=st.session_state.running)

            st.button("Generate Cover Letter", key='open_source_gen_key', on_click=generate_open_source,
                      disabled=st.session_state.running)

        elif llm_tab == "Open AI LLMs":
            cover_letter_generator = None
            st.session_state.cover_letter_stream = ""
            selected_model = st.selectbox("Select Open AI Model", options=LLMHelper.AVAILABLE_MODELS_OPENAI,
                                          disabled=st.session_state.running)
            open_ai_key = st.text_input("Enter your open ai API key", type='password')
            st.button("Generate Cover Letter", key='open_ai_gen_key', disabled=st.session_state.running,
                      on_click=generate_openai)

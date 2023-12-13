import time
import streamlit as st
import LLMHelper


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


if 'running' not in st.session_state:
    st.session_state.running = False

st.session_state.cover_letter_stream = ""
st.set_page_config(page_title='Cover Letter Generator', layout="wide")
st.markdown("## Cover Letter Generator")
info = st.expander("Information")
info.write(f"This project aims to:\n"
           f"- Explore various open-source Large Language Models (LLMs).\n"
           f"- Compare them to OpenAI models for performance. \n"
           f"- Highlight benefits of open-source LLMs: \n"
           f"  - Faster generation on OpenAI models due to non-local execution.\n"
           f"  - Run open-source models on CPU with 10GB RAM (Around 5-min generation time).\n"
           f"  - Significantly faster generation on GPUs. \n"
           f"  - Free of cost and user-data ownership.\n\n"
           f"Checkout my profile: https://zayedupal.github.io")

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

        elif llm_tab == "Open AI Models":
            cover_letter_generator = None
            st.session_state.cover_letter_stream = ""
            selected_model = st.selectbox("Select Open AI Model", options=LLMHelper.AVAILABLE_MODELS_OPENAI,
                                          disabled=st.session_state.running)
            open_ai_key = st.text_input("Enter your open ai API key")
            st.button("Generate Cover Letter", key='open_ai_gen_key', disabled=st.session_state.running,
                      on_click=generate_openai)

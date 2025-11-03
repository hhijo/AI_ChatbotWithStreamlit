import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:")

def load_text_generator():
    text_generator = pipeline("text-generation", model="gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generator

    SYSTEM_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions and help with tasks."
        "You are also a great listener and can provide empathetic support."
        "Use simple, easy to understand language and avoid using jargon."
        "If you don't know the answer, say 'I don't know' and offer to help find the information."
    )

    # Build the convo prompt
    def build_conversation_prompt(chat_history, user_questio):
        formated_conversation = []
        for previous_question, previous_answer in chat_history:
            formated_conversation.append(f"User: {previous_question}\nAssistant: {previous_answer}")

        formated_conversation.append(f"User: {user_question}\nAssistant:")
        return SYSTEM_INSTRUCTIONS + "\n" + "\n".join(formated_conversation)

    
    st.title("AI Chatbot")
    st.caption("Ask me anything!")


    # Sidebar for config
    with st.sidebar:
        st.header("Model Controls/Config")
        max_new_tokens = st.slider("Max New Tokens", min_value=10, max_value=200, value=50, step=10)
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

        if st.button("Clear Chat"):
            st.session_state.chat_history = ["start new chat"]
            st.success("Chat history cleared")

    # Display chat history
    for user_message, ai_reply in st.session_state.chat_history:
        st.chat_message("user").markdown(user_message)
        st.chat_message("assistant").markdown(ai_reply)


    # User input
    user_input = st.chat_input("Ask me anything!")
    if user_input:
        st.chat_message("user").markdown(user_input)

        with st.spinner("Thinking..."):
            text_generator = load_text_generator()
            prompt_text = build_conversation_prompt(st.session_state.chat_history, user_input)

            generation_output = text_generator(
                prompt_text,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=text_generator.tokenizer.eos_token_id,
                eos_token_id=text_generator.tokenizer.eos_token_id,
            )[0]['generated_text']

            # Extracting the model answer from the generated text
            if "Assistant:" in generation_output:
                generated_answer = generation_output.split("Assistant:")[0].strip()

    # Displaying and storing the chatbot response
    st.chat_message("assistant").markdown(generated_answer)
    st.session_state.chat_history.append((user_input, generated_answer))

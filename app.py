import os
import streamlit as st
import streamlit.components.v1 as components
from utils.dfa_minimization import load_dfa_minimization_model, predict_dfa_minimization,load_tokenizer
from utils.regex_to_epsilon_nfa import load_regex_to_e_nfa_model,predict_regex_to_e_nfa
from utils.e_nfa_to_dfa import load_e_nfa_to_dfa_model,predict_e_nfa_to_dfa
from utils.push_down_automata import load_PDA_model,predict_PDA_transitions
from utils.graphviz.graphviz_regex_to_e_nfa import epsilon_nfa_to_dot
from utils.graphviz.graphviz_minimized_dfa import minimized_dfa_to_dot
from utils.graphviz.graphviz_dfa import dfa_output_to_dot
from utils.graphviz.graphviz_pda import pda_output_to_dot
from utils.llm import setup_llm
from utils.conversations import load_conversation_history
from langchain_core.messages import HumanMessage
from utils.classes.regex_conversion_stack import RegexConversionStack
from utils.classes.e_nfa_conversion_stack import enfaConversionStack
from utils.classes.dfa_minimization_stack import DfaMinimizationStack
from utils.classes.push_down_automata_stack import PushDownAutomataStack
from utils.text_extraction.dfa_minimization_image_to_text import extract_dfa_text_from_image
from utils.text_extraction.e_nfa_image_to_text import extract_e_nfa_text_from_image
from streamlit_scroll_to_top import scroll_to_here

st.set_page_config(
    page_title='State Forge',
    page_icon='⚙️',
    layout='wide'
)

st.markdown('# State Forge')

models_root = './models'
models = [
    {"name": "DFA-Minimization", "path": os.path.join(models_root, "dfa_minimization")},
    {"name": "Regex-to-ε-NFA", "path": os.path.join(models_root, "regex_to_e_nfa")},
    {"name": "e_NFA-to-DFA", "path": os.path.join(models_root, "e_nfa_to_dfa")},
    {"name": "PDA", "path": os.path.join(models_root, "pda")},
]

# Validate that model paths exist
valid_models = []
for model_config in models:
    if os.path.isdir(model_config["path"]):
        valid_models.append(model_config)
    else:
        st.warning(f"Model path not found: {model_config['path']}")

if not valid_models:
    st.error("No valid models available.")
    st.stop()

model_names = [m["name"] for m in valid_models]
selected_name = st.sidebar.selectbox('Choose Converter', model_names, index=0)


selected_model = next(m for m in valid_models if m["name"] == selected_name)
st.session_state.selected_model = selected_model


if selected_model['name'] == "Regex-to-ε-NFA":
    if "regex_stack" not in st.session_state:
        st.session_state.regex_stack = RegexConversionStack()

if selected_model['name'] == "e_NFA-to-DFA":
    if "e_nfa_stack" not in st.session_state:
        st.session_state.e_nfa_stack = enfaConversionStack()

if selected_model['name'] == "DFA-Minimization":
    if "dfa_stack" not in st.session_state:
        st.session_state.dfa_stack = DfaMinimizationStack()

if selected_model['name'] == "PDA":
    if "pda_stack" not in st.session_state:
        st.session_state.pda_stack = PushDownAutomataStack()


def load_model(model_name: str):

    if model_name == "DFA-Minimization":
        dfa_minimization_model =load_dfa_minimization_model("models/dfa_minimization/dfa_minimizer_transformer.pt","models/dfa_minimization/dfa_minimizer_tokenizer.pkl")
        return dfa_minimization_model, None, None
    elif model_name == "Regex-to-ε-NFA":
        regex_to_e_nfa_model,stoi, itos = load_regex_to_e_nfa_model("models/regex_to_e_nfa/transformer_regex_to_e_nfa.pt","models/regex_to_e_nfa/regex_to_e_nfa_tokenizer.pkl")
        return regex_to_e_nfa_model,stoi, itos
    elif model_name == "e_NFA-to-DFA":
        e_nfa_to_dfa_model = load_e_nfa_to_dfa_model("models/e_nfa_to_dfa/transformer_model.pt")
        return e_nfa_to_dfa_model, None, None
    elif model_name == "PDA":
        pda_model = load_PDA_model("models/pda/pda.pth")
        return pda_model, None, None

    return None  # Replace with actual model


def clear_on_convert():
    if st.session_state.conversion_result:
        st.session_state.conversion_result = None
        st.session_state.conversion_graph = None
        st.session_state.diagram_png_bytes = None
        st.session_state.latest_input_regex = None
        st.session_state.regex_to_e_nfa_transition = None
        st.session_state.regex_to_e_nfa_used  = False
        st.session_state.latest_input_e_nfa = None
        st.session_state.e_nfa_to_dfa_transition = None
        st.session_state.e_nfa_to_dfa_used  = False
        st.session_state.latest_input_pda = None
        st.session_state.pda_transition = None
        st.session_state.pda_used  = False


st.session_state.pressed_once = False

# Input area with dynamic placeholder based on selected model
input_placeholder = {
    "DFA-Minimization": "Enter your DFA description (states, transitions, etc.)",
    "Regex-to-ε-NFA": "Enter your regular expression",
    "e_NFA-to-DFA": "Enter your ∈-NFA description",
    "PDA": "Enter your language example string...\nEg:- aabb (a^nb^n)"
    # Add more placeholders for other models
}.get(selected_model['name'], "Enter your input here")

input_img_bytes = None
img_input = None
user_input = ""


# if selected_model['name'] == "DFA-Minimization" or selected_model['name'] == "e_NFA-to-DFA":
#     img_input =  st.file_uploader("Upload image of DFA or NFA",type=['png','jpg','jpeg','svg'])
    

# user_input = st.text_area("Input", placeholder=input_placeholder)

user_input = ""
if selected_model['name'] == "DFA-Minimization" or selected_model['name'] == "e_NFA-to-DFA":
    img_input = st.file_uploader("Upload image of DFA or NFA", type=['png','jpg','jpeg','svg'])
    if img_input:
        with st.spinner("Extracting DFA transitions from image..."):
            try:
                image_bytes = img_input.read()
                if selected_model['name'] == "DFA-Minimization":
                    user_input = extract_dfa_text_from_image(image_bytes)
                elif selected_model['name'] == "e_NFA-to-DFA":
                    user_input = extract_e_nfa_text_from_image(image_bytes)
                st.success("Text extracted from image successfully!")
            except Exception as e:
                st.error(f"Failed to extract text: {e}")
    
user_input = st.text_area("Input", placeholder=input_placeholder, value=user_input)


if selected_model['name'] == "Regex-to-ε-NFA":
    st.session_state.latest_input_regex = user_input

if selected_model['name'] == "e_NFA-to-DFA":
    st.session_state.latest_input_e_nfa = user_input

if selected_model['name'] == "DFA-Minimization":
    st.session_state.latest_input_dfa = user_input

if selected_model['name'] == "PDA":
    st.session_state.latest_input_pda = user_input

if st.button("Convert", type="primary"):
    if not user_input.strip():
        st.warning("Please enter something to convert.")
    else:
        with st.spinner(f"Converting using {selected_model['name']}..."):
            
            model,stoi,itos = load_model(selected_model['name'])
            
            result = None
            graph =  None
            png_bytes = None

            if selected_model['name'] == "Regex-to-ε-NFA":
                result = predict_regex_to_e_nfa(user_input,model,stoi,itos)
                st.session_state.regex_to_e_nfa_transition = result
                st.session_state.regex_stack.push(user_input,result)
                st.session_state.is_pressed_convert = True
                if "regex_to_e_nfa_used" in st.session_state: 
                    st.session_state.regex_to_e_nfa_used = False
                graph =epsilon_nfa_to_dot(result)
                png_bytes = graph.pipe(format="png")

            elif selected_model['name'] == "DFA-Minimization":
                result = predict_dfa_minimization(model,user_input)
                st.session_state.dfa_to_minimized_dfa_transition = result
                st.session_state.dfa_stack.push(user_input,result)
                st.session_state.is_pressed_convert = True
                if "dfa_to_minimized_dfa_used" in st.session_state: 
                    st.session_state.dfa_to_minimized_dfa_used = False
                graph = minimized_dfa_to_dot(result)
                png_bytes = graph.pipe(format="png")

            elif selected_model['name'] == "e_NFA-to-DFA":
                result = predict_e_nfa_to_dfa(model,user_input)
                st.session_state.e_nfa_to_dfa_transition = result
                st.session_state.e_nfa_stack.push(user_input,result)
                st.session_state.is_pressed_convert = True
                if "e_nfa_to_dfa_used" in st.session_state: 
                    st.session_state.e_nfa_to_dfa_used = False
                graph =dfa_output_to_dot(result)
                png_bytes = graph.pipe(format="png")

            elif selected_model['name'] == "PDA":
                result = predict_PDA_transitions(model,user_input)
                st.session_state.pda_transition = result
                st.session_state.pda_stack.push(user_input,result)
                st.session_state.is_pressed_convert = True
                if "pda_used" in st.session_state: 
                    st.session_state.pda_used = False
                graph =pda_output_to_dot(result)
                png_bytes = graph.pipe(format="png")
            
            st.session_state.conversion_result = result
            st.session_state.conversion_graph  = graph
            st.session_state.diagram_png_bytes = png_bytes



if 'conversion_result' in st.session_state and "diagram_png_bytes" in st.session_state:
    st.subheader("Conversion Result:")
    st.code(st.session_state.conversion_result, language="text")
    st.subheader("Generated Diagram:")
    st.graphviz_chart(st.session_state.conversion_graph.source)

    if st.session_state.diagram_png_bytes:
        st.subheader("Download Diagram as PNG")
        st.download_button(
            label="⬇️ Download (PNG)",
            data=st.session_state.diagram_png_bytes,
            file_name="diagram.png",
            mime="image/png"
        )
            

if selected_model['name'] == "Regex-to-ε-NFA" or selected_model['name'] == "e_NFA-to-DFA" or selected_model['name'] == "DFA-Minimization" or selected_model['name'] == "PDA":
    if 'app' not in st.session_state:
        st.session_state.app,st.session_state.config = setup_llm()

    if "messages" not in st.session_state:
        st.session_state.messages = load_conversation_history(
            st.session_state.app,
            st.session_state.config
        )
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create human message and invoke the app
                    human_message = HumanMessage(content=prompt)
                    output = st.session_state.app.invoke(
                        {"messages": [human_message]}, 
                        st.session_state.config
                    )
                        
                    # Get assistant response
                    assistant_response = output["messages"][-1].content
                        
                        # Display assistant response
                    st.markdown(assistant_response)
                        
                        # Add assistant response to chat history
                    st.session_state.messages.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
                        
                except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })



# Styled scroll to top button
st.markdown('''
    <style>
    .scroll-to-top {
        position: fixed;
        bottom: 120px;
        right: 20px;
        z-index: 9999;
        text-decoration: none;
    }
    
    .scroll-to-top button {
        background: #1e3a8a;
        color: white;
        border: 2px solid #1e3a8a;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4);
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .scroll-to-top button:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.5);
    }
    
    .scroll-to-top button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.3);
    }
    </style>
    
    <a class="scroll-to-top" target="_self" href="#section-1">
        <button>
            TOP
        </button>
    </a>
''', unsafe_allow_html=True)


st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info(f"Selected Model: **{selected_model['name']}**")
st.sidebar.write(f"Model Path: `{selected_model['path']}`")

st.sidebar.markdown("---")
st.sidebar.subheader('Controls')



if st.sidebar.button("Clear Chat History",type="secondary"):
    if selected_model['name'] == "Regex-to-ε-NFA":
        st.session_state.messages = []
        raise st.rerun()
    if selected_model['name'] == "DFA-Minimization":
        st.session_state.messages = []
        raise st.rerun()
    if selected_model['name'] == "e_NFA-to-DFA":
        st.session_state.messages = []
        raise st.rerun()
    if selected_model['name'] == "PDA":
        st.session_state.messages = []
        raise st.rerun()
if st.sidebar.button("View your conversion history"):
    @st.dialog("Conversion History")
    def conversion_history():
        # st.write("Conversion History")
        # history = st.session_state.regex_stack.all_items()
        # if not history:
        #     st.info("No conversions found yet.")
        #     return
        
        # for idx, item in enumerate(history[::-1], start=1):  # Show most recent first
        #     st.markdown(f"### 🔢 Conversion {idx}")
        #     st.markdown(f"**Regex:** `{item['regex']}`")
        #     st.markdown("**Conversion Result:**")
        #     st.code(item['conversion'], language='text')
        #     st.markdown("---")
        if selected_model['name'] == "Regex-to-ε-NFA":
            history = st.session_state.regex_stack.all_items()
            if not history:
                st.info("No conversions found yet.")
                return
            
            for idx, item in enumerate(history[::-1], start=1):
                st.markdown(f"### 🔢 Conversion {idx}")
                st.markdown(f"**Regex:** `{item['regex']}`")
                st.markdown("**Conversion Result:**")
                st.code(item['conversion'], language='text')
                st.markdown("---")

        elif selected_model['name'] == "DFA-Minimization":
            history = st.session_state.dfa_stack.all_items()
            if not history:
                st.info("No conversions found yet.")
                return
            
            for idx, item in enumerate(history[::-1], start=1):
                st.markdown(f"### 🔢 Conversion {idx}")
                st.markdown(f"**DFA Input:** `{item['regex']}`")  # 'regex' is used as the key — rename later
                st.markdown("**Minimized DFA:**")
                st.code(item['conversion'], language='text')
                st.markdown("---")

        elif selected_model['name'] == "e_NFA-to-DFA":
            history = st.session_state.e_nfa_stack.all_items()
            if not history:
                st.info("No conversions found yet.")
                return
            
            for idx, item in enumerate(history[::-1], start=1):
                st.markdown(f"### 🔢 Conversion {idx}")
                st.markdown(f"**E NFA Input:** `{item['regex']}`")  # 'regex' is used as the key — rename later
                st.markdown("**Converted DFA:**")
                st.code(item['conversion'], language='text')
                st.markdown("---")

        elif selected_model['name'] == "PDA":
            history = st.session_state.pda_stack.all_items()
            if not history:
                st.info("No conversions found yet.")
                return
            
            for idx, item in enumerate(history[::-1], start=1):
                st.markdown(f"### 🔢 Conversion {idx}")
                st.markdown(f"**Context-Free Input String:** `{item['cf_string']}`")
                st.markdown("**Conversion Result:**")
                st.code(item['conversion'], language='text')
                st.markdown("---")
    conversion_history()


st.sidebar.markdown("---")
if selected_model['name'] == "Regex-to-ε-NFA":
    st.sidebar.markdown(f"**Messages in conversation:** {len(st.session_state.messages)}")
if selected_model['name'] == "DFA-Minimization":
    st.sidebar.markdown(f"**Messages in conversation:** {len(st.session_state.messages)}")
if selected_model['name'] == "PDA":
    st.sidebar.markdown(f"**Messages in conversation:** {len(st.session_state.messages)}")
if selected_model['name'] == "e_NFA-to-DFA":
    st.sidebar.markdown(f"**Messages in conversation:** {len(st.session_state.messages)}")

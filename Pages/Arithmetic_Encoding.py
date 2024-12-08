import streamlit as st
import pyae

st.sidebar.success("Choose an algorithm")

if "msg" not in st.session_state:
    st.session_state.msg = ""

if "encoded_msg" not in st.session_state:
    st.session_state.encoded_msg = 0

if "AE" not in st.session_state:
    st.session_state.AE = ""

st.markdown("# Arithmetic Encoding")

msg = st.text_input("Input a string to encode", key="msg")

encoded_msg_container = st.container()

if st.button("Encode the string"):
    frequency_table = {char: msg.count(char) for char in msg}
    st.session_state.AE = pyae.ArithmeticEncoding(frequency_table=frequency_table, save_stages=True)
    st.session_state.encoded_msg, _, _, _ = st.session_state.AE.encode(msg=msg, probability_table=st.session_state.AE.probability_table)

encoded_msg_container.text_input("Encoded string", st.session_state.encoded_msg)

# Decoding
st.markdown("# Decoding")

decoded_msg_container = st.container()
decoded_msg_string = ""

if st.button("Decode the binary string:"):
    decoded_msg_list, _ = st.session_state.AE.decode(
        encoded_msg=st.session_state.encoded_msg,
        msg_length=len(msg),
        probability_table=st.session_state.AE.probability_table
    )
    decoded_msg_string = "".join(decoded_msg_list)

decoded_msg_container.text_input("Decoded message", value=decoded_msg_string, disabled=True, key="decoded_msg")

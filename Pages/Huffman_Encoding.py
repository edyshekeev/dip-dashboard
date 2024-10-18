import streamlit as st

class Node:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

def calculate_frequencies(word):
    frequencies = {}
    for char in word:
        if char not in frequencies:
            freq = word.count(char)
            frequencies[char] = freq
            nodes.append(Node(char, freq))

def build_huffman_tree():
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        
        merged = Node(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        nodes.append(merged)

    return nodes[0]

def generate_huffman_codes(node, current_code, codes):
    if node is None:
        return

    if node.char is not None:
        codes[node.char] = current_code

    generate_huffman_codes(node.left, current_code + '0', codes)
    generate_huffman_codes(node.right, current_code + '1', codes)

def huffman_encoding(word):
    global nodes
    nodes = []
    calculate_frequencies(word)
    root = build_huffman_tree()
    codes = {}
    generate_huffman_codes(root, '', codes)
    return codes

def huffman_decoding(encoded_word, codes):
    current_code = ''
    decoded_chars = []

    # Invert the codes dictionary to get the reverse mapping
    code_to_char = {v: k for k, v in codes.items()}

    for bit in encoded_word:
        current_code += bit
        if current_code in code_to_char:
            decoded_chars.append(code_to_char[current_code])
            current_code = ''

    return ''.join(decoded_chars)

st.sidebar.success("Choose an algorithm")

if "msg" not in st.session_state:
    st.session_state.msg = ""

if "encoded_msg" not in st.session_state:
    st.session_state.encoded_msg = 0

if "codes" not in st.session_state:
    st.session_state.codes = {}

st.markdown("#Encoding")

msg = st.text_input("Input a string to encode", key="msg")

encoded_msg = st.container()

if st.button("Encode the string"):
    frequency_table = {}
    for i in range(len(msg)):
        if msg[i] not in frequency_table.keys():
            frequency_table.update({msg[i]:1})
        else:
            frequency_table[msg[i]] += 1 
    nodes = []
    st.session_state.codes = huffman_encoding(msg)
    st.session_state.encoded_msg = ''.join(st.session_state.codes[char] for char in msg)

encoded_msg.text_input("Encoded string", st.session_state.encoded_msg)

st.markdown("#Decoding")

decoded_msg = st.container()

decoded_msg_string = ""

if st.button("Decode the binary string:"):
    decoded_msg_string = huffman_decoding(st.session_state.encoded_msg, st.session_state.codes)

decoded_msg.text_input("Decoded message", value=decoded_msg_string, disabled=True, key="decoded_msg")
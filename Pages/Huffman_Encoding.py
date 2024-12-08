import streamlit as st

class Node:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

def calculate_frequencies(word):
    frequencies = {char: word.count(char) for char in word}
    nodes = [Node(char, freq) for char, freq in frequencies.items()]
    return nodes

def build_huffman_tree(nodes):
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x.freq)
        left, right = nodes.pop(0), nodes.pop(0)
        merged = Node(freq=left.freq + right.freq)
        merged.left, merged.right = left, right
        nodes.append(merged)
    return nodes[0]

def generate_huffman_codes(node, current_code="", codes={}):
    if node is None:
        return
    if node.char:
        codes[node.char] = current_code
    generate_huffman_codes(node.left, current_code + '0', codes)
    generate_huffman_codes(node.right, current_code + '1', codes)
    return codes

def huffman_encoding(word):
    nodes = calculate_frequencies(word)
    root = build_huffman_tree(nodes)
    codes = generate_huffman_codes(root)
    return codes

def huffman_decoding(encoded_word, codes):
    code_to_char = {v: k for k, v in codes.items()}
    current_code, decoded_chars = "", []
    for bit in encoded_word:
        current_code += bit
        if current_code in code_to_char:
            decoded_chars.append(code_to_char[current_code])
            current_code = ''
    return ''.join(decoded_chars)

st.sidebar.success("Choose an algorithm")

if "msg" not in st.session_state:
    st.session_state.msg = ""

if "encoded_msg_huffman" not in st.session_state:
    st.session_state.encoded_msg_huffman = ""

if "codes" not in st.session_state:
    st.session_state.codes = {}

st.markdown("# Huffman Encoding")

msg = st.text_input("Input a string to encode using Huffman", key="msg_huffman")

encoded_msg_huffman_container = st.container()

if st.button("Encode with Huffman"):
    st.session_state.codes = huffman_encoding(msg)
    st.session_state.encoded_msg_huffman = ''.join(st.session_state.codes[char] for char in msg)

encoded_msg_huffman_container.text_input("Encoded string (Huffman)", st.session_state.encoded_msg_huffman)

st.markdown("# Huffman Decoding")

decoded_msg_huffman_container = st.container()
decoded_msg_huffman = ""

if st.button("Decode Huffman"):
    decoded_msg_huffman = huffman_decoding(st.session_state.encoded_msg_huffman, st.session_state.codes)

decoded_msg_huffman_container.text_input("Decoded message (Huffman)", value=decoded_msg_huffman, disabled=True, key="decoded_msg_huffman")

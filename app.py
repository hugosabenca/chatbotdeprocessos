import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx
import os

# --- 1. LEITURA SEGURA DA CHAVE DE API (A MUDANÇA PRINCIPAL) ---
# O Streamlit lê o "Secret" que configuramos no painel online.
try:
    # A linha mais importante: pega a chave do "cofre" do Streamlit Cloud
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except (KeyError, FileNotFoundError):
    st.error("Chave de API do Google não configurada nos 'Secrets' do Streamlit.")
    st.stop() # Interrompe a execução se a chave não for encontrada

# --- Funções Auxiliares (já corrigidas) ---

def get_documents_text(uploaded_files):
    """Extrai o texto de uma lista de arquivos PDF e DOCX."""
    text = ""
    for doc_file in uploaded_files:
        if doc_file.name.endswith('.docx'):
            try:
                doc = docx.Document(doc_file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Erro ao ler o arquivo DOCX {doc_file.name}: {e}")
        elif doc_file.name.endswith('.pdf'):
            try:
                pdf_reader = PdfReader(doc_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                st.error(f"Erro ao ler o arquivo PDF {doc_file.name}: {e}")
    return text

def get_text_chunks(text):
    """Divide o texto em pedaços (chunks) menores."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Cria e salva um banco de dados de vetores a partir dos chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Cria a cadeia de conversação com o modelo e o prompt melhorado."""
    prompt_template = """
    Assuma a persona de um especialista de processos da Dox Brasil Pinheiral. Você é prestativo,
    confiante e responde de forma direta e natural. Aja como se o conhecimento fosse seu, não como se
    estivesse lendo um documento.

    REGRAS IMPORTANTES:
    1. NUNCA use frases como "com base no contexto", "segundo o documento", "a informação fornecida diz".
    2. Responda diretamente à pergunta do usuário.
    3. Se a informação não estiver disponível, diga de forma natural, como por exemplo: "Não encontrei os detalhes sobre esse processo específico" ou "Essa informação não está no meu escopo de conhecimento".

    Use as informações abaixo para formular sua resposta final.
    ---
    INFORMAÇÕES DISPONÍVEIS:
    {context}
    ---
    
    PERGUNTA DO USUÁRIO:
    {question}

    SUA RESPOSTA DIRETA E HUMANA:
    """
    model = genai.GenerativeModel('gemini-2.5-pro')
    return model

# --- Interface Principal do Streamlit ---

st.set_page_config(page_title="Chatbot Dox Brasil", page_icon="🤖")
st.title("🤖 Chatbot de Processos Dox Brasil Pinheiral")
st.write("Faça o upload dos documentos da empresa e faça perguntas sobre eles!")

# Barra lateral para upload (sem pedir a chave!)
with st.sidebar:
    st.header("Documentos")
    uploaded_files = st.file_uploader("Faça upload dos seus PDFs ou DOCX aqui", accept_multiple_files=True)
    if st.button("Processar Documentos"):
        if not uploaded_files:
            st.warning("Por favor, faça o upload de pelo menos um documento.")
        else:
            with st.spinner("Processando..."):
                raw_text = get_documents_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Documentos processados com sucesso!")

# Lógica do Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Qual é a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if not os.path.exists("faiss_index"):
        st.warning("Por favor, processe os documentos na barra lateral primeiro.")
        st.stop()

    with st.spinner("Pensando..."):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])
        
        model = get_ conversational_chain()
        # O prompt já está dentro da função, então passamos o contexto e a pergunta
        prompt_completo = model.prompt_template.format(context=context, question=user_question)
        response = model.generate_content(prompt_completo)

        bot_response = response.text
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)
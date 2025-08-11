import streamlit as st
import google.generativeai as genai
import docx
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
import os

# --- Configurações Iniciais e Funções ---

# Configura a página do Streamlit
st.set_page_config(page_title="Chatbot Dox Brasil", page_icon="🤖")

def get_documents_text(uploaded_files):
    """Extrai o texto de uma lista de arquivos PDF e DOCX."""
    text = ""
    for doc_file in uploaded_files:
        # Verifica se o arquivo é um DOCX
        if doc_file.name.endswith('.docx'):
            try:
                doc = docx.Document(doc_file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Erro ao ler o arquivo DOCX {doc_file.name}: {e}")
        
        # Verifica se o arquivo é um PDF
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

def get_vector_store(text_chunks, api_key):  # <-- 1. Adicionamos o api_key como parâmetro
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key  # <-- 2. Passamos a chave diretamente aqui
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Assuma a persona de um especialista de processos da [NOME DA SUA EMPRESA]. Você é prestativo,
    confiante e responde de forma direta e natural. Aja como se o conhecimento fosse seu, não como se
    estivesse lendo um documento.

    **REGRAS IMPORTANTES:**
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

# --- Interface do Streamlit ---

st.title("🤖 Chatbot de Processos Dox Brasil Pinheiral")
st.write("Faça o upload dos seus PDFs e faça perguntas sobre eles!")

# Barra lateral para upload e configuração
with st.sidebar:
    st.header("Configuração")
    
    # Input para a chave da API do Google
    api_key = st.text_input("Sua chave da API do Google Gemini", type="password")
    if api_key:
        genai.configure(api_key=api_key)

    # Upload dos arquivos PDF
    pdf_docs = st.file_uploader("Faça upload dos seus PDFs aqui", accept_multiple_files=True)

    if st.button("Processar Documentos"):
        if not api_key:
            st.warning("Por favor, insira sua chave de API do Google.")
        elif not pdf_docs:
            st.warning("Por favor, faça o upload de pelo menos um PDF.")
        else:
            with st.spinner("Processando..."):
                raw_text = get_documents_text(pdf_docs) # Agora chamamos a nova função
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Documentos processados com sucesso!")

# --- Lógica do Chat ---

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obtém a pergunta do usuário
if user_question := st.chat_input("Qual é a sua dúvida?"):
    # Adiciona a pergunta do usuário ao histórico e exibe
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Verifica se a API e os documentos estão prontos
    if not api_key:
        st.warning("Por favor, insira e configure sua chave de API na barra lateral.")
    elif not os.path.exists("faiss_index"):
        st.warning("Por favor, processe os documentos na barra lateral primeiro.")
    else:
        with st.spinner("Pensando..."):
            # Carrega o vector store e busca documentos relevantes
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key # <-- A MESMA CORREÇÃO: Passamos a chave aqui também
            )
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # Permite carregar o índice
            docs = new_db.similarity_search(user_question)
            
            # Cria o contexto a partir dos documentos encontrados
            context = "\n".join([doc.page_content for doc in docs])
            
            # Monta o prompt
            prompt_completo = f"""
            Você é um assistente da Dox Brasil. Sua tarefa é responder perguntas sobre nossos processos
            com base no contexto fornecido. Responda de forma clara, objetiva e em português. Se a resposta não
            estiver no contexto, diga que você não tem essa informação.

            Contexto:
            {context}

            Pergunta:
            {user_question}

            Resposta:
            """

            # Gera a resposta com o Gemini
            model = get_conversational_chain()
            response = model.generate_content(prompt_completo)

            # Adiciona a resposta do bot ao histórico e exibe
            bot_response = response.text
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)
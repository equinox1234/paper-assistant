import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 🚨 这里的导入路径改为了 langchain_classic (适配 LangChain v1.0+)
# ==========================================
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ==========================================
# 0. 页面配置与 UI 设计
# ==========================================
st.set_page_config(page_title="论文研读助手", page_icon="📚", layout="wide")
st.title("📚 个人科研论文研读助手")
st.markdown("上传 PDF 文献，AI 自动解析，随时提问。")

# 左侧侧边栏：配置 API Key
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("请输入 API Key:", type="password",
                             value="")
    base_url = "https://api.siliconflow.cn/v1"
    st.info("💡 提示：项目使用了硅基流动的 Qwen 模型和 BGE 向量模型。")

# ==========================================
# 1. 初始化核心引擎 (缓存在 Session State 中)
# ==========================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 2. 上传与解析 PDF
# ==========================================
uploaded_file = st.file_uploader("请上传一篇 PDF 论文", type="pdf")

if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("正在努力阅读文献中，请稍候..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url, model="BAAI/bge-m3")
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

            st.session_state.vectorstore = vectorstore
            st.success("文献阅读完成！现在你可以向我提问了。")

        except Exception as e:
            st.error(f"解析失败: {e}")
        finally:
            os.remove(tmp_file_path)

# ==========================================
# 3. 聊天交互与问答
# ==========================================
if st.session_state.vectorstore is not None:

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("例如：这篇文章的核心创新点是什么？")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="Qwen/Qwen2.5-7B-Instruct", temperature=0)
        retriever = st.session_state.vectorstore.as_retriever()

        template = """
        你是一个严谨的科研助手。请基于以下【参考文献片段】回答用户的问题。
        如果文献中没有提及，请明确回答“文献中未提及”，绝不捏造数据。

        【参考文献片段】：
        {context}

        用户提问：{input}
        """
        prompt = ChatPromptTemplate.from_template(template)

        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = rag_chain.invoke({"input": user_question})
                ai_answer = response["answer"]
                st.markdown(ai_answer)

                with st.expander("查看引用的原文片段"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**片段 {i + 1}** (来自第 {doc.metadata.get('page', '未知')} 页):")
                        st.caption(doc.page_content)

        st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
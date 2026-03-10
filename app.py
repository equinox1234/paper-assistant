# ==========================================
# 0. 页面配置与 UI 设计
# ==========================================
st.set_page_config(page_title="论文研读助手", page_icon="📚", layout="wide")
st.title("📚 个人科研论文研读助手")
st.markdown("上传 PDF 文献，AI 自动解析，随时提问。")

# 左侧侧边栏：配置 API Key
with st.sidebar:
    st.header("⚙️ 配置中心")
    
    # 修改点：不再强制要求填入值，加一句温馨提示
    user_key = st.text_input(
        "请输入 API Key (访客请直接留空，使用内置体验通道):", 
        type="password", 
        value=""
    )
    base_url = "https://api.siliconflow.cn/v1"
    st.info("💡 提示：面试官/访客无需填写 Key，直接上传 PDF 即可体验。项目使用了 Qwen 模型和 BGE 向量模型。")

# 修改点：智能密钥选择逻辑
# 如果用户在网页上填了 key，就用用户的 user_key
if user_key:
    api_key = user_key
else:
    # 如果用户没填，程序就去 Streamlit 的隐形保险箱里拿我们预设的 Key
    try:
        api_key = st.secrets["SILICONFLOW_API_KEY"]
    except Exception:
        api_key = "" # 防止本地运行没有 secrets 时报错
        st.warning("⚠️ 未检测到系统内置密钥，请手动输入 API Key。")

import streamlit as st

st.set_page_config(page_title="Input Resources", layout="wide")

st.title("📁 Input Resources")

st.write("Upload documents to provide context for AI reasoning.")

# Checklist
st.subheader("✅ Document Checklist")
col1, col2 = st.columns(2)
with col1:
    st.checkbox("Technical Manuals", value=False)
    st.checkbox("Financial Reports", value=False)
with col2:
    st.checkbox("Maintenance Logs", value=False)
    st.checkbox("Parts Inventory", value=False)

st.divider()

# Drag and Drop Boxes
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🛠️ Technical Resource")
    tech_file = st.file_uploader("Upload technical manual (PDF, TXT)", type=["pdf", "txt"], key="tech")
    if tech_file:
        st.success(f"Uploaded: {tech_file.name}")

with col_b:
    st.subheader("💰 Financial Report")
    fin_file = st.file_uploader("Upload financial report (PDF, CSV)", type=["pdf", "csv"], key="fin")
    if fin_file:
        st.success(f"Uploaded: {fin_file.name}")

if st.button("Process Documents for RAG", type="primary"):
    with st.spinner("Indexing documents into Supabase Vector DB..."):
        # Placeholder for Langchain/Supabase logic
        import time
        time.sleep(2)
    st.success("Documents processed and ready for LLM reasoning!")

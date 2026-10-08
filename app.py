"""Workshop home: Learn → Build → Apply → Integrate."""
import streamlit as st
st.set_page_config(page_title='NLP Workshop', page_icon='📚')
st.title('From Classical NLP to Generative AI')
st.header('Building an Intelligent Academic Assistant')
st.write('A 2-Day Hands-On NLP Workshop for Engineering Students.')
st.subheader('Learn → Build → Apply → Integrate')
st.markdown('''- **Learn:** preprocessing, representations, embeddings and Transformers.
- **Build:** lexical search, semantic retrieval and transparent RAG.
- **Apply:** deterministic question-paper analysis and grounded study support.
- **Integrate:** the Intelligent Academic Assistant combines these capabilities.''')
st.subheader('Learning progression')
st.write('Raw Text → Preprocessing → Tokenization → Vocabulary → One-Hot Encoding → Bag of Words → N-Grams → TF-IDF → Cosine Similarity → Keyword Search → Word Embeddings → Sentence Embeddings → Semantic Search → Attention → Transformers → LLMs → Chunking → Retrieval → RAG → Intelligent Academic Assistant')
st.success('All ten pages are functional in working/reference form. The final Assistant is ready; 49 algorithm TODO boundaries remain unchanged.')
st.page_link('pages/10_Academic_Assistant.py', label='Open Intelligent Academic Assistant', icon='📚')
st.write('Choose a lab in the sidebar. Begin with workshop/SETUP_CHECKLIST.md and docs/STUDENT_GUIDE.md.')
st.caption('The institution and dataset are fictional. Generation is optional; supplied sources do not certify generated claims. Task17 reliability is audited. Task18 release preparation remains pending.')

st.caption('Before class: python -m workshop.preflight --full-classroom. Unit-specific assistant generation is withheld when requested evidence is missing; pure RAG Lab retrieval stays unchanged.')


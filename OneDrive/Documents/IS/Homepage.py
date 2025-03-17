import streamlit as st

st.set_page_config(page_title="IS project", page_icon="📊")

page = st.sidebar.selectbox("Select a Page", 
    ["Homepage", "Machine Learning", "Machine Learning Models", "Neural Network", "Neural Network Models"])

# if page != "Homepage":
#         st.session_state.page_history.append(page)
        
if page == "Homepage":
    st.write("# Intelligent System Project")
    st.write("Select a page from the sidebar to explore Machine Learning or Neural Network models.")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Machine Learning"):
            st.switch_page("pages/ProcessMachine_Learning.py")
    
    with col2:
        if st.button("Neural Network"):
            st.switch_page("pages/ProcessNeural_Network.py")



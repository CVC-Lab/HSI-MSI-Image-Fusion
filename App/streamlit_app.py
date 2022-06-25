# Contents of ~/my_app/streamlit_app.py
import streamlit as st

def main_page():
    st.markdown("# HSI-MSI-Image-Fusion")
    st.sidebar.markdown("# HSI-MSI-Image-Fusion")
    st.markdown(
        """
        The problem is super resolution for enhanced identification of target regions of interest (TROI). The solution to this problem is to combine two low resolution multispectral and hyperspectral video streams into a single super-resolution stream.
        """
    )

def page2():
    st.markdown("# Results")
    st.sidebar.markdown("# Results")


page_names_to_funcs = {
    "Main Page": main_page,
    "Results": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
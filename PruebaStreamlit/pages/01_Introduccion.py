import streamlit as st
import webbrowser

st.title('')


# URL dashboard de Power BI
power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiMzIzYzg3MTgtYjIxOS00MWM0LTliYjctOWE0MmI5OTczMGE0IiwidCI6IjBlMGNiMDYwLTA5YWQtNDlmNS1hMDA1LTY4YjliNDlhYTFmNiIsImMiOjR9&pageName=ReportSectiond0a2b6e7747a69360de0"

# Configurar sección
st.write("<h2>Tablero - KPI's</h2>", unsafe_allow_html=True)

# Insertar el dashboard en Streamlit
st.components.v1.iframe(power_bi_url, width=800, height=600)


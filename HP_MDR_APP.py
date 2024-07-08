import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

if 'model' not in st.session_state:
    model = joblib.load('model.pkl')
    st.session_state["model"] = model
else:
    model = st.session_state["model"]

st.set_page_config(layout="wide")


def set_background():
    page_bg_img = '''
    <style>
    .css-1nnpbs {width: 100vw}
    h1 {padding: 0.75rem 0px 0.75rem;margin-top: 2rem;box-shadow: 0px 3px 5px gray;}
    h2 {background-color: #B266FF;margin-top: 2vh;border-left: red solid 0.6vh}
    .css-1avcm0n {background: rgb(14, 17, 23, 0)}
    .css-18ni7ap {background: #B266FF;z-index:1;height:3rem}
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    .css-z5fcl4 {padding: 0 1rem 1rem}
    .css-12ttj6m {box-shadow: 0.05rem 0.05rem 0.2rem 0.1rem rgb(192, 192, 192);margin:0 calc(20% + 0.5rem);}
    .css-1cbqeqj {text-align: center;}
    .css-1x8cf1d {background: #00800082}
    .css-1x8cf1d:hover {background: #00800033}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background()
st.markdown(
    "<h1 style='text-align: center'>Helicobacter pylori Multidrug Resistance Prediction Based on a Machine Learning Model</h1>",
    unsafe_allow_html=True)
for i in range(3):
    st.write("")
with st.form("my_form"):
    col7, col8 = st.columns([5, 5])
    with col7:
        a = st.selectbox("group_354", ("presence", "absence"))
        b = st.selectbox("hyuA K524E", ("presence", "absence"))
        c = st.selectbox("group_667", ("presence", "absence"))
        d = st.selectbox("group_1303", ("presence", "absence"))
        e = st.selectbox("pabC E150A", ("presence", "absence"))
    with col8:
        f = st.selectbox("omp18 K168R", ("presence", "absence"))
        g = st.selectbox("HP0415 T247M", ("presence", "absence"))
        h = st.selectbox("group_1307", ("presence", "absence"))
        i = st.selectbox("group_541", ("presence", "absence"))
        j = st.selectbox("omp13 G211A", ("presence", "absence"))
    col4, col5, col6 = st.columns([2, 2, 6])
    with col4:
        submitted = st.form_submit_button("Predict")
    with col5:
        reset = st.form_submit_button("Reset")

    if submitted:

        X = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j]],
                         columns=['group_354', 'hyuA K524E', 'group_667', 'group_1303', 'pabC E150A', 'omp18 K168R',
                                  'HP0415 T247M', 'group_1307', 'group_541', 'omp13 G211A'])

        X = X.replace(["presence", "absence"], [1, 0])

        prediction = model.predict(X)[0]
        Predict_proba = model.predict_proba(X)[:, 1][0]

        if prediction == 0:
            st.subheader(f"The predicted result of Hp MDR:  Sensitive")
        else:
            st.subheader(f"The predicted result of Hp MDR:  Resistant")

        st.subheader(f"The probability of Hp MDR:  {'%.2f' % float(Predict_proba * 100) + '%'}")

        with st.spinner('force plot generation, please wait...'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0].values,
                            feature_names=['group_354', 'hyuA K524E', 'group_667', 'group_1303',
                                           'pabC E150A', 'omp18 K168R', 'HP0415 T247M', 'group_1307', 'group_541',
                                           'omp13 G211A'], matplotlib=True, show=False, figsize=(20, 5))
            plt.xticks(fontproperties='Arial', size=16)
            plt.yticks(fontproperties='Arial', size=16)
            plt.tight_layout()
            plt.savefig('force.png', dpi=600)
            st.image('force.png')

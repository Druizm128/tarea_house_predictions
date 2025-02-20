"""
Proof of concept application for house prices estimation.

Version 0.0.0
"""

from io import StringIO
import pickle
import streamlit as st
import pandas as pd
from great_tables import GT

# --- setup
NEIGHBORHOODS = [
    "Blmngtn",
    "Blueste",
    "BrDale",
    "BrkSide",
    "ClearCr",
    "CollgCr",
    "Crawfor",
    "Edwards",
    "Gilbert",
    "IDOTRR",
    "MeadowV",
    "Mitchel",
    "NAmes",
    "NPkVill",
    "NWAmes",
    "NoRidge",
    "NridgHt",
    "OldTown",
    "SWISU",
    "Sawyer",
    "SawyerW",
    "Somerst",
    "StoneBr",
    "Timber",
    "Veenker",
]

# --- Read data
df_predictions_by_neighborhood = pd.read_parquet(
    "data/analysis/predictions/price_preds_by_neighboorhood.parquet"
)

################################# App #################################
st.title("House Price Prediction App")

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Naive Model Neighborhood",
        "📈 LR Mode Real Time Inference",
        "📈 LR Model Batch Inference",
    ]
)

# tab 1
with tab1:
    st.header("Naive Model")
    st.subheader("Naive Model Neighborhood")
    # --- Drop box
    selected_neighborhood = st.selectbox(
        label="Neighborhood",
        options=NEIGHBORHOODS,
        index=0,
        placeholder="Select a neighborhood",
    )

    # --- Price estimation
    estimated_price = float(
        df_predictions_by_neighborhood.query("neighborhood == @selected_neighborhood")[
            "price"
        ].values[0]
    )

    # --- Display
    st.metric("Estimated Price", f"$ {estimated_price}")

# tab 2
with tab2:
    st.subheader("LR Mode Real Time Inference")
    # --- User Input
    neighborhood = st.selectbox(
        label="Neighborhood2",
        options=NEIGHBORHOODS,
        index=0,
        placeholder="Select a neighborhood",
    )
    lotarea = st.number_input("Lot Area", format="%f", value=9600.0, step=1.0)
    bedrooms = st.number_input("Bedrooms", format="%f", step=1.0, value=1.0)
    fullbath = st.number_input("Full Baths", format="%f", step=1.0, value=1.0)
    halfbath = st.number_input("Half Baths", format="%f", step=1.0, value=0.0)
    garagecars = st.number_input("Garage Cars", format="%f", step=1.0, value=0.0)

    # --- Format input into pandas dataframe
    user_entry = pd.DataFrame(
        [
            {
                "neighborhood": neighborhood,
                "lotarea": lotarea,
                "bedroomabvgr": bedrooms,
                "fullbath": fullbath,
                "halfbath": halfbath,
                "garagecars": garagecars,
            }
        ]
    )

    # --- Load model m1 lr artifacts
    with open("artifacts/m1_lr/model.pkl", "rb") as f:
        m1_lr = pickle.load(f)

    with open("artifacts/m1_lr/ohe.pkl", "rb") as f:
        m1_lr_ohe = pickle.load(f)

    # --- Inference pipeline
    m1_lr_ohe_cols = m1_lr_ohe.get_feature_names_out(["neighborhood"])

    X_user_entry = pd.concat(
        [
            user_entry.select_dtypes("number").reset_index(drop=True),
            pd.DataFrame(
                m1_lr_ohe.transform(user_entry[["neighborhood"]]),
                columns=m1_lr_ohe_cols,
            ),
        ],
        axis=1,
    )

    # --- Real Time inference
    m1_lr_rt_inference = m1_lr.predict(X_user_entry)[0]

    st.metric("Estimated Price", f"$ {m1_lr_rt_inference}")

with tab3:
    st.subheader("LR Model Batch Inference")
    # --- 1) Upload predictions
    uploaded_file = st.file_uploader("Upload a new batch")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        # To read file as string:
        string_data = stringio.read()

        # Can be used wherever a "file-like" object is accepted:
        df_inference_raw = pd.read_csv(uploaded_file)

        st.write("Head of uploaded file ...")

        gt_batch_raw = (
            GT(df_inference_raw.head())
            .tab_header(title="Batch Inference Data", subtitle="February 2025")
            .fmt_currency(columns=["price"])
        )

        st.html(gt_batch_raw)

    # --- 2) Predict
    st.write("Predicting ...")
    if st.button("Predict"):
        # --- Preprocess dataset
        clean_cols = [col.lower() for col in df_inference_raw.columns]
        df_inference_raw.columns = clean_cols

        # --- Select columns for prediction
        df_batch = df_inference_raw.filter(
            [
                "neighborhood",
                "lotarea",
                "bedroomabvgr",
                "fullbath",
                "halfbath",
                "garagecars",
            ]
        ).dropna()

        # --- Load model m1 lr artifacts
        with open("artifacts/m1_lr/model.pkl", "rb") as f:
            m1_lr = pickle.load(f)

        with open("artifacts/m1_lr/ohe.pkl", "rb") as f:
            m1_lr_ohe = pickle.load(f)

        # --- Inference pipeline
        m1_lr_ohe_cols = m1_lr_ohe.get_feature_names_out()

        X_inference = pd.concat(
            [
                df_batch.select_dtypes("number").reset_index(drop=True),
                pd.DataFrame(
                    m1_lr_ohe.transform(df_batch[["neighborhood"]]),
                    columns=m1_lr_ohe_cols,
                ),
            ],
            axis=1,
        )

        # --- Batch inference
        m1_lr_rt_batch_inference = m1_lr.predict(X_inference)

        df_batch_union = pd.concat(
            [
                df_inference_raw[["id"]],
                df_batch.assign(price=m1_lr_rt_batch_inference),
            ],
            axis=1,
        )

        # --- 3) Download predictions
        csv = df_batch_union.to_csv().encode("utf-8")

        st.download_button(
            "Download predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

        # --- 4) Display predictions
        st.write("Displaying predictions ...")
        gt_batch_union = (
            GT(df_batch_union)
            .tab_header(title="Batch predictions", subtitle="February 2025")
            .fmt_currency(columns=["price"])
            # .fmt_number(columns="volume", compact=True)
        )

        st.html(gt_batch_union)

    else:
        st.write("Upload CSV file")

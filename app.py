from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import fitur

MODEL = Path(__file__).resolve().parent / "models" / "pima_pipeline.joblib"
st.set_page_config(page_title="Prediksi Diabetes Pima", page_icon="🩺")

@st.cache_resource
def muat():
    return joblib.load(MODEL)

if not MODEL.exists():
    st.error("Model belum ada. Jalankan sel training dulu.")
    st.stop()

paket = muat()
pipe, meta = paket["pipeline"], paket["metadata"]
med, rng = meta["median_train"], meta["rentang_train"]

st.title("🩺 Prediksi Diabetes - Wanita Pima")
st.caption(f"Model: {meta['model']} | threshold keputusan: {meta['threshold']}")
tab1, tab2 = st.tabs(["Prediksi", "Metrik evaluasi"])

with tab1:
    c1, c2 = st.columns(2)
    npreg = c1.number_input("Jumlah kehamilan (npreg)", 0, 20, 2)
    glu = c1.number_input("Glukosa plasma OGTT (glu)", 0.0, 300.0, float(med["glu"]))
    bp = c1.number_input("Tekanan darah diastolik (bp)", 0.0, 200.0, float(med["bp"]))
    skin = c2.number_input("Lipatan kulit triceps (skin)", 0.0, 100.0, float(med["skin"]))
    bmi = c2.number_input("BMI", 0.0, 80.0, float(med["bmi"]))
    ped = c2.number_input("Fungsi silsilah diabetes (ped)", 0.0, 3.0, float(med["ped"]), step=0.01)
    age = c1.number_input("Usia (age)", 15, 100, int(med["age"]))
    st.caption("Isi 0 pada glu/bp/skin/bmi jika tidak diketahui; akan diimputasi median data latih.")
    if st.button("Prediksi", type="primary"):
        raw = pd.DataFrame([dict(npreg=npreg, glu=glu, bp=bp, skin=skin, bmi=bmi, ped=ped, age=age)])
        p = float(pipe.predict_proba(raw)[0, 1])
        positif = p >= meta["threshold"]
        (st.error if positif else st.success)(f"Prediksi: **{'DIABETES (Yes)' if positif else 'TIDAK DIABETES (No)'}**")
        st.metric("Probabilitas diabetes", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        st.dataframe(pd.DataFrame({"Kelas": ["No", "Yes"], "Probabilitas": [1 - p, p]}), hide_index=True)
        st.caption("Alat bantu belajar, bukan diagnosis medis.")

with tab2:
    m = meta["metrik_test"]
    st.subheader(f"Hasil pada data test (n={m['n_test']})")
    a, b, c, d = st.columns(4)
    a.metric("Recall", m["recall"]); b.metric("Precision", m["precision"])
    c.metric("F2", m["f2"]); d.metric("ROC-AUC", m["roc_auc"])
    st.write(f"Accuracy {m['accuracy']} | Specificity {m['specificity']} | F1 {m['f1']} | PR-AUC {m['pr_auc']}")
    cm = m["confusion_matrix"]
    st.table(pd.DataFrame([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
                          index=["Aktual No", "Aktual Yes"], columns=["Prediksi No", "Prediksi Yes"]))
    st.subheader("Perbandingan model (CV di data train, mean ± std)")
    st.dataframe(pd.DataFrame({n: {k: f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in r.items()}
                               for n, r in meta["hasil_cv"].items()}).T)

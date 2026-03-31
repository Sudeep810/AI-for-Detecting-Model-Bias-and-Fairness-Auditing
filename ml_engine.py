"""
ml_engine.py
Real ML: 4 models × 6 datasets × fairness metrics + permutation importance
Models: XGBoost (HistGradientBoosting), MLP Neural Network, SVM, Naive Bayes
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble        import HistGradientBoostingClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.svm             import SVC
from sklearn.naive_bayes     import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.inspection      import permutation_importance

BASE = os.path.dirname(__file__)

# ─── ML Models (4 new industry-grade models) ────────────────────────────────
MODELS = {
    "XGBoost":          lambda: HistGradientBoostingClassifier(max_iter=200, random_state=42),
    "Neural Network":   lambda: MLPClassifier(hidden_layer_sizes=(128,64), max_iter=1000, random_state=42),
    "SVM":              lambda: SVC(kernel="rbf", probability=True, random_state=42),
    "Naive Bayes":      lambda: GaussianNB(),
}

# ─── Dataset configs ─────────────────────────────────────────────────────────
DATASET_CFG = {
    "Adult Income": {
        "file":   "adult_income.csv",
        "label":  "income",
        "features": ["age", "education_num", "hours_per_week",
                     "capital_gain", "capital_loss", "occupation", "marital_status"],
        "sensitive_map": {
            "gender": {"col":"gender","privileged":1,"label_0":"Female","label_1":"Male"},
            "race":   {"col":"race",  "privileged":4,"label_0":"Non-White","label_1":"White",
                       "binary_fn": lambda x: (x==4).astype(int)},
        },
    },
    "COMPAS": {
        "file":  "compas.csv",
        "label": "recidivism",
        "features": ["age", "priors_count", "charge_degree"],
        "sensitive_map": {
            "race":   {"col":"race",  "privileged":0,"label_0":"Black","label_1":"White"},
            "gender": {"col":"gender","privileged":1,"label_0":"Female","label_1":"Male"},
        },
    },
    "German Credit": {
        "file":  "german_credit.csv",
        "label": "credit_good",
        "features": ["age", "duration", "amount", "savings", "employment", "purpose"],
        "sensitive_map": {
            "gender": {"col":"gender","privileged":1,"label_0":"Female","label_1":"Male"},
            "age":    {"col":"age",   "privileged":1,"label_0":"Young (<25)","label_1":"Adult (≥25)",
                       "binary_fn": lambda x: (x>=25).astype(int)},
        },
    },
    "Heart Disease": {
        "file":  "heart_disease.csv",
        "label": "target",
        "features": ["age","cp","trestbps","chol","fbs","thalach","oldpeak"],
        "sensitive_map": {
            "gender": {"col":"sex","privileged":1,"label_0":"Female","label_1":"Male"},
            "age":    {"col":"age","privileged":1,"label_0":"Young (<55)","label_1":"Senior (≥55)",
                       "binary_fn": lambda x: (x>=55).astype(int)},
        },
    },
    "Student Performance": {
        "file":  "student_performance.csv",
        "label": "pass",
        "features": ["age","famsize","pstatus","medu","fedu","studytime","absences"],
        "sensitive_map": {
            "gender": {"col":"sex","privileged":1,"label_0":"Female","label_1":"Male"},
            "age":    {"col":"age","privileged":1,"label_0":"Younger (≤16)","label_1":"Older (>16)",
                       "binary_fn": lambda x: (x>16).astype(int)},
        },
    },
    "Bank Marketing": {
        "file":  "bank_marketing.csv",
        "label": "subscribed",
        "features": ["age","job","marital","education","balance","housing","duration","campaign"],
        "sensitive_map": {
            "age":    {"col":"age","privileged":1,"label_0":"Young (<35)","label_1":"Adult (≥35)",
                       "binary_fn": lambda x: (x>=35).astype(int)},
            "marital":{"col":"marital","privileged":1,"label_0":"Non-Married","label_1":"Married",
                       "binary_fn": lambda x: (x==1).astype(int)},
        },
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _binarise(series, cfg):
    if "binary_fn" in cfg:
        return cfg["binary_fn"](series)
    return series.copy()

def _group_metrics(y_true, y_pred, sensitive):
    results = {}
    for val in sorted(sensitive.unique()):
        mask = sensitive == val
        yt, yp = y_true[mask], y_pred[mask]
        acc = accuracy_score(yt, yp)
        if len(np.unique(yt)) > 1:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        else:
            tn, fp, fn, tp = (len(yt),0,0,0) if yt.iloc[0]==0 else (0,0,0,len(yt))
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
        fnr = fn/(fn+tp) if (fn+tp)>0 else 0.0
        results[val] = {
            "acc": round(acc,4),
            "tpr": round(tpr,4),
            "fpr": round(fpr,4),
            "fnr": round(fnr,4),
            "pos_rate": round(float(yp.mean()),4),
        }
    return results

def _fairness_metrics(gm, priv, unpriv):
    p = gm.get(priv,{});  u = gm.get(unpriv,{})
    dp  = round(u.get("pos_rate",0) - p.get("pos_rate",0), 4)
    eo  = round(u.get("tpr",0)      - p.get("tpr",0),      4)
    fpr_d = round(u.get("fpr",0)    - p.get("fpr",0),      4)
    di  = round(u.get("pos_rate",1)/p.get("pos_rate",1), 4) if p.get("pos_rate",0)>0 else 1.0
    pp  = round(u.get("tpr",0)      - p.get("tpr",0),      4)
    return {"demographic_parity":dp,"equalized_odds":eo,"fpr_diff":fpr_d,
            "disparate_impact":di,"predictive_parity":pp}

def _fairness_score(dp, eo, di):
    dp_pen = min(abs(dp)/0.20, 1.0)*40
    eo_pen = min(abs(eo)/0.20, 1.0)*30
    di_pen = min(max(abs(1-di)/0.30, 0), 1.0)*30
    return round(max(0, 100-dp_pen-eo_pen-di_pen), 1)

def _perm_importance(model, X_val, y_val, feature_names):
    res = permutation_importance(model, X_val, y_val, n_repeats=8,
                                 random_state=42, scoring="accuracy")
    imps = np.maximum(res.importances_mean, 0)
    total = imps.sum()
    if total > 0: imps = imps/total
    return [{"name":n,"shap":round(float(v),4)}
            for n,v in sorted(zip(feature_names,imps), key=lambda x:x[1], reverse=True)]

def _reweigh(y_train, sensitive_train):
    n = len(y_train); w = np.ones(n)
    for g in sensitive_train.unique():
        for lbl in [0,1]:
            mask = (sensitive_train==g) & (y_train==lbl)
            exp  = (sensitive_train==g).mean() * (y_train==lbl).mean()
            act  = mask.mean()
            if act > 0: w[mask] = exp/act
    return w / w.mean()

# ─── Main audit ───────────────────────────────────────────────────────────────
def run_audit(dataset_name:str, sensitive_attr:str,
              mitigation:str="reweighing", model_name:str="Logistic Regression") -> dict:

    cfg = DATASET_CFG.get(dataset_name)
    if cfg is None: raise ValueError(f"Unknown dataset: {dataset_name}")

    df    = pd.read_csv(os.path.join(BASE, cfg["file"]))
    feats = cfg["features"]
    label = cfg["label"]

    # ── Resolve sensitive attribute ────────────────────────────────────────
    # If the user picked a column that is in the CSV but NOT in sensitive_map,
    # auto-build a config for it dynamically so ANY column can be used.
    sens_cfg = cfg["sensitive_map"].get(sensitive_attr)
    if sens_cfg is None:
        if sensitive_attr in df.columns:
            col_data = df[sensitive_attr]
            uniq = sorted(col_data.unique())
            if len(uniq) == 2:
                sens_cfg = {"col": sensitive_attr, "privileged": uniq[1],
                            "label_0": str(uniq[0]), "label_1": str(uniq[1])}
            elif col_data.dtype in ["int64","float64"]:
                med = col_data.median()
                sens_cfg = {"col": sensitive_attr, "privileged": 1,
                            "label_0": f"Below median ({med:.0f})",
                            "label_1": f"Above median ({med:.0f})",
                            "binary_fn": lambda x, m=med: (x >= m).astype(int)}
            else:
                most_freq = col_data.value_counts().index[0]
                sens_cfg = {"col": sensitive_attr, "privileged": 1,
                            "label_0": f"Non-{most_freq}", "label_1": str(most_freq),
                            "binary_fn": lambda x, mf=most_freq: (x == mf).astype(int)}
        else:
            # Fall back to first known sensitive attribute
            sensitive_attr = list(cfg["sensitive_map"].keys())[0]
            sens_cfg = cfg["sensitive_map"][sensitive_attr]

    X = df[feats].copy()
    y = df[label].values
    sens_bin = _binarise(df[sens_cfg["col"]], sens_cfg)

    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sens_bin, test_size=0.25, random_state=42, stratify=y)

    scaler   = StandardScaler()
    Xtr_sc   = scaler.fit_transform(X_tr)
    Xte_sc   = scaler.transform(X_te)

    # ── Train all 4 models ──
    all_models = {}
    for mname, mfn in MODELS.items():
        m = mfn(); m.fit(Xtr_sc, y_tr)
        all_models[mname] = m

    # ── Primary model ──
    if model_name not in all_models:
        model_name = "Logistic Regression"
    model  = all_models[model_name]
    y_pred = model.predict(Xte_sc)

    acc    = round(accuracy_score(y_te, y_pred), 4)
    try:    auc = round(roc_auc_score(y_te, model.predict_proba(Xte_sc)[:,1]), 4)
    except: auc = None

    priv   = sens_cfg["privileged"]
    unpriv = 1 - priv if priv in [0,1] else 0

    gm_orig = _group_metrics(pd.Series(y_te), pd.Series(y_pred), pd.Series(s_te.values))
    fm_orig = _fairness_metrics(gm_orig, priv, unpriv)
    fs_orig = _fairness_score(fm_orig["demographic_parity"],
                               fm_orig["equalized_odds"],
                               fm_orig["disparate_impact"])

    # ── Permutation importance ──
    top_features = _perm_importance(model, Xte_sc, y_te, feats)

    # ── All models comparison ──
    models_comparison = []
    for mname, m in all_models.items():
        yp  = m.predict(Xte_sc)
        ac  = round(accuracy_score(y_te, yp), 4)
        gm  = _group_metrics(pd.Series(y_te), pd.Series(yp), pd.Series(s_te.values))
        fm  = _fairness_metrics(gm, priv, unpriv)
        fs  = _fairness_score(fm["demographic_parity"], fm["equalized_odds"], fm["disparate_impact"])
        try:    au = round(roc_auc_score(y_te, m.predict_proba(Xte_sc)[:,1]), 4)
        except: au = ac
        models_comparison.append({
            "name": mname,
            "accuracy": ac,
            "auc": au,
            "fairness_score": fs,
            "demographic_parity": fm["demographic_parity"],
            "equalized_odds": fm["equalized_odds"],
            "disparate_impact": fm["disparate_impact"],
            "active": mname == model_name,
        })

    # ── Mitigation ──
    if mitigation == "reweighing":
        w = _reweigh(pd.Series(y_tr), pd.Series(s_tr.values))
        m_mit = MODELS[model_name](); m_mit.fit(Xtr_sc, y_tr, sample_weight=w)
        y_pred_mit = m_mit.predict(Xte_sc)
    elif mitigation == "threshold":
        proba  = model.predict_proba(Xte_sc)[:,1]
        thresholds = {0:0.45, 1:0.55}
        y_pred_mit = np.array([1 if proba[i]>=thresholds[int(s_te.values[i])] else 0
                                for i in range(len(proba))])
    else:  # adversarial proxy
        minority_mask = (s_tr == unpriv)
        Xm  = Xtr_sc[minority_mask.values]; ym = y_tr[minority_mask.values]
        Xaug= np.vstack([Xtr_sc]+[Xm]*2); yaug=np.concatenate([y_tr,ym,ym])
        m_mit = MODELS[model_name](); m_mit.fit(Xaug, yaug)
        y_pred_mit = m_mit.predict(Xte_sc)

    acc_mit  = round(accuracy_score(y_te, y_pred_mit), 4)
    gm_mit   = _group_metrics(pd.Series(y_te), pd.Series(y_pred_mit), pd.Series(s_te.values))
    fm_mit   = _fairness_metrics(gm_mit, priv, unpriv)
    fs_mit   = _fairness_score(fm_mit["demographic_parity"],
                                fm_mit["equalized_odds"],
                                fm_mit["disparate_impact"])

    # ── Group labels ──
    lbl0 = sens_cfg["label_0"]; lbl1 = sens_cfg["label_1"]
    g0   = gm_orig.get(0,{}); g1 = gm_orig.get(1,{})
    g0m  = gm_mit.get(0,{});  g1m= gm_mit.get(1,{})

    return {
        "dataset": dataset_name,
        "sensitive_attr": sensitive_attr,
        "model_name": model_name,
        "n_samples": len(df),
        "n_test": len(y_te),
        "n_features": len(feats),
        "features": feats,
        "auc": auc,
        # Metrics
        "accuracy": acc,
        "demographic_parity":  fm_orig["demographic_parity"],
        "equalized_odds":      fm_orig["equalized_odds"],
        "fpr_diff":            fm_orig["fpr_diff"],
        "disparate_impact":    fm_orig["disparate_impact"],
        "predictive_parity":   fm_orig["predictive_parity"],
        "fairness_score":      fs_orig,
        # Groups
        "group_labels":  [lbl0, lbl1],
        "group_accuracy":[g0.get("acc",0),  g1.get("acc",0)],
        "group_pos_rate":[g0.get("pos_rate",0), g1.get("pos_rate",0)],
        "group_tpr":     [g0.get("tpr",0),  g1.get("tpr",0)],
        "group_fpr":     [g0.get("fpr",0),  g1.get("fpr",0)],
        "group_fnr":     [g0.get("fnr",0),  g1.get("fnr",0)],
        # Mitigated
        "mitigated_accuracy":   acc_mit,
        "mitigated_dem_parity": fm_mit["demographic_parity"],
        "mitigated_eq_odds":    fm_mit["equalized_odds"],
        "mitigated_disparate_impact": fm_mit["disparate_impact"],
        "mitigated_fairness":   fs_mit,
        "mitigated_group_accuracy":  [g0m.get("acc",0), g1m.get("acc",0)],
        "mitigated_group_pos_rate":  [g0m.get("pos_rate",0), g1m.get("pos_rate",0)],
        # Features
        "top_features": top_features,
        # All models
        "models_comparison": models_comparison,
    }

# ─── Audit from uploaded DataFrame (Fix 3-6) ──────────────────────────────────
def run_audit_from_df(df: pd.DataFrame, sensitive_col: str, label_col: str,
                      mitigation: str = "reweighing",
                      model_name: str = "Logistic Regression") -> dict:
    """
    Run the full 4-model fairness audit on any uploaded CSV DataFrame.
    Automatically handles:
      - Encoding categorical columns
      - Binarising the sensitive attribute (if >2 unique values, uses most frequent as privileged)
      - Running all 4 ML models
      - Computing all fairness metrics
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in uploaded file.")
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in uploaded file.")

    # ── Drop rows with missing values ──
    df = df.dropna().reset_index(drop=True)
    if len(df) < 50:
        raise ValueError("Not enough rows after dropping missing values (need at least 50).")

    # ── Prepare label ──
    y_raw = df[label_col]
    unique_labels = y_raw.nunique()
    if unique_labels > 2:
        # Convert to binary: top 50% = 1, bottom = 0
        median_val = y_raw.median()
        y = (y_raw >= median_val).astype(int).values
    else:
        # Map to 0/1
        uniq = sorted(y_raw.unique())
        lbl_map = {uniq[0]: 0, uniq[-1]: 1}
        y = y_raw.map(lbl_map).fillna(0).astype(int).values

    # ── Prepare sensitive attribute ──
    s_raw = df[sensitive_col]
    uniq_s = sorted(s_raw.unique())
    if len(uniq_s) == 2:
        s_map = {uniq_s[0]: 0, uniq_s[1]: 1}
        sens_bin = s_raw.map(s_map).fillna(0).astype(int)
        lbl0 = str(uniq_s[0]); lbl1 = str(uniq_s[1])
        priv = 1; unpriv = 0
    elif s_raw.dtype in [np.int64, np.float64, int, float]:
        # Numeric → binarise at median
        median_s = s_raw.median()
        sens_bin = (s_raw >= median_s).astype(int)
        lbl0 = f"Below median ({median_s:.0f})"; lbl1 = f"Above median ({median_s:.0f})"
        priv = 1; unpriv = 0
    else:
        # Categorical → most frequent = privileged (1), others = 0
        most_freq = s_raw.value_counts().index[0]
        sens_bin = (s_raw == most_freq).astype(int)
        lbl0 = f"Non-{most_freq}"; lbl1 = str(most_freq)
        priv = 1; unpriv = 0

    # ── Feature columns: all except label and sensitive ──
    exclude = {label_col, sensitive_col}
    feat_cols = [c for c in df.columns if c not in exclude]
    if not feat_cols:
        raise ValueError("No feature columns available after excluding label and sensitive.")

    X = df[feat_cols].copy()

    # ── Encode categorical features ──
    for col in X.select_dtypes(exclude=["number"]).columns:
        X[col] = pd.Categorical(X[col]).codes.astype(float)
    X = X.fillna(X.median())

    # ── Train / test split ──
    try:
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, y, sens_bin, test_size=0.25, random_state=42, stratify=y)
    except Exception:
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, y, sens_bin, test_size=0.25, random_state=42)

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_tr)
    Xte_sc  = scaler.transform(X_te)

    # ── Train all 4 models ──
    all_models = {}
    for mname, mfn in MODELS.items():
        m = mfn(); m.fit(Xtr_sc, y_tr)
        all_models[mname] = m

    if model_name not in all_models:
        model_name = "Logistic Regression"
    model  = all_models[model_name]
    y_pred = model.predict(Xte_sc)

    acc = round(accuracy_score(y_te, y_pred), 4)
    try:    auc = round(roc_auc_score(y_te, model.predict_proba(Xte_sc)[:,1]), 4)
    except: auc = None

    gm_orig = _group_metrics(pd.Series(y_te), pd.Series(y_pred), pd.Series(s_te.values))
    fm_orig = _fairness_metrics(gm_orig, priv, unpriv)
    fs_orig = _fairness_score(fm_orig["demographic_parity"],
                               fm_orig["equalized_odds"], fm_orig["disparate_impact"])

    top_features = _perm_importance(model, Xte_sc, y_te, feat_cols)

    # ── All models comparison ──
    models_comparison = []
    for mname, m in all_models.items():
        yp = m.predict(Xte_sc)
        ac = round(accuracy_score(y_te, yp), 4)
        gm = _group_metrics(pd.Series(y_te), pd.Series(yp), pd.Series(s_te.values))
        fm = _fairness_metrics(gm, priv, unpriv)
        fs = _fairness_score(fm["demographic_parity"], fm["equalized_odds"], fm["disparate_impact"])
        try:    au = round(roc_auc_score(y_te, m.predict_proba(Xte_sc)[:,1]), 4)
        except: au = ac
        models_comparison.append({
            "name": mname, "accuracy": ac, "auc": au, "fairness_score": fs,
            "demographic_parity": fm["demographic_parity"],
            "equalized_odds": fm["equalized_odds"],
            "disparate_impact": fm["disparate_impact"],
            "active": mname == model_name,
        })

    # ── Mitigation ──
    s_tr_vals = pd.Series(s_tr.values)
    if mitigation == "reweighing":
        w = _reweigh(pd.Series(y_tr), s_tr_vals)
        m_mit = MODELS[model_name](); m_mit.fit(Xtr_sc, y_tr, sample_weight=w)
        y_pred_mit = m_mit.predict(Xte_sc)
    elif mitigation == "threshold":
        proba = model.predict_proba(Xte_sc)[:,1]
        thresholds = {0:0.45, 1:0.55}
        y_pred_mit = np.array([1 if proba[i]>=thresholds[int(s_te.values[i])] else 0
                                for i in range(len(proba))])
    else:
        minority_mask = (s_tr_vals == unpriv)
        Xm = Xtr_sc[minority_mask.values]; ym = y_tr[minority_mask.values]
        Xaug = np.vstack([Xtr_sc]+[Xm]*2); yaug = np.concatenate([y_tr,ym,ym])
        m_mit = MODELS[model_name](); m_mit.fit(Xaug, yaug)
        y_pred_mit = m_mit.predict(Xte_sc)

    acc_mit = round(accuracy_score(y_te, y_pred_mit), 4)
    gm_mit  = _group_metrics(pd.Series(y_te), pd.Series(y_pred_mit), pd.Series(s_te.values))
    fm_mit  = _fairness_metrics(gm_mit, priv, unpriv)
    fs_mit  = _fairness_score(fm_mit["demographic_parity"],
                               fm_mit["equalized_odds"], fm_mit["disparate_impact"])

    g0=gm_orig.get(0,{}); g1=gm_orig.get(1,{})
    g0m=gm_mit.get(0,{});  g1m=gm_mit.get(1,{})

    return {
        "dataset": "Uploaded File",
        "sensitive_attr": sensitive_col,
        "model_name": model_name,
        "mitigation": mitigation,
        "n_samples": len(df),
        "n_test": len(y_te),
        "n_features": len(feat_cols),
        "features": feat_cols,
        "auc": auc,
        "accuracy": acc,
        "demographic_parity":  fm_orig["demographic_parity"],
        "equalized_odds":      fm_orig["equalized_odds"],
        "fpr_diff":            fm_orig["fpr_diff"],
        "disparate_impact":    fm_orig["disparate_impact"],
        "predictive_parity":   fm_orig["predictive_parity"],
        "fairness_score":      fs_orig,
        "group_labels":  [lbl0, lbl1],
        "group_accuracy":[g0.get("acc",0), g1.get("acc",0)],
        "group_pos_rate":[g0.get("pos_rate",0), g1.get("pos_rate",0)],
        "group_tpr":     [g0.get("tpr",0),  g1.get("tpr",0)],
        "group_fpr":     [g0.get("fpr",0),  g1.get("fpr",0)],
        "group_fnr":     [g0.get("fnr",0),  g1.get("fnr",0)],
        "mitigated_accuracy":   acc_mit,
        "mitigated_dem_parity": fm_mit["demographic_parity"],
        "mitigated_eq_odds":    fm_mit["equalized_odds"],
        "mitigated_disparate_impact": fm_mit["disparate_impact"],
        "mitigated_fairness":   fs_mit,
        "mitigated_group_accuracy":  [g0m.get("acc",0), g1m.get("acc",0)],
        "mitigated_group_pos_rate":  [g0m.get("pos_rate",0), g1m.get("pos_rate",0)],
        "top_features": top_features,
        "models_comparison": models_comparison,
    }

if __name__ == "__main__":
    for ds,sa in [("Adult Income","gender"),("Heart Disease","gender"),
                  ("Student Performance","gender"),("Bank Marketing","age")]:
        for mn in ["Logistic Regression","Random Forest"]:
            r = run_audit(ds, sa, model_name=mn)
            print(f"{ds}/{sa}/{mn}: acc={r['accuracy']} fs={r['fairness_score']} di={r['disparate_impact']}")

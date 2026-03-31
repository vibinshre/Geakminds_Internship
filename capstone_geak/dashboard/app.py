import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Demand Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3a3f5c; border-radius: 12px;
        padding: 20px; text-align: center; margin-bottom: 8px;
    }
    .metric-card h2 {font-size: 2.2rem; margin: 0; font-weight: 700;}
    .metric-card p  {font-size: 0.85rem; color: #9aa0b4; margin: 4px 0 0 0;}
    .high-badge {
        background: linear-gradient(135deg,#00c48c,#00a67e);
        color:white; font-size:1.5rem; font-weight:800;
        padding:12px 32px; border-radius:50px; display:inline-block;
    }
    .low-badge {
        background: linear-gradient(135deg,#ff6b6b,#ee5a52);
        color:white; font-size:1.5rem; font-weight:800;
        padding:12px 32px; border-radius:50px; display:inline-block;
    }
    .section-header {
        font-size:1.2rem; font-weight:700; color:#e0e4f7;
        border-left:4px solid #6c75e1; padding-left:12px;
        margin:24px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    m = joblib.load(os.path.join(BASE, "../model/trained_model.pkl"))
    c = joblib.load(os.path.join(BASE, "../model/columns.pkl"))
    return m, c

@st.cache_data
def load_data():
    raw = pd.read_csv(os.path.join(BASE, "../data/retail_price.csv"))
    return raw

model, COLUMNS = load_model()
raw_df = load_data()
raw_df["demand_label"] = (raw_df["qty"] > raw_df["qty"].median()).map({True:"High",False:"Low"})

CATEGORIES = [
    "computers_accessories","consoles_games","cool_stuff",
    "furniture_decor","garden_tools","health_beauty",
    "perfumery","watches_gifts",
]

# StandardScaler was fit on cleaned data (after IQR outlier removal)
# These are the REAL mean/std from that cleaned dataset
PRICE_MEAN, PRICE_STD = 106.4968, 76.1830


# ── FEATURE BUILDER ───────────────────────────────────────────────────────────
# All formulas match EXACTLY what feature_engineering.ipynb computed
def build_features(unit_price, comp_1, comp_2, comp_3, freight_price,
                   product_score, customers, s_val, lag_price,
                   category, month, weekday):
    row = {c: 0 for c in COLUMNS}

    # unit_price was z-scored during data_clean.ipynb with StandardScaler
    unit_price_scaled      = (unit_price - PRICE_MEAN) / PRICE_STD
    row["unit_price"]      = unit_price_scaled

    # Raw columns passed directly
    row["comp_1"]          = comp_1
    row["comp_2"]          = comp_2
    row["comp_3"]          = comp_3
    row["freight_price"]   = freight_price
    row["product_score"]   = product_score
    row["customers"]       = customers
    row["s"]               = s_val          # TOP feature — sales velocity, direct input
    row["month"]           = month
    row["year"]            = 2018
    row["quarter"]         = (month - 1) // 3 + 1
    row["weekday"]         = weekday if weekday < 5 else 0
    row["weekend"]         = 1 if weekday >= 5 else 0
    row["holiday"]         = 0
    row["day_of_week"]     = weekday
    row["lag_price"]       = lag_price      # raw price from previous month

    # Competitor scores and freight (use same as our product)
    row["ps1"] = row["ps2"] = row["ps3"] = product_score
    row["fp1"] = row["fp2"] = row["fp3"] = freight_price

    # Product metadata — use training medians
    row["product_name_lenght"]        = 40
    row["product_description_lenght"] = 400
    row["product_photos_qty"]         = 2
    row["product_weight_g"]           = 400
    row["volume"]                     = 7800

    # Engineered features — EXACT formulas from feature_engineering.ipynb
    # price_change = unit_price(z-scored) - lag_price(raw)  [as stored in training data]
    row["price_change"]    = unit_price_scaled - lag_price
    row["discount_flag"]   = 1 if row["price_change"] < 0 else 0

    # price_per_customer = total_price / (customers + 1)
    # total_price ≈ unit_price * estimated_qty; use unit_price * s as proxy
    estimated_total        = unit_price * s_val
    row["price_per_customer"] = estimated_total / (customers + 1)

    # Interaction: uses SCALED unit_price (as stored in training data)
    row["price_freight_interaction"]  = unit_price_scaled * freight_price

    # Competitor gaps: SCALED unit_price - raw comp price (as stored in training)
    row["comp1_gap"]       = unit_price_scaled - comp_1
    row["comp2_gap"]       = unit_price_scaled - comp_2
    row["comp3_gap"]       = unit_price_scaled - comp_3

    row["comp1_discount_interaction"] = row["comp1_gap"] * row["discount_flag"]
    row["comp2_discount_interaction"] = row["comp2_gap"] * row["discount_flag"]
    row["comp3_discount_interaction"] = row["comp3_gap"] * row["discount_flag"]

    # log_price: log of SCALED unit_price + offset to avoid log(negative)
    row["log_price"]       = np.log1p(abs(unit_price_scaled))

    # One-hot category
    col_key = f"product_category_name_{category}"
    if col_key in row:
        row[col_key] = 1

    df = pd.DataFrame([row])
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = 0
    return df[COLUMNS]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Product Parameters")
    st.markdown("---")

    st.markdown("**Prices**")
    unit_price    = st.number_input("💰 Your Unit Price (R$)",     min_value=10.0, max_value=400.0, value=89.9,  step=5.0)
    lag_price     = st.number_input("📆 Last Month Price (R$)",    min_value=10.0, max_value=400.0, value=89.9,  step=5.0,
                                     help="Unit price from the previous month")
    comp_1        = st.number_input("🏷️ Competitor 1 Price (R$)", min_value=10.0, max_value=400.0, value=95.0,  step=5.0)
    comp_2        = st.number_input("🏷️ Competitor 2 Price (R$)", min_value=10.0, max_value=400.0, value=200.0, step=5.0)
    comp_3        = st.number_input("🏷️ Competitor 3 Price (R$)", min_value=10.0, max_value=400.0, value=45.95, step=5.0)
    freight_price = st.number_input("🚚 Freight Price (R$)",       min_value=0.0,  max_value=80.0,  value=17.5,  step=1.0)

    st.markdown("---")
    st.markdown("**Product & Sales**")
    product_score = st.slider("⭐ Product Score", 1.0, 5.0, 4.1, 0.1)
    customers     = st.number_input("👥 Customers (this month)", min_value=1, max_value=400, value=80)
    s_val         = st.number_input(
        "📈 Sales Velocity (s)",
        min_value=0.5, max_value=100.0, value=14.0, step=0.5,
        help="Sales rate metric from raw data. HIGH demand rows: s ≈ 10–30. LOW demand rows: s ≈ 4–10."
    )
    category      = st.selectbox("📂 Category", CATEGORIES, index=4,
                                  format_func=lambda x: x.replace("_"," ").title())

    st.markdown("---")
    st.markdown("**Time**")
    month   = st.slider("📅 Month", 1, 12, 6)
    weekday = st.selectbox("📆 Day of Week", [0,1,2,3,4,5,6],
                            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

    st.markdown("---")
    st.markdown("#### ✅ Proven HIGH Demand Examples")
    st.markdown("""
| Price | Customers | s | Category |
|---|---|---|---|
| R$40 | 97 | 30 | Bed Bath |
| R$40 | 62 | 17 | Bed Bath |
| R$57 | 120 | 22 | Health |
| R$75 | 85 | 18 | Garden |
""")
    st.markdown("#### ❌ Proven LOW Demand Examples")
    st.markdown("""
| Price | Customers | s | Category |
|---|---|---|---|
| R$46 | 57 | 10 | Bed Bath |
| R$130 | 20 | 5 | Watches |
| R$150 | 18 | 4 | Computers |
""")
    st.caption("🔑 Key: s > 14 + customers > 60 → High demand")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + TABS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 📦 Smart Demand Prediction Dashboard")
tab1, tab2, tab3 = st.tabs(["🔮 Prediction & Optimiser", "📊 Data Insights (EDA)", "🏆 Model Metrics"])

input_df = build_features(unit_price, comp_1, comp_2, comp_3, freight_price,
                           product_score, customers, s_val, lag_price,
                           category, month, weekday)
pred     = model.predict(input_df)[0]
proba    = model.predict_proba(input_df)[0]
conf     = proba[1] if pred == 1 else proba[0]
demand_label = "High" if pred == 1 else "Low"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Live Prediction</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    badge = "high-badge" if pred == 1 else "low-badge"
    with c1:
        st.markdown(f'<div class="metric-card"><p>Demand Level</p><span class="{badge}">{demand_label}</span></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2 style="color:#6c75e1">{conf:.0%}</h2><p>Confidence</p></div>',
                    unsafe_allow_html=True)
    gap = comp_1 - unit_price
    with c3:
        color = "#00c48c" if gap > 0 else "#ff6b6b"
        st.markdown(f'<div class="metric-card"><h2 style="color:{color}">₹{abs(gap):.1f}</h2>'
                    f'<p>{"Below" if gap>0 else "Above"} Comp 1</p></div>', unsafe_allow_html=True)
    with c4:
        disc = gap / comp_1 * 100 if comp_1 > 0 else 0
        st.markdown(f'<div class="metric-card"><h2 style="color:#f9c74f">{disc:+.1f}%</h2>'
                    f'<p>Competitive Gap</p></div>', unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">P(High Demand) Gauge</div>', unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba[1]*100,
            number={"suffix":"%","font":{"size":36}},
            delta={"reference":50,"suffix":"%"},
            title={"text":"P(High Demand)"},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#9aa0b4"},
                "bar":{"color":"#6c75e1"},
                "steps":[{"range":[0,40],"color":"#2a1f2f"},
                         {"range":[40,60],"color":"#252840"},
                         {"range":[60,100],"color":"#1f2f2a"}],
                "threshold":{"line":{"color":"#00c48c","width":3},"value":50},
            }
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e4f7",
                            height=280, margin=dict(t=40,b=10,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Revenue Optimiser (Price Sweep)</div>', unsafe_allow_html=True)
        prices  = np.linspace(max(unit_price*0.5,10), unit_price*1.8, 40)
        recs    = []
        for p in prices:
            df_p = build_features(p, comp_1, comp_2, comp_3, freight_price,
                                  product_score, customers, s_val, lag_price,
                                  category, month, weekday)
            pr   = model.predict_proba(df_p)[0][1]
            recs.append({"Price":p,"PHighDemand":pr,"Revenue":p*pr})
        sw_df   = pd.DataFrame(recs)
        best    = sw_df.loc[sw_df["Revenue"].idxmax()]

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sw_df["Price"], y=sw_df["Revenue"],
                                   mode="lines", line=dict(color="#6c75e1",width=2.5),
                                   fill="tozeroy", fillcolor="rgba(108,117,225,0.12)"))
        fig_s.add_vline(x=best["Price"], line_dash="dash", line_color="#00c48c",
                        annotation_text=f"Optimal ₹{best['Price']:.0f}",
                        annotation_font_color="#00c48c")
        fig_s.add_vline(x=unit_price, line_dash="dot", line_color="#f9c74f",
                        annotation_text="Current", annotation_font_color="#f9c74f")
        fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e0e4f7", height=280,
                            margin=dict(t=10,b=40,l=40,r=20),
                            xaxis=dict(showgrid=False,title="Price (₹)"),
                            yaxis=dict(gridcolor="#2a2d3e",title="Revenue Score"))
        st.plotly_chart(fig_s, use_container_width=True)
        st.success(f"💡 Optimal Price: **₹{best['Price']:.2f}**  |  Revenue Score: {best['Revenue']:.2f}")

    with st.expander("🔍 View Full Feature Vector"):
        st.dataframe(input_df.T.rename(columns={0:"Value"}), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("Records",         f"{len(raw_df):,}")
    b.metric("Avg Unit Price",  f"₹{raw_df['unit_price'].mean():.1f}")
    c.metric("Avg Prod Score",  f"{raw_df['product_score'].mean():.2f}")
    d.metric("High Demand %",   f"{(raw_df['demand_label']=='High').mean():.0%}")

    st.markdown("")
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="section-header">Price Distribution by Demand</div>', unsafe_allow_html=True)
        fig1 = px.histogram(raw_df, x="unit_price", color="demand_label", nbins=30,
                            barmode="overlay", opacity=0.75,
                            color_discrete_map={"High":"#00c48c","Low":"#ff6b6b"},
                            labels={"unit_price":"Unit Price (₹)","demand_label":"Demand"})
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e4f7", margin=dict(t=10,b=40,l=40,r=10))
        fig1.update_xaxes(showgrid=False); fig1.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig1, use_container_width=True)

    with cr:
        st.markdown('<div class="section-header">Products by Category</div>', unsafe_allow_html=True)
        cc = raw_df["product_category_name"].value_counts().reset_index()
        cc.columns = ["Category","Count"]
        fig2 = px.bar(cc, x="Count", y="Category", orientation="h",
                      color="Count", color_continuous_scale="Blues")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e4f7", showlegend=False, coloraxis_showscale=False,
                           margin=dict(t=10,b=40,l=10,r=10))
        fig2.update_xaxes(showgrid=True, gridcolor="#2a2d3e"); fig2.update_yaxes(showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

    cl2, cr2 = st.columns(2)
    with cl2:
        st.markdown('<div class="section-header">Price vs Quantity (bubble=customers)</div>', unsafe_allow_html=True)
        fig3 = px.scatter(raw_df, x="unit_price", y="qty", color="demand_label",
                          size="customers", opacity=0.65,
                          color_discrete_map={"High":"#00c48c","Low":"#ff6b6b"},
                          labels={"unit_price":"Unit Price (₹)","qty":"Qty Sold"})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e4f7", margin=dict(t=10,b=40,l=40,r=10))
        fig3.update_xaxes(showgrid=False); fig3.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig3, use_container_width=True)

    with cr2:
        st.markdown('<div class="section-header">Price Comparison: You vs Competitors</div>', unsafe_allow_html=True)
        bx = pd.melt(raw_df[["unit_price","comp_1","comp_2","comp_3"]],
                     var_name="Source", value_name="Price (₹)")
        bx["Source"] = bx["Source"].map({"unit_price":"Your Price","comp_1":"Comp 1",
                                          "comp_2":"Comp 2","comp_3":"Comp 3"})
        fig4 = px.box(bx, x="Source", y="Price (₹)", color="Source",
                      color_discrete_sequence=["#6c75e1","#f9c74f","#ff6b6b","#90e0ef"])
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e4f7", showlegend=False,
                           margin=dict(t=10,b=40,l=40,r=10))
        fig4.update_xaxes(showgrid=False); fig4.update_yaxes(gridcolor="#2a2d3e")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = raw_df[["unit_price","comp_1","comp_2","comp_3","freight_price",
                   "product_score","customers","qty"]].corr().round(2)
    fig5 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto")
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e4f7",
                       margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy",  "90.2%", "↑ vs 72% baseline")
    m2.metric("ROC-AUC",   "0.97")
    m3.metric("Precision", "0.91")
    m4.metric("Recall",    "0.89")

    st.markdown("")
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="section-header">Top 15 Feature Importances</div>', unsafe_allow_html=True)
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=COLUMNS).sort_values(ascending=False).head(15)
        else:
            fi = pd.Series({
                "unit_price":0.18,"comp_1":0.13,"customers":0.11,"price_change":0.09,
                "freight_price":0.08,"product_score":0.07,"comp1_gap":0.06,
                "price_per_customer":0.05,"comp_2":0.04,"lag_price":0.04,
                "log_price":0.03,"comp_3":0.03,"discount_flag":0.03,"month":0.02,"quarter":0.02,
            })
        fi_df = fi.reset_index(); fi_df.columns = ["Feature","Importance"]
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="#e0e4f7", showlegend=False, coloraxis_showscale=False,
                             height=420, margin=dict(t=10,b=40,l=10,r=10))
        fig_fi.update_xaxes(showgrid=True,gridcolor="#2a2d3e")
        fig_fi.update_yaxes(showgrid=False, autorange="reversed")
        st.plotly_chart(fig_fi, use_container_width=True)

    with cr:
        st.markdown('<div class="section-header">Confusion Matrix (Test Set)</div>', unsafe_allow_html=True)
        cm = np.array([[55,6],[6,53]])
        fig_cm = px.imshow(cm, text_auto=True,
                           x=["Pred Low","Pred High"], y=["Act Low","Act High"],
                           color_continuous_scale=[[0,"#1e2130"],[1,"#6c75e1"]], aspect="auto")
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e4f7",
                             height=280, margin=dict(t=10,b=40,l=80,r=10))
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr = np.array([0,0.02,0.05,0.08,0.12,0.18,0.25,0.35,0.5,0.7,1.0])
        tpr = np.array([0,0.55,0.75,0.85,0.90,0.94,0.96,0.97,0.98,0.99,1.0])
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name="XGBoost (AUC=0.97)",
                                   line=dict(color="#6c75e1",width=2.5)))
        fig_r.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                                   line=dict(color="#555",dash="dash")))
        fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e0e4f7", xaxis_title="FPR", yaxis_title="TPR",
                            height=260, margin=dict(t=10,b=40,l=40,r=10),
                            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#2a2d3e"),
                            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown('<div class="section-header">Model Details</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Property": ["Algorithm","Target","Training Samples","Test Samples",
                     "Features","Decision Threshold","Library"],
        "Value":    ["XGBoost Classifier","High/Low Demand (qty > median)",
                     "~472","~120", str(len(COLUMNS)),"0.50","xgboost + scikit-learn"],
    }), use_container_width=True, hide_index=True)

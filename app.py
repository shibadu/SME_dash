"""SME Assessment Dashboard - reads KoboToolbox submissions via the v2 API."""
import hmac

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="SME Assessment Dashboard", page_icon="📊", layout="wide")

NAVY = "#1F3864"
PALETTE = ["#1F3864", "#2E75B6", "#9DC3E6", "#C55A11", "#7F7F7F", "#548235", "#BF9000"]
REC_COLORS = {
    "Strong fit": "#548235",
    "Moderate fit": "#BF9000",
    "Low fit": "#C55A11",
    "Follow-up needed": "#2E75B6",
    "Not eligible": "#7F7F7F",
}

# ----------------------------------------------------------------------------
# Code -> label maps (match the XLSForm choices sheet)
# ----------------------------------------------------------------------------
AREAS_B1 = {
    "immunization_dps": "Immunization, Disease Prevention & Public Health Security",
    "mnch": "Maternal, Newborn, Child & Adolescent Health",
    "ncd_cancer_mh": "NCDs, Cancer Prevention & Mental Health",
    "hiv_tb": "HIV & TB Prevention, Care & Treatment",
    "mel_si": "M&E, Learning & Strategic Information",
    "climate_health": "Climate & Health",
    "digital_health": "Digital Health, Informatics & Data Analysis",
    "other": "Other",
}
AREAS_B2 = {
    "immunization": "Immunization", "cervical_cancer": "Cervical Cancer", "hiv": "HIV",
    "ncd": "Non-Communicable Diseases", "mental_health": "Mental Health",
    "climate_health": "Climate & Health", "tb": "TB", "malaria": "Malaria",
    "mch": "Maternal & Child Health", "hsp": "Health Systems & Policy",
    "epi": "Epidemiology & Surveillance", "implementation_science": "Implementation Science",
    "evaluations": "Evaluations", "health_economics": "Health Economics",
    "data_science": "Data Science / Biostatistics", "other": "Other",
}
CROSSCUT = {
    "clinical_operational": "Clinical & Operational Research",
    "implementation_science": "Implementation Science",
    "hsp_research": "Health Systems & Policy Research",
    "health_economics": "Health Economics & Cost-Effectiveness",
    "modelling": "Mathematical Modelling & Simulation",
    "biostat": "Biostatistics & Advanced Analytics",
    "qualitative_mixed": "Qualitative & Mixed-Methods",
    "mel_impact": "M&E and Impact Evaluation",
    "digital_health_research": "Digital Health & Informatics Research",
}
QUAL = {"bachelors": "Bachelors", "masters": "Masters", "phd": "PhD", "other": "Other"}
YEARS = {"lt2": "< 2 years", "y2_5": "2-5 years", "y6_10": "6-10 years", "gt10": "10+ years"}
PUB = {"published": "Peer-reviewed published", "in_process": "In the process", "none": "No publication record"}
MENTOR = {"mentor": "As a mentor", "mentee": "As a mentee", "both": "Both", "no": "No"}
HOURS = {"lt2": "< 2 hrs", "h2_5": "2-5 hrs", "h5_10": "5-10 hrs", "h10_plus": "> 10 hrs"}
WILLING = {"1": "1 - Not willing", "2": "2 - Slightly", "3": "3 - Moderate", "4": "4 - Very", "5": "5 - Extremely"}
SUPPORT = {
    "mentorship": "Mentorship from senior expert", "training": "Training",
    "stats_support": "Statistical support", "protected_time": "Protected time",
    "institutional_support": "Institutional support", "other": "Other",
}
YESNO = {"yes": "Yes", "no": "No"}
A1 = {"yes": "Yes", "no": "No", "unsure": "Unsure"}
REC = {
    "strong": "Strong fit", "moderate": "Moderate fit", "low": "Low fit",
    "followup": "Follow-up needed", "not_eligible": "Not eligible",
}
REC_ORDER = list(REC.values())

SCORE_PARTS = {  # column -> (label, max points)
    "p_expertise": ("Domain expertise", 30),
    "p_pub": ("Publications", 25),
    "p_will": ("Willingness", 25),
    "p_acad": ("Academic", 10),
    "p_mentor": ("Mentorship", 10),
}


# ----------------------------------------------------------------------------
# Password gate
# ----------------------------------------------------------------------------
def check_password() -> bool:
    if st.session_state.get("auth_ok"):
        return True
    expected = st.secrets.get("app_password")
    if not expected:
        st.error("`app_password` is not set in the app secrets.")
        return False
    st.markdown(f"<h2 style='color:{NAVY}'>SME Assessment Dashboard</h2>", unsafe_allow_html=True)
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if hmac.compare_digest(pw.encode(), str(expected).encode()):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Loading submissions from KoboToolbox...")
def load_data(server: str, form_id: str, token: str) -> pd.DataFrame:
    url = f"{server.rstrip('/')}/api/v2/assets/{form_id}/data/"
    headers = {"Authorization": f"Token {token}"}
    params = {"format": "json", "limit": 1000}
    records = []
    while url:
        r = requests.get(url, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        records.extend(payload.get("results", []))
        url, params = payload.get("next"), None  # 'next' already carries the query string
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # Kobo prefixes grouped questions as 'group/question' - keep the question name only
    df.columns = [c if c.startswith("_") else c.split("/")[-1] for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["consent", "a1_willing", "b1_primary", "target_area", "c1_qual", "c3_years",
              "d2_grant", "e1_grant_training", "e2_mentor", "e3_hours", "f1_willing",
              "total_score", "rec_code", "rec_label", "respondent_name", "respondent_contact",
              "a2_more_info", "a2_info_required", "interviewer_comments", "_submission_time"]:
        if c not in df.columns:
            df[c] = pd.NA
    for c in list(SCORE_PARTS) + ["total_score"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["submitted"] = pd.to_datetime(df["_submission_time"], errors="coerce", utc=True).dt.tz_localize(None)
    df["rec"] = df["rec_code"].map(REC)
    df["primary_area"] = df["b1_primary"].map(AREAS_B1).fillna(df["b1_primary"])
    df["aligned_area"] = df["target_area"].map(AREAS_B1).fillna(df["target_area"])
    return df


def multi_counts(series: pd.Series, labels: dict) -> pd.DataFrame:
    s = series.dropna().astype(str).str.split()
    s = s.explode().dropna()
    out = s.map(lambda x: labels.get(x, x)).value_counts().rename_axis("Option").reset_index(name="Respondents")
    return out


def single_counts(series: pd.Series, labels: dict, order=None) -> pd.DataFrame:
    s = series.dropna().astype(str).map(lambda x: labels.get(x, x))
    out = s.value_counts().rename_axis("Option").reset_index(name="Respondents")
    if order:
        out["Option"] = pd.Categorical(out["Option"], [labels[k] for k in order if k in labels], ordered=True)
        out = out.sort_values("Option")
    return out


def hbar(df: pd.DataFrame, title: str, height: int = 340):
    if df.empty:
        st.caption(f"{title}: no data")
        return
    d = df.sort_values("Respondents")
    fig = px.bar(d, x="Respondents", y="Option", orientation="h", text="Respondents", title=title)
    fig.update_traces(marker_color=NAVY, textposition="outside", cliponaxis=False)
    fig.update_layout(height=max(height, 40 * len(d) + 90), margin=dict(l=0, r=20, t=50, b=0),
                      yaxis_title=None, xaxis_title=None, title_font_size=15)
    st.plotly_chart(fig, width="stretch")


def vbar(df: pd.DataFrame, title: str, height: int = 320):
    if df.empty:
        st.caption(f"{title}: no data")
        return
    fig = px.bar(df, x="Option", y="Respondents", text="Respondents", title=title)
    fig.update_traces(marker_color=NAVY, textposition="outside", cliponaxis=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=50, b=0), xaxis_title=None,
                      yaxis_title=None, title_font_size=15)
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------
# Connect to Kobo
# ----------------------------------------------------------------------------
missing = [k for k in ("kobo_token", "kobo_form_id") if not st.secrets.get(k)]
if missing:
    st.error(f"Missing secrets: {', '.join(missing)}. See `.streamlit/secrets.toml.example`.")
    st.stop()

SERVER = st.secrets.get("kobo_server", "https://kf.kobotoolbox.org")

try:
    raw = load_data(SERVER, st.secrets["kobo_form_id"], st.secrets["kobo_token"])
except requests.HTTPError as e:
    st.error(f"KoboToolbox returned an error ({e.response.status_code}). "
             "Check the API token, form ID and server URL in the secrets.")
    st.stop()
except requests.RequestException:
    st.error("Could not reach KoboToolbox. Check your connection and the server URL.")
    st.stop()

# ----------------------------------------------------------------------------
# Header + sidebar
# ----------------------------------------------------------------------------
st.markdown(f"<h1 style='color:{NAVY};margin-bottom:0'>SME Assessment Dashboard</h1>", unsafe_allow_html=True)
st.caption("Subject matter expert matching - live from KoboToolbox")

with st.sidebar:
    if st.button("Refresh data", width="stretch"):
        load_data.clear()
        st.rerun()
    if st.button("Sign out", width="stretch"):
        st.session_state.clear()
        st.rerun()

if raw.empty:
    st.info("No submissions yet.")
    st.stop()

df = prepare(raw)

scored = df[df["total_score"].notna()]
consented = df[df["consent"] == "yes"]
willing = df[(df["consent"] == "yes") & (df["a1_willing"] == "yes")]

# ----------------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------------
k = st.columns(6)
k[0].metric("Submissions", len(df))
k[1].metric("Consented", len(consented))
k[2].metric("Willing to be an SME", len(willing))
k[3].metric("Scored", len(scored))
k[4].metric("Strong fit", int((df["rec_code"] == "strong").sum()))
k[5].metric("Average score", f"{scored['total_score'].mean():.1f}" if len(scored) else "-")

tabs = st.tabs(["Overview", "Expertise", "Background & capacity", "SME shortlist", "Follow-up", "Data"])

# ---------------------------- Overview --------------------------------------
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        funnel = pd.DataFrame({
            "Stage": ["Submitted", "Consented", "Willing to be an SME", "Fully scored"],
            "Respondents": [len(df), len(consented), len(willing), len(scored)],
        })
        fig = px.funnel(funnel, x="Respondents", y="Stage", title="Respondent pipeline")
        fig.update_traces(marker_color=NAVY)
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=50, b=0), title_font_size=15)
        st.plotly_chart(fig, width="stretch")
    with c2:
        rc = df["rec"].dropna().value_counts().reindex(REC_ORDER).dropna().rename_axis("Recommendation").reset_index(name="Respondents")
        if rc.empty:
            st.caption("No recommendations yet")
        else:
            fig = px.bar(rc, x="Recommendation", y="Respondents", text="Respondents", color="Recommendation",
                         color_discrete_map=REC_COLORS, title="Recommendation")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=340, showlegend=False, margin=dict(l=0, r=0, t=50, b=0),
                              xaxis_title=None, yaxis_title=None, title_font_size=15)
            st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        if len(scored):
            fig = px.histogram(scored, x="total_score", nbins=10, range_x=[0, 100], title="Total score distribution")
            fig.update_traces(marker_color=NAVY)
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=50, b=0), xaxis_title="Total score (0-100)",
                              yaxis_title="Respondents", title_font_size=15, bargap=0.05)
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No scored respondents yet")
    with c4:
        if len(scored):
            parts = pd.DataFrame([
                {"Component": f"{lab} (max {mx})", "Average": scored[col].mean(), "Max": mx}
                for col, (lab, mx) in SCORE_PARTS.items()
            ])
            parts["% of maximum"] = (parts["Average"] / parts["Max"] * 100).round(0)
            fig = px.bar(parts.sort_values("% of maximum"), x="% of maximum", y="Component", orientation="h",
                         text="% of maximum", title="Average score by component (% of maximum)")
            fig.update_traces(marker_color="#2E75B6", texttemplate="%{text:.0f}%", textposition="outside", cliponaxis=False)
            fig.update_layout(height=320, margin=dict(l=0, r=20, t=50, b=0), xaxis_range=[0, 110],
                              xaxis_title=None, yaxis_title=None, title_font_size=15)
            st.plotly_chart(fig, width="stretch")

    if df["submitted"].notna().any():
        daily = df.set_index("submitted").resample("W").size().rename("Submissions").reset_index()
        fig = px.line(daily, x="submitted", y="Submissions", markers=True, title="Submissions per week")
        fig.update_traces(line_color=NAVY)
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=50, b=0), xaxis_title=None, title_font_size=15)
        st.plotly_chart(fig, width="stretch")

# ---------------------------- Expertise -------------------------------------
with tabs[1]:
    base = willing if len(willing) else df
    st.caption(f"Based on {len(base)} respondent(s) willing to be an SME.")
    c1, c2 = st.columns(2)
    with c1:
        hbar(single_counts(base["b1_primary"], AREAS_B1), "Primary area of expertise (B1)")
    with c2:
        hbar(multi_counts(base["b2_secondary"], AREAS_B2) if "b2_secondary" in base else pd.DataFrame(),
             "Secondary areas (B2)")
    hbar(multi_counts(base["b3_cross"], CROSSCUT) if "b3_cross" in base else pd.DataFrame(),
         "Specialized research areas (B3)")

    st.subheader("Area alignment vs. primary expertise")
    if base["target_area"].notna().any() and base["b1_primary"].notna().any():
        ct = pd.crosstab(base["aligned_area"], base["primary_area"])
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x="Primary expertise (B1)", y="Most aligned area", color="Respondents"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, width="stretch")

# ------------------------ Background & capacity ------------------------------
with tabs[2]:
    base = willing if len(willing) else df
    c1, c2, c3 = st.columns(3)
    with c1:
        vbar(single_counts(base["c1_qual"], QUAL, list(QUAL)), "Highest qualification (C1)")
    with c2:
        vbar(single_counts(base["c3_years"], YEARS, list(YEARS)), "Research experience (C3)")
    with c3:
        vbar(single_counts(base["f1_willing"], WILLING, list(WILLING)), "Willingness to champion protocols (F1)")
    c4, c5 = st.columns(2)
    with c4:
        hbar(multi_counts(base["d1_pub"], PUB) if "d1_pub" in base else pd.DataFrame(), "Publication record (D1)", 260)
    with c5:
        vbar(single_counts(base["e2_mentor"], MENTOR, list(MENTOR)), "Prior mentorship experience (E2)")
    c6, c7, c8 = st.columns(3)
    with c6:
        vbar(single_counts(base["e3_hours"], HOURS, list(HOURS)), "Monthly hours available (E3)")
    with c7:
        vbar(single_counts(base["d2_grant"], YESNO, list(YESNO)), "Grant writing experience (D2)")
    with c8:
        vbar(single_counts(base["e1_grant_training"], YESNO, list(YESNO)), "Trained in grant writing (E1)")
    hbar(multi_counts(base["f2_support"], SUPPORT) if "f2_support" in base else pd.DataFrame(), "Support needed (F2)")

# ---------------------------- Shortlist -------------------------------------
with tabs[3]:
    st.caption("Ranked by total score. Contains personal contact details - handle accordingly.")
    sl = scored.sort_values("total_score", ascending=False).copy()
    sl["Rank"] = range(1, len(sl) + 1)
    show = sl.assign(
        Name=sl["respondent_name"], Contact=sl["respondent_contact"],
        **{"Primary area": sl["primary_area"], "Most aligned area": sl["aligned_area"],
           "Total": sl["total_score"], "Recommendation": sl["rec"]},
        **{lab: sl[col] for col, (lab, _) in SCORE_PARTS.items()},
    )[["Rank", "Name", "Contact", "Primary area", "Most aligned area", "Total", "Recommendation"]
      + [lab for lab, _ in SCORE_PARTS.values()]]
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={"Total": st.column_config.ProgressColumn("Total", min_value=0, max_value=100, format="%.1f")})
    st.download_button("Download shortlist (CSV)", show.to_csv(index=False).encode("utf-8"),
                       "sme_shortlist.csv", "text/csv")

# ----------------------------- Follow-up ------------------------------------
with tabs[4]:
    fu = df[df["rec_code"] == "followup"]
    st.metric("Respondents requesting more information", len(fu))
    if len(fu):
        out = fu[["submitted", "respondent_name", "respondent_contact", "a2_info_required"]].rename(columns={
            "submitted": "Submitted", "respondent_name": "Name", "respondent_contact": "Contact",
            "a2_info_required": "Information required"})
        st.dataframe(out, width="stretch", hide_index=True)
    else:
        st.caption("No follow-up requests.")

# ------------------------------- Data ---------------------------------------
with tabs[5]:
    st.caption("All submissions (question names shown without group prefixes).")
    st.dataframe(df.drop(columns=["submitted", "rec", "primary_area", "aligned_area"], errors="ignore"),
                 width="stretch", hide_index=True)
    st.download_button("Download data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "sme_assessment_data.csv", "text/csv")
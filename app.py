import io

import pandas as pd
import streamlit as st

from parser import parse_pdf, consolidate

st.set_page_config(page_title="Transaction Summary Dashboard", layout="wide")

st.title("Bank Statement Consolidated Dashboard")
st.caption(
    "Upload your ICICI 'Statement of Transactions' PDFs (e.g. 10 months). "
    "For every unique person/counterparty, see how much they sent you and how much you sent them."
)

uploaded_files = st.file_uploader(
    "Upload statement PDFs", type=["pdf"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload one or more PDF statements to get started.")
    st.stop()

@st.cache_data(show_spinner=False)
def _parse_all(file_bytes_list):
    frames = []
    errors = []
    for name, data in file_bytes_list:
        try:
            df = parse_pdf(io.BytesIO(data), source_name=name)
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            errors.append((name, str(e)))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()
    return combined, errors


file_bytes_list = [(f.name, f.getvalue()) for f in uploaded_files]
with st.spinner(f"Parsing {len(file_bytes_list)} file(s)..."):
    transactions, errors = _parse_all(file_bytes_list)

if errors:
    for name, err in errors:
        st.error(f"Failed to parse **{name}**: {err}")

if transactions.empty:
    st.warning("No transactions could be extracted from the uploaded file(s).")
    st.stop()

holder_names = transactions["account_holder"].dropna().unique().tolist()
holder_label = holder_names[0] if holder_names else "the account holder"

st.success(
    f"Parsed {len(transactions)} transactions from {len(uploaded_files)} file(s) "
    f"for **{holder_label}** ({transactions['date'].min().date()} to {transactions['date'].max().date()})."
)

summary = consolidate(transactions)

# ---- Top metrics ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique People/Counterparties", f"{len(summary):,}")
c2.metric("Total They Sent You (INR)", f"{summary['They Sent You (INR)'].sum():,.2f}")
c3.metric("Total You Sent Them (INR)", f"{summary['You Sent Them (INR)'].sum():,.2f}")
c4.metric("Net (INR)", f"{summary['Net (INR)'].sum():,.2f}")

st.caption(
    "**They Sent You** = money that person sent into this account (credit). "
    "**You Sent Them** = money this account sent to that person (debit)."
)

st.divider()

# ---- Filters ----
fc1, fc2 = st.columns([2, 1])
search = fc1.text_input("Search name")
min_txn = fc2.number_input(
    "Min total transactions with this person", min_value=0, value=0, step=1
)

filtered = summary.copy()
if search:
    filtered = filtered[filtered["Name"].str.contains(search, case=False, na=False)]
if min_txn:
    filtered = filtered[
        (filtered["Times They Sent You"] + filtered["Times You Sent Them"]) >= min_txn
    ]

st.subheader("Summary: How Much Each Person Sent You / You Sent Them")
st.dataframe(
    filtered.style.format({
        "They Sent You (INR)": "{:,.2f}",
        "You Sent Them (INR)": "{:,.2f}",
        "Net (INR)": "{:,.2f}",
        "First Txn": lambda d: d.strftime("%d-%b-%Y") if pd.notna(d) else "",
        "Last Txn": lambda d: d.strftime("%d-%b-%Y") if pd.notna(d) else "",
    }),
    use_container_width=True,
    height=400,
)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download summary (CSV)", data=csv_bytes,
    file_name="consolidated_summary.csv", mime="text/csv",
)

st.divider()

# ---- Charts ----
chart_c1, chart_c2 = st.columns(2)
top_recv = summary.nlargest(10, "They Sent You (INR)")[["Name", "They Sent You (INR)"]].set_index("Name")
top_sent = summary.nlargest(10, "You Sent Them (INR)")[["Name", "You Sent Them (INR)"]].set_index("Name")
chart_c1.subheader("Top 10: They Sent You")
chart_c1.bar_chart(top_recv)
chart_c2.subheader("Top 10: You Sent Them")
chart_c2.bar_chart(top_sent)

st.divider()

# ---- Per-person full ledger (date-wise, every transaction) ----
st.subheader("Per-Person Transaction Ledger (every transaction, with dates)")
st.caption(
    "Expand any person below to see every individual transaction between you and them, "
    "across all uploaded files, with dates and direction — e.g. 'Claude sent you 20,000 on "
    "18th, you sent Claude 10,000 on 19th, Claude sent you 2,000 on 20th'."
)

detail_df = transactions.copy()
detail_df["name_norm"] = detail_df["name"].fillna("OTHERS").str.strip().str.upper()

# map raw names to the same canonicalized names used in `summary`
from parser import _canonicalize_names  # noqa: E402
canon_map = _canonicalize_names(detail_df["name_norm"].unique())
detail_df["name_canon"] = detail_df["name_norm"].map(canon_map)
detail_df["Direction"] = detail_df["type"].map({"Credit": "They -> You", "Debit": "You -> Them"})

ledger_rows = filtered
if len(filtered) > 100 and not search:
    st.info(
        f"{len(filtered)} people match — showing the top 100 by amount received here. "
        "Use the search box above to jump to a specific person."
    )
    ledger_rows = filtered.head(100)

for _, row in ledger_rows.iterrows():
    name = row["Name"]
    person_txns = detail_df[detail_df["name_canon"] == name].sort_values("date")
    header = (
        f"{name} — They sent you Rs.{row['They Sent You (INR)']:,.0f} "
        f"({row['Times They Sent You']}x) | You sent them Rs.{row['You Sent Them (INR)']:,.0f} "
        f"({row['Times You Sent Them']}x)"
    )
    with st.expander(header):
        view = person_txns[["date", "Direction", "amount", "balance", "source_file", "remarks"]]
        st.dataframe(
            view.style.format({"amount": "{:,.2f}", "balance": "{:,.2f}"}),
            use_container_width=True,
        )
        person_csv = view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download this person's transactions (CSV)", data=person_csv,
            file_name=f"{name}_transactions.csv", mime="text/csv",
            key=f"dl_{name}",
        )

st.divider()

with st.expander("All raw parsed transactions (every row from every file)"):
    st.dataframe(
        transactions.drop(columns=["s_no"], errors="ignore").style.format(
            {"amount": "{:,.2f}", "balance": "{:,.2f}"}
        ),
        use_container_width=True,
    )
    all_csv = transactions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download all raw transactions (CSV)", data=all_csv,
        file_name="all_transactions.csv", mime="text/csv",
    )

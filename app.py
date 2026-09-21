import io

import pandas as pd
import streamlit as st

from parser import parse_pdf, consolidate, _canonicalize_names

st.set_page_config(page_title="Transaction Summary Dashboard", layout="wide")

st.title("Bank Statement Consolidated Dashboard")
st.caption(
    "Upload statement PDFs (ICICI, Canara Bank, or Karur Vysya Bank supported, e.g. 10 months each). "
    "Each account holder gets their own report, shown one after another. For every "
    "unique person/counterparty, see how much they sent the account holder and how "
    "much the account holder sent them."
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

transactions["account_key"] = (
    transactions["bank"].fillna("")
    + " | " + transactions["account_no"].fillna("")
    + " | " + transactions["account_holder"].fillna("")
)
account_keys = transactions["account_key"].unique().tolist()

st.success(
    f"Parsed {len(transactions)} transactions from {len(uploaded_files)} file(s), "
    f"covering **{len(account_keys)}** distinct account(s)."
)


def render_account_dashboard(account_df: pd.DataFrame, account_label: str, key_prefix: str):
    st.header(account_label)

    files_used = sorted(account_df["source_file"].dropna().unique().tolist())
    st.caption(
        f"{len(account_df)} transactions from {len(files_used)} file(s): "
        + ", ".join(files_used)
        + f"  ·  {account_df['date'].min().date()} to {account_df['date'].max().date()}"
    )

    summary = consolidate(account_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique People/Counterparties", f"{len(summary):,}")
    c2.metric("Total They Sent You (INR)", f"{summary['They Sent You (INR)'].sum():,.2f}")
    c3.metric("Total You Sent Them (INR)", f"{summary['You Sent Them (INR)'].sum():,.2f}")
    c4.metric("Net (INR)", f"{summary['Net (INR)'].sum():,.2f}")

    st.caption(
        "**They Sent You** = money that person sent into this account (credit). "
        "**You Sent Them** = money this account sent to that person (debit)."
    )

    fc1, fc2 = st.columns([2, 1])
    search = fc1.text_input("Search name", key=f"{key_prefix}_search")
    min_txn = fc2.number_input(
        "Min total transactions with this person", min_value=0, value=0, step=1,
        key=f"{key_prefix}_min_txn",
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
        file_name=f"{key_prefix}_consolidated_summary.csv", mime="text/csv",
        key=f"{key_prefix}_dl_summary",
    )

    chart_c1, chart_c2 = st.columns(2)
    top_recv = (
        summary.sort_values("They Sent You (INR)", ascending=False)
        .head(10)[["Name", "They Sent You (INR)"]].set_index("Name")
    )
    top_sent = (
        summary.sort_values("You Sent Them (INR)", ascending=False)
        .head(10)[["Name", "You Sent Them (INR)"]].set_index("Name")
    )
    chart_c1.subheader("Top 10: They Sent You")
    chart_c1.bar_chart(top_recv)
    chart_c2.subheader("Top 10: You Sent Them")
    chart_c2.bar_chart(top_sent)

    st.subheader("Per-Person Transaction Ledger (every transaction, with dates)")
    st.caption(
        "Expand any person below to see every individual transaction between the "
        "account holder and them, across all uploaded files, with dates and direction."
    )

    detail_df = account_df.copy()
    detail_df["name_norm"] = detail_df["name"].fillna("OTHERS").str.strip().str.upper()
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
                file_name=f"{key_prefix}_{name}_transactions.csv", mime="text/csv",
                key=f"{key_prefix}_dl_{name}",
            )

    with st.expander("All raw parsed transactions for this account"):
        st.dataframe(
            account_df.drop(columns=["s_no", "account_key"], errors="ignore").style.format(
                {"amount": "{:,.2f}", "balance": "{:,.2f}"}
            ),
            use_container_width=True,
        )
        all_csv = account_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all raw transactions for this account (CSV)", data=all_csv,
            file_name=f"{key_prefix}_all_transactions.csv", mime="text/csv",
            key=f"{key_prefix}_dl_all",
        )


for idx, key in enumerate(account_keys):
    account_df = transactions[transactions["account_key"] == key]
    bank = account_df["bank"].dropna().iloc[0] if account_df["bank"].notna().any() else "Unknown Bank"
    holder = account_df["account_holder"].dropna().iloc[0] if account_df["account_holder"].notna().any() else "Unknown Holder"
    acct_no = account_df["account_no"].dropna().iloc[0] if account_df["account_no"].notna().any() else "Unknown A/c"
    label = f"Account {idx + 1}: {holder} — {bank} Bank (A/c {acct_no})"

    render_account_dashboard(account_df, label, key_prefix=f"acct{idx}")
    if idx < len(account_keys) - 1:
        st.divider()
        st.divider()

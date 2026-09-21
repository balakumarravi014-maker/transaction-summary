"""Parser for ICICI Bank 'Statement of Transactions' PDFs.

Extracts one row per transaction with: date, counterparty name, type
(credit/debit), amount, balance, remarks, and source file/account info.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import BinaryIO, Union

import pdfplumber
import pandas as pd

DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
SNO_RE = re.compile(r"^\d+$")
ACCOUNT_RE = re.compile(
    r"Statement of Transactions in (.+?) Account no\.\s*(\S+)", re.IGNORECASE
)

HEADER_WORDS = {
    "Transaction", "Withdrawal", "Deposit", "Balance", "S", "No.",
    "Cheque", "Number", "Remarks", "Date", "Amount", "(INR)",
}
FOOTER_PREFIXES = (
    "www.icici", "Please call", "Never share", "Dial your Bank",
)


@dataclass
class ColumnBounds:
    withdrawal_min: float
    withdrawal_max: float
    deposit_min: float
    deposit_max: float
    balance_min: float


def _line_groups(words, tol=2.5):
    """Group words into visual lines, sorted top-to-bottom, left-to-right."""
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    current = []
    current_top = None
    for w in words:
        if current_top is None or abs(w["top"] - current_top) <= tol:
            current.append(w)
            current_top = w["top"] if current_top is None else current_top
        else:
            lines.append(sorted(current, key=lambda w: w["x0"]))
            current = [w]
            current_top = w["top"]
    if current:
        lines.append(sorted(current, key=lambda w: w["x0"]))
    return lines


def _find_column_bounds(page) -> ColumnBounds | None:
    words = page.extract_words()
    by_text = {}
    for w in words:
        by_text.setdefault(w["text"], []).append(w)

    def x0(text):
        return by_text[text][0]["x0"] if text in by_text else None

    withdrawal_x = x0("Withdrawal")
    deposit_x = x0("Deposit")
    balance_x = x0("Balance")
    if withdrawal_x is None or deposit_x is None or balance_x is None:
        return None
    return ColumnBounds(
        withdrawal_min=withdrawal_x - 20,
        withdrawal_max=(withdrawal_x + deposit_x) / 2 + 15,
        deposit_min=(withdrawal_x + deposit_x) / 2 + 15,
        deposit_max=(deposit_x + balance_x) / 2 + 15,
        balance_min=(deposit_x + balance_x) / 2 + 15,
    )


def _is_header_or_footer(line_text: str, first_word_text: str) -> bool:
    if first_word_text in HEADER_WORDS:
        return True
    if line_text.strip().startswith(FOOTER_PREFIXES):
        return True
    if line_text.strip().isdigit() and len(line_text.strip()) <= 3:
        return True
    return False


def parse_pdf(source: Union[str, BinaryIO], source_name: str | None = None) -> pd.DataFrame:
    """Parse one ICICI statement PDF into a DataFrame of transactions."""
    rows = []
    account_holder = None
    account_no = None
    bounds: ColumnBounds | None = None

    with pdfplumber.open(source) as pdf:
        full_text_first_page = pdf.pages[0].extract_text() or ""
        m = ACCOUNT_RE.search(full_text_first_page)
        if m:
            account_holder = None
            account_no = m.group(2)
        holder_m = re.search(r"^([A-Z][A-Z\s.]+?)\s+Your Base Branch", full_text_first_page, re.MULTILINE)
        if holder_m:
            account_holder = holder_m.group(1).strip()

        for page in pdf.pages:
            if bounds is None:
                bounds = _find_column_bounds(page)
            if bounds is None:
                continue

            lines = _line_groups(page.extract_words())
            pending_name = None
            i = 0
            while i < len(lines):
                line = lines[i]
                text = " ".join(w["text"] for w in line)
                first_word = line[0]["text"]

                if _is_header_or_footer(text, first_word):
                    i += 1
                    continue

                is_data_line = (
                    SNO_RE.match(first_word) is not None
                    and len(line) >= 2
                    and DATE_RE.match(line[1]["text"]) is not None
                )

                if is_data_line:
                    sno = first_word
                    date = line[1]["text"]
                    withdrawal = None
                    deposit = None
                    balance = None
                    for w in line[2:]:
                        try:
                            val = float(w["text"].replace(",", ""))
                        except ValueError:
                            continue
                        if bounds.withdrawal_min <= w["x0"] < bounds.withdrawal_max:
                            withdrawal = val
                        elif bounds.deposit_min <= w["x0"] < bounds.deposit_max:
                            deposit = val
                        elif w["x0"] >= bounds.balance_min:
                            balance = val

                    # gather remarks lines until next name-label / data-line
                    remarks_lines = []
                    j = i + 1
                    while j < len(lines):
                        nxt = lines[j]
                        nxt_text = " ".join(w["text"] for w in nxt)
                        nxt_first = nxt[0]["text"]
                        if _is_header_or_footer(nxt_text, nxt_first):
                            j += 1
                            continue
                        nxt_is_data = (
                            SNO_RE.match(nxt_first) is not None
                            and len(nxt) >= 2
                            and DATE_RE.match(nxt[1]["text"]) is not None
                        )
                        if nxt_is_data:
                            break
                        # Heuristic: a short line (<=2 words, all title/upper case,
                        # no digits) right before the NEXT data line is the name
                        # label for the next row, not remarks for this one.
                        is_short_label = (
                            len(nxt) <= 3
                            and not any(c.isdigit() for c in nxt_text)
                            and j + 1 < len(lines)
                        )
                        if is_short_label:
                            # peek ahead: is the line after this a data line?
                            k = j + 1
                            while k < len(lines) and _is_header_or_footer(
                                " ".join(w["text"] for w in lines[k]), lines[k][0]["text"]
                            ):
                                k += 1
                            if k < len(lines):
                                after = lines[k]
                                after_text = " ".join(w["text"] for w in after)
                                after_is_data = (
                                    SNO_RE.match(after[0]["text"]) is not None
                                    and len(after) >= 2
                                    and DATE_RE.match(after[1]["text"]) is not None
                                )
                                if after_is_data:
                                    break
                        remarks_lines.append(nxt_text)
                        j += 1

                    remarks = " ".join(remarks_lines)
                    name = pending_name
                    if not name or "TRXN" in name.upper():
                        name = _guess_name_from_remarks(remarks)

                    if withdrawal is not None:
                        rows.append({
                            "source_file": source_name,
                            "account_no": account_no,
                            "account_holder": account_holder,
                            "s_no": sno,
                            "date": date,
                            "name": name,
                            "type": "Debit",
                            "amount": withdrawal,
                            "balance": balance,
                            "remarks": remarks,
                        })
                    if deposit is not None:
                        rows.append({
                            "source_file": source_name,
                            "account_no": account_no,
                            "account_holder": account_holder,
                            "s_no": sno,
                            "date": date,
                            "name": name,
                            "type": "Credit",
                            "amount": deposit,
                            "balance": balance,
                            "remarks": remarks,
                        })

                    pending_name = None
                    i = j
                    continue

                # Not a data line: could be a name-label line preceding the
                # next data line, or a stray footer/header fragment.
                is_short_label = len(line) <= 3 and not any(c.isdigit() for c in text)
                if is_short_label:
                    pending_name = text
                i += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def _guess_name_from_remarks(remarks: str) -> str:
    if not remarks:
        return "OTHERS"
    upper = remarks.upper()
    if "INT.PD" in upper or "INTEREST" in upper:
        return "INTEREST"
    parts = [p.strip() for p in remarks.split("/") if p.strip()]
    if not parts:
        return "OTHERS"
    if parts[0].upper() == "UPI" and len(parts) >= 2:
        return parts[1]
    if parts[0].upper() in {"CMS", "NEFT", "IMPS", "RTGS", "ECS"} and len(parts) >= 2:
        return parts[-1]
    return parts[0]


def _canonicalize_names(names) -> dict:
    """Merge names that are simple prefix-truncations of a longer name for
    the same party (bank statements truncate remark text inconsistently
    month to month, e.g. 'INDIUM SOFTWARE INDIA LIMIT' vs '...LIMITED').
    Only merges when the shared prefix is at least 8 characters, to avoid
    collapsing genuinely different short names (e.g. 'CRED' vs 'CRED Club').
    """
    uniq = sorted({n.strip().upper() for n in names if n}, key=len, reverse=True)
    finalized = []
    mapping = {}
    for n in uniq:
        match = next((f for f in finalized if len(n) >= 8 and f.startswith(n)), None)
        if match:
            mapping[n] = match
        else:
            mapping[n] = n
            finalized.append(n)
    return mapping


SUMMARY_COLUMNS = [
    "Name", "Times They Sent You", "They Sent You (INR)",
    "Times You Sent Them", "You Sent Them (INR)", "Net (INR)",
    "First Txn", "Last Txn",
]


def consolidate(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate parsed transactions into one row per unique counterparty
    name, from the statement account holder's point of view:
    'They Sent You' = money that person sent to the account holder (a
    Credit/Deposit on the statement); 'You Sent Them' = money the account
    holder sent to that person (a Debit/Withdrawal on the statement).
    """
    if transactions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    df = transactions.copy()
    df["name"] = df["name"].fillna("OTHERS").str.strip().str.upper()
    canon_map = _canonicalize_names(df["name"].unique())
    df["name"] = df["name"].map(canon_map)

    # they -> you (credit/deposit), you -> them (debit/withdrawal)
    they_sent = df[df["type"] == "Credit"].groupby("name")["amount"].agg(["count", "sum"])
    you_sent = df[df["type"] == "Debit"].groupby("name")["amount"].agg(["count", "sum"])
    date_range = df.groupby("name")["date"].agg(["min", "max"])

    summary = pd.DataFrame(index=df["name"].unique())
    summary.index.name = "Name"
    summary["Times They Sent You"] = they_sent["count"]
    summary["They Sent You (INR)"] = they_sent["sum"]
    summary["Times You Sent Them"] = you_sent["count"]
    summary["You Sent Them (INR)"] = you_sent["sum"]
    summary = summary.fillna(0)
    summary["Times They Sent You"] = summary["Times They Sent You"].astype(int)
    summary["Times You Sent Them"] = summary["Times You Sent Them"].astype(int)
    summary["Net (INR)"] = summary["They Sent You (INR)"] - summary["You Sent Them (INR)"]
    summary["First Txn"] = date_range["min"]
    summary["Last Txn"] = date_range["max"]
    summary = summary.reset_index().sort_values(
        by=["They Sent You (INR)", "You Sent Them (INR)"], ascending=False
    )
    return summary

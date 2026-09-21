"""Parsers for bank 'Statement of Transactions' PDFs.

Currently supports ICICI Bank and Canara Bank statement layouts. Each
parser extracts one row per transaction with: date, counterparty name
(best-effort), type (credit/debit), amount, balance, remarks, and
source file/account info, into a common DataFrame schema.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO, Union

import pdfplumber
import pandas as pd

COLUMNS = [
    "source_file", "bank", "account_no", "account_holder", "s_no",
    "date", "name", "type", "amount", "balance", "remarks",
]


@dataclass
class ColumnBounds:
    """x-position bounds for the two amount columns (col_a = the one that
    appears further left on the page, col_b = further right) and where the
    balance column begins. Each bank's parser maps col_a/col_b onto
    withdrawal/deposit according to that bank's actual column order.
    """
    col_a_min: float
    col_a_max: float
    col_b_min: float
    col_b_max: float
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


def _find_column_bounds(page, col_a_label, col_b_label, balance_label) -> ColumnBounds | None:
    words = page.extract_words()
    by_text = {}
    for w in words:
        by_text.setdefault(w["text"], []).append(w)

    def x0(text):
        return by_text[text][0]["x0"] if text in by_text else None

    a_x = x0(col_a_label)
    b_x = x0(col_b_label)
    bal_x = x0(balance_label)
    if a_x is None or b_x is None or bal_x is None:
        return None
    return ColumnBounds(
        col_a_min=a_x - 20,
        col_a_max=(a_x + b_x) / 2 + 15,
        col_b_min=(a_x + b_x) / 2 + 15,
        col_b_max=(b_x + bal_x) / 2 + 15,
        balance_min=(b_x + bal_x) / 2 + 15,
    )


def _amount_at(word, bounds: ColumnBounds):
    try:
        val = float(word["text"].replace(",", ""))
    except ValueError:
        return None, None
    if bounds.col_a_min <= word["x0"] < bounds.col_a_max:
        return "a", val
    if bounds.col_b_min <= word["x0"] < bounds.col_b_max:
        return "b", val
    if word["x0"] >= bounds.balance_min:
        return "balance", val
    return None, None


def _detect_bank(first_page_text: str) -> str | None:
    if "Statement of Transactions in" in first_page_text and "ICICI" in first_page_text.upper():
        return "ICICI"
    if "Statement for A/c" in first_page_text or "CNRB" in first_page_text.upper():
        return "CANARA"
    return None


def parse_pdf(source: Union[str, BinaryIO], source_name: str | None = None) -> pd.DataFrame:
    """Parse a bank statement PDF, auto-detecting the bank format."""
    with pdfplumber.open(source) as pdf:
        first_page_text = pdf.pages[0].extract_text() or ""
        bank = _detect_bank(first_page_text)
        if bank == "ICICI":
            rows = _parse_icici(pdf, first_page_text, source_name)
        elif bank == "CANARA":
            rows = _parse_canara(pdf, first_page_text, source_name)
        else:
            raise ValueError(
                "Unrecognized statement format — only ICICI Bank and Canara "
                "Bank 'Statement of Transactions' PDFs are currently supported."
            )

    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# ICICI Bank
# ---------------------------------------------------------------------------

ICICI_DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
ICICI_SNO_RE = re.compile(r"^\d+$")
ICICI_ACCOUNT_RE = re.compile(
    r"Statement of Transactions in (.+?) Account no\.\s*(\S+)", re.IGNORECASE
)
ICICI_HEADER_WORDS = {
    "Transaction", "Withdrawal", "Deposit", "Balance", "S", "No.",
    "Cheque", "Number", "Remarks", "Date", "Amount", "(INR)",
}
ICICI_FOOTER_PREFIXES = (
    "www.icici", "Please call", "Never share", "Dial your Bank",
)


def _icici_is_header_or_footer(line_text: str, first_word_text: str) -> bool:
    if first_word_text in ICICI_HEADER_WORDS:
        return True
    if line_text.strip().startswith(ICICI_FOOTER_PREFIXES):
        return True
    if line_text.strip().isdigit() and len(line_text.strip()) <= 3:
        return True
    return False


def _parse_icici(pdf, first_page_text: str, source_name: str | None) -> list[dict]:
    rows = []
    account_holder = None
    account_no = None
    bounds: ColumnBounds | None = None

    m = ICICI_ACCOUNT_RE.search(first_page_text)
    if m:
        account_no = m.group(2)
    holder_m = re.search(r"^([A-Z][A-Z\s.]+?)\s+Your Base Branch", first_page_text, re.MULTILINE)
    if holder_m:
        account_holder = holder_m.group(1).strip()

    for page in pdf.pages:
        if bounds is None:
            # ICICI column order left-to-right: Withdrawal, Deposit, Balance
            bounds = _find_column_bounds(page, "Withdrawal", "Deposit", "Balance")
        if bounds is None:
            continue

        lines = _line_groups(page.extract_words())
        pending_name = None
        i = 0
        while i < len(lines):
            line = lines[i]
            text = " ".join(w["text"] for w in line)
            first_word = line[0]["text"]

            if _icici_is_header_or_footer(text, first_word):
                i += 1
                continue

            is_data_line = (
                ICICI_SNO_RE.match(first_word) is not None
                and len(line) >= 2
                and ICICI_DATE_RE.match(line[1]["text"]) is not None
            )

            if is_data_line:
                sno = first_word
                date = line[1]["text"]
                withdrawal = None
                deposit = None
                balance = None
                for w in line[2:]:
                    col, val = _amount_at(w, bounds)
                    if col == "a":
                        withdrawal = val
                    elif col == "b":
                        deposit = val
                    elif col == "balance":
                        balance = val

                remarks_lines = []
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    nxt_text = " ".join(w["text"] for w in nxt)
                    nxt_first = nxt[0]["text"]
                    if _icici_is_header_or_footer(nxt_text, nxt_first):
                        j += 1
                        continue
                    nxt_is_data = (
                        ICICI_SNO_RE.match(nxt_first) is not None
                        and len(nxt) >= 2
                        and ICICI_DATE_RE.match(nxt[1]["text"]) is not None
                    )
                    if nxt_is_data:
                        break
                    is_short_label = (
                        len(nxt) <= 3
                        and not any(c.isdigit() for c in nxt_text)
                        and j + 1 < len(lines)
                    )
                    if is_short_label:
                        k = j + 1
                        while k < len(lines) and _icici_is_header_or_footer(
                            " ".join(w["text"] for w in lines[k]), lines[k][0]["text"]
                        ):
                            k += 1
                        if k < len(lines):
                            after = lines[k]
                            after_is_data = (
                                ICICI_SNO_RE.match(after[0]["text"]) is not None
                                and len(after) >= 2
                                and ICICI_DATE_RE.match(after[1]["text"]) is not None
                            )
                            if after_is_data:
                                break
                    remarks_lines.append(nxt_text)
                    j += 1

                remarks = " ".join(remarks_lines)
                name = pending_name
                if not name or "TRXN" in name.upper():
                    name = _icici_guess_name(remarks)

                date_val = pd.to_datetime(date, format="%d.%m.%Y", errors="coerce")
                if withdrawal is not None:
                    rows.append({
                        "source_file": source_name, "bank": "ICICI",
                        "account_no": account_no, "account_holder": account_holder,
                        "s_no": sno, "date": date_val, "name": name, "type": "Debit",
                        "amount": withdrawal, "balance": balance, "remarks": remarks,
                    })
                if deposit is not None:
                    rows.append({
                        "source_file": source_name, "bank": "ICICI",
                        "account_no": account_no, "account_holder": account_holder,
                        "s_no": sno, "date": date_val, "name": name, "type": "Credit",
                        "amount": deposit, "balance": balance, "remarks": remarks,
                    })

                pending_name = None
                i = j
                continue

            is_short_label = len(line) <= 3 and not any(c.isdigit() for c in text)
            if is_short_label:
                pending_name = text
            i += 1

    return rows


def _icici_guess_name(remarks: str) -> str:
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


# ---------------------------------------------------------------------------
# Canara Bank
# ---------------------------------------------------------------------------

CANARA_DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
CANARA_ACCOUNT_RE = re.compile(r"Statement for A/c\s+(\S+)\s+between", re.IGNORECASE)
CANARA_HOLDER_RE = re.compile(r"^Name\s+(.+)$", re.MULTILINE)
CANARA_HEADER_WORDS = {"Date", "Particulars", "Deposits", "Withdrawals", "Balance"}
CANARA_IMPS_RE = re.compile(r"MB-IMPS-\s*(DR|CR)/([^/]+)/([^/]+)/\D*(\d{3,6})")


def _canara_is_header_or_footer(line_text: str, first_word_text: str) -> bool:
    text = line_text.strip()
    if first_word_text in CANARA_HEADER_WORDS:
        return True
    if re.match(r"^page\s+\d+$", text, re.IGNORECASE):
        return True
    return False


CANARA_JUNK_LINE_RE = re.compile(r"^(Chq:|\d{1,2}:\d{2}:\d{2}/)")


def _parse_canara(pdf, first_page_text: str, source_name: str | None) -> list[dict]:
    rows = []
    account_holder = None
    account_no = None
    bounds: ColumnBounds | None = None
    seen_opening_balance = False

    m = CANARA_ACCOUNT_RE.search(first_page_text)
    if m:
        account_no = m.group(1)
    holder_m = CANARA_HOLDER_RE.search(first_page_text)
    if holder_m:
        account_holder = holder_m.group(1).strip()

    for page in pdf.pages:
        if bounds is None:
            # Canara column order left-to-right: Deposits, Withdrawals, Balance
            bounds = _find_column_bounds(page, "Deposits", "Withdrawals", "Balance")
        if bounds is None:
            continue

        lines = _line_groups(page.extract_words())
        buffer_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            text = " ".join(w["text"] for w in line)
            first_word = line[0]["text"]
            stripped = text.strip()

            if _canara_is_header_or_footer(text, first_word):
                i += 1
                continue

            if stripped.startswith("Opening Balance") or stripped.startswith("Closing Balance"):
                buffer_lines = []
                seen_opening_balance = True
                i += 1
                continue

            is_data_line = CANARA_DATE_RE.match(first_word) is not None and len(line) >= 2

            if is_data_line and seen_opening_balance:
                date = first_word
                deposit = None
                withdrawal = None
                balance = None
                inline_words = []
                for w in line[1:]:
                    col, val = _amount_at(w, bounds)
                    if col == "a":
                        deposit = val
                    elif col == "b":
                        withdrawal = val
                    elif col == "balance":
                        balance = val
                    else:
                        try:
                            float(w["text"].replace(",", ""))
                        except ValueError:
                            inline_words.append(w["text"])

                if inline_words:
                    buffer_lines.append(" ".join(inline_words))
                particulars = " ".join(buffer_lines)
                buffer_lines = []

                name = _canara_guess_name(particulars, account_holder)
                date_val = pd.to_datetime(date, format="%d-%m-%Y", errors="coerce")

                if withdrawal is not None:
                    rows.append({
                        "source_file": source_name, "bank": "CANARA",
                        "account_no": account_no, "account_holder": account_holder,
                        "s_no": None, "date": date_val, "name": name, "type": "Debit",
                        "amount": withdrawal, "balance": balance, "remarks": particulars,
                    })
                if deposit is not None:
                    rows.append({
                        "source_file": source_name, "bank": "CANARA",
                        "account_no": account_no, "account_holder": account_holder,
                        "s_no": None, "date": date_val, "name": name, "type": "Credit",
                        "amount": deposit, "balance": balance, "remarks": particulars,
                    })
                i += 1
                continue

            if seen_opening_balance and not CANARA_JUNK_LINE_RE.match(stripped):
                buffer_lines.append(text)
            i += 1

    return rows


def _canara_guess_name(particulars: str, account_holder: str | None) -> str:
    if not particulars:
        return "OTHERS"
    upper = particulars.upper()

    m = CANARA_IMPS_RE.search(particulars)
    if m:
        direction, name_field, bank_code, last4 = m.groups()
        if direction.upper() == "CR":
            return name_field.strip()
        return f"IMPS TO {bank_code.strip()} ****{last4[-4:]}"

    if particulars.upper().startswith("MB/"):
        neft_m = re.match(r"^MB/\d+/(.*?)/\d{6,}", particulars)
        if neft_m:
            return neft_m.group(1).strip()

    if "UPI/" in upper:
        parts = [p.strip() for p in particulars.split("/") if p.strip()]
        idx = next((i for i, p in enumerate(parts) if p.upper() == "UPI"), None)
        if idx is not None and idx + 1 < len(parts):
            return parts[idx + 1]

    if "DRAWDOWN" in upper:
        ref_m = re.search(r"(\d{6,})", particulars)
        ref = ref_m.group(1) if ref_m else ""
        return f"OD/DRAWDOWN ACCOUNT {ref}".strip()

    if "CASH DEPOSIT" in upper or "CASH WITHDRAWAL" in upper:
        return " ".join(particulars.split()[:4]).strip()

    if "CREDIT CARD" in upper:
        return "CREDIT CARD DUES"

    if upper.startswith("IMPS SC") or upper.startswith("NEFT SC") or "SERVICE CHARGE" in upper:
        return "BANK CHARGES"

    if "INTEREST" in upper:
        return "INTEREST"

    # Fallback: first few words of the narration, excluding "Chq:" trailers.
    first_chunk = particulars.split(" Chq:")[0].strip()
    words = first_chunk.split()
    return " ".join(words[:5]) if words else "OTHERS"


# ---------------------------------------------------------------------------
# Consolidation (bank-agnostic)
# ---------------------------------------------------------------------------

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

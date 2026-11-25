# Smartspend
Expense manager for students
1. smartspend/__init__.py
"""SmartSpend - AI-based expense tracker for students."""
__version__ = "1.0.0"
2. smartspend/database.py
"""database.py - SQLite storage layer for SmartSpend (CRUD operations)."""
import sqlite3
from datetime import date

DB_NAME = "smartspend.db"


def get_connection(db_name=DB_NAME):
    return sqlite3.connect(db_name)


def create_tables(db_name=DB_NAME):
    conn = get_connection(db_name)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            date      TEXT NOT NULL,
            category  TEXT NOT NULL,
            amount    REAL NOT NULL,
            note      TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS budget (
            month   TEXT PRIMARY KEY,
            limit_  REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ---------- CREATE ----------
def add_expense(category, amount, note="", exp_date=None, db_name=DB_NAME):
    if amount <= 0:
        raise ValueError("Amount must be greater than zero")
    exp_date = exp_date or date.today().isoformat()
    conn = get_connection(db_name)
    conn.execute(
        "INSERT INTO expenses (date, category, amount, note) VALUES (?, ?, ?, ?)",
        (exp_date, category, amount, note),
    )
    conn.commit()
    conn.close()


# ---------- READ ----------
def get_all_expenses(db_name=DB_NAME):
    conn = get_connection(db_name)
    rows = conn.execute(
        "SELECT id, date, category, amount, note FROM expenses ORDER BY date"
    ).fetchall()
    conn.close()
    return rows


# ---------- UPDATE ----------
def update_expense(exp_id, category, amount, note, db_name=DB_NAME):
    conn = get_connection(db_name)
    conn.execute(
        "UPDATE expenses SET category = ?, amount = ?, note = ? WHERE id = ?",
        (category, amount, note, exp_id),
    )
    conn.commit()
    conn.close()


# ---------- DELETE ----------
def delete_expense(exp_id, db_name=DB_NAME):
    conn = get_connection(db_name)
    conn.execute("DELETE FROM expenses WHERE id = ?", (exp_id,))
    conn.commit()
    conn.close()


# ---------- BUDGET ----------
def set_budget(month, limit_amount, db_name=DB_NAME):
    conn = get_connection(db_name)
    conn.execute(
        "INSERT OR REPLACE INTO budget (month, limit_) VALUES (?, ?)",
        (month, limit_amount),
    )
    conn.commit()
    conn.close()


def get_budget(month, db_name=DB_NAME):
    conn = get_connection(db_name)
    row = conn.execute(
        "SELECT limit_ FROM budget WHERE month = ?", (month,)).fetchone()
    conn.close()
    return row[0] if row else None
3. smartspend/categorizer.py
"""categorizer.py - keyword based automatic categorisation of expense notes."""

CATEGORY_KEYWORDS = {
    "Food":          ["lunch", "dinner", "breakfast", "snack", "tea", "coffee",
                      "canteen", "pizza", "burger", "samosa", "mess", "juice"],
    "Transport":     ["bus", "auto", "metro", "train", "cab", "uber", "ola",
                      "petrol", "fuel", "ticket"],
    "Education":     ["book", "notebook", "xerox", "print", "course", "exam",
                      "fees", "stationery", "pen"],
    "Entertainment": ["movie", "game", "netflix", "party", "outing", "concert"],
    "Recharge":      ["recharge", "wifi", "internet", "data pack", "mobile"],
    "Shopping":      ["clothes", "shirt", "shoes", "bag", "amazon", "flipkart"],
    "Health":        ["medicine", "doctor", "pharmacy", "hospital", "gym"],
}


def auto_category(note):
    """Return a category name for a note, or 'Other' if nothing matches."""
    text = (note or "").lower()
    for category, words in CATEGORY_KEYWORDS.items():
        for word in words:
            if word in text:
                return category
    return "Other"
4. smartspend/analysis.py
"""analysis.py - data cleaning and summaries using Pandas and NumPy."""
import numpy as np
import pandas as pd


def to_dataframe(rows):
    """Convert database rows into a clean DataFrame."""
    df = pd.DataFrame(rows, columns=["id", "date", "category", "amount", "note"])
    if df.empty:
        df["month"] = pd.Series(dtype=str)   # keep the column so later code works
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])        # remove invalid rows
    df = df[df["amount"] > 0]                         # remove non-positive amounts
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df.reset_index(drop=True)


def category_summary(df):
    return (df.groupby("category")["amount"].sum()
              .sort_values(ascending=False).round(2))


def monthly_summary(df):
    return df.groupby("month")["amount"].sum().round(2)


def daily_average(df, month):
    data = df[df["month"] == month]
    if data.empty:
        return 0.0
    days = data["date"].dt.day.nunique()
    return round(float(np.sum(data["amount"]) / days), 2)


def budget_status(df, month, limit_amount):
    spent = float(df[df["month"] == month]["amount"].sum())
    remaining = limit_amount - spent
    used_pct = round(spent / limit_amount * 100, 1) if limit_amount else 0.0
    if used_pct >= 100:
        level = "EXCEEDED"
    elif used_pct >= 80:
        level = "WARNING"
    else:
        level = "SAFE"
    return {"spent": round(spent, 2), "remaining": round(remaining, 2),
            "used_percent": used_pct, "status": level}


def weekday_pattern(df):
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    totals = df.groupby(df["date"].dt.dayofweek)["amount"].sum()
    return totals.reindex(range(7), fill_value=0).rename(dict(enumerate(names)))
5. smartspend/predictor.py
"""predictor.py - simple machine learning forecast with Scikit-learn."""
import numpy as np
from sklearn.linear_model import LinearRegression


def predict_month_end(df, month):
    """Estimate total spending for the month from the cumulative spending trend.

    A Linear Regression model is fitted on (day of month -> cumulative spend)
    and used to predict the cumulative value on the last day of the month.
    """
    data = df[df["month"] == month].copy()
    if data["date"].dt.day.nunique() < 5:
        return None                                    # not enough data
    daily = data.groupby(data["date"].dt.day)["amount"].sum()
    days_in_month = data["date"].dt.days_in_month.iloc[0]
    last_day = int(data["date"].dt.day.max())
    daily = daily.reindex(range(1, last_day + 1), fill_value=0)
    X = np.array(daily.index).reshape(-1, 1)
    y = daily.cumsum().values
    model = LinearRegression().fit(X, y)
    forecast = float(model.predict([[days_in_month]])[0])
    return round(max(forecast, float(y[-1])), 2)


def saving_tips(df, month, limit_amount=None):
    """Rule based suggestions generated from the spending pattern."""
    data = df[df["month"] == month]
    tips = []
    if data.empty:
        return ["Add some expenses to receive saving tips."]
    total = data["amount"].sum()
    by_cat = data.groupby("category")["amount"].sum().sort_values(ascending=False)
    top, top_amt = by_cat.index[0], by_cat.iloc[0]
    share = top_amt / total * 100
    if share > 35:
        tips.append(f"{top} takes {share:.0f}% of your spending. "
                    "Try to reduce it by 10%.")
    for cat in ("Entertainment", "Shopping"):
        if cat in by_cat.index and by_cat[cat] / total > 0.20:
            tips.append(f"{cat} is above 20% of the total. "
                        "Consider setting a weekly cap.")
    if limit_amount and total > 0.8 * limit_amount:
        tips.append("You have used more than 80% of your monthly budget.")
    if not tips:
        tips.append("Your spending looks balanced. Keep tracking regularly.")
    return tips
6. smartspend/charts.py
"""charts.py - Matplotlib charts for the SmartSpend report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def category_bar_chart(summary, filename="category_chart.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary.index, summary.values, color="gray")
    ax.set_title("Category-wise Spending")
    ax.set_xlabel("Category")
    ax.set_ylabel("Amount (Rs.)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return filename


def monthly_line_chart(monthly, filename="monthly_chart.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color="black")
    ax.set_title("Monthly Spending Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (Rs.)")
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return filename
7. smartspend/sample_data.py
"""sample_data.py - generates sample student expenses for testing."""
import random
from datetime import date, timedelta
from .database import add_expense

NOTES = {
    "Food": ["canteen lunch", "evening tea", "dinner mess", "snack",
             "pizza with friends"],
    "Transport": ["bus ticket", "auto fare", "petrol"],
    "Education": ["notebook", "xerox and print", "stationery pen"],
    "Entertainment": ["movie night", "game top-up"],
    "Recharge": ["mobile recharge", "wifi pack"],
}


def load_sample(db_name, start=date(2026, 9, 1), days=60, seed=7):
    random.seed(seed)
    for i in range(days):
        d = start + timedelta(days=i)
        for _ in range(random.randint(1, 3)):
            cat = random.choices(list(NOTES), weights=[50, 20, 12, 10, 8])[0]
            note = random.choice(NOTES[cat])
            ranges = {"Food": (30, 180), "Transport": (10, 120),
                      "Education": (20, 250), "Entertainment": (100, 400),
                      "Recharge": (99, 299)}
            amount = random.randint(*ranges[cat])
            add_expense(cat, amount, note, d.isoformat(), db_name)
8. demo.py
"""demo.py - command line demo of SmartSpend (no Streamlit needed)."""
import os
from smartspend import database as db
from smartspend import analysis as an
from smartspend import predictor as pr
from smartspend import charts
from smartspend.categorizer import auto_category
from smartspend.sample_data import load_sample

DB = "smartspend_demo.db"


def main():
    if os.path.exists(DB):
        os.remove(DB)
    db.create_tables(DB)

    # CREATE: add an expense and let SmartSpend choose the category
    note = "canteen lunch"
    db.add_expense(auto_category(note), 120, note, "2026-08-31", DB)
    load_sample(DB)

    # READ
    df = an.to_dataframe(db.get_all_expenses(DB))
    print("Total records:", len(df))
    print(df.head())

    print("\nCategory summary")
    cat = an.category_summary(df)
    print(cat)

    print("\nMonthly summary")
    monthly = an.monthly_summary(df)
    print(monthly)

    month = "2026-09"
    db.set_budget(month, 8500, DB)
    print("\nDaily average:", an.daily_average(df, month))
    print("Budget status:", an.budget_status(df, month, db.get_budget(month, DB)))

    # AI part: forecast using only the first 15 days, as a mid-month check
    partial = df[(df["month"] == month) & (df["date"].dt.day <= 15)]
    print("\nForecast from first 15 days:", pr.predict_month_end(partial, month))
    print("Actual month total        :", float(monthly[month]))
    print("\nTips:")
    for t in pr.saving_tips(df, month, 8500):
        print("-", t)

    charts.category_bar_chart(cat, "docs/images/category_chart.png")
    charts.monthly_line_chart(monthly, "docs/images/monthly_chart.png")

    # UPDATE and DELETE
    db.update_expense(1, "Food", 150, "canteen lunch (edited)", DB)
    db.delete_expense(2, DB)
    print("\nRecords after update/delete:", len(db.get_all_expenses(DB)))


if __name__ == "__main__":
    main()
9. app.py
"""app.py - Streamlit interface for SmartSpend.  Run:  streamlit run app.py"""
import streamlit as st
from smartspend import database as db
from smartspend import analysis as an
from smartspend import predictor as pr
from smartspend.categorizer import auto_category
from datetime import date

db.create_tables()
st.title("SmartSpend - Student Expense Tracker")

menu = st.sidebar.radio(
    "Menu", ["Add Expense", "View / Delete", "Reports", "AI Insights"])

if menu == "Add Expense":
    st.header("Add Expense")
    exp_date = st.date_input("Date", date.today())
    note = st.text_input("Note (example: canteen lunch)")
    amount = st.number_input("Amount (Rs.)", min_value=0.0, step=10.0)
    suggested = auto_category(note)
    category = st.text_input("Category", value=suggested)
    if st.button("Save"):
        if amount > 0:
            db.add_expense(category, amount, note, exp_date.isoformat())
            st.success("Expense saved")
        else:
            st.error("Enter an amount greater than zero")

elif menu == "View / Delete":
    st.header("All Expenses")
    df = an.to_dataframe(db.get_all_expenses())
    st.dataframe(df)
    del_id = st.number_input("ID to delete", min_value=0, step=1)
    if st.button("Delete"):
        db.delete_expense(int(del_id))
        st.success("Deleted. Reload the page to refresh.")

elif menu == "Reports":
    st.header("Reports")
    df = an.to_dataframe(db.get_all_expenses())
    if df.empty:
        st.info("No data yet.")
    else:
        month = st.selectbox("Month", sorted(df["month"].unique()))
        limit_amount = st.number_input(
            "Monthly budget (Rs.)", min_value=0.0, value=3000.0)
        db.set_budget(month, limit_amount)
        st.write(an.budget_status(df, month, limit_amount))
        st.bar_chart(an.category_summary(df[df["month"] == month]))
        st.line_chart(an.monthly_summary(df))

elif menu == "AI Insights":
    st.header("AI Insights")
    df = an.to_dataframe(db.get_all_expenses())
    if df.empty:
        st.info("No data yet.")
    else:
        month = st.selectbox("Month", sorted(df["month"].unique()))
        forecast = pr.predict_month_end(df, month)
        st.write("Predicted month-end spending:",
                 forecast if forecast else "Not enough data")
        for tip in pr.saving_tips(df, month):
            st.write("- " + tip)
10. tests/test_smartspend.py
"""Automated tests for SmartSpend. Run with:  pytest"""
import pytest

from smartspend import analysis as an
from smartspend import database as db
from smartspend import predictor as pr
from smartspend.categorizer import auto_category


@pytest.fixture
def test_db(tmp_path):
    """A fresh, empty database file for every test."""
    path = str(tmp_path / "test.db")
    db.create_tables(path)
    return path


def test_categorizer():
    assert auto_category("canteen lunch") == "Food"
    assert auto_category("bus ticket") == "Transport"
    assert auto_category("xyz unknown") == "Other"
    assert auto_category("") == "Other"


def test_crud(test_db):
    db.add_expense("Food", 100, "lunch", "2026-09-01", test_db)
    rows = db.get_all_expenses(test_db)
    assert len(rows) == 1
    db.update_expense(rows[0][0], "Food", 150, "lunch edited", test_db)
    assert db.get_all_expenses(test_db)[0][3] == 150
    db.delete_expense(rows[0][0], test_db)
    assert db.get_all_expenses(test_db) == []


def test_invalid_amount(test_db):
    with pytest.raises(ValueError):
        db.add_expense("Food", -5, "bad", "2026-09-01", test_db)


def test_budget_roundtrip(test_db):
    assert db.get_budget("2026-09", test_db) is None
    db.set_budget("2026-09", 3000, test_db)
    assert db.get_budget("2026-09", test_db) == 3000


def test_budget_status(test_db):
    for d, a in [("2026-09-01", 400), ("2026-09-02", 450)]:
        db.add_expense("Food", a, "lunch", d, test_db)
    df = an.to_dataframe(db.get_all_expenses(test_db))
    assert an.budget_status(df, "2026-09", 1000)["status"] == "WARNING"
    assert an.budget_status(df, "2026-09", 800)["status"] == "EXCEEDED"
    assert an.budget_status(df, "2026-09", 5000)["status"] == "SAFE"


def test_forecast_needs_data(test_db):
    db.add_expense("Food", 100, "lunch", "2026-09-01", test_db)
    df = an.to_dataframe(db.get_all_expenses(test_db))
    assert pr.predict_month_end(df, "2026-09") is None


def test_forecast_never_below_spent(test_db):
    for day in range(1, 11):
        db.add_expense("Food", 100, "lunch", f"2026-09-{day:02d}", test_db)
    df = an.to_dataframe(db.get_all_expenses(test_db))
    assert pr.predict_month_end(df, "2026-09") >= 1000


def test_saving_tips_empty():
    df = an.to_dataframe([])
    assert pr.saving_tips(df, "2026-09") == ["Add some expenses to receive saving tips."]
11. requirements.txt
pandas
numpy
matplotlib
scikit-learn
streamlit
pytest

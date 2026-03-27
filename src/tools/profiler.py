# Data profiling tool 
import pandas as pd
from langchain_core.tools import tool
from src.tools.loader import _data_cache

EXPECTED_SCHEMA = {
    "order_number": "object",
    "line_item": "int64",
    "order_date": "object",
    "customer_code": "object",
    "region": "object",
    "product_code": "object",
    "product_name": "object",
    "category": "object",
    "sales_channel": "object",
    "volume": "int64",
    "unit_price": "float64",
    "discount_pct": "float64",
    "discount_amount": "float64",
    "gross_revenue": "float64",
    "revenue": "float64",
    "transaction_status": "object",
}

def _profile_numeric(df: pd.DataFrame) -> str:
    """Profile numeric columns"""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        return "No numeric columns found.\n"
    
    stats = df[numeric_cols].describe().round(2)
    return f"NUMERIC COLUMNS\n{stats.to_string()}\n"

def _profile_categorical(df: pd.DataFrame) -> str:
    """Profile categorical columns"""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not cat_cols:
        return "No categorical columns found.\n"
    
    report = "CATEGORICAL COLUMNS\n"
    for col in cat_cols:
        unique_count = df[col].nunique()
        top_values = df[col].value_counts().head(5)
        report += f"\n  {col}: {unique_count} unique values\n"

        for val, count in top_values.items():
            pct = count/len(df) * 100
            report += f"    {val}: {count} ({pct:.1f}%)\n"

    return report

def _profile_nulls(df: pd.DataFrame) -> str:
    """Profile null values."""
    nulls = df.isnull().sum()
    total = len(df)

    report = "NULL VALUES\n"
    has_nulls = False

    for col, count in nulls.items():
        if count > 0:
            has_nulls = True
            pct = count/total * 100
            report +=  f"  {col}: {count} ({pct:.1f}%)\n"

    if not has_nulls:
        report += "  No null values found.\n"
    return report

def _check_schema(df: pd.DataFrame) -> str:
    """Compare actual schema against expected."""
    report = "SCHEMA COMPARISON\n"
    actual = dict(df.dtypes)

    missing_cols = set(EXPECTED_SCHEMA.keys()) - set(actual.keys())
    extra_cols = set(actual.keys()) - set(EXPECTED_SCHEMA.keys())
    type_mismatches = []

    for col, expected_type in EXPECTED_SCHEMA.items():
        if col in actual:
            actual_type = str(actual[col])
            if actual_type != expected_type:
                type_mismatches.append(f"  {col}: expected {expected_type}, got {actual_type}")

    if not missing_cols and not extra_cols and not type_mismatches:
        report += " Schema matches expected definition.\n"
    else:
        if missing_cols:
            report += f"  Missing columns: {', '.join(missing_cols)}\n"
        if extra_cols:
            report += f"  Extra columns: {', '.join(extra_cols)}\n"
        if type_mismatches:
            report += "  Type mismatches:\n" + "\n".join(type_mismatches) + "\n"

    return report

@tool
def profile_data(date: str = None) -> str:
    """Generate a comprehensive profile of the loaded sales data.

    Provides row count, column types, value distributions, null analysis,
    and schema comparison against the expected definition.

    Args:
        date: Date key for data to profile (e.g., '2025-02-01' or
            '2025-02-01_to_2025-02-07'). Must match a key in loaded data.

    Returns:
        Comprehensive data profile report.
    """
    if not date:
        return "Please specify which data to profile. Provide a date (e.g., '2025-02-01') or date range (e.g., '2025-02-01_to_2025-02-07')."

    if date not in _data_cache:
        return f"No data found for '{date}'. Load it first using load_data or load_date_range."
    
    df = _data_cache[date]
    report = f"DATA PROFILE FOR {date}\n"
    report += "=" * 50 + "\n\n"

    #shape
    report += f"SHAPE\n"
    report += f"  Rows: {len(df)}\n"
    report += f"  Columns: {len(df.columns)}\n"
    report += f"  Column names: {', '.join(df.columns.tolist())}\n\n"

    #Date coverage
    if "order_date" in df.columns:
        report += f"DATE COVERAGE\n"
        report += f"  Min date: {df['order_date'].min()}\n"
        report += f"  Max date: {df['order_date'].max()}\n"
        report += f"  Distinct dates: {df['order_date'].nunique()}\n\n"

    # Schema comparison
    report += _check_schema(df) + "\n"

    # Null analysis
    report += _profile_nulls(df) + "\n"

    # Numeric stats
    report += _profile_numeric(df) + "\n"

    # Categorical distributions
    report += _profile_categorical(df)

    return report
    


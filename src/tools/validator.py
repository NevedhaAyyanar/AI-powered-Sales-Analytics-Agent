# Data quality validation tool 
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from src.tools.loader import _data_cache, _download_csv

NUMERIC_COLS = ["volume", "unit_price", "discount_pct", "discount_amount", "gross_revenue", "revenue"]

def _get_dim_product() -> pd.DataFrame:
    """Load dim_product from cache or Azure blob storage"""
    if "dim_product" not in _data_cache:
        try:
            _data_cache["dim_product"] = _download_csv(blob_path="dim_product.csv")
        except Exception:
            return None
    return _data_cache["dim_product"]

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to appropriate types, handling errors."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
        
def _check_revenue_mismatch(df: pd.DataFrame) -> dict:
    """Check: revenue != gross_revenue - discount_amount"""
    df_check = _coerce_numeric(df)
    df_check = df_check.dropna(subset=["gross_revenue", "discount_amount", "revenue"])
    df_check["expected_revenue"] = df_check["gross_revenue"] - df_check["discount_amount"]
    mismatches = df_check[~np.isclose(df_check["revenue"], df_check["expected_revenue"],atol=0.01)]
    
    return{
        "check": "Revenue Calculation Mismatch",
        "description": "revenue != gross_revenue - discount_amount",
        "issues_found": len(mismatches),
        "severity": "HIGH" if len(mismatches) > 0 else "PASS",
        "details": mismatches[["order_number", "line_item", "gross_revenue", "discount_amount", "revenue", "expected_revenue"]].head(10).to_string(index=False) if len(mismatches) > 0 else None,
        "remediation": "Recalculate revenue as gross_revenue - discount_amount"
    }

def _check_duplicate_keys(df: pd.DataFrame) -> dict:
    """Check: duplicate order number + line item combinations"""
    duplicate_check = df[df.duplicated(subset=["order_number", "line_item"], keep=False)]

    return{
        "check" : "Duplicate composite keys",
        "description": "Same order_number + line_item appears more than once",
        "issues_found": len(duplicate_check),
        "severity": "HIGH" if len(duplicate_check) >0 else "PASS",
        "details": duplicate_check[["order_number", "line_item", "product_code", "revenue"]].head(10).to_string(index=False) if len(duplicate_check) > 0 else None,
        "remediation": "Deduplicate rows or correct line_item numbering"
    }

def _check_line_item_gaps(df: pd.DataFrame) -> dict:
     """Check: line_item sequence gaps within orders"""
     issues = []
     for order, group in df.groupby("order_number"):
         items = sorted(group["line_item"].to_list())
         expected = list(range(1, len(items)+1))
         if items != expected:
             issues.append({
                 "order_number": order,
                 "line_items": items,
                 "expected": expected
             })

     details = None
     if issues:
         details_df = pd.DataFrame(issues[:10])
         details = details_df.to_string(index=False)

     return {
        "check": "Line Item Sequence Gaps",
        "description": "Line items not sequential (1, 2, 3...) within an order",
        "issues_found": len(issues),
        "severity": "MEDIUM" if len(issues) > 0 else "PASS",
        "details": details,
        "remediation": "Renumber line items sequentially within each order"
     }


def _check_gross_revenue_mismatch(df: pd.DataFrame) -> dict:
    """Check: gross_revenue != volume * unit_price"""
    df_check =_coerce_numeric(df)
    df_check = df_check.dropna(subset=["volume", "unit_price"])
    df_check["expected_gross"] = df_check["volume"] * df_check["unit_price"]
    mismatches = df_check[~np.isclose(df_check["gross_revenue"], df_check["expected_gross"], atol=0.01)]

    return {
        "check": "Gross Revenue Mismatch",
        "description": "gross_revenue != volume * unit_price",
        "issues_found": len(mismatches),
        "severity": "HIGH" if len(mismatches) > 0 else "PASS",
        "details": mismatches[["order_number", "line_item", "volume", "unit_price", "gross_revenue", "expected_gross"]].head(10).to_string(index=False) if len(mismatches) > 0 else None,
        "remediation": "Recalculate gross_revenue as volume * unit_price"
    }

def _check_discount_amount_mismatch(df: pd.DataFrame) -> dict:
    """Check: discount_amount != volume * unit_price * discount_pct / 100"""
    df_check = _coerce_numeric(df)
    df_check = df_check.dropna(subset=["volume", "unit_price", "discount_pct"])
    df_check["expected_discount"] = df_check["volume"] * df_check["unit_price"] * df_check["discount_pct"] / 100
    mismatches = df_check[~np.isclose(df_check["discount_amount"], df_check["expected_discount"], atol=0.01)]

    return {
        "check": "Discount Amount Mismatch",
        "description": "discount_amount != volume * unit_price * discount_pct / 100",
        "issues_found": len(mismatches),
        "severity": "MEDIUM" if len(mismatches) > 0 else "PASS",
        "details": mismatches[["order_number", "line_item", "volume", "unit_price", "discount_pct", "discount_amount", "expected_discount"]].head(10).to_string(index=False) if len(mismatches) > 0 else None,
        "remediation": "Recalculate discount_amount using the formula"
    }

def _check_product_consistency(df: pd.DataFrame) -> dict:
    """Check: product_code maps to different product_name vs dim_product"""
    dim = _get_dim_product()
    if dim is None:
        return{
            "check": "Product Code Consistency",
            "description": "product_code maps to correct product_name from dim_product",
            "issues_found": -1,
            "severity": "SKIPPED",
            "details": "Could not load dim_product.csv from Azure",
            "remediation": None
        }
    
    merged = df.merge(dim[["product_code","product_name"]], on = "product_code", how="left",suffixes=("_sales","_dim"))
    if "product_name_sales" in merged.columns and "product_name_dim" in merged.columns:
        mismatches = merged[merged["product_name_sales"] != merged["product_name_dim"]]
    else:
        mismatches = merged[merged["product_name"].isna()] if "product_name" in merged.columns else pd.DataFrame()

    return {

        "check": "Product Code Consistency",
        "description": "product_code maps to correct product_name from dim_product",
        "issues_found": len(mismatches),
        "severity": "MEDIUM" if len(mismatches) > 0 else "PASS",
        "details": mismatches[["order_number", "product_code"]].drop_duplicates().head(10).to_string(index=False) if len(mismatches) > 0 else None,
        "remediation": "Correct product_name to match dim_product lookup"
    }

def _check_null_values(df: pd.DataFrame) -> dict:
    """Check: null values in any column"""
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]

    details = None
    if len(nulls) > 0:
        details = "\n".join([f"  {col}: {count} nulls" for col, count in nulls.items()])

    return {
        "check": "Null Values",
        "description": "Missing values in columns",
        "issues_found": int(nulls.sum()) if len(nulls) > 0 else 0,
        "severity": "MEDIUM" if len(nulls) > 0 else "PASS",
        "details": details,
        "remediation": "Investigate and fill or remove null records"
    }

def _check_negative_values(df: pd.DataFrame) -> dict:
    """Check: negative values in numeric columns"""
    df_check = _coerce_numeric(df)

    issues = []
    for col in NUMERIC_COLS:
        if col in df.columns:
            negatives = df_check[df_check[col] < 0]
            if len(negatives) > 0:
                issues.append(f"  {col}: {len(negatives)} negative values")

    return {
        "check": "Negative Values",
        "description": "Negative values in numeric fields",
        "issues_found": len(issues),
        "severity": "HIGH" if len(issues) > 0 else "PASS",
        "details": "\n".join(issues) if issues else None,
        "remediation": "Review negative values — may indicate returns/credits or data errors"
    }

def _check_duplicate_rows(df: pd.DataFrame) -> dict:
    """Check: exact duplicate rows"""
    dupes = df[df.duplicated(keep=False)]

    return {
        "check": "Exact Duplicate Rows",
        "description": "Fully identical rows in the dataset",
        "issues_found": len(dupes),
        "severity": "HIGH" if len(dupes) > 0 else "PASS",
        "details": dupes[["order_number", "line_item", "product_code", "revenue"]].head(10).to_string(index=False) if len(dupes) > 0 else None,
        "remediation": "Remove exact duplicate rows"
    }

def _check_outliers(df: pd.DataFrame) -> dict:
    """Check: statistical outliers using IQR method"""
    df_check = _coerce_numeric(df)
    issues = []
    for col in NUMERIC_COLS:
        if col in df_check.columns:
            col_data = df_check[col].dropna()
            if len(col_data) == 0:
                continue
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df_check[(df_check[col] < lower) | (df_check[col] > upper)]
            if len(outliers) > 0:
                issues.append(f"  {col}: {len(outliers)} outliers (range: {lower:.2f} to {upper:.2f})")

    return {
        "check": "Statistical Outliers",
        "description": "Values outside 1.5x IQR range",
        "issues_found": len(issues),
        "severity": "LOW" if len(issues) > 0 else "PASS",
        "details": "\n".join(issues) if issues else None,
        "remediation": "Review outliers — may be valid high-volume orders or data errors"
    }

def _check_invalid_dates(df: pd.DataFrame) -> dict:
    """Check for unparseable dates in order_date."""
    parsed = pd.to_datetime(df["order_date"], format="mixed", errors="coerce")
    invalid = df[parsed.isna()]
    details = None
    if len(invalid) > 0:
        details = f"Unparseable dates ({len(invalid)} rows):\n" + invalid[["order_number", "line_item", "order_date"]].head(10).to_string(index=False)
    return {
        "check": "Invalid Dates",
        "description": "Unparseable dates in order_date",
        "issues_found": len(invalid),
        "severity": "HIGH" if len(invalid) > 0 else "PASS",
        "details": details,
        "remediation": "Fix invalid date values to a valid date format"
    }


_CHECK_MAP = {
    "revenue_mismatch": _check_revenue_mismatch,
    "duplicate_keys": _check_duplicate_keys,
    "line_item_gaps": _check_line_item_gaps,
    "gross_revenue_mismatch": _check_gross_revenue_mismatch,
    "discount_mismatch": _check_discount_amount_mismatch,
    "product_consistency": _check_product_consistency,
    "null_values": _check_null_values,
    "negative_values": _check_negative_values,
    "duplicate_rows": _check_duplicate_rows,
    "outliers": _check_outliers,
    "invalid_dates": _check_invalid_dates,
}

@tool
def validate_data(check_type: str = None, date: str = None) -> str:
    """Run data quality checks on the loaded sales data.

        Args:
        check_type: Optional check name. Valid values: revenue_mismatch,
            duplicate_keys, line_item_gaps, gross_revenue_mismatch,
            discount_mismatch, product_consistency, null_values,
            negative_values, duplicate_rows, outliers, invalid_dates
        date: Date key for data to validate (e.g., '2025-02-01' or
            '2025-02-01_to_2025-02-07'). Must match a key in loaded data.

        Returns:
        Data quality report with issues found, severity, and remediation suggestions."""
    
    if not date:
        return "Please specify which data to validate. Provide a date (e.g., '2025-02-01') or date range (e.g., '2025-02-01_to_2025-02-07'). Use load_data without a date to see available files."
    
    if date not in _data_cache:
        return f"No data found for '{date}'. Load it first using load_data or load_date_range."
    
    df = _data_cache[date]

    if check_type:
        if check_type not in _CHECK_MAP:
            return f"Unknown check: {check_type}. Valid checks: {', '.join(_CHECK_MAP.keys())}"
        results = [_CHECK_MAP[check_type](df)]
    else:
        results = [check_fn(df) for check_fn in _CHECK_MAP.values()]
    
    total_issues = sum(r["issues_found"] for r in results if r["issues_found"] > 0)
    report = f"DATA QUALITY REPORT for {date} ({len(results)} checks run)\n"
    report += f"Total issues found: {total_issues}\n"
    report += "=" * 50 + "\n\n"

    for r in results:
        status = "PASS" if r["severity"] == "PASS" else f"FAIL [{r['severity']}]"
        report += f"{r['check']}: {status}\n"
        if r["issues_found"] > 0:
            report += f"  Issues: {r['issues_found']}\n"
            report += f"  Description: {r['description']}\n"
            if r["details"]:
                report += f"  Sample:\n{r['details']}\n"
            if r["remediation"]:
                report += f"  Fix: {r['remediation']}\n"
        report += "\n"

    return report


@tool
def detect_anomalies(column: str = None, method: str = "iqr", date: str = None) -> str:
    """Find statistical outliers in the loaded sales data.

    Args:
        column: Optional column name to check. If not provided, checks all numeric columns.
            Valid columns: volume, unit_price, discount_pct, discount_amount, gross_revenue, revenue
        method: Detection method - 'iqr' (default) or 'zscore'
        date: Date key for data to analyze (e.g., '2025-02-01' or
            '2025-02-01_to_2025-02-07'). Must match a key in loaded data.

    Returns:
        List of anomalous rows with context.
    """
    if not date:
        return "Please specify which data to analyze. Provide a date (e.g., '2025-02-01') or date range (e.g., '2025-02-01_to_2025-02-07')."

    if date not in _data_cache:
        return f"No data found for '{date}'. Load it first using load_data or load_date_range."
    
    if method not in ("iqr","zscore"):
        return f"Invalid method: {method}. Valid methods: iqr, zscore"

    df = _coerce_numeric(_data_cache[date])    
    numeric_cols = ["volume", "unit_price", "discount_pct", "discount_amount", "gross_revenue", "revenue"]

    if column:
        if column not in numeric_cols:
            return f"Invalid column: {column}. Valid columns: {', '.join(numeric_cols)}"
        check_cols = [column]
    else:
        check_cols = numeric_cols

    report = f"ANOMALY DETECTION for {date} ({method.upper()} method)\n"
    report += "=" * 50 + "\n\n"

    for col in check_cols:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        if method == "zscore":
            mean = col_data.mean()
            std = col_data.std()
            if std == 0:
                continue
            z_scores = (df[col] - mean) / std
            outliers = df[z_scores.abs() > 3]
        else:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]

        report += f"{col}: {len(outliers)} anomalies detected\n"
        if len(outliers) > 0:
            report += outliers[["order_number", "line_item", col, "product_code", "region"]].head(10).to_string(index=False)
            report += "\n"
        report += "\n"

    return report
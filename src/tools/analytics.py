# Trend and segment analysis tools 
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from src.tools.loader import _data_cache
from src.tools.validator import _coerce_numeric

def _filter_by_status(df: pd.DataFrame, status: str = "Settled") -> pd.DataFrame:
    """Filter DataFrame by transaction status if specified."""
    if status and "transaction_status" in df.columns:
        return df[df["transaction_status"] == status].copy()
    return df.copy()

def _aggregate_by_period(df: pd.DataFrame, metric: str, period: str) -> pd.DataFrame:
    """Aggregate metric by time period"""

    VALID_METRICS = {"revenue", "volume", "orders"}
    if metric not in VALID_METRICS:
        return f"Invalid metric '{metric}'. Choose from: {', '.join(VALID_METRICS)}"
    
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", errors="coerce")

    if period == "weekly":
        df["period"] = df["order_date"].dt.strftime("%Y-W%U")
    else:
        df["period"] = df["order_date"].dt.date
    
    if metric == "revenue":
        agg = df.groupby("period")["revenue"].sum().reset_index()
    elif metric == "volume":
        agg = df.groupby("period")["volume"].sum().reset_index()
    elif metric == "orders":
        agg = df.groupby("period")["order_number"].nunique().reset_index()
        agg = agg.rename(columns = {"order_number": "orders"})

    agg = agg.sort_values("period").reset_index(drop=True)
    return agg


def _calculate_growth(agg: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Add growth rate and moving average columns."""
    agg = agg.copy()
    agg["growth_pct"] = agg[metric_col].pct_change() * 100
    agg["moving_avg_7d"] = agg[metric_col].rolling(window=7, min_periods=1).mean()
    return agg

def _detect_spikes(agg: pd.DataFrame, metric_col: str, threshold: float = 1.5) -> pd.DataFrame:
    """Detect spikes/dips relative to moving average."""
    agg = agg.copy()
    agg["ratio"] = agg[metric_col] / agg["moving_avg_7d"]
    spikes = agg[(agg["ratio"] > threshold) | (agg["ratio"] < 1 / threshold)]
    return spikes

def _revenue_leakage(df: pd.DataFrame) -> str:
    """Calculate revenue impact of data quality issues."""
    
    df = _coerce_numeric(df)
    df = df.dropna(subset=["volume", "unit_price", "gross_revenue", "discount_amount", "discount_pct", "revenue"])
    report = "REVENUE LEAKAGE ANALYSIS\n"
    total_revenue = df["revenue"].sum()

    expected_revenue = df["gross_revenue"] - df["discount_amount"]
    revenue_mismatch = df[~np.isclose(df["revenue"], expected_revenue, atol=0.01)]
    revenue_impact = abs(revenue_mismatch["revenue"].sum() - (revenue_mismatch["gross_revenue"] - revenue_mismatch["discount_amount"]).sum())

    expected_gross = df["volume"] * df["unit_price"]
    gross_mismatch = df[~np.isclose(df["gross_revenue"], expected_gross, atol=0.01)]
    gross_impact = abs(gross_mismatch["gross_revenue"].sum() - (gross_mismatch["volume"] * gross_mismatch["unit_price"]).sum())

    expected_discount = df["volume"] * df["unit_price"] * df["discount_pct"] / 100
    discount_mismatch = df[~np.isclose(df["discount_amount"], expected_discount, atol=0.01)]
    discount_impact = abs(discount_mismatch["discount_amount"].sum() - expected_discount[discount_mismatch.index].sum())

    duplicates = df[df.duplicated(keep=False)]
    duplicate_revenue = duplicates["revenue"].sum() - duplicates.drop_duplicates()["revenue"].sum()

    total_leakage = revenue_impact + gross_impact + discount_impact + duplicate_revenue
    pct_of_total = (total_leakage / total_revenue * 100) if total_revenue > 0 else 0

    report += f"  Total revenue: {total_revenue:,.2f}\n"
    report += f"  Revenue calculation errors: {revenue_impact:,.2f} ({len(revenue_mismatch)} rows)\n"
    report += f"  Gross revenue errors: {gross_impact:,.2f} ({len(gross_mismatch)} rows)\n"
    report += f"  Discount errors: {discount_impact:,.2f} ({len(discount_mismatch)} rows)\n"
    report += f"  Duplicate row impact: {duplicate_revenue:,.2f} ({len(duplicates)} rows)\n"
    report += f"  TOTAL LEAKAGE: {total_leakage:,.2f} ({pct_of_total:.2f}% of total revenue)\n"

    return report


@tool
def analyze_trends(metric: str = "revenue", period: str = "daily", date: str = None, status: str = "Settled") -> str:
    """Analyze time-series trends in the loaded sales data.

    Provides aggregations, growth rates, moving averages, spike detection,
    and revenue leakage analysis.

    Args:
        metric: Metric to analyze - 'revenue' (default), 'volume', or 'orders'
        period: Time period - 'daily' (default) or 'weekly'
        date: Date key for data to analyze (e.g., '2025-02-01' or
            '2025-02-01_to_2025-02-07'). Must match a key in loaded data.
        status: transaction status filter - 'Settled' (default)

    Returns:
        Trend analysis report with growth rates, moving averages, and spike detection.
    """

    if not date:
       return "Please specify which data to analyze. Provide a date (e.g., '2025-02-01') or date range (e.g., '2025-02-01_to_2025-02-07')."

    if date not in _data_cache:
        return f"No data found for '{date}'. Load it first using load_data or load_date_range."
    
    if metric not in ("revenue", "volume", "orders"):
        return f"Invalid metric: {metric}. Valid metrics: revenue, volume, orders"

    if period not in ("daily", "weekly"):
        return f"Invalid period: {period}. Valid periods: daily, weekly"

    if status and status not in ("Settled", "Unsettled"):
        return f"Invalid status: {status}. Valid values: Settled, Unsettled"
    
    df = _filter_by_status(_data_cache[date], status)
    df = _coerce_numeric(df)

    if len(df) == 0:
        return f"No {status} transactions found in the data."
    
    agg = _aggregate_by_period(df, metric, period)
    
    metric_col = "orders" if metric == "orders" else metric
    agg = _calculate_growth(agg, metric_col)

    status_label = f", {status} only" if status else ""
    report = f"TREND ANALYSIS for {date} ({metric}, {period}{status_label})\n"
    report += "=" * 50 + "\n\n"
  
    report += f"SUMMARY\n"
    report += f"  Total {metric}: {agg[metric_col].sum():,.2f}\n"
    report += f"  Average {period} {metric}: {agg[metric_col].mean():,.2f}\n"
    report += f"  Min: {agg[metric_col].min():,.2f}\n"
    report += f"  Max: {agg[metric_col].max():,.2f}\n"
    report += f"  Std Dev: {agg[metric_col].std():,.2f}\n\n"

    report += f"{period.upper()} BREAKDOWN\n"
    display_cols = ["period", metric_col, "growth_pct", "moving_avg_7d"]
    agg_display = agg[display_cols].copy()
    agg_display["growth_pct"] = agg_display["growth_pct"].round(1)
    agg_display["moving_avg_7d"] = agg_display["moving_avg_7d"].round(2)
    report += agg_display.to_string(index=False) + "\n\n"

    spikes = _detect_spikes(agg, metric_col)
    if len(spikes) > 0:
        report += f"SPIKES/DIPS DETECTED ({len(spikes)})\n"
        for _, row in spikes.iterrows():
            direction = "SPIKE" if row["ratio"] > 1 else "DIP"
            report += f"  {row['period']}: {direction} — {metric_col}={row[metric_col]:,.2f}, "
            report += f"moving avg={row['moving_avg_7d']:,.2f} ({row['ratio']:.2f}x)\n"
        report += "\n"
    else:
        report += "SPIKES/DIPS: None detected\n\n"

    if metric == "revenue":
        report += _revenue_leakage(df)

    return report

@tool
def segment_analysis(dimension: str, metric: str = "revenue", date: str = None, status: str = "Settled") -> str:
    """Analyze data by segments — slice and dice by any dimension.

    Provides grouped aggregations, rankings, percentage shares, and
    concentration analysis.

    Args:
        dimension: Dimension to segment by - 'region', 'sales_channel',
            'category', 'customer_code', or 'transaction_status'
        metric: Metric to analyze - 'revenue' (default), 'volume', 'orders', or 'discount'
        date: Date key for data to analyze (e.g., '2025-02-01' or
            '2025-02-01_to_2025-02-07'). Must match a key in loaded data.
        status: transaction status filter - 'Settled' (default)

    Returns:
        Segment analysis report with rankings and concentration metrics.
    """
    if not date:
        return "Please specify which data to analyze. Provide a date (e.g., '2025-02-01') or date range (e.g., '2025-02-01_to_2025-02-07')."

    if date not in _data_cache:
        return f"No data found for '{date}'. Load it first using load_data or load_date_range."

    valid_dims = ["region", "sales_channel", "category", "customer_code", "transaction_status"]
    if dimension not in valid_dims:
        return f"Invalid dimension: {dimension}. Valid dimensions: {', '.join(valid_dims)}"

    if metric not in ("revenue", "volume", "orders", "discount"):
        return f"Invalid metric: {metric}. Valid metrics: revenue, volume, orders, discount"

    if status and status not in ("Settled", "Unsettled"):
        return f"Invalid status: {status}. Valid values: Settled, Unsettled"

    df = _filter_by_status(_data_cache[date], status)
    df = _coerce_numeric(df)

    if len(df) == 0:
        return f"No {status} transactions found in the data."

    if dimension not in df.columns:
        return f"Column '{dimension}' not found in the data."
    
    if metric == "orders":
        agg = df.groupby(dimension)["order_number"].nunique().reset_index()
        agg = agg.rename(columns = {"order_number": "orders"})
        metric_col = "orders"
    elif metric == "discount":
        agg = df.groupby(dimension)["discount_amount"].sum().reset_index()
        metric_col = "discount_amount"
    else:
        agg = df.groupby(dimension)[metric].sum().reset_index()
        metric_col = metric
    
    agg = agg.sort_values(metric_col, ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    total = agg[metric_col].sum()
    agg["pct_share"] = (agg[metric_col]/total * 100).round(1)
    agg["cumulative_pct"] = agg["pct_share"].cumsum().round(1)

    status_label = f", {status} only" if status else ""
    report = f"SEGMENT ANALYSIS for {date} ({dimension} by {metric}{status_label})\n"
    report += "=" * 50 + "\n\n"

    report += "RANKINGS\n"
    display_cols = ["rank", dimension, metric_col, "pct_share", "cumulative_pct"]
    report += agg[display_cols].to_string(index=False) + "\n\n"

    top_n = min(2, len(agg))
    top_share = agg.head(top_n)["pct_share"].sum()
    report += "CONCENTRATION\n"
    report += f"  Top {top_n} {dimension}s account for {top_share:.1f}% of {metric}\n"

    above_80 = agg[agg["cumulative_pct"] <= 80]
    count_80 = len(above_80) + 1 if len(above_80) < len(agg) else len(above_80)
    report += f"  {count_80} of {len(agg)} {dimension}s drive ~80% of {metric}\n\n"

    df_copy = df.copy()
    df_copy["order_date"] = pd.to_datetime(df_copy["order_date"], format="mixed", errors="coerce")
    df_copy["week"] = df_copy["order_date"].dt.strftime("%Y-W%U")
    weeks = sorted(df_copy["week"].dropna().unique())

    if len(weeks) >= 2:
        first_week = weeks[0]
        last_week = weeks[-1]

        if metric == "orders":
            w1 = df_copy[df_copy["week"] == first_week].groupby(dimension)["order_number"].nunique()
            w2 = df_copy[df_copy["week"] == last_week].groupby(dimension)["order_number"].nunique()
        elif metric == "discount":
            w1 = df_copy[df_copy["week"] == first_week].groupby(dimension)["discount_amount"].sum()
            w2 = df_copy[df_copy["week"] == last_week].groupby(dimension)["discount_amount"].sum()
        else:
            w1 = df_copy[df_copy["week"] == first_week].groupby(dimension)[metric].sum()
            w2 = df_copy[df_copy["week"] == last_week].groupby(dimension)[metric].sum()

        wow = pd.DataFrame({"week_first": w1, "week_last": w2}).fillna(0)
        wow["change_pct"] = ((wow["week_last"] - wow["week_first"]) / wow["week_first"] * 100).round(1)
        wow = wow.replace([np.inf, -np.inf], 0)

        report += f"WEEK-OVER-WEEK (Week {first_week} vs Week {last_week})\n"
        report += wow.to_string() + "\n"

    return report
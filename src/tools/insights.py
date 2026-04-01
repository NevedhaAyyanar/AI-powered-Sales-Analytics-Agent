# Product and customer analysis tools 
import pandas as pd
from itertools import combinations
from langchain_core.tools import tool
from src.tools.loader import _data_cache, _download_csv

def _get_dim_product() -> pd.DataFrame:
    if "dim_product" not in _data_cache:
        _data_cache["dim_product"] = _download_csv("dim_product.csv")
    return _data_cache["dim_product"]

def _get_dim_customer() -> pd.DataFrame:
    if "dim_customer" not in _data_cache:
        _data_cache["dim_customer"] = _download_csv("dim_customer.csv")
    return _data_cache["dim_customer"]

def _filter_by_status(df: pd.DataFrame, status: str = "Settled") -> pd.DataFrame:
    if status and "transaction_status" in df.columns:
        return df[df["transaction_status"] == status].copy()
    return df.copy()


@tool
def analyze_products(date: str = None, status: str = "Settled") -> str:
    """Analyze product performance with enriched dimension data.

    Provides revenue/volume rankings, price compliance checks against
    min/max price bands, and discount depth analysis by product.

    Args:
        date: Date key for data to analyze (e.g., '2025-02-01' or
            '2025-02-01_to_2025-02-07'). Must match a key in loaded data.
        status: Transaction status filter - 'Settled' (default) or 'Unsettled'

    Returns:
        Product performance report with rankings, price flags, and discount analysis.
    """
    if not date:
        return "Please specify a date or date range to analyze."

    if date not in _data_cache:
        return f"No data found for '{date}'. Load it first using load_data or load_date_range."

    if status and status not in ("Settled", "Unsettled"):
        return f"Invalid status: {status}. Valid values: Settled, Unsettled"

    df = _filter_by_status(_data_cache[date], status)
    if len(df) == 0:
        return f"No {status} transactions found."
    
    dim_product = _get_dim_product()
    merged = df.merge(dim_product, on="product_code", how="left")

    report = f"PRODUCT ANALYSIS for {date} ({status} only)\n"
    report += "=" * 50 + "\n\n"


    # Revenue & Volume Rankings by Product
    product_aggregation = merged.groupby(["product_code", "product_name", "category"]).agg(
        total_revenue = ("revenue","sum"),
        total_volume = ("volume","sum"),
        order_count = ("order_number", "nunique"),
        average_discount_percent = ("discount_pct", "mean")).reset_index()
    
    product_aggregation = product_aggregation.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    product_aggregation["rank"] = range(1, len(product_aggregation)+1)

    report += "TOP 10 PRODUCTS BY REVENUE\n"
    top10 = product_aggregation.head(10)[["rank", "product_code", "product_name", "category",
                                "total_revenue", "total_volume", "order_count"]]
    report += top10.to_string(index=False) + "\n\n"

    report += "BOTTOM 5 PRODUCTS BY REVENUE\n"
    bottom5 = product_aggregation.tail(5)[["rank", "product_code", "product_name", "total_revenue", "total_volume"]]
    report += bottom5.to_string(index=False) + "\n\n"

    # Price Compliance Check against min/max price bands
    price_check = merged[["product_code", "product_name", "unit_price", "min_price", "max_price"]].copy()
    violations = price_check[
        (price_check["unit_price"] < price_check["min_price"]) |
        (price_check["unit_price"] > price_check["max_price"])]
    
    if len(violations) > 0:
        report += f"PRICE VIOLATIONS ({len(violations)} rows)\n"
        violation_summary = violations.groupby(["product_code", "product_name", "min_price", "max_price"]).agg(
            violation_count=("unit_price", "count"),
            min_actual=("unit_price", "min"),
            max_actual=("unit_price", "max")
        ).reset_index()
        report += violation_summary.to_string(index=False) + "\n\n"
    else:
        report += "PRICE VIOLATIONS: None — all prices within min/max bands\n\n"

    # Discount Depth Analysis by Category
    report += "DISCOUNT DEPTH BY CATEGORY\n"
    disc_by_cat = merged.groupby("category").agg(
        avg_discount_pct=("discount_pct", "mean"),
        max_discount_pct=("discount_pct", "max"),
        total_discount_amount=("discount_amount", "sum"),
        total_revenue=("revenue", "sum")
    ).reset_index()

    disc_by_cat["discount_to_revenue_pct"] = (
        disc_by_cat["total_discount_amount"] / (disc_by_cat["total_revenue"] + disc_by_cat["total_discount_amount"]) * 100
    ).round(1)
    disc_by_cat["avg_discount_pct"] = disc_by_cat["avg_discount_pct"].round(1)
    disc_by_cat = disc_by_cat.sort_values("discount_to_revenue_pct", ascending=False)
    report += disc_by_cat.to_string(index=False) + "\n\n"


    # Heaviest Discounted Products
    report += "TOP 5 MOST DISCOUNTED PRODUCTS\n"
    disc_by_prod = product_aggregation.sort_values("average_discount_percent", ascending=False).head(5)
    report += disc_by_prod[["product_code", "product_name", "average_discount_percent",
                             "total_revenue", "total_volume"]].to_string(index=False) + "\n"

    return report

@tool
def analyze_customers(date: str = None, status: str = "Settled") -> str:
    """Analyze customer behavior with enriched dimension data.

    Provides revenue rankings, order frequency, average order value,
    settlement behavior, and Pareto analysis by customer.

    Args:
        date: Date key for data to analyze (e.g., '2025-02-01' or
            '2025-02-01_to_2025-02-07'). Must match a key in loaded data.
        status: Transaction status filter - 'Settled' (default) or 'Unsettled'

    Returns:
        Customer analysis report with rankings, frequency, and risk indicators.
    """
    if not date:
        return "Please specify a date or date range to analyze."

    if date not in _data_cache:
        return f"No data found for '{date}'. Load it first using load_data or load_date_range."

    if status and status not in ("Settled", "Unsettled"):
        return f"Invalid status: {status}. Valid values: Settled, Unsettled"

    df = _filter_by_status(_data_cache[date], status)
    if len(df) == 0:
        return f"No {status} transactions found."

    dim_customer = _get_dim_customer()
    full_df = _data_cache[date]

    report = f"CUSTOMER ANALYSIS for {date} ({status} only)\n"
    report += "=" * 50 + "\n\n"
    
    merged = df.merge(dim_customer, on="customer_code", how="left")
    cust_agg = merged.groupby(["customer_code", "customer_name", "region"]).agg(
        total_revenue=("revenue", "sum"),
        order_count=("order_number", "nunique"),
        total_volume=("volume", "sum"),
        line_item_count=("line_item", "count")
    ).reset_index()

    cust_agg["avg_order_value"] = (cust_agg["total_revenue"] / cust_agg["order_count"]).round(2)
    cust_agg = cust_agg.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    cust_agg["rank"] = range(1, len(cust_agg) + 1)

    total_revenue = cust_agg["total_revenue"].sum()
    cust_agg["pct_share"] = (cust_agg["total_revenue"] / total_revenue * 100).round(1)
    cust_agg["cumulative_pct"] = cust_agg["pct_share"].cumsum().round(1)

    report += "TOP 10 CUSTOMERS BY REVENUE\n"
    top10 = cust_agg.head(10)[["rank", "customer_code", "customer_name", "region",
                                "total_revenue", "order_count", "avg_order_value", "pct_share"]]
    report += top10.to_string(index=False) + "\n\n"

    # Pareto Analysis: How many customers drive 80% of revenue?
    above_80 = cust_agg[cust_agg["cumulative_pct"] <= 80]
    count_80 = len(above_80) + 1 if len(above_80) < len(cust_agg) else len(above_80)
    report += "PARETO ANALYSIS\n"
    report += f"  {count_80} of {len(cust_agg)} customers drive ~80% of revenue\n"
    report += f"  Top 1 customer: {cust_agg.iloc[0]['pct_share']:.1f}% of total revenue\n\n"

   # Order Frequency Distribution 
    freq_dist = cust_agg["order_count"].describe()
    report += "ORDER FREQUENCY\n"
    report += f"  Avg orders per customer: {freq_dist['mean']:.1f}\n"
    report += f"  Max orders: {freq_dist['max']:.0f}\n"
    report += f"  Customers with 1 order: {len(cust_agg[cust_agg['order_count'] == 1])}\n"
    report += f"  Customers with 5+ orders: {len(cust_agg[cust_agg['order_count'] >= 5])}\n\n"

    # Settlement Behavior (uses full unfiltered data)
    settlement = full_df.groupby("customer_code").agg(
        total_orders=("order_number", "nunique"),
        total_revenue=("revenue", "sum")
    ).reset_index()

    settled_df = full_df[full_df["transaction_status"] == "Settled"]
    settled_orders = settled_df.groupby("customer_code")["order_number"].nunique().reset_index()
    settled_orders.columns = ["customer_code", "settled_orders"]

    settlement = settlement.merge(settled_orders, on="customer_code", how="left")
    settlement["settled_orders"] = settlement["settled_orders"].fillna(0)
    settlement["settlement_rate"] = (settlement["settled_orders"] / settlement["total_orders"] * 100).round(1)
    risky = settlement[settlement["settlement_rate"] < 70].sort_values("total_revenue", ascending=False)

    if len(risky) > 0:
        risky_enriched = risky.merge(dim_customer, on="customer_code", how="left")
        report += f"SETTLEMENT RISK ({len(risky)} customers with <70% settlement rate)\n"
        display = risky_enriched[["customer_code", "customer_name", "total_orders",
                                   "settled_orders", "settlement_rate", "total_revenue"]].head(10)
        report += display.to_string(index=False) + "\n\n"
    else:
        report += "SETTLEMENT RISK: All customers above 70% settlement rate\n\n"

    # Regional Summary
    report += "REVENUE BY REGION\n"
    region_agg = merged.groupby("region").agg(
        revenue=("revenue", "sum"),
        customers=("customer_code", "nunique"),
        orders=("order_number", "nunique")
    ).reset_index()
    region_agg["revenue_per_customer"] = (region_agg["revenue"] / region_agg["customers"]).round(2)
    region_agg = region_agg.sort_values("revenue", ascending=False)
    report += region_agg.to_string(index=False) + "\n"

    return report

@tool
def analyze_basket(date: str = None, status: str = "Settled") -> str:
    """Analyze order baskets — product co-occurrence and basket composition.

    Identifies which products are frequently bought together, average
    basket sizes, and category mix patterns.

    Args:
        date: Date key for data to analyze (e.g., '2025-02-01' or
            '2025-02-01_to_2025-02-07'). Must match a key in loaded data.
        status: Transaction status filter - 'Settled' (default) or 'Unsettled'

    Returns:
        Basket analysis report with co-purchase patterns and basket metrics.
    """
    if not date:
        return "Please specify a date or date range to analyze."

    if date not in _data_cache:
        return f"No data found for '{date}'. Load it first using load_data or load_date_range."

    if status and status not in ("Settled", "Unsettled"):
        return f"Invalid status: {status}. Valid values: Settled, Unsettled"

    df = _filter_by_status(_data_cache[date], status)
    if len(df) == 0:
        return f"No {status} transactions found."

    dim_product = _get_dim_product()

    report = f"BASKET ANALYSIS for {date} ({status} only)\n"
    report += "=" * 50 + "\n\n"

    df_with_cat = df.merge(dim_product[["product_code", "category"]], on="product_code", how="left")

    basket = df_with_cat.groupby("order_number").agg(
        items=("line_item", "count"),
        unique_products=("product_code", "nunique"),
        revenue=("revenue", "sum"),
        categories=("category", "nunique")).reset_index()
    
    report += "BASKET SIZE METRICS\n"
    report += f"  Total orders: {len(basket)}\n"
    report += f"  Avg items per order: {basket['items'].mean():.1f}\n"
    report += f"  Avg unique products per order: {basket['unique_products'].mean():.1f}\n"
    report += f"  Avg revenue per order: {basket['revenue'].mean():.2f}\n"
    report += f"  Avg categories per order: {basket['categories'].mean():.1f}\n\n"

    # Basket Size Distribution 
    size_dist = basket["items"].value_counts().sort_index()
    report += "BASKET SIZE DISTRIBUTION\n"
    for size, count in size_dist.items():
        report += f"  {size} items: {count} orders ({count/len(basket)*100:.1f}%)\n"
    report += "\n"

    # Product Co-occurrence Analysis
    order_products = df.groupby("order_number")["product_code"].apply(set).reset_index()
    pair_counts = {}

    for _, row in order_products.iterrows():
        products = sorted(row["product_code"])
        if len(products) >= 2:
            for p1, p2 in combinations(products, 2):
                pair = (p1, p2)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if pair_counts:
        pairs_df = pd.DataFrame([
            {"product_1": k[0], "product_2": k[1], "co_occurrence": v}
            for k, v in pair_counts.items()
        ])
        pairs_df = pairs_df.sort_values("co_occurrence", ascending=False).reset_index(drop=True)

        # Enrich with product names
        name_map = dict(zip(dim_product["product_code"], dim_product["product_name"]))
        pairs_df["product_1_name"] = pairs_df["product_1"].map(name_map)
        pairs_df["product_2_name"] = pairs_df["product_2"].map(name_map)

        total_orders = len(order_products)
        pairs_df["frequency_pct"] = (pairs_df["co_occurrence"] / total_orders * 100).round(1)

        report += "TOP 15 PRODUCT PAIRS (bought together)\n"
        top_pairs = pairs_df.head(15)[["product_1_name", "product_2_name",
                                        "co_occurrence", "frequency_pct"]]
        report += top_pairs.to_string(index=False) + "\n\n"
    else:
        report += "PRODUCT PAIRS: No multi-product orders found\n\n"


    # Category Mix
    merged = df.merge(dim_product[["product_code", "category"]], on="product_code", how="left")
    cat_per_order = merged.groupby("order_number")["category"].apply(
        lambda x: frozenset(x.unique())
    ).reset_index()

    cat_combo_counts = cat_per_order["category"].value_counts().head(10)
    report += "TOP 10 CATEGORY COMBINATIONS PER ORDER\n"
    for combo, count in cat_combo_counts.items():
        cats = ", ".join(sorted(combo))
        report += f"  {cats}: {count} orders ({count/len(cat_per_order)*100:.1f}%)\n"

    return report
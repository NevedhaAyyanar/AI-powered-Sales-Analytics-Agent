# Azure Blob Storage data loader 
import pandas as pd
from io import StringIO
from azure.storage.blob import BlobServiceClient
from langchain_core.tools import tool
from src.config import AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME

#In-memory cache to avoid re-downloading the same files
_data_cache = {}

def _get_container_client():
    """Create and return Azure Blob container client"""
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    return blob_service.get_container_client(AZURE_CONTAINER_NAME)

def _download_csv(blob_path: str) -> pd.DataFrame:
    """Download CSV from Azure Blob Storage and return as DataFrame"""
    container = _get_container_client()
    blob_client = container.get_blob_client(blob_path)
    csv_content = blob_client.download_blob().readall().decode("utf-8")
    return pd.read_csv(StringIO(csv_content))

@tool
def load_data(date: str=None) -> str:
    """Load sales data from Azure blob storage. 
    If date is provided (format: YYYY-MM-DD),load the specific day's file. 
    If date is not provided, list all available files in blob storage.
    Args: date: optional date string in YYYY-MM-DD format (e.g., "2023-01-01")
    Returns: Summary of loaded data or list of available files."""
    
    if not date:
        # List all blobs in the container
        container = _get_container_client()
        blobs = container.list_blobs(name_starts_with="daily_sales/")
        files = sorted([
            blob.name.split("/")[-1]
            for blob in blobs
            if blob.name.endswith(".csv")
        ])
        return f"Available files ({len(files)}):\n" + "\n".join(files)
    
    year = date[:4]
    month = date[5:7]
    filename = f"sales_{date}.csv"
    blob_path = f"daily_sales/{year}/{month}/{filename}"

    if date in _data_cache:
        df = _data_cache[date]
    else:
        try:
            df = _download_csv(blob_path=blob_path)
            _data_cache[date] = df
        except Exception as e:
            return f"Could not load {filename}: {str(e)}. Use load_data without a date to see available files."
        
    summary =(
        f"Loaded {filename} from Azure Blob Storage.\n"
        f"Order Date range: {df['order_date'].min()} to {df['order_date'].max()}\n"
        f"Total orders: {df['order_number'].nunique()}\n"
        f"Total revenue: {df['revenue'].sum():,.2f}\n"
        f"Channels: {', '.join(df['sales_channel'].unique())}\n"
    )

    return summary

@tool
def load_date_range(start_date: str, end_date:str) -> str:
        """Load sales data for a range of dates and combine into one dataset.

    Args:
        start_date: Start date in YYYY-MM-DD format (e.g., '2025-02-01')
        end_date: End date in YYYY-MM-DD format (e.g., '2025-02-07')

    Returns:
        Summary of the combined dataset.
    """
        range_key = f"{start_date}_to_{end_date}"
        loaded_frames = []
        missing_files = []

        if range_key in _data_cache:
            combined = _data_cache[range_key]
        else:
            date_range = pd.date_range(start=start_date, end=end_date)
            
            for single_date in date_range:
                date_str = single_date.strftime("%Y-%m-%d")
                year = date_str[:4]
                month = date_str[5:7]
                filename = f"sales_{date_str}.csv"
                blob_path = f"daily_sales/{year}/{month}/{filename}"

                if date_str in _data_cache:
                    loaded_frames.append(_data_cache[date_str])
                else:
                    try:
                        df = _download_csv(blob_path=blob_path)
                        _data_cache[date_str] = df
                        loaded_frames.append(df)
                    except Exception:
                        missing_files.append(filename)
        
            if not loaded_frames:
                return f"No files found for the date range {start_date} to {end_date}."
        
            combined = pd.concat(loaded_frames, ignore_index=True)
            _data_cache[range_key] = combined

        summary = (
        f"Data loaded from {start_date} to {end_date}\n"
        f"Order Date range: {combined['order_date'].min()} to {combined['order_date'].max()}\n"
        f"Total orders: {combined['order_number'].nunique()}\n"
        f"Total revenue: {combined['revenue'].sum():,.2f}\n"
        f"Channels: {', '.join(combined['sales_channel'].unique())}"
        )

        if missing_files:
            summary += f"\n\nMissing files: {', '.join(missing_files)}"

        return summary















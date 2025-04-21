import logging
import re
import json
import os
import sys
from concurrent import futures
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import jdatetime
import pandas as pd
from pytse_client import config, symbols_data, translations, tse_settings
from pytse_client.scraper.symbol_scraper import MarketSymbol
from pytse_client.tse_settings import TSE_CLIENT_TYPE_DATA_URL
from pytse_client.utils import persian, requests_retry_session
from requests import HTTPError
from requests.sessions import Session
from tenacity import retry, retry_if_exception_type, wait_random
from tenacity.before_sleep import before_sleep_log

logger = logging.getLogger(config.LOGGER_NAME)


def _handle_ticker_index(symbol):
    ticker_index = symbols_data.get_ticker_index(symbol)

    if ticker_index is None:
        market_symbol = get_symbol_info(symbol)
        if market_symbol is not None:
            symbols_data.append_symbol_to_file(market_symbol)
            ticker_index = market_symbol.index
    return ticker_index


def _extract_ticker_client_types_data(ticker_index: str) -> List:
    url = TSE_CLIENT_TYPE_DATA_URL.format(ticker_index)
    with requests_retry_session() as session:
        response = session.get(url, timeout=5)
    data = response.text.split(";")
    data = [row.split(",") for row in data]
    return data


def _create_financial_index_from_text_response(data):
    data = pd.DataFrame(re.split(r"\;", data))
    columns = ["date", "high", "low", "open", "close", "volume", "__"]
    data[columns] = data[0].str.split(",", expand=True)
    return data.drop(columns=["__", 0])


def _adjust_data_frame(df, include_jdate):
    df.date = pd.to_datetime(df.date, format="%Y%m%d")
    if include_jdate:
        df["jdate"] = ""
        df.jdate = df.date.apply(
            lambda gregorian: jdatetime.date.fromgregorian(date=gregorian)
        )


def _adjust_data_frame_for_fIndex(df, include_jdate):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    if include_jdate:
        df["jdate"] = df.date.apply(
            lambda gregorian: jdatetime.date.fromgregorian(date=gregorian)
        )


def download(
    symbols: Union[List, str],
    write_to_csv: bool = False,
    include_jdate: bool = False,
    base_path: str = config.DATA_BASE_PATH,
    adjust: bool = False,
) -> Dict[str, pd.DataFrame]:
    if symbols == "all":
        symbols = symbols_data.all_symbols()
    elif isinstance(symbols, str):
        symbols = [symbols]

    df_list = {}
    future_to_symbol = {}
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        session = requests_retry_session()
        for symbol in symbols:
            if (
                symbol.isnumeric()
                and symbols_data.get_ticker_index(symbol) is None
            ):
                ticker_indexes = [symbol]
            else:
                ticker_index = _handle_ticker_index(symbol)
                if ticker_index is None:
                    raise Exception(f"Cannot find symbol: {symbol}")
                ticker_indexes = symbols_data.get_ticker_old_index(symbol)
                ticker_indexes.insert(0, ticker_index)

            for index in ticker_indexes:
                future = executor.submit(
                    download_ticker_daily_record, index, session
                )

                future_to_symbol[future] = symbol

        for future in futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df: pd.DataFrame = future.result()
            except pd.errors.EmptyDataError as ex:
                logger.error(
                    f"Cannot read daily trade records for symbol: {symbol}",
                    extra={"Error": ex},
                )
                continue
            df = df.iloc[::-1].reset_index(drop=True)
            df = df.rename(columns=translations.HISTORY_FIELD_MAPPINGS)
            df = df.drop(columns=["<PER>", "<TICKER>"])
            _adjust_data_frame(df, include_jdate)

            if symbol in df_list:
                df_list[symbol] = (
                    pd.concat(
                        [df_list[symbol], df], ignore_index=True, sort=False
                    )
                    .sort_values("date")
                    .reset_index(drop=True)
                )
            else:
                df_list[symbol] = df

            if adjust:
                df_list[symbol] = adjust_price(df_list[symbol])

            if write_to_csv:
                Path(base_path).mkdir(parents=True, exist_ok=True)
                if adjust:
                    df_list[symbol].to_csv(
                        f"{base_path}/{symbol}-ت.csv", index=False
                    )
                else:
                    df_list[symbol].to_csv(
                        f"{base_path}/{symbol}.csv", index=False
                    )

    if len(df_list) != len(symbols):
        print("Warning, download did not complete, re-run the code")
    session.close()
    return df_list


def adjust_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust historical records of stock

    There is a capital increase/profit sharing,
    if today "Final Close Price" is not equal to next day
    "Yesterday Final Close Price" by using this ratio,
    performance adjustment of stocks is achieved

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with historical records.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted historical records.

    Notes
    -----
    DataFrame can not be empty or else it makes runtime error
    Type of DataFrame must be RangeIndex to make proper range of records
    that need to be modified

    diff: list
        list of indexs of the day after capital increase/profit sharing
    ratio_list: List
        List of ratios to adjust historical data of stock
    ratio: Float
        ratio = df.loc[i].adjClose / df.loc[i+1].yesterday

    Description
    -----------
    # Note: adjustment does not include Tenth and twentieth days
    df.index = range(0,101,1)
    # step is 1
    step = df.index.step
    diff = [10,20]
    ratio_list = [0.5, 0.8]
    df.loc[0:10-step, [open,...]] * ratio[0]
    df.loc[10:20-step, [open,...]] * ratio[1]
    """
    if df.empty or not isinstance(df.index, pd.core.indexes.range.RangeIndex):
        return df

    new_df = df.copy()
    step = new_df.index.step
    diff = list(new_df.index[new_df.shift(1).adjClose != new_df.yesterday])
    if len(diff) > 0:
        diff.pop(0)
    ratio = 1
    ratio_list = []
    for i in diff[::-1]:
        ratio *= (
            new_df.loc[i, "yesterday"] / new_df.shift(1).loc[i, "adjClose"]
        )
        ratio_list.insert(0, ratio)
    for i, k in enumerate(diff):
        if i == 0:
            start = new_df.index.start
        else:
            start = diff[i - 1]
        end = diff[i] - step
        new_df.loc[
            start:end,
            [
                "open",
                "high",
                "low",
                "close",
                "adjClose",
                "yesterday",
            ],
        ] = round(
            new_df.loc[
                start:end,
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjClose",
                    "yesterday",
                ],
            ]
            * ratio_list[i]
        )

    return new_df


@retry(
    retry=retry_if_exception_type(HTTPError),
    wait=wait_random(min=1, max=4),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def download_ticker_daily_record(ticker_index: str, session: Session):
    url = tse_settings.TSE_TICKER_EXPORT_DATA_ADDRESS.format(ticker_index)
    response = session.get(url, timeout=10)
    if 400 <= response.status_code < 500:
        logger.error(
            f"Cannot read daily trade records from the url: {url}",
            extra={"response": response.text, "code": response.status_code},
        )

    response.raise_for_status()

    data = StringIO(response.text)
    return pd.read_csv(data)


@retry(
    retry=retry_if_exception_type(HTTPError),
    wait=wait_random(min=1, max=4),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def download_fIndex_record(fIndex: str, session: Session):
    url = tse_settings.TSE_FINANCIAL_INDEX_EXPORT_DATA_ADDRESS.format(fIndex)
    response = session.get(url, timeout=10)
    if 400 <= response.status_code < 500:
        logger.error(
            f"Cannot read daily trade records from the url: {url}",
            extra={"response": response.text, "code": response.status_code},
        )

    response.raise_for_status()
    data = response.text
    if not data or ";" not in data or "," not in data:
        raise ValueError(
            f"""Invalid response from the url: {url}.
                         \nExpected valid financial index data."""
        )
    df = _create_financial_index_from_text_response(data)
    return df


def download_financial_indexes(
    symbols: Union[List, str],
    write_to_csv: bool = False,
    include_jdate: bool = False,
    base_path: str = config.FINANCIAL_INDEX_BASE_PATH,
) -> Dict[str, pd.DataFrame]:
    if symbols == "all":
        symbols = symbols_data.all_financial_index()
    elif isinstance(symbols, str):
        symbols = [symbols]

    df_list = {}
    future_to_symbol = {}
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        session = requests_retry_session()
        for symbol in symbols:
            symbol = persian.replace_persian(symbol)
            if (
                symbol.isnumeric()
                and symbols_data.get_financial_index(symbol) is None
            ):
                financial_index = symbol
            else:
                financial_index = symbols_data.get_financial_index(symbol)
            if financial_index is None:
                raise Exception(f"Cannot find financial index: {symbol}")

            future = executor.submit(
                download_fIndex_record, financial_index, session
            )

            future_to_symbol[future] = symbol

        for future in futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df: pd.DataFrame = future.result()
            except pd.errors.EmptyDataError as ex:
                logger.error(
                    f"Cannot read daily trade records for symbol: {symbol}",
                    extra={"Error": ex},
                )
                continue
            _adjust_data_frame_for_fIndex(df, include_jdate)
            df_list[symbol] = df

            if write_to_csv:
                Path(base_path).mkdir(parents=True, exist_ok=True)
                df_list[symbol].to_csv(
                    f"{base_path}/{symbol}.csv", index=False
                )

    if len(df_list) != len(symbols):
        print("Warning, download did not complete, re-run the code")
    session.close()
    return df_list


def download_client_types_records(
    symbols: Union[List, str],
    write_to_csv: bool = False,
    include_jdate: bool = False,
    base_path: str = config.CLIENT_TYPES_DATA_BASE_PATH,
):
    if symbols == "all":
        symbols = symbols_data.all_symbols()
    elif isinstance(symbols, str):
        symbols = [symbols]

    df_list = {}
    future_to_symbol = {}
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        for symbol in symbols:
            ticker_index = _handle_ticker_index(symbol)
            if ticker_index is None:
                raise Exception(f"Cannot find symbol: {symbol}")
            future = executor.submit(
                download_ticker_client_types_record, ticker_index
            )
            future_to_symbol[future] = symbol
        for future in futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            df: pd.DataFrame = future.result()
            # ignore failures
            if df is None:
                continue
            _adjust_data_frame(df, include_jdate)
            df_list[symbol] = df
            if write_to_csv:
                Path(base_path).mkdir(parents=True, exist_ok=True)
                df.to_csv(f"{base_path}/{symbol}.csv")

    if len(df_list) != len(symbols):
        print(
            """could not download client types for all the symbols make
        sure you have what you need or re-run the function"""
        )
    return df_list


@retry(
    retry=retry_if_exception_type(HTTPError),
    wait=wait_random(min=1, max=4),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def download_ticker_client_types_record(ticker_index: Optional[str]):
    data = _extract_ticker_client_types_data(ticker_index)
    if len(data) == 1:
        logger.warning(
            f"""Cannot create client types data for ticker{ticker_index}
             from data: {data}""",
            extra={"ticker_index": ticker_index},
        )
        return None
    client_types_data_frame = pd.DataFrame(
        data,
        columns=[
            "date",
            "individual_buy_count",
            "corporate_buy_count",
            "individual_sell_count",
            "corporate_sell_count",
            "individual_buy_vol",
            "corporate_buy_vol",
            "individual_sell_vol",
            "corporate_sell_vol",
            "individual_buy_value",
            "corporate_buy_value",
            "individual_sell_value",
            "corporate_sell_value",
        ],
    )
    for i in [
        "individual_buy_",
        "individual_sell_",
        "corporate_buy_",
        "corporate_sell_",
    ]:
        client_types_data_frame[f"{i}mean_price"] = client_types_data_frame[
            f"{i}value"
        ].astype(float) / client_types_data_frame[f"{i}vol"].astype(float)
    client_types_data_frame[
        "individual_ownership_change"
    ] = client_types_data_frame["corporate_sell_vol"].astype(
        float
    ) - client_types_data_frame[
        "corporate_buy_vol"
    ].astype(
        float
    )
    return client_types_data_frame


def download_option_data(
    symbols: Union[List, str] = "all",
    json_path: Optional[str] = None,
    write_to_csv: bool = False,
    write_to_excel: bool = False,
    base_path: str = "option_data",
    include_jdate: bool = False,
    adjust: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Download historical option data for given symbols from a JSON file.
    
    Parameters
    ----------
    symbols : Union[List, str], optional
        List of symbols, single symbol, or "all" to download all symbols from JSON (default is "all").
    json_path : Optional[str], optional
        Path to symbols_option.json. If None, looks for symbols_option.json next to the executed script.
    write_to_csv : bool, optional
        If True, saves data as CSV files in base_path (default is False).
    write_to_excel : bool, optional
        If True, saves data as Excel files in base_path (default is False).
    base_path : str, optional
        Directory to save CSV/Excel files (default is "option_data").
    include_jdate : bool, optional
        If True, includes Persian date column (JDate) (default is False).
    adjust : bool, optional
        If True, adjusts prices for capital increase/profit sharing (default is False).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping symbols to their historical option data DataFrames.
    """
    # Determine JSON path (next to executed script if not specified)
    if json_path is None:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        json_path = os.path.join(script_dir, "symbols_option.json")
    
    # Read symbols from JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            symbols_data = json.load(f)
        all_symbols = [list(item.keys())[0] for item in symbols_data if item]
        symbol_indices = {list(item.keys())[0]: item[list(item.keys())[0]]["index"] for item in symbols_data if item}
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_path}")
        return {}
    except Exception as e:
        logger.error(f"Error reading JSON file {json_path}: {e}")
        return {}

    # Handle symbols input
    if symbols == "all":
        symbols = all_symbols
    elif isinstance(symbols, str):
        symbols = [symbols]

    # Validate symbols
    valid_symbols = [s for s in symbols if s in all_symbols]
    if not valid_symbols:
        logger.error("No valid symbols provided or found in JSON")
        return {}

    df_list = {}
    future_to_symbol = {}
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        session = requests_retry_session()
        for symbol in valid_symbols:
            ticker_index = symbol_indices.get(symbol)
            if ticker_index is None:
                logger.error(f"No index found for symbol: {symbol}")
                continue

            future = executor.submit(
                download_ticker_daily_record, ticker_index, session
            )
            future_to_symbol[future] = symbol

        for future in futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df: pd.DataFrame = future.result()
            except pd.errors.EmptyDataError as ex:
                logger.error(
                    f"Cannot read daily option records for symbol: {symbol}",
                    extra={"Error": ex},
                )
                continue
            except Exception as e:
                logger.error(
                    f"Error downloading option data for {symbol}: {e}"
                )
                continue

            # Process DataFrame
            df = df.iloc[::-1].reset_index(drop=True)
            df = df.rename(columns=translations.HISTORY_FIELD_MAPPINGS)
            df = df.drop(columns=["<PER>", "<TICKER>"], errors="ignore")
            _adjust_data_frame(df, include_jdate)

            # Rename columns to match desired format
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'adjClose': 'Final',
                'value': 'Value',
                'volume': 'Volume',
                'jdate': 'JDate',
                'yesterday': 'Yesterday',
                'count': 'Count'
            })
            df['Symbol'] = symbol
            df['Date'] = pd.to_datetime(df['Date']).dt.date

            if adjust:
                df = adjust_price(df)

            df_list[symbol] = df

            # Save to CSV or Excel
            if write_to_csv or write_to_excel:
                Path(base_path).mkdir(parents=True, exist_ok=True)
                try:
                    if write_to_csv:
                        df.to_csv(f"{base_path}/{symbol}.csv", index=False)
                    if write_to_excel:
                        df.to_excel(f"{base_path}/{symbol}.xlsx", index=False, engine='openpyxl')
                except Exception as e:
                    logger.error(f"Error saving option data for {symbol}: {e}")

    if len(df_list) != len(valid_symbols):
        print("Warning, option data download did not complete, re-run the code")
    session.close()
    return df_list


def get_symbol_id(symbol_name: str):
    url = tse_settings.TSE_SYMBOL_ID_URL.format(symbol_name.strip())
    response = requests_retry_session().get(url, timeout=10)
    try:
        response.raise_for_status()
    except HTTPError:
        raise Exception("Sorry, tse server did not respond")

    symbol_full_info = response.text.split(";")[0].split(",")
    if persian.replace_arabic(symbol_name) == symbol_full_info[0].strip():
        return symbol_full_info[2]  # symbol id
    return None


def get_symbol_info(symbol_name: str):
    url = tse_settings.TSE_SYMBOL_ID_URL.format(symbol_name.strip())
    response = requests_retry_session().get(url, timeout=10)
    try:
        response.raise_for_status()
    except HTTPError:
        raise Exception(f"{symbol_name}: Sorry, tse server did not respond")

    symbols = response.text.split(";")
    market_symbol = MarketSymbol(
        code=None,
        symbol=None,
        index=None,
        name=None,
        old=[],
    )
    for symbol_full_info in symbols:
        if symbol_full_info.strip() == "":
            continue
        symbol_full_info = symbol_full_info.split(",")
        if persian.replace_arabic(symbol_full_info[0]) == symbol_name:
            # if symbol id is active
            if symbol_full_info[7] == "1":
                market_symbol.symbol = persian.replace_arabic(
                    symbol_full_info[0]
                )
                market_symbol.name = persian.replace_arabic(
                    symbol_full_info[1]
                )
                market_symbol.index = symbol_full_info[2]  # active symbol id
            else:
                market_symbol.old.append(symbol_full_info[2])  # old symbol id

    if market_symbol.index is None:
        return None
    return market_symbol
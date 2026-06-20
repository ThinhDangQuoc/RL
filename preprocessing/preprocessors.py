import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)].copy()
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)

    # Make sure we do not propagate inf/-inf into the environment.
    # Technical indicators (e.g. CCI) can occasionally generate inf values.
    data = data.replace([np.inf, -np.inf], np.nan)

    # Ensure each trading day has the same number of tickers (STOCK_DIM) so
    # the environment state/observation length stays constant.
    all_tickers = sorted(df.tic.unique().tolist())
    unique_dates = sorted(data.datadate.unique().tolist())
    if len(unique_dates) == 0:
        return data

    turbulence_by_date = None
    if 'turbulence' in data.columns:
        turbulence_by_date = (
            data[['datadate', 'turbulence']]
            .drop_duplicates(subset=['datadate'])
            .set_index('datadate')['turbulence']
        )

    full_index = pd.MultiIndex.from_product(
        [unique_dates, all_tickers],
        names=['datadate', 'tic'],
    )

    data = data.set_index(['datadate', 'tic']).reindex(full_index).reset_index()

    # Fill turbulence per-date (it's a market-level signal shared by all tickers).
    if turbulence_by_date is not None:
        data['turbulence'] = data['datadate'].map(turbulence_by_date)

    value_cols = [c for c in data.columns if c not in ['datadate', 'tic']]
    per_ticker_cols = [c for c in value_cols if c != 'turbulence']

    # Fill per-ticker features without looking ahead.
    # IMPORTANT: Do not use backward fill (bfill) here; it leaks future information
    # within the same split window and can also fabricate values for days when a
    # ticker truly has no data.
    if len(per_ticker_cols) > 0:
        data[per_ticker_cols] = data.groupby('tic')[per_ticker_cols].ffill()

    # Turbulence is market-level and shared across tickers; fill forward by date.
    if 'turbulence' in data.columns:
        data['turbulence'] = data['turbulence'].fillna(method='ffill')

    # Remaining NaNs (e.g., tickers that have no history in the window) become 0.
    data[value_cols] = data[value_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    data = data.sort_values(['datadate', 'tic'], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df



def preprocess_data():
    """data preprocessing pipeline"""

    df = load_dataset(file_name=config.TRAINING_DATA_FILE)
    tickers = getattr(config, 'TICKERS', None)
    if tickers:
        df = df[df['tic'].isin(tickers)].copy()
    # get data after a configured start date
    start_date = getattr(config, 'DATA_START_DATE', 20090000)
    df = df[df.datadate >= start_date]
    # calcualte adjusted price
    df_preprocess = calcualte_price(df)
    # add technical indicators using stockstats
    df_final = add_technical_indicator(df_preprocess)

    # Some indicators can produce inf/-inf (e.g., divide by zero in CCI).
    # Convert to NaN so we can fill them safely.
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)

    # fill the missing values at the beginning
    df_final.fillna(method='bfill', inplace=True)

    # Ensure no remaining NaN/inf.
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0.0, inplace=True)
    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets

    df_price_pivot = df.pivot(index='datadate', columns='tic', values='adjcp').sort_index()
    unique_date = df_price_pivot.index.to_list()

    # start after a year (trading days)
    start = 252
    if len(unique_date) <= start:
        return pd.DataFrame({'datadate': df_price_pivot.index, 'turbulence': [0] * len(unique_date)})

    turbulence_index = [0] * start
    count = 0

    for i in range(start, len(unique_date)):
        current_price = df_price_pivot.iloc[[i]]
        hist_price = df_price_pivot.iloc[0:i]

        # Only keep tickers with sufficient non-NaN history and a value on the current date.
        eligible = hist_price.count(axis=0)
        eligible = eligible[eligible >= start].index
        if len(eligible) == 0:
            turbulence_index.append(0)
            continue

        current_row = current_price.iloc[0]
        eligible = [c for c in eligible if not pd.isna(current_row[c])]
        if len(eligible) < 2:
            turbulence_index.append(0)
            continue

        hist_eligible = hist_price[eligible].ffill()
        # Drop any remaining rows with NaNs (typically at the beginning of a ticker's history)
        hist_eligible = hist_eligible.dropna(axis=0, how='any')
        if hist_eligible.shape[0] < 2:
            turbulence_index.append(0)
            continue

        cov_temp = hist_eligible.cov()
        current_temp = (current_price[eligible] - hist_eligible.mean(axis=0))

        cov = cov_temp.values
        if not np.isfinite(cov).all():
            turbulence_index.append(0)
            continue

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        temp = current_temp.values.dot(inv_cov).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = float(temp[0][0])
            else:
                # avoid large outlier because calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0

        turbulence_index.append(turbulence_temp)

    turbulence_index = pd.DataFrame({'datadate': df_price_pivot.index, 'turbulence': turbulence_index})
    return turbulence_index











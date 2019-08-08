import pandas as pd
import logging
import numpy as np

log = logging.getLogger(__name__)


def load_csv(path):
    """ Loads csv data to a Dataframe.

    Expects to find atleast "date" column. Column names are case-insensitive.

    Parameters
    ----------
    path : str
        Filepath.

    Returns
    -------
    Dataframe
        Lowercase columns, datetimeindex named "date".

    """

    df = pd.read_csv(path, infer_datetime_format=True)
    df.columns = [col.lower() for col in df.columns]  # Make lower case
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)
    msg = 'Read csv "{}", rows: {}'.format(path.split('/')[-1], len(df))
    log.debug(msg)
    return df


def resample_ohlc(df, timeframe):
    """ Resamples to new longer timeframe

    Uses forward fill (a.k.a zero-order-hold) to fill missing values.
    Columns other than ['open', 'high', 'low', 'close', 'volume']
    are dropped.

    Assumes data is 24h/7

    Last resampled row is dropped if it's partial.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with datetime index and has least some of the following named columns
        ['open', 'high', 'low', 'close', 'volume']

    timeframe : str
        Name of the timeframe. Ex. '1Min', '15Min', '1H', '1D'

    Returns
    -------
    pandas.DataFrame
        Resampled.
    """

    # TODO: Make preserve column order
    df = df.copy()

    supported_cols = ['open', 'high', 'low', 'close', 'volume']
    original_cols = df.columns
    for col in original_cols:
        if col not in supported_cols:
            raise ValueError('Column "{}" is not supported for resampling.'.format(col))

    original_start = df.index[0]
    # Replace index with fake one that starts at day 00:00 (normalize=True does this)
    df.index = pd.date_range(start=df.index[0], periods=len(df.index), freq=pd.infer_freq(df.index),
                                 normalize=True)

    resampled = df.resample(timeframe)
    rules = dict()

    columns = df.columns
    if 'open' in columns:
        rules['open'] = 'first'
    if 'high' in columns:
        rules['high'] = 'max'
    if 'low' in columns:
        rules['low'] = 'min'
    if 'close' in columns:
        rules['close'] = 'last'
    if 'volume' in columns:
        rules['volume'] = 'sum'

    resampled = resampled.agg(rules).fillna(method='ffill')

    # Correct index
    new_index = pd.date_range(start=original_start, periods=len(resampled.index), freq=timeframe)

    resampled.index = new_index

    if original_start != resampled.index[0]:
        err = ValueError('Resampling changed the start time of the data.')
        log.error(err); raise err

    if len(df.index) % len(resampled.index) != 0:
        resampled = resampled.drop(resampled.tail(1).index, axis=0)  # Drop last if its partial

    # Bring back original column order
    resampled = resampled[original_cols]

    return resampled


def extract_states_and_labels(df, step_len):
    """ Extracts normalized states and labels.

    close,high,low are normalized using last close of the state.
    Volume is normalized using last volume of the state.

    Column order must be [close, high, low, volume]

    """

    col_order = ('close', 'high', 'low', 'volume')
    df = df.copy()
    df_vol = None
    if 'volume' in df.columns:
        if tuple(df.columns) != col_order:
            raise ValueError('Bad column order {}. Expected {}'.format(df.columns, col_order))
        df_vol = df[['volume']]
        df = df.drop('volume', axis=1)
    else:
        if tuple(df.columns) != col_order[:3]:
            raise ValueError('Bad column order {}. Expected {}'.format(df.columns, col_order[:3]))

    tot_states = len(df) - step_len  # Last one is ignored bc it won't have known label

    states = np.empty([tot_states + 1, 1, step_len, len(df.columns)])
    states_vol = np.empty([tot_states + 1, 1, step_len, 1])

    labels = np.empty([tot_states, 1])

    # Create states by "sliding a window", 1 extra state is made
    for i in range(tot_states + 1):
        states[i, 0, ...] = df.iloc[i:i + step_len].values
        if df_vol is not None:
            states_vol[i, 0, ...] = df_vol.iloc[i:i + step_len].values
    # Compute labels and normalize the states
    for i in range(tot_states):
        current_price = states[i, 0, -1, 0]
        next_price = states[i+1, 0, -1, 0]

        labels[i, 0] = next_price / current_price

        # Normalize
        states[i, 0, ...] /= current_price

        if df_vol is not None:
            total_volume = np.sum(states_vol[i, 0, ...])
            if total_volume <= 1e-7:
                states_vol[i, 0, ...] = 0.
            else:
                states_vol[i, 0, ...] /= total_volume

    if df_vol is not None:
        states = np.concatenate([states, states_vol], axis=-1)

    return states[:-1, ...], labels  # Drop the extra last state bc it doesnt have label
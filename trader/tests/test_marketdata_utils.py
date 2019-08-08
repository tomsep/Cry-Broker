import pytest
from numpy.testing import assert_almost_equal, assert_equal
from pandas.testing import assert_index_equal


from trader.marketdata_utils import *


def test_resampling():

    df_source = load_csv('./test_data_5min_repaired.csv')
    df_source = df_source[np.flip(df_source.columns.values)]  # reverse cols

    # Resample
    df_desired_resampled = pd.read_csv('./test_data_30m_resampled.csv')
    df_desired_resampled = df_desired_resampled[np.flip(df_desired_resampled.columns.values)]
    df_desired_resampled.set_index('date', drop=True, inplace=True)
    df_desired_resampled.index = pd.to_datetime(df_desired_resampled.index, infer_datetime_format=True)

    df_resampled = resample_ohlc(df_source, '30min')

    # Compare columns, ignore "date" col from df_desired
    assert_equal(np.array(df_resampled.columns), np.array(df_desired_resampled.columns))

    # Compare values
    assert_almost_equal(df_resampled.values, df_desired_resampled.values)

    # Check index
    assert_index_equal(df_resampled.index, df_desired_resampled.index, check_names=False)


def test_extract_states_and_labels():
    """ Test first and last states (and labels) that their values are correct.
    Uses a test data file. """

    df_orig = load_csv('./test_data_5min_repaired.csv')
    df_orig = df_orig[['close', 'high', 'low']]

    state_length = 4
    total_states = 19 - state_length

    # Manually compute the first state
    first_label = df_orig.iloc[4].values[0] / df_orig.iloc[3].values[0]
    first_state = df_orig.iloc[:4].values / df_orig.iloc[3].values[0]

    # Manually the last state (Note: last row is never included on any state bc it wouldn't have known next price)
    last_label = df_orig.iloc[-1].values[0] / df_orig.iloc[-2].values[0]
    last_state = df_orig.iloc[-5:-1].values / df_orig.iloc[-2].values[0]

    states, labels = extract_states_and_labels(df_orig, step_len=state_length)

    assert_equal(np.shape(states), (total_states, 1, state_length, 3))
    assert_equal(np.shape(labels), (total_states, 1))

    assert_almost_equal(states[0, 0, ...], first_state)
    assert_almost_equal(states[-1, 0, ...], last_state)

    assert_almost_equal(labels[0, 0], first_label)
    assert_almost_equal(labels[-1, 0], last_label)


def test_extract_states_and_labels_with_volume():
    """ Test first and last states (and labels) that their values are correct.
        Uses a test data file. """

    df_orig = load_csv('./test_data_5min_repaired.csv')
    df_orig = df_orig[['close', 'high', 'low', 'volume']]

    state_length = 4
    total_states = 19 - state_length

    # Manually compute the first state
    first_label = df_orig.iloc[4].values[0] / df_orig.iloc[3].values[0]
    first_state = df_orig.iloc[:4].values / df_orig.iloc[3].values[0]

    # First state volume
    first_state_vol = df_orig[['volume']].iloc[:4].values / np.sum(df_orig[['volume']].iloc[:4].values)

    # Manually the last state (Note: last row is never included on any state bc it wouldn't have known next price)
    last_label = df_orig.iloc[-1].values[0] / df_orig.iloc[-2].values[0]
    last_state = df_orig.iloc[-5:-1].values / df_orig.iloc[-2].values[0]
    last_state_vol = df_orig[['volume']].iloc[-5:-1].values / np.sum(df_orig[['volume']].iloc[-5:-1].values)

    states, labels = extract_states_and_labels(df_orig, step_len=state_length)

    assert_equal(np.shape(states), (total_states, 1, state_length, 4))
    assert_equal(np.shape(labels), (total_states, 1))

    # Test first 3 channels (last dim), i.e. test close, high, low
    assert_almost_equal(states[0, 0, :, :3], first_state[..., :3])
    assert_almost_equal(states[-1, 0, :, :3], last_state[..., :3])

    assert_almost_equal(labels[0, 0], first_label)
    assert_almost_equal(labels[-1, 0], last_label)

    # Test volume channel
    assert_almost_equal(states[0, 0, :, -1:], first_state_vol)
    assert_almost_equal(states[-1, 0, :, -1:], last_state_vol)

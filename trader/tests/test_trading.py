import pytest
from decimal import Decimal as D
from pprint import pprint
from numpy.testing import assert_array_equal, assert_almost_equal
import os
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from trader.trading import *


def assert_nested_dicts_equal(actual, desired):
    """ For a dict like

    {'mainkey1': {'subkey1': val1, 'subkey2': val2},
    'mainkey2': {'subkey1': val3, 'subkey2': val4}}

    """

    # Check main level keys
    actual_keys = sorted(actual.keys())
    desired_keys = sorted(desired.keys())
    assert_array_equal(actual_keys, desired_keys)

    # Check second level
    for key in actual.keys():
        actual_keys = sorted(actual[key].keys())
        desired_keys = sorted(desired[key].keys())
        assert_array_equal(actual_keys, desired_keys)

    # Check values
    for main_key in actual.keys():
        for sub_key in actual[main_key].keys():
            assert actual[main_key][sub_key] == desired[main_key][sub_key]


def assert_list_of_dicts(actual, desired, sort_key):
    """ Assert a list of dicts like [{'key1': val1, 'key2': val2}, ...] """

    assert len(actual) == len(desired)

    actual = sorted(actual, key=lambda x: x[sort_key])
    desired = sorted(desired, key=lambda x: x[sort_key])

    for i in range(len(actual)):

        actual_keys = sorted(actual[i].keys())
        desired_keys = sorted(desired[i].keys())

        assert_array_equal(actual_keys, desired_keys)

        for key in actual_keys:
            assert actual[i][key] == desired[i][key]


class TestPortfolio:

    def test_hide_value(self):

        portf = {'USDT': {'value': D('100'), 'qty': D('100')},
                 'BTC': {'value': D('50'), 'qty': D('0.1')},
                 'ETH': {'value': D('80'), 'qty': D('1')}}

        prices = {'BTC': {'USDT': D('500')},
                  'ETH': {'USDT': D('80')}}

        hide_value = D('60')

        desired = {'USDT': {'value': D('40'), 'qty': D('40')},
                   'BTC': {'value': D('0'), 'qty': D('0')},
                   'ETH': {'value': D('20'), 'qty': D('0.25')}}

        actual = Portfolio._hide_value(portf, prices, hide_value)

        assert_nested_dicts_equal(actual, desired)

    def test_get(self):

        # Create Portfolio
        balances = {'USDT': D('100'),
                    'BTC': D('0.1'),
                    'ETH': D('1'),
                    'IGNORE': D('999')}

        assets = ['BTC', 'USDT', 'ETH']

        prices = {'BTC': {'USDT': D('500')},
                  'ETH': {'USDT': D('80')}}

        portf = Portfolio(balances, assets)

        # Test without deduction
        desired = {'USDT': {'value': D('100'), 'qty': D('100'), 'alloc': D('100') / D('230')},
                   'BTC': {'value': D('50'), 'qty': D('0.1'), 'alloc': D('50') / D('230')},
                   'ETH': {'value': D('80'), 'qty': D('1'), 'alloc': D('80') / D('230')}}

        actual = portf.get(prices)
        assert_nested_dicts_equal(actual, desired)

        # Test with deduction
        hide_value = D('60')

        desired = {'USDT': {'value': D('40'), 'qty': D('40'), 'alloc': D('40')/D('60')},
                   'BTC': {'value': D('0'), 'qty': D('0'), 'alloc': D('0')/D('60')},
                   'ETH': {'value': D('20'), 'qty': D('0.25'), 'alloc': D('20')/D('60')}}

        actual = portf.get(prices, deduct=hide_value)

        assert_nested_dicts_equal(actual, desired)

        # Test with non-D hide_value
        hide_value = 60
        with pytest.raises(ValueError):
            portf.get(prices, deduct=hide_value)

    def test_get_value_changes(self):

        balances = {'USDT': D('100'),
                    'BTC': D('0.1'),
                    'ETH': D('1'),
                    'IGNORE': D('999')}

        assets = ['BTC', 'USDT', 'ETH']

        prices = {'BTC': {'USDT': D('500')},
                  'ETH': {'USDT': D('80')}}

        portf = Portfolio(balances, assets)

        portf_dict = portf.get(prices)

        alloc_dict = {'USDT': D('0.5'),
                      'BTC': D('0.2'),
                      'ETH': D('0.3')}

        desired = {'USDT': {'value': D('0.5') * D('230') - D('100')},
                   'BTC': {'value': D('0.2') * D('230') - D('50')},
                   'ETH': {'value': D('0.3') * D('230') - D('80')}}

        actual = portf.get_value_changes(portf_dict, alloc_dict)

        assert_nested_dicts_equal(actual, desired)

    def test_as_alloc_vector(self):

        portf = {'USDT': {'value': D('100'), 'qty': D('100'), 'alloc': D('0.5')},
                 'ETH': {'value': D('50'), 'qty': D('0.1'), 'alloc': D('0.3')},
                 'BTC': {'value': D('80'), 'qty': D('1'), 'alloc': D('0.2')}}

        assets = ['USDT', 'BTC', 'ETH']

        actual = Portfolio.as_alloc_vector(portf, assets)

        desired = np.array([[0.5, 0.2, 0.3]])

        assert_almost_equal(actual, desired)

    def test_as_alloc_dict(self):
        portf_vec = np.array([[0.5, 0.2, 0.3]])

        assets = ['USDT', 'BTC', 'ETH']
        actual = Portfolio.as_alloc_dict(portf_vec, assets)

        desired = {'USDT': D('0.5'), 'ETH': D('0.3'), 'BTC': D('0.2')}

        assert_list_of_dicts([actual], [desired], sort_key='USDT')


class TestOrderFactory:

    def test_value_flows_and_to_real_markets(self):

        changes = {'USDT': {'value': D('-4')},
                   'BTC': {'value': D('15')},
                   'ETH': {'value': D('-11')},
                   'IGNORE': {'value': D('0')}}

        flows = OrderFactory._value_flows(changes)

        prices = {'BTC': {'USDT': D('500')},
                  'ETH': {'USDT': D('80'), 'BTC': D('40')}}

        actual = OrderFactory._flows_to_real_markets(flows, prices)

        desired = [{'pair': 'BTCUSDT', 'side': 'buy', 'value': Decimal('4'), 'base': 'BTC', 'quote': 'USDT'},
                   {'pair': 'ETHBTC', 'side': 'sell', 'value': Decimal('11'), 'base': 'ETH', 'quote': 'BTC'}]

        assert_list_of_dicts(actual, desired, sort_key='pair')

    def test_orders_from_flows(self):

        flows = [{'pair': 'BTCUSDT', 'side': 'buy', 'value': Decimal('4'), 'base': 'BTC', 'quote': 'USDT'},
                 {'pair': 'ETHBTC', 'side': 'sell', 'value': Decimal('11'), 'base': 'ETH', 'quote': 'BTC'}]

        prices = {'BTC': {'USDT': D('500')},
                  'ETH': {'USDT': D('80'), 'BTC': D('40')}}

        actual = OrderFactory._orders_from_flows(flows, prices)

        desired = [{'base': 'BTC',
                    'pair': 'BTCUSDT',
                    'qty': D('0.008'),
                    'quote': 'USDT',
                    'side': 'buy',
                    'type': 'market'},

                   {'base': 'ETH',
                    'pair': 'ETHBTC',
                    'qty': D('0.1375'),
                    'quote': 'BTC',
                    'side': 'sell',
                    'type': 'market'}]

        assert_list_of_dicts(actual, desired, sort_key='pair')

    def test_apply_min_value_notional(self):
        """

        min_value = 10
        0.008 BTC * 1250 = 10  (OK!)
        5 ETH * 0.7 = 3.5      (NOT ENOUGH -> qty is made 0)

        """

        prices = {'BTC': {'USDT': D('1250')},
                  'ETH': {'USDT': D('0.7'), 'BTC': D('40')}}

        orders = [{'base': 'BTC',
                   'pair': 'BTCUSDT',
                   'qty': D('0.008'),
                   'quote': 'USDT',
                   'side': 'buy',
                   'type': 'market'},

                  {'base': 'ETH',
                   'pair': 'ETHBTC',
                   'qty': D('5'),
                   'quote': 'BTC',
                   'side': 'sell',
                   'type': 'market'}]

        min_value = D('10')

        OrderFactory._apply_min_value_notional_filter(orders, prices, min_value)

        desired = [{'base': 'BTC',
                    'pair': 'BTCUSDT',
                    'qty': D('0.008'),
                    'quote': 'USDT',
                    'side': 'buy',
                    'type': 'market'},

                   {'base': 'ETH',
                    'pair': 'ETHBTC',
                    'qty': D('0'),
                    'quote': 'BTC',
                    'side': 'sell',
                    'type': 'market'}]

        assert_list_of_dicts(orders, desired, sort_key='pair')

    def test_lot_filter(self):

        filters = {'BTCUSDT': [{'filterType': 'LOT_SIZE', 'minQty': '0.0100', 'stepSize': '0.001000'}],
                   'ETHBTC': [{'filterType': 'LOT_SIZE', 'minQty': '5.000', 'stepSize': '0.1000'}]}

        orders = [{'base': 'BTC',
                   'pair': 'BTCUSDT',
                   'qty': D('0.0105'),
                   'quote': 'USDT',
                   'side': 'buy',
                   'type': 'market'},

                  {'base': 'ETH',
                   'pair': 'ETHBTC',
                   'qty': D('4'),
                   'quote': 'BTC',
                   'side': 'sell',
                   'type': 'market'}]

        OrderFactory._apply_lot_filter(orders, filters)

        desired = [{'base': 'BTC',
                   'pair': 'BTCUSDT',
                   'qty': D('0.01'),
                   'quote': 'USDT',
                   'side': 'buy',
                   'type': 'market'},

                  {'base': 'ETH',
                   'pair': 'ETHBTC',
                   'qty': D('0'),
                   'quote': 'BTC',
                   'side': 'sell',
                   'type': 'market'}]


        assert_list_of_dicts(orders, desired, sort_key='pair')

@pytest.mark.skip(reason='Not up-to-date')
def test_market_data_to_input_format():

    client = Client(None, None)
    pairs = ['BTCUSDT', 'ETHUSDT']
    data = get_market_data(client, pairs)

    data_trimmed = {}
    for pair, df in data.items():
        data_trimmed[pair] = trim_market_data(df)

    close = data_trimmed[pairs[0]].iloc[-1].close
    previous_close = data_trimmed[pairs[0]].iloc[-2].close

    prev_close_normalized = float(previous_close) / float(close)

    states = market_data_to_input_format(data_trimmed, pairs)

    assert_array_equal(np.shape(states), [1, 2, 50, 3])

    assert_almost_equal(states[0, 0, -2, 0], prev_close_normalized)


class TestDataLogger:

    def test_log_portf_before_and_after(self, tmp_path):

        portf_dict = {'USDT': {'value': D('100'), 'qty': D('100'), 'alloc': D('100') / D('230')},
                      'BTC': {'value': D('50'), 'qty': D('0.1'), 'alloc': D('50') / D('230')},
                      'ETH': {'value': D('80'), 'qty': D('1'), 'alloc': D('80') / D('230')}}

        dlog = DataLogger(tmp_path)

        # Construct desired result
        loop_id = pd.to_datetime('2000-01-01 12:00', utc=True)
        deduct_val = D('10.55')
        desired_data = [
            ['2000-01-01 12:00:00+00:00', 'BTC', '50.00000000', '0.10000000', '0.21739130', '10.55000000'],
            ['2000-01-01 12:00:00+00:00', 'ETH', '80.00000000', '1.00000000', '0.34782609', '10.55000000'],
            ['2000-01-01 12:00:00+00:00', 'USDT', '100.00000000', '100.00000000', '0.43478261', '10.55000000']
        ]
        cols = ['loop_id', 'asset', 'value', 'quantity', 'allocation', 'deducted_value']
        desired = pd.DataFrame(desired_data, columns=cols, dtype=str)

        # Test port before
        fname = 'portf_before.csv'
        dlog.log_portf_before(portf_dict, deduct_val, loop_id)
        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)
        assert_frame_equal(actual, desired)

        # Test portf after
        fname = 'portf_after.csv'
        dlog.log_portf_after(portf_dict, deduct_val, loop_id)
        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)
        assert_frame_equal(actual, desired)

        # Test appending functionality while also using id without timezone
        loop_id = pd.to_datetime('2000-01-01 12:00')
        dlog.log_portf_after(portf_dict, deduct_val, loop_id)
        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)
        desired = pd.concat([desired, desired], ignore_index=True)
        assert_frame_equal(actual, desired)

    def test_log_portf_target_allocs(self, tmp_path):

        alloc_dict = {'USDT': D('0.333333333333333'), 'BTC': D('0.66666666666'), 'ETH': D('0')}
        fname = 'portf_target_allocs.csv'

        dlog = DataLogger(tmp_path)

        # Construct desired result
        loop_id = pd.to_datetime('2000-01-01 12:00', utc=True)
        desired_data = [
            ['2000-01-01 12:00:00+00:00', 'BTC', '0.66666667'],
            ['2000-01-01 12:00:00+00:00', 'ETH', '0.00000000'],
            ['2000-01-01 12:00:00+00:00', 'USDT', '0.33333333']
        ]
        cols = ['loop_id', 'asset', 'allocation']
        desired = pd.DataFrame(desired_data, columns=cols, dtype=str)

        # Test
        dlog.log_portf_target_allocs(alloc_dict, loop_id)
        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)
        assert_frame_equal(actual, desired)

        # Test appending functionality while also using id without timezone
        loop_id = pd.to_datetime('2000-01-01 12:00')
        desired = pd.concat([desired, desired], ignore_index=True)
        dlog.log_portf_target_allocs(alloc_dict, loop_id)
        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)
        assert_frame_equal(actual, desired)

    def test_log_orders(self, tmp_path):

        columns = ['loop_id', 'symbol', 'orderId', 'clientOrderId', 'transactTime', 'price', 'origQty',
                   'executedQty', 'cummulativeQuoteQty', 'status', 'timeInForce', 'type', 'side']

        orders = [{'symbol': 'BTCUSDT',
                   'orderId': 499729097,
                   'clientOrderId': 'PzfU3fMEcz3lfTQ4vVPfwl',
                   'transactTime': 1562652005157,
                   'price': '0.00000000',
                   'origQty': '0.00494600',
                   'executedQty': '0.00494600',
                   'cummulativeQuoteQty': '62.49350136',
                   'status': 'FILLED',
                   'timeInForce': 'GTC',
                   'type': 'MARKET',
                   'side': 'SELL',
                   'fills': [{'price': '12635.16000000',
                              'qty': '0.00494600',
                              'commission': '0.00140918',
                              'commissionAsset': 'BNB',
                              'tradeId': 151079353}]},
                  {'symbol': 'BNBUSDT',
                   'orderId': 184151590,
                   'clientOrderId': 'aKOTu0YJpAK3JuSwW12V7R',
                   'transactTime': 1562652004857,
                   'price': '0.00000000',
                   'origQty': '2.05000000',
                   'executedQty': '2.05000000',
                   'cummulativeQuoteQty': '68.19919500',
                   'status': 'FILLED',
                   'timeInForce': 'GTC',
                   'type': 'MARKET',
                   'side': 'SELL',
                   'fills': [{'price': '33.26790000',
                              'qty': '2.05000000',
                              'commission': '0.00153784',
                              'commissionAsset': 'BNB',
                              'tradeId': 33828101}]}]
        fname = 'order_results.csv'
        desired_data = [
            ['2000-01-01 12:00:00+00:00', 'BNBUSDT', '184151590', 'aKOTu0YJpAK3JuSwW12V7R', '1562652004857',
             '0.00000000', '2.05000000', '2.05000000', '68.19919500', 'FILLED', 'GTC', 'MARKET', 'SELL'],
            ['2000-01-01 12:00:00+00:00', 'BTCUSDT', '499729097', 'PzfU3fMEcz3lfTQ4vVPfwl', '1562652005157',
             '0.00000000', '0.00494600', '0.00494600', '62.49350136', 'FILLED', 'GTC', 'MARKET', 'SELL']
        ]

        desired = pd.DataFrame(desired_data, columns=columns, dtype=str)

        loop_id = pd.to_datetime('2000-01-01 12:00', utc=True)
        dlog = DataLogger(tmp_path)

        dlog.log_order_results(orders, loop_id)
        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)
        assert_frame_equal(actual, desired)

        # Test appending functionality while also using id without timezone
        loop_id = pd.to_datetime('2000-01-01 12:00')
        desired_new = pd.concat([desired, desired], ignore_index=True)
        dlog.log_order_results(orders, loop_id)
        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)
        assert_frame_equal(actual, desired_new)

        # Test ignoring of Exceptions within the responses
        orders.insert(0, ValueError('Some error.'))
        desired_new = pd.concat([desired, desired, desired], ignore_index=True)
        dlog.log_order_results(orders, loop_id)
        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)
        assert_frame_equal(actual, desired_new)

    def test_log_general(self, tmp_path):

        loop_id = pd.to_datetime('2000-01-01 12:00', utc=True)
        version = '0.1'
        agent_id = 'algo'
        min_notational = D('20.000666666')
        fname = 'general_info.csv'
        columns = ['loop_id', 'agent_id', 'program_version', 'min_notational']

        dlog = DataLogger(tmp_path)

        dlog.log_general(loop_id, agent_id, min_notational, version)

        actual = pd.read_csv(os.path.join(tmp_path, fname), dtype=str)

        desired = [
            ['2000-01-01 12:00:00+00:00', agent_id, version, str(round(min_notational, 8))]
        ]

        desired = pd.DataFrame(desired, columns=columns, dtype=str)
        assert_frame_equal(actual, desired)


def test_compute_total_commission():

    # Order (most keys deleted for readability)
    order = {'symbol': 'BTCUSDT', 'clientOrderId': '4JYTQecXIHmnPbmQ1PfFAx',
             'fills': [
                 {'price': '9987.56000000', 'qty': '0.00561500', 'commission': '0.00157422', 'commissionAsset': 'BNB',
                  'tradeId': 156109628},
                 {'price': '9988.77000000', 'qty': '0.00561500', 'commission': '0.00300001', 'commissionAsset': 'BNB',
                  'tradeId': 156109629}]}

    actual = compute_total_commission(order)
    desired = {'commission': D('0.00457423'), 'commissionAsset': 'BNB'}

    assert_list_of_dicts([actual], [desired], sort_key='commission')


    order['fills'][1]['commissionAsset'] = 'USDT'

    actual = compute_total_commission(order)

    desired = {'commission': D('0'), 'commissionAsset': 'MIXED'}

    assert_list_of_dicts([actual], [desired], sort_key='commission')


@pytest.mark.parametrize(
    "now, offset, desired",
     [('10:05', '0min', '10:30'),
      ('10:05', '10min', '10:10'),
      ('10:35', '0min', '11:00'),
      ('10:35', '10min', '10:40'),
      ('10:30', '25min', '10:55')
      ])
def test_get_new_target_time(now, offset, desired):

    now = pd.Timestamp('2000-01-01 ' + now, tz='UTC')
    offset = pd.Timedelta(offset)

    actual = get_new_target_time(offset, now)
    desired = pd.Timestamp('2000-01-01 ' + desired, tz='UTC')
    assert actual == desired


@pytest.mark.parametrize(
    "now, offset, desired",
     [('10:05', '30min', ValueError),
      ('10:05', '-5min', ValueError),
      ])
def test_get_new_target_time_exception(now, offset, desired):

    now = pd.Timestamp('2000-01-01 ' + now, tz='UTC')
    offset = pd.Timedelta(offset)

    with pytest.raises(desired):
        get_new_target_time(offset, now)

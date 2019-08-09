import os
import logging
import pandas as pd
import numpy as np
from binance.client import Client
from threading import Thread
from binance.enums import *
import decimal
from decimal import Decimal
from copy import deepcopy
import time
from binance.exceptions import BinanceAPIException
from functools import partial
import warnings

from trader.marketdata_utils import extract_states_and_labels, resample_ohlc
from trader.database import Database

log = logging.getLogger(__name__)


def wait(until):
    print('Waiting until {}'.format(until))
    while pd.to_datetime('now', utc=True) < until:
        time.sleep(0.01)


def get_new_target_time(offset, now):

    if pd.Timedelta('0min') <= offset < pd.Timedelta('30min'):
        pass
    else:
        raise ValueError('Offsets 0 <= x < 30 only are supported. Not "{}"'.format(offset))
    rounded = now.floor('30min') + offset
    if rounded <= now:
        wait_until = rounded + pd.Timedelta(minutes=30)
    else:
        wait_until = rounded
    return wait_until

def warmup_model(model):
    """ Run once to allow tensorflow to optimize its runtime """
    model.predict(np.ones((1, 3, 50, 3)), np.ones((1, 4)))


def run_trading(client, model, pairs, assets, base_fund_value, min_notional_value, db_path, trader_info, test=True):

    is_binance_open(client)

    binance_rules = _gather_filters(client, pairs, ['BNBBTC', 'ETHBTC', 'BNBETH'])  # TODO: remove alt_pairs hack

    warmup_model(model)

    db = Database(db_path)
    db.register_trader(trader_info)

    while True:

        print('-----------------------------')

        is_binance_open(client)

        portf = Portfolio(account_balances(client, assets), assets)
        print('balances', portf.balances())

        target_time = get_new_target_time(trader_info['offset'], pd.Timestamp('now', tz='UTC'))
        print(target_time)

        if test and input('Skip wait? [Y/n]: ') == 'Y':
            print('Wait skipped!')
        else:
            wait(target_time)

        start_time = time.time()

        db.write_balance_before(portf.balances(), target_time)

        start, end = get_fetch_start_end(target_time, pd.Timedelta('5min'), minutes=1500)
        data = get_market_datas_parallel(client, pairs, start, end, Client.KLINE_INTERVAL_5MINUTE)

        for key, value in data.items():
            data[key] = resample_ohlc(value, '30min')
            assert len(data[key]) == 50

        prices = prices_from_market_data(data, pairs)

        db.write_decision_prices(prices, target_time)

        supplement_prices(prices)

        states = market_data_to_input_format(data, pairs)

        portf_dict = portf.get(prices, deduct=base_fund_value)
        portf_total_value = Portfolio.total_value(portf_dict)
        print('Portfolio total value:', portf_total_value, 'USDT')

        db.write_allocs_before(portf_to_alloc_dict(portf_dict), target_time)

        # Get prediction
        portf_vec = Portfolio.as_alloc_vector(portf_dict, assets)
        print('PortfVec Before:', portf_vec)

        portf_target_alloc_vec = model.predict(states, portf_vec)
        print('Preds:', portf_target_alloc_vec)
        portf_target_allocs = Portfolio.as_alloc_dict(portf_target_alloc_vec, assets)

        db.write_allocs_predictions(portf_target_allocs, target_time)

        value_changes = portf.get_value_changes(portf_dict, portf_target_allocs)

        order_factory = OrderFactory(binance_rules, min_notional_value)
        orders = order_factory.get_orders(value_changes, prices)
        print('orders planned:', orders)

        order_executer = OrderExecuter()
        if test:
            print('TEST MODE, NO TRADING EXECUTED!')
        else:
            responses = order_executer.apply_market_orders(client, orders)
            for resp in responses:
                if type(resp) is dict:  # Throw away exceptions
                    resp.update(compute_total_commission(resp))
                    db.write_order(resp, target_time)
                print(resp)

        print('Trading loop time:', time.time() - start_time)

        # Print portfolio after
        balances = account_balances(client, assets)
        portf = Portfolio(balances, assets)
        portf_dict = portf.get(prices, deduct=base_fund_value)
        portf_total_value = Portfolio.total_value(portf_dict)
        print('Portfolio total value after:', portf_total_value, 'USDT')

        db.write_allocs_after(portf_to_alloc_dict(portf_dict), target_time)
        portf_vec = Portfolio.as_alloc_vector(portf_dict, assets)
        print('PortfVec After:', portf_vec)

        # Get balances after
        portf_after = Portfolio(account_balances(client, assets), assets)
        db.write_balance_after(portf_after.balances(), target_time)


class Portfolio:
    """
    Allows to construct portfolio of assets and their values, quantities and allocations.
    When provided with current market prices.

    Allows deducting (i.e. hiding) a fixed value from the portfolio.

    Value means the cash value of the asset's holding amount.
    quantity (qty) is the actual amount.

    """

    def __init__(self, balances, assets):

        # Considers only free balances
        balances_dict = {}
        for name, amt in balances.items():
            if name in assets:
                balances_dict[name] = amt

        self._balances = balances_dict

    def balances(self):
        return deepcopy(self._balances)

    def get(self, prices, deduct=None):

        portf = {}

        # Values and qtys
        for asset, qty in deepcopy(self._balances).items():
            portf[asset] = {}
            portf[asset]['qty'] = qty
            if asset == 'USDT':
                portf[asset]['value'] = qty
            else:
                portf[asset]['value'] = qty * prices[asset]['USDT']

        # Do deductions
        if deduct is not None:
            if type(deduct) != Decimal:
                err = ValueError('Deduction value should be type Decimal, was {}'.format(type(deduct)))
                log.error(err)
                raise err
            portf = Portfolio._hide_value(portf, prices, deduct)

        # Compute total value
        total_value = Decimal('0')
        for asset, data in portf.items():
            total_value += data['value']

        # Allocations
        for asset, data in portf.items():
            portf[asset]['alloc'] = portf[asset]['value'] / total_value

        # Out {'USDT': {'value', 'qty', 'alloc'}}
        return portf

    @staticmethod
    def total_value(portf):

        total = Decimal('0')
        assets = portf.keys()
        for asset in assets:
            total += portf[asset]['value']
        return total

    @staticmethod
    def as_alloc_vector(portf, assets):

        portf_vec = np.zeros((1, len(assets)))

        for i, asset in enumerate(assets):
            portf_vec[0, i] = float(portf[asset]['alloc'])

        return portf_vec

    @staticmethod
    def as_alloc_dict(portf_vec, assets):

        portf = {}
        for i, asset in enumerate(assets):
            portf[asset] = Decimal(str(portf_vec[0, i]))

        return portf

    @staticmethod
    def get_value_changes(portf, allocs):
        """ Computes value changes between allocs and portf.
        I.e. allocs.as_values - portf

        Parameters
        ----------
        portf : dict
            Portfolio dictionary, e.g. {'USDT': {'value', 'qty', 'alloc'}, ...}
        allocs : dict
            Allocations, e.g. {'USDT': Decimal('0.3'), 'BTC': Decimal('0.7')}

        Returns
        -------
        dict
            E.g. {'USDT': {'value': Decimal('30')} , 'BTC': {'value': Decimal('70')}}

        """

        changes = {}

        value_to_manage = Decimal('0')
        for _, data in portf.items():
            value_to_manage += data['value']

        for asset, alloc in allocs.items():
            changes[asset] = {}
            target_val = value_to_manage * alloc
            changes[asset]['value'] = target_val - portf[asset]['value']

        return changes

    @staticmethod
    def vector(portf, data):
        pass

    @staticmethod
    def _hide_value(portf, prices, hide_value):
        """ Subtracts 'hide_value' from portfolio."""

        portf_hidden = deepcopy(portf)

        # Hide value
        for asset, data in portf.items():
            val = data['value'] - hide_value

            if val < Decimal('0'):
                val = Decimal('0')

            portf_hidden[asset]['value'] = val

        # Hide qty
        for asset, data in portf.items():

            if asset == 'USDT':
                qty = data['qty'] - hide_value

            else:
                val_as_qty = hide_value / prices[asset]['USDT']
                qty = data['qty'] - val_as_qty

            if qty < Decimal('0'):
                qty = Decimal('0')

            portf_hidden[asset]['qty'] = qty

        return portf_hidden


class OrderExecuter:

    def apply_market_orders(self, client, orders):

        responses = []

        for order in orders:
            symbol = order['pair']
            quantity = str(order['qty'])
            side = order['side']

            if order['qty'].is_zero():
                continue  # Skip zero order

            if side not in ['buy', 'sell']:
                err = ValueError('Order had unknown side "{}"'.format(side))
                log.error(err)
                raise err

            try:
                if side == 'buy':
                    resp = client.order_market_buy(symbol=symbol, quantity=quantity)
                else:
                    resp = client.order_market_sell(symbol=symbol, quantity=quantity)

                responses.append(resp)
                log.info('Executed order:', order)
                log.debug(resp)

            except BinanceAPIException as err:
                log.error(err)
                responses.append(err)

        return responses


class OrderFactory:
    """
    Create orders from value changes

    """
    def __init__(self, rules, global_min_notional):

        self.global_min_notional = global_min_notional
        self.rules = rules

    def get_orders(self, value_changes, prices):

        flows = OrderFactory._value_flows(value_changes)
        flows = OrderFactory._flows_to_real_markets(flows, prices)

        orders = OrderFactory._orders_from_flows(flows, prices)

        OrderFactory._apply_lot_filter(orders, self.rules)
        OrderFactory._apply_min_value_notional_filter(orders, prices, self.global_min_notional)

        return orders

    @staticmethod
    def _value_flows(changes):

        changes_pos = [(key, data['value']) for key, data in changes.items() if data['value'] > Decimal('0')]
        changes_neg = [(key, data['value']) for key, data in changes.items() if data['value'] < Decimal('0')]

        changes_pos = sorted(changes_pos, key=lambda x: x[1], reverse=True)  # most positive first
        changes_neg = sorted(changes_neg, key=lambda x: x[1])  # most negative first

        flows = []

        for buy_asset, to_buy in changes_pos:
            if to_buy == Decimal('0'):
                continue

            for i, item in enumerate(changes_neg):
                sell_asset, to_sell = item
                if to_sell.is_zero():
                    continue

                elif to_buy > abs(to_sell):
                    # Use all of to_sell and continue to next
                    to_buy -= abs(to_sell)
                    flows.append({'from': sell_asset, 'to': buy_asset, 'value': abs(to_sell)})
                    to_sell = Decimal('0')

                    changes_neg[i] = (sell_asset, to_sell)
                    continue

                else:
                    # Fill fully to_buy and break
                    flows.append({'from': sell_asset, 'to': buy_asset, 'value': to_buy})
                    to_sell += to_buy
                    to_buy = Decimal('0')
                    changes_neg[i] = (sell_asset, to_sell)
                    break

        return flows

    @staticmethod
    def _flows_to_real_markets(flows, prices):

        orders = []
        for flow in flows:

            base, quote = flow['from'], flow['to']
            ord_type = 'sell'

            # Switch to buy order if needed (not all markets are available)
            if base not in prices.keys() or quote not in prices[base].keys():
                base, quote = flow['to'], flow['from']  # reversed
                ord_type = 'buy'

                # Confirm that the new base/quote exists
                if base not in prices.keys() or quote not in prices[base].keys():
                    err = ValueError('No price info for {}'.format(base + quote))
                    log.error(err)
                    raise err

            order = {'pair': base + quote, 'value': flow['value'], 'side': ord_type, 'base': base, 'quote': quote}
            orders.append(order)

        return orders

    @staticmethod
    def _orders_from_flows(flows, prices):

        orders = []
        for flow in flows:
            base, quote = flow['base'], flow['quote']
            side = flow['side']
            order_type = 'market'

            order = {'pair': base + quote, 'side': side, 'type': order_type,
                     'qty': flow['value'] / prices[base]['USDT'], 'base': base, 'quote': quote}
            orders.append(order)

        return orders

    @staticmethod
    def _find_filter_by_name(rules, name):
        return list(filter(lambda x: x['filterType'] == name, rules))[0]

    @staticmethod
    def _apply_lot_filter(orders, rules):

        for order in orders:
            fltr = OrderFactory._find_filter_by_name(rules[order['pair']], 'LOT_SIZE')
            # max_qty = fltr('maxQty').rstrip('0')
            min_qty = fltr['minQty'].rstrip('0')  # removing trailing zeros because it confuses Decimal.quantize
            step_size = fltr['stepSize'].rstrip('0')

            qty = order['qty']

            qty = qty.quantize(Decimal(step_size), rounding=decimal.ROUND_DOWN)

            if qty < Decimal(min_qty):
                qty = Decimal('0')

            order['qty'] = qty

    @staticmethod
    def _apply_min_value_notional_filter(orders, prices, min_value):

        for order in orders:

            notional = prices[order['base']]['USDT'] * order['qty']
            if notional < Decimal(min_value):
                order['qty'] = Decimal('0')


def _apply_price_filter(price, filters):
    fltr = _find_filter_by_name(filters, 'PRICE_FILTER')
    # max_price = fltr('maxPrice').rstrip('0')
    min_price = fltr['minPrice'].rstrip('0')  # removing trailing zeros becaus it confuses Decimal.quantize
    tick_size = fltr['tickSize'].rstrip('0')

    price = price.quantize(Decimal(tick_size), rounding=decimal.ROUND_DOWN)

    if price < Decimal(min_price):
        price = Decimal('0')

    return price


def _gather_filters(client, pairs, alt_pairs):
    # See binance doc on filters
    # https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#filters

    filters = {}
    for pair in pairs + alt_pairs:
        filters[pair] = client.get_symbol_info(pair)['filters']

    return filters


def account_balances(client, assets):
    balances = client.get_account()['balances']

    balances_dict = {}
    for item in balances:
        item_name = item['asset']
        if item_name in assets:
            balances_dict[item_name] = Decimal(item['free'])

    return balances_dict


def prices_from_market_data(data, pairs):
    prices = {}

    for pair in pairs:
        df = data[pair]
        name = pair.split('USDT')[0]
        prices[name] = {'USDT': Decimal(str(df.iloc[-1].close))}

    return prices


def supplement_prices(prices):
    # TODO: Temporary hack to add price entries for some assets
    prices['BNB']['ETH'] = None
    prices['ETH']['BTC'] = None
    prices['BNB']['BTC'] = None


def portfolio_allocations(balances, prices):
    allocs = {}

    # Total value in USDT
    total = Decimal('0')

    for asset, balance in balances.items():
        if asset == 'USDT':
            total += balance
        else:
            total += balance * prices[asset]

    # Compute allocs
    for asset, balance in balances.items():
        if asset == 'USDT':
            allocs[asset] = balance / total
        else:
            allocs[asset] = balance * prices[asset] / total

    return allocs, total


def get_market_data(client, pair, start, end, interval):
    """ Fetches data for pairs.

    Usually takes 0.5-1 seconds per pair.

    Returns
    -------
    Dict of DataFrames where keys are pair names i.e. 'BNBUSDT'.

    """

    klines = client.get_historical_klines(pair, interval, start, end)
    klines = np.array(klines)[:, :6]  # Discard some columns

    df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df = pd.DataFrame(df).set_index('date')
    df.index = pd.to_datetime(df.index, infer_datetime_format=True, unit='ms', utc=True)
    return df.astype(np.float64)


def get_market_datas_parallel(client, pairs, start, end, interval):

    # To get result back from thread
    def wrap_output_variable(client, pair, results):
        results[pair] = get_market_data(client, pair, str(start), str(end), interval)

    threads = []

    results = {} #[None] * len(pairs)

    for i, pair in enumerate(pairs):
        t = Thread(target=wrap_output_variable, args=(client, pair, results))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # results_dict = {}
    # for pair, df in results:
    #     results_dict[pair] = df

    return results


def is_binance_open(client):

    my_now = int(pd.to_datetime('now', utc=True).timestamp() * 1000)
    server_now = client.get_server_time()['serverTime']

    if abs(my_now - server_now) > 5000:  # ms
        err = ValueError('Time difference too large. Server {}, PC {}'.format(server_now, my_now))
        log.error(err)
        raise err

    status = client.get_system_status()

    if status['status'] != 0:
        err = ValueError('Binance is under maintenance. Message {}'.format(status))
        log.error(err)
        raise err

    return True


def trim_market_data(df):
    """ Trims to length 50 based on current time.

    Examples
    --------
    Last index 13:00.

    If now > 13:15 then last index is dropped (excluded).
    If now <= 13:15 then last index is included.

    Let's say now=13:10
    The output's last index is then 12:30.

    Returns
    -------
    DataFrame
        Length 50. Copy of the original.

    Raises
    ------
    ValueError
        If after trimming the last index is not correct (data wasn't recent).

    """

    df = df.copy()

    desired_last_open = pd.to_datetime('now', utc=True).round('30min') - pd.to_timedelta('30min')

    if df.index[-1] > desired_last_open:
        # Drop last one
        df = df.iloc[:-1]

    if df.index[-1] != desired_last_open:
        err = ValueError('Data to trim is not recent. Last index was {}.'.format(df.index[-1]))
        log.error(err)
        raise err

    return df.tail(50)


def execute_order(client, pair, qty, price, order_rules):

    min_price = order_rules[pair]['minPrice']
    max_price = order_rules[pair]['maxPrice']

    price = str(price)
    order = client.create_order(
        symbol=pair,
        side=SIDE_BUY,
        type=ORDER_TYPE_LIMIT,
        timeInForce=TIME_IN_FORCE_GTC,
        quantity=qty,
        price='0.00001')

    return order


def apply_trading_filters(price, qty, filters):
    qty = _apply_lot_filter(qty, filters)
    price = _apply_price_filter(price, filters)
    qty = _apply_min_notional_filter(price, qty, filters)
    return price, qty


def market_data_to_input_format(df_dict, pairs):

    df_list = [df_dict[pair].copy() for pair in pairs]

    states = np.empty((1, len(df_list), 50, 3))
    for i, df in enumerate(df_list):
        df = df[['close', 'high', 'low']]
        df = df.append(df.tail(2))  # Because the state extractor wants to have enough data for labels
        df.index = pd.date_range(start=df.index[0], periods=52, freq='30min')
        states_, _ = extract_states_and_labels(df, 50)
        states[0, i, ...] = states_[0, 0, ...]

    return states


def decimal_to_str(val, prec):
    return '{:f}'.format(round(val, prec))


def portf_to_alloc_dict(portf_dict):
    alloc_dict = {}
    for key, subdict in portf_dict.items():
        alloc_dict[key] = subdict['alloc']
    return alloc_dict


def compute_total_commission(response):

    fills = response['fills']

    # commission assets should be same
    asset = fills[0]['commissionAsset']

    total = Decimal('0')
    for fill in fills:
        total += Decimal(fill['commission'])

        if fill['commissionAsset'] != asset:
            msg = 'Mixed commission assets. ClientOrderID: "{}"'.format(response['clientOrderId'])
            log.warning(msg)
            warnings.warn(msg)
            total = Decimal('0')
            asset = 'MIXED'
            break

    return {'commission': total, 'commissionAsset': asset}


def get_fetch_start_end(target_time, source_interval, minutes):
    end = target_time - source_interval
    start = end - pd.Timedelta('{}min'.format(minutes)) + source_interval
    return start, end
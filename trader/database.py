import sqlite3
import pandas as pd
import logging
import uuid
from decimal import Decimal as D
from queue import Queue
from threading import Thread
import numpy as np

log = logging.getLogger(__name__)


class Database:

    def __init__(self, db_path, precision=(10, 8)):
        self.prec = precision  # tuple(x, y)
        self.db_path = db_path
        self.queue = Queue()

        sqlite3.register_adapter(pd.Timestamp, lambda x: int(x.timestamp()))
        sqlite3.register_adapter(uuid.UUID, lambda x: x.hex)
        sqlite3.register_adapter(D, lambda x: decimal_to_str(x, self.prec[1]))
        sqlite3.register_adapter(pd.Timedelta, lambda x: x.seconds)

        sqlite3.register_converter('Time', lambda x: pd.to_datetime(x, utc=True))
        sqlite3.register_converter('TraderID', lambda x: uuid.UUID(x))

        self.conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys = 1')

        self._writer_thread = Thread(target=Database._writer_worker, args=(self.conn, self.queue))
        self._writer_thread.daemon = True
        self._writer_thread.start()

        self.trader_id = None

    def register_trader(self, trader_info):
        """ Write trader info and check that it doesn't conflict with previous data."""
        self.trader_id = trader_info['trader_id']
        model_name, version, cycle_time, offset = trader_info['model_name'], trader_info['version'], \
                                                  trader_info['cycle_time'], trader_info['offset']

        c = ' '.join(
            ['CREATE TABLE IF NOT EXISTS Trader',
             '(TraderID   TEXT NOT NULL,',
             'ModelName   TEXT,',
             'Version     TEXT,',
             'CycleTime   INT,',
             'Offset      INT,',
             'PRIMARY KEY (TraderID))'])

        with self.conn as conn:
            conn.execute(c)

        # Test if trader already in table
        c = ' '.join(['SELECT * FROM Trader WHERE TraderID=? LIMIT 1'], )
        with self.conn as conn:
            res = conn.execute(c, (self.trader_id,)).fetchone()
            if res is not None:
                if res[:] != (self.trader_id, model_name, version, cycle_time.seconds, offset.seconds):
                    msg = 'TradeID "{}" already exists with different other values: {}, {}, {}, {}.'.format(*res)
                    err = ValueError(msg)
                    log.error(err)
                    raise err

            # Insert if wasnt
            else:
                c = ' '.join(
                    ['INSERT INTO Trader',
                     '(TraderID, ModelName, Version, CycleTime, Offset)'
                     'VALUES (?, ?, ?, ?, ?)'])
                conn.execute(c, (self.trader_id, model_name, version, cycle_time, offset))

    def write_decision_prices(self, prices, time):
        self._check_registration()

        # Create table
        c = ' '.join(
            ['CREATE TABLE IF NOT EXISTS DecisionPrice',
             '(TraderID  TEXT NOT NULL,',
             'Time      INT NOT NULL,',
             'BTC       DECIMAL({}, {}),'.format(*self.prec),
             'ETH       DECIMAL({}, {}),'.format(*self.prec),
             'BNB       DECIMAL({}, {}),'.format(*self.prec),
             'PRIMARY KEY (TraderID, Time),'
             'FOREIGN KEY(TraderID) REFERENCES Trader(TraderID))'])

        self.queue.put((c, tuple()))

        # Insert
        c = ' '.join(
            ['INSERT INTO DecisionPrice',
             '(TraderID, Time, BTC, ETH, BNB)',
             'VALUES (?, ?, ?, ?, ?)'])

        self.queue.put((c, (self.trader_id, time, prices['BTC']['USDT'], prices['ETH']['USDT'], prices['BNB']['USDT'])))

    def write_order(self, order, time):

        self._check_registration()

        # Create table (if doesnt exist)
        c = ' '.join(
            ['CREATE TABLE IF NOT EXISTS ExecutedOrder',
             '(TraderID             TEXT NOT NULL,',
             'Time                  INT NOT NULL,',
             'Symbol                TEXT NOT NULL,',
             'OrderID               INT NOT NULL,',
             'ClientOrderID         TEXT NOT NULL,',
             'TransactTime          INT NOT NULL,',
             'Price                 DECIMAL({}, {}) NOT NULL,',
             'OrigQty               DECIMAL({}, {}) NOT NULL,',
             'ExecutedQty           DECIMAL({}, {}) NOT NULL,',
             'CumulativeQuoteQty    DECIMAL({}, {}) NOT NULL,',
             'Status                TEXT NOT NULL,',
             'TimeInForce           TEXT NOT NULL,',
             'Type                  TEXT NOT NULL,',
             'Side                  TEXT NOT NULL,',
             'Commission            DECIMAL({}, {}) NOT NULL,',
             'CommissionAsset       TEXT NOT NULL,',
             'PRIMARY KEY (ClientOrderID),',
             'FOREIGN KEY(TraderID) REFERENCES Trader(TraderID))']).format(*(self.prec * 5))

        self.queue.put((c, tuple()))

        # Insert
        c = ' '.join(
            ['INSERT INTO ExecutedOrder',
             '(TraderID, Time, Symbol, OrderID, ClientOrderID, TransactTime, Price, OrigQty, ExecutedQty,',
             'CumulativeQuoteQty, Status, TimeInForce, Type, Side, Commission, CommissionAsset)',
             'VALUES (?, ? ,? ,? ,? ,? ,? ,?, ?, ?, ?, ?, ?, ?, ?, ?)'
             ])

        keys = ['symbol', 'orderId', 'clientOrderId', 'transactTime', 'price', 'origQty', 'executedQty',
                'cummulativeQuoteQty', 'status', 'timeInForce', 'type', 'side', 'commission', 'commissionAsset']
        vals = tuple([order[key] for key in keys])
        self.queue.put((c, (self.trader_id, time) + vals))

    def write_allocs_before(self, allocs, time):
        self._write_allocations(allocs, time, 'AllocationBefore')

    def write_allocs_after(self, allocs, time):
        self._write_allocations(allocs, time, 'AllocationAfter')

    def write_allocs_predictions(self, allocs, time):
        self._write_allocations(allocs, time, 'AllocationPrediction')

    def write_balance_before(self, balances, time):
        self._write_balance(balances, time, 'BalanceBefore')

    def write_balance_after(self, balances, time):
        self._write_balance(balances, time, 'BalanceAfter')

    def _write_balance(self, balances, time, table_name):
        self._check_registration()

        c = ' '.join([
            'CREATE TABLE IF NOT EXISTS {}',
            '(TraderID  TEXT NOT NULL,',
            'Time       INT NOT NULL,',
            'USDT       DECIMAL({}, {}),',
            'BTC        DECIMAL({}, {}),',
            'ETH        DECIMAL({}, {}),',
            'BNB        DECIMAL({}, {}),',
            'PRIMARY KEY (TraderID, Time),',
            'FOREIGN KEY(TraderID) REFERENCES Trader(TraderID))'
        ]).format(*((table_name,) + self.prec * 4))

        self.queue.put((c, tuple()))

        c = ' '.join([
            'INSERT INTO {} (TraderID, Time, USDT, BTC, ETH, BNB)',
            'VALUES (? ,? ,?, ?, ?, ?)'
        ]).format(table_name)

        assets = ['USDT', 'BTC', 'ETH', 'BNB']

        self.queue.put((c, (self.trader_id, time) + tuple([balances[key] for key in assets])))

    def _write_allocations(self, allocs, time, table_name):
        self._check_registration()

        c = ' '.join([
            'CREATE TABLE IF NOT EXISTS {}',
            '(TraderID  TEXT NOT NULL,',
            'Time       INT NOT NULL,',
            'USDT       REAL,',
            'BTC        REAL,',
            'ETH        REAL,',
            'BNB        REAL,',
            'PRIMARY KEY (TraderID, Time),',
            'FOREIGN KEY(TraderID) REFERENCES Trader(TraderID))'
        ]).format(*((table_name,) + self.prec * 4))

        self.queue.put((c, tuple()))

        c = ' '.join([
            'INSERT INTO {} (TraderID, Time, USDT, BTC, ETH, BNB)',
            'VALUES (?, ?, ?, ?, ?, ?)'
        ]).format(table_name)

        vals = tuple([allocs[key] for key in ['USDT', 'BTC', 'ETH', 'BNB']])

        if abs(1 - sum(vals)) > 0.01:
            err = ValueError('Allocations do not sum to one. Sum was {}'.format(sum(vals)))
            log.error(err)
            raise err
        else:
            self.queue.put((c, (self.trader_id, time) + vals))

    def _check_registration(self):
        if self.trader_id == None:
            err = RuntimeError('You must register trader before calling writing methods.')
            log.error(err)
            raise err

    @staticmethod
    def _writer_worker(connection, queue):
        while True:
            command, data = queue.get()
            with connection as conn:
                conn.execute(command, data)
            queue.task_done()

def decimal_to_str(val, prec):
    return '{:f}'.format(round(val, prec))


def new_uuid():
    return uuid.uuid4().hex


def validate_uuid(id):
    uuid.UUID(id)


class DataQuery:

    def __init__(self, db_path, timeout=50):

        self.conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)

    def recently_active_ids(self, after):
        c = ' '.join([
            'SELECT TraderID FROM BalanceAfter WHERE Time > ? GROUP BY TraderID'
        ])
        return pd.read_sql(c, self.conn, params=(after,))

    def trader_info(self, trader_id=None):
        if trader_id is not None:
            c = ' '.join([
                'SELECT * FROM Trader WHERE TraderID=?'
            ])
            params = (trader_id,)
        else:
            c = ' '.join([
                'SELECT * FROM Trader'
            ])
            params = tuple()

        return pd.read_sql(c, self.conn, params=params)

    def decision_price(self, trader_id):
        c = ' '.join([
            'SELECT * FROM DecisionPrice WHERE TraderID=? ORDER BY Time ASC'
        ])
        return pd.read_sql(c, self.conn, params=(trader_id,), parse_dates='Time', index_col='Time')

    def balance_before(self, trader_id):
        c = ' '.join([
            'SELECT * FROM BalanceBefore WHERE TraderID=? ORDER BY Time ASC'
        ])
        return pd.read_sql(c, self.conn, params=(trader_id,), parse_dates='Time', index_col='Time')

    def balance_after(self, trader_id):
        c = ' '.join([
            'SELECT * FROM BalanceAfter WHERE TraderID=? ORDER BY Time ASC'
        ])
        return pd.read_sql(c, self.conn, params=(trader_id,), parse_dates='Time', index_col='Time')

    def executed_orders(self, trader_id):
        c = ' '.join([
            'SELECT * FROM ExecutedOrder WHERE TraderID=? ORDER BY Time ASC'
        ])
        return pd.read_sql(c, self.conn, params=(trader_id,), parse_dates='Time', index_col='Time')


def get_relative_price_changes(prices, cycle_time):
    prices_prev = prices.drop('TraderID', axis=1).astype(np.float64)
    prices_next = prices_prev.copy()
    prices_next.index = prices_next.index - pd.Timedelta(cycle_time)

    relative_price_change = prices_next.divide(prices_prev, axis='columns')

    return drop_nan_rows(relative_price_change)


def get_usdt_balances(prices, balances):
    balances = balances.drop('TraderID', axis=1).astype(np.float64)
    prices = prices.drop('TraderID', axis=1)

    usdt_holdings = balances[['USDT']]
    usdt_balances = balances.multiply(prices, axis='columns')
    usdt_balances[['USDT']] = usdt_holdings

    usdt_balances = drop_nan_rows(usdt_balances)
    return usdt_balances


def get_usdt_balances_after_period(usdt_balances_after, rel_price_change):
    usdt_holdings = usdt_balances_after[['USDT']]
    usdt_balances_after_period = usdt_balances_after.multiply(rel_price_change)
    usdt_balances_after_period.update(usdt_holdings)
    return drop_nan_rows(usdt_balances_after_period)


def get_portfolio_relative_changes(usdt_balances_after_period, usdt_balances_before):
    balance_before = usdt_balances_before.sum(axis=1)
    balance_after = usdt_balances_after_period.sum(axis=1)
    change = balance_after.divide(balance_before)
    return drop_nan_rows(change)


def log_cumsum(portf_relative_change):
    portf_relative_change_cum = portf_relative_change.copy()
    portf_relative_change_cum[:] = np.cumsum(np.log(portf_relative_change.values))
    return portf_relative_change_cum


def relative_uniform_portfolio(rel_price_change):
    uniform_portf = (rel_price_change / len(rel_price_change.columns)).sum(axis=1)
    return uniform_portf


def drop_nan_rows(df):
    df = df.copy()
    if type(df) == pd.Series:
        return df[~df.isnull()]
    else:
        return df[~df.isnull().any(axis=1)]


def get_trading_volume(prices, orders):
    """ Computes trading volume as USDT (pandas series) """

    # Creates mapping from pairs to symbols, e.g. BNBETH -> ETH
    quotes = ['USDT', 'BTC', 'ETH', 'BNB']
    pair_to_quote_map = binance_pairs_map(bases=quotes, quotes=quotes, mode='quote')

    # Using loc (instead of df['colname'] to suppress SettingWithCopyWarning
    orders = orders.loc[:, ['Symbol', 'CumulativeQuoteQty']]
    orders['QuoteSymbol'] = orders['Symbol'].map(pair_to_quote_map)

    orders = orders.join(prices)
    orders['CumulativeUSDTQty'] = np.nan

    # Use default index temporarily (0,1,2..) bc "Time" may have duplicates.
    orders['Time'] = orders.index
    orders.reset_index(drop=True, inplace=True)

    # Compute USDT values for each quote
    for quote in quotes:
        df = orders[orders['QuoteSymbol'] == quote]

        if quote == 'USDT':
            df = df['CumulativeQuoteQty']
        else:
            df = df['CumulativeQuoteQty'] * df[quote]
            orders.drop(quote, axis=1, inplace=True)  # Quote col not needed anymore

        df.name = 'CumulativeUSDTQty'
        orders.update(df)

    # Restore time index
    orders.set_index('Time', inplace=True)

    volume = orders['CumulativeUSDTQty']
    volume = volume.groupby(level=0).sum()

    return volume


def get_commissions(prices, orders):
    """ Computes trading commissions as USDT (pandas series) """

    comm_assets = ['USDT', 'BTC', 'ETH', 'BNB']
    orders = orders.loc[:, ['Commission', 'CommissionAsset']]
    orders = orders.join(prices)
    orders['CommissionUSDT'] = np.nan

    # Use default index temporarily (0,1,2..) bc "Time" may have duplicates.
    orders['Time'] = orders.index
    orders.reset_index(drop=True, inplace=True)

    # Compute USDT values for each asset type
    for asset in comm_assets:
        df = orders[orders['CommissionAsset'] == asset]

        if asset == 'USDT':
            df = df['Commission']
        else:
            df = df['Commission'] * df[asset]
            orders.drop(asset, axis=1, inplace=True)  # Quote col not needed anymore

        df.name = 'CommissionUSDT'
        orders.update(df)

    # Restore time index
    orders.set_index('Time', inplace=True)

    comms = orders['CommissionUSDT']
    comms = comms.groupby(level=0).sum()

    return comms


def binance_pairs_map(bases, quotes, mode='both'):
    # Creates mapping from pairs to symbols, e.g. BNBETH -> base:BNB quote:ETH
    pair_to_quote_map = {}
    for base in bases:
        for quote in quotes:
            if mode == 'both':
                pair_to_quote_map[base + quote] = (base, quote)
            elif mode == 'base':
                pair_to_quote_map[base + quote] = base
            elif mode == 'quote':
                pair_to_quote_map[base + quote] = quote
            else:
                raise ValueError('Mode must be one of [both, base, quote]!')

    return pair_to_quote_map


def get_slippage_for_usdt_quotes(prices, orders):
    """ Slippage only for USDT quote, e.g. ETHUSDT. Values as relative values actualPrice / desiredPrice - 1"""

    # Creates mapping from pairs to symbols, e.g. BNBETH -> ETH
    quotes = ['USDT', 'BTC', 'ETH', 'BNB']
    bases = ['BTC', 'ETH', 'BNB']
    pair_to_base_map = binance_pairs_map(bases=bases, quotes=quotes, mode='base')

    # Using loc (instead of df['colname'] to suppress SettingWithCopyWarning
    orders = orders.loc[:, ['Symbol', 'ExecutedQty', 'CumulativeQuoteQty']]
    orders['BaseSymbol'] = orders['Symbol'].map(pair_to_base_map)
    orders['Slippage'] = np.nan
    orders = orders.join(prices)

    # Use default index temporarily (0,1,2..) bc "Time" may have duplicates.
    orders['Time'] = orders.index
    orders.reset_index(drop=True, inplace=True)

    # Compute actual prices for each USDT quotes only
    for base in bases:
        df = orders[orders['Symbol'] == base + 'USDT']

        actual_price = df['CumulativeQuoteQty'] / df['ExecutedQty']
        slippage = actual_price / df[base].astype(np.float64)

        slippage.name = 'Slippage'
        orders.update(slippage)

    # Restore time index
    orders.set_index('Time', inplace=True)

    slippage = orders['Slippage'].dropna() - 1.

    return slippage

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import os
import time

from trader.database import get_usdt_balances, log_cumsum, relative_uniform_portfolio, \
    get_usdt_balances_after_period, get_portfolio_relative_changes, get_relative_price_changes, \
    get_slippage_for_usdt_quotes, get_trading_volume

from webplot.app_setup import app_setup

app, cache, db, conf = app_setup(os.environ['WEBPLOT_CONF'])

helptips = {}
helptips['perf'] = html.P(
    'Example: Relative performance of 5% means that the trader performed +5%-units better than the market.')

helptips['vol'] = html.P(
    'Dollar amount for all trades executed.')

helptips['slippage'] = html.P(
    'Slippage is the difference between the expected price and the price at which the trade was executed. '
    'Value 0.1% (x-axis) means that the actual price was 0.1% WORSE than expected, i.e. the more negative the better.')

helptips['summary'] = """
Artificial intelligence can be used to trade cryptocurrencies.

This website summarises various metrics for all currently active traders.

Traders are operating on [Binance](https://www.binance.com).



Author: Tomi Sepponen



"""

colors = {'header': '#1b417d', 'bg': '#E6E6FA'}
horiz_padding = '20px'

app.layout = html.Div(style={'background': colors['bg']}, children=[
    html.H1(children='Performance Report', style={'background': colors['header'], 'color': '#ffffff', 'text-align': 'center',
                                                'margin': '0px', 'padding': '25px',
                                                  'text-shadow': '0 -3px 3px rgba(0,0,0,0.25)'}),

    dcc.Markdown(helptips['summary'], style={'background': colors['bg'], 'margin': '0px', 'line-height': '75%',
                                             'padding': '18px'}),

    dcc.Dropdown(
            id='select-model-ddown',
            value='',
            style={'width': '100%'}
        ),

    html.Div(children=[dcc.Graph(id='performance_graph'), html.Div(children=[helptips['perf']], className='help-tip')],
             style={'position': 'relative', 'background': colors['bg'], 'padding-top': horiz_padding}),

    html.Div(children=[dcc.Graph(id='volume_graph'), html.Div(children=[helptips['vol']], className='help-tip')],
                 style={'position': 'relative', 'background':  colors['bg'], 'padding-top': horiz_padding}),

    html.Div(children=[dcc.Graph(id='slippage_graph'), html.Div(children=[helptips['slippage']], className='help-tip')],
                     style={'position': 'relative', 'background':  colors['bg'], 'padding-top': horiz_padding,
                            'padding-bottom': horiz_padding}),

    html.Div(id='hidden-div', style={'display': 'none'})
])


@app.callback(
    Output('select-model-ddown', 'options'), [Input('select-model-ddown', 'id')])
def update_agent_list(_):

    recent_ids = get_recent_trader_ids(minutes=conf['recent_cutoff']).sort_values('TraderID')

    options = [{'label': 'All Active Traders', 'value': ''}]

    for trader_id in recent_ids.values:
        trader_id = trader_id[0]
        info = db.trader_info(trader_id)
        name = info['ModelName'].values[0]
        label = '{} ({})'.format(name, trader_id[:4])
        options.append({'label': label, 'value': trader_id})

    return options


@app.callback(
    Output('performance_graph', 'figure'),
    [Input('select-model-ddown', 'value')])
def update_performance_graph(trader_id):

    if trader_id != '':
        y, yb = trader_performance(trader_id)
        y = (y - 1.) * 100.
        yb = (yb - 1.) * 100.
        figure = {
            'data': [
                go.Scatter(x=yb.index, y=yb, line={'color': 'grey', 'dash': 'dash'}, name='Market'),
                go.Scatter(x=y.index, y=y, name='Trader')
            ],
            'layout': {'title': 'Trader Absolute Performance (%)'}
        }

    else:
        recent_ids = get_recent_trader_ids(minutes=conf['recent_cutoff'])
        data = []

        all_y_diff = []  # collect relative performances

        for trader_id in recent_ids.sort_values('TraderID').values:
            trader_id = trader_id[0]
            info = db.trader_info(trader_id)
            name = info['ModelName'].values[0].split('_')[-1]
            y, yb = trader_performance(trader_id)
            y *= 100.
            yb *= 100.

            y_diff = y - yb
            all_y_diff.append(y_diff)

            label = '{} ({})'.format(name, trader_id[:4])
            data.append(go.Scatter(x=y_diff.index, y=y_diff, name=label))

        # Compute MEAN performance
        # Floor indices and to_frame
        for i in range(len(all_y_diff)):
            all_y_diff[i].index = all_y_diff[i].index.floor('30min')
            all_y_diff[i] = all_y_diff[i].to_frame(str(i))

        combined_df = all_y_diff[0]
        for i in range(1, len(all_y_diff)):
            combined_df = combined_df.join(all_y_diff[i])

        mean_performance = combined_df.mean(axis=1, skipna=True)

        data.insert(0, go.Scatter(x=mean_performance.index, y=mean_performance, name='MEAN',
                                  line={'color': 'black'}))

        figure = {
            'data': data,
            'layout': {'title': 'Performance Relative To The Market (%-units)'}
        }

    return figure


@app.callback(
    Output('slippage_graph', 'figure'),
    [Input('select-model-ddown', 'value')])
def update_slippage_graph(trader_id):

    if trader_id != '':
        slippage = trader_slippage(trader_id)

        # Convert to %units
        slippage *= 100.

        data = [go.Histogram(x=slippage)]

    else:
        recent_ids = get_recent_trader_ids(minutes=conf['recent_cutoff'])

        combined_slippage = None
        for trader_id in recent_ids.sort_values('TraderID').values:
            trader_id = trader_id[0]
            slippage = trader_slippage(trader_id)
            if combined_slippage is None:
                combined_slippage = slippage
            else:
                combined_slippage = combined_slippage.append(slippage)

        # Convert to %units
        combined_slippage *= 100.

        data = [go.Histogram(x=combined_slippage)]

    figure = {
        'data': data,
        'layout': {'title': 'Slippage (%-units)'}
    }

    return figure


@app.callback(
    Output('volume_graph', 'figure'),
    [Input('select-model-ddown', 'value')])
def update_volume_graph(trader_id):

    if trader_id != '':
        traders_selected = [[trader_id]]
    else:
        traders_selected = get_recent_trader_ids(minutes=conf['recent_cutoff'])
        traders_selected = traders_selected.sort_values('TraderID').values
    data = []

    total = 0

    for trader_id in traders_selected:
        trader_id = trader_id[0]
        info = db.trader_info(trader_id)
        prices, _, _, orders = database_query(trader_id)

        volume = get_trading_volume(prices, orders)
        total += volume.sum()
        name = info['ModelName'].values[0].split('_')[-1]

        label = '{} ({})'.format(name, trader_id[:4])
        data.append(go.Bar(x=volume.index, y=volume, name=label))

    figure = {
        'data': data,
        'layout': {'title': 'Trading volume\n(Total ${})'.format(int(total))}
    }

    return figure


def trader_performance(trader):
    prices, balances_before, balances_after, _ = database_query(trader)
    rel_price_change = get_relative_price_changes(prices, '30min')

    usdt_balances_before = get_usdt_balances(prices, balances_before)
    usdt_balances_after = get_usdt_balances(prices, balances_after)

    usdt_balances_after_period = get_usdt_balances_after_period(usdt_balances_after, rel_price_change)
    portf_relative_change = get_portfolio_relative_changes(usdt_balances_after_period, usdt_balances_before)

    portf_cumsum = log_cumsum(portf_relative_change)
    y = portf_cumsum.apply(np.exp)

    # Benchmark
    uniform = relative_uniform_portfolio(rel_price_change)
    bench_portf_cumsum = log_cumsum(uniform)
    bench_portf_cumsum = bench_portf_cumsum[y.index]  # Drop indices not in x
    y_bench = bench_portf_cumsum.apply(np.exp)

    return y, y_bench  # both type Series


def trader_slippage(trader):
    prices, _, _, orders = database_query(trader)
    slippage = get_slippage_for_usdt_quotes(prices, orders)
    return slippage


@cache.memoize()
def database_query(trader):
    prices = db.decision_price(trader)
    balances_before = db.balance_before(trader)
    balances_after = db.balance_after(trader)
    executed_orders = db.executed_orders(trader)
    return prices, balances_before, balances_after, executed_orders


@cache.cached()
def get_all_traders():
    info = db.trader_info()
    return info


@cache.memoize()
def get_recent_trader_ids(minutes=60):
    recent_ids = db.recently_active_ids(int(time.time() - minutes * 60))
    return recent_ids


def run_server():
    app.run_server(debug=False, host=os.environ['WEBPLOT_HOST'], port=int(os.environ['WEBPLOT_PORT']))
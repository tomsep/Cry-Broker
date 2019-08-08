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
    get_usdt_balances_after_period, get_portfolio_relative_changes, get_relative_price_changes

from webplot.app_setup import app_setup

app, cache, db, conf = app_setup(os.environ['WEBPLOT_CONF'])

app.layout = html.Div(children=[
    html.H1(children='Performance Report'),

    dcc.Dropdown(
        id='select-model-ddown',
        value=''
    ),

    html.Div(children='''
        Select trader to plot
    '''),

    dcc.Graph(
        id='graph'
    ),
    html.Div(id='hidden-div', style={'display': 'none'})
])


@app.callback(
    Output('select-model-ddown', 'options'), [Input('select-model-ddown', 'id')])
def update_agent_list(_):

    recent_ids = get_recent_trader_ids(minutes=conf['recent_cutoff']).sort_values('TraderID')

    options = [{'label': 'ALL', 'value': ''}]

    for trader_id in recent_ids.values:
        trader_id = trader_id[0]
        info = db.trader_info(trader_id)
        name = info['ModelName'].values[0]
        label = '{} ({})'.format(name, trader_id[:4])
        options.append({'label': label, 'value': trader_id})

    return options


@app.callback(
    Output('graph', 'figure'),
    [Input('select-model-ddown', 'value')])
def update_graph(trader_id):

    if trader_id != '':
        x, y = trader_performance(trader_id)
        xb, yb = market_performance(trader_id)

        figure = {
            'data': [
                go.Scatter(x=xb, y=yb, line={'color': 'grey', 'dash': 'dash'}, name='Market'),
                go.Scatter(x=x, y=y, name='Trader')
            ],
            'layout': {'title': 'Trader Performance'}
        }

    else:
        recent_ids = get_recent_trader_ids(minutes=conf['recent_cutoff'])
        data = []

        for trader_id in recent_ids.sort_values('TraderID').values:
            trader_id = trader_id[0]
            info = db.trader_info(trader_id)
            name = info['ModelName'].values[0]
            x, y = trader_performance(trader_id)
            xb, yb = market_performance(trader_id)

            y_diff = y - yb
            label = '{} ({})'.format(name, trader_id[:4])
            data.append(go.Scatter(x=x, y=y_diff, name=label))

        figure = {
            'data': data,
            'layout': {'title': 'Performance Relative To The Market'}
        }

    return figure


def trader_performance(trader):

    prices, balances_before, balances_after = database_query(trader)
    rel_price_change = get_relative_price_changes(prices, '30min')

    usdt_balances_before = get_usdt_balances(prices, balances_before)
    usdt_balances_after = get_usdt_balances(prices, balances_after)

    usdt_balances_after_period = get_usdt_balances_after_period(usdt_balances_after, rel_price_change)
    portf_relative_change = get_portfolio_relative_changes(usdt_balances_after_period, usdt_balances_before)

    portf_cumsum = log_cumsum(portf_relative_change)[:-1]
    x = portf_cumsum.index
    y = np.exp(portf_cumsum.values)
    return x, y


def market_performance(trader):
    # TODO: query all prices and take the first existing item for each time entry
    prices, balances_before, balances_after = database_query(trader)
    rel_price_change = get_relative_price_changes(prices, '30min')
    uniform = relative_uniform_portfolio(rel_price_change)
    portf_cumsum = log_cumsum(uniform)[:-1]
    x = portf_cumsum.index
    y = np.exp(portf_cumsum.values)
    return x, y


@cache.memoize()
def database_query(trader):
    prices = db.decision_price(trader)
    balances_before = db.balance_before(trader)
    balances_after = db.balance_after(trader)
    return prices, balances_before, balances_after


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
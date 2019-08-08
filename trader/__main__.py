import yaml
import fire
import logging
from decimal import Decimal
from binance.client import Client
import time
import os
import pandas as pd

from trader.trading import run_trading
from trader.database import validate_uuid, new_uuid
from trader.utils import PushMessager
from trader.lite_model_inference import LiteModel


log = logging.getLogger(__name__)


def load_settings(path):

    with open(path) as f:
        sett = yaml.safe_load(f)

    id = sett['trader_info']['trader_id']
    try:
        validate_uuid(id)
    except Exception as err:
        print('Bad TraderID: {}. New ID: "{}"'.format(id, new_uuid()))
        raise err

    sett['min_notional'] = Decimal(sett['min_notional'])
    sett['base_fund_value'] = Decimal(sett['base_fund_value'])
    sett['trader_info']['cycle_time'] = pd.Timedelta(sett['trader_info']['cycle_time'], tz='UTC')
    sett['trader_info']['offset'] = pd.Timedelta(sett['trader_info']['offset'], tz='UTC')

    model_load_path = sett['model_load_path']
    if not os.path.isfile(model_load_path):
        raise FileNotFoundError('Model file doesnt exist "{}"'.format(model_load_path))

    # Load keys
    with open(sett['key_file_path']) as f:
        apikeys_data = yaml.safe_load(f)

    sett.update(apikeys_data)

    return sett


class Deeptrader:

    def start_trading(self, config):
        # TODO: option to disable CUDA

        sett = load_settings(config)

        trading_execution_handler(sett)

def get_model(sett):
    model = LiteModel(sett['model_load_path'])
    return model

def trading_execution_handler(setts):

    test_mode = setts['test_orders_only']
    if test_mode:
        input('USING ONLY TESTING ORDERS! Enter to continue: ')
    else:
        input('USING REAL ORDERS! Enter to continue: ')

    agnt = get_model(setts)

    client = Client(setts['api_key'], setts['api_secret'])

    msg_client = PushMessager(setts['pushbullet_token'])

    try:

        db_path = setts['database_path'] if not test_mode else setts['test_database_path']

        run_trading(client, agnt, setts['pairs'], setts['assets'],
                    setts['base_fund_value'], setts['min_notional'],
                    msg_client, db_path, setts['trader_info'], test=test_mode)
    except Exception as err:
        if not test_mode:
            resp = None
            for i in range(10):
                resp = msg_client.push(type(err), str(err))
                if resp.status_code == 200:
                    break
                print('Retrying push message soon...')
                time.sleep(60*5)
            else:
                print('Failed to push message to PushBullet. Last request response was {}'.format(resp.text))
                raise err
        raise err

if __name__ == '__main__':

    fire.Fire(Deeptrader, name='deeptrader')
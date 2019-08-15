import yaml
import os
import dash
from flask_caching import Cache
from trader.database import DataQuery


def app_setup(config):

    config = os.path.abspath(config)

    with open(config) as f:
        sett = yaml.safe_load(f)

    os.environ['WEBPLOT_PORT'] = str(sett['port'])
    os.environ['WEBPLOT_HOST'] = sett['host']

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    cache = Cache(app.server, config={'CACHE_TYPE': 'redis', 'CACHE_REDIS_HOST': sett['redis']['host'],
                                      'CACHE_REDIS_PORT': sett['redis']['port'],
                                      'CACHE_REDIS_PASSWORD': sett['redis']['password'],
                                      'CACHE_THRESHOLD': sett['cache_threshold'],
                                      'CACHE_DEFAULT_TIMEOUT': sett['cache_time_to_live']})
    cache.clear()
    db = DataQuery(sett['db_path'])

    conf = ReadOnlyDict(sett)

    return app, cache, db, conf

class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("Cannot modify ReadOnlyDict")
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__
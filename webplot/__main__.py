import fire
import os


def run(config='./webplot.yaml'):
    config = os.path.abspath(config)
    os.environ['WEBPLOT_CONF'] = config
    from webplot.plotting import run_server
    run_server()


if __name__ == '__main__':

    fire.Fire(run, name='webplot')
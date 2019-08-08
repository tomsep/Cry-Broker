import logging
import requests

log = logging.getLogger(__name__)


class PushMessager:

    def __init__(self, token):
        self.access_token = token  # PushBullet access token

    def push(self, title, body):
        """ Push message to PushBullet app """
        api_url = 'https://api.pushbullet.com/v2/pushes'
        data = {
            'type': 'note',
            'title': title,
            'body': body
        }
        resp = requests.post(api_url, data=data,
                             auth=(self.access_token, ''))
        return resp

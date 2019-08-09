import requests
import json


class Telegram:

    def __init__(self, token):
        self._token = token
        self._base_url = 'https://api.telegram.org/bot{}/'.format(self._token)
        self._chat_id = Telegram._get_chat_id(self._base_url)

    def send_message(self, text):
        url = self._base_url + 'sendMessage?text={}&chat_id={}'.format(text, self._chat_id)
        Telegram.get_url(url)

    @staticmethod
    def get_url(url):
        response = requests.get(url)
        content = response.content.decode('utf8')
        return content

    @staticmethod
    def _get_chat_id(base_url):
        """Get chat ID. Chat must not be empty."""
        res = Telegram._get_updates(base_url)
        chat_id = res['result'][-1]['message']['chat']['id']
        return chat_id

    @staticmethod
    def _get_updates(base_url):
        url = base_url + 'getUpdates'
        js = Telegram.get_json_from_url(url)
        return js

    @staticmethod
    def get_json_from_url(url):
        content = Telegram.get_url(url)
        js = json.loads(content)
        return js



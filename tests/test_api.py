import logging
import pytest
import requests
from unittest.mock import patch, Mock, ANY
from http.client import HTTPMessage

from utils import makeRequestWithRetry

class TestMakeRequestWithRetry:
    logging.getLogger('urllib3').setLevel(logging.DEBUG)

    @patch("requests.Session.request")
    def test_get(self, mock_response):
        status_code = 200
        mock_response.return_value.status_code = status_code

        response = makeRequestWithRetry('GET', 'https://test.com', retries=2)
        assert response.status_code == status_code
        mock_response.assert_called_once_with('GET', 'https://test.com',
                                             headers=ANY, data=ANY, params=ANY)

    @patch("requests.Session.request")
    def test_put(self, mock_response):
        status_code = 201
        mock_response.return_value.status_code = status_code
        
        data = {
            "key1": "value1",
            "key2": "value2"
        }

        params = {
            "param1": "value1"
        }
        
        response = makeRequestWithRetry('POST', 
                                        'https://test.com',
                                        data=data,
                                        headers={"Authorization": "my_token"},
                                        params=params,
                                        retries=2)
        
        assert response.status_code == status_code
        mock_response.assert_called_once_with('POST', 
                                             'https://test.com',
                                             data=data,
                                             headers={"Authorization": "my_token"},
                                             params=params)

    @patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
    def test_success_after_retries(self, mock_response):
        mock_response.return_value.getresponse.side_effect = [
            Mock(status=500, msg=HTTPMessage()),
            Mock(status=502, msg=HTTPMessage()),
            Mock(status=200, msg=HTTPMessage()),
            Mock(status=429, msg=HTTPMessage()),
        ]

        response = makeRequestWithRetry('GET', 
                                        'https://test.com', 
                                        retries=5, 
                                        backoff_factor=0.1)

        assert response.status_code == 200
        assert mock_response.call_count == 3

    # comment out test since httpbin can be unstable and we don't want to rely
    # on it for tests. uncomment and see debug log to see retry attempts
    '''def test_httpbin(self):
        response = makeRequestWithRetry('GET', 
                                        'https://httpbin.org/status/500', 
                                        retries=4, 
                                        backoff_factor=0.1)
    '''
import logging
import pytest
import requests

from utils import makeRequestWithRetry

class TestMakeRequestWithRetry:
    logging.getLogger('urllib3').setLevel(logging.DEBUG)

    def test_get(self):
        # should work and return 200
        response = makeRequestWithRetry('GET', 'https://httpbin.org/get')
        assert response.status_code == 200
        assert response.json()['url'] == 'https://httpbin.org/get'
    
    def test_500(self):
        # use an error code that triggers retries (should show up in debug log)
        with pytest.raises(Exception):
            response_500 = makeRequestWithRetry('GET', 'https://httpbin.org/status/500', retries=3, backoff_factor=0.1)
    
    def test_404(self):
        # error code that won't trigger retries
        with pytest.raises(Exception):
            response_404 = makeRequestWithRetry('GET', 'https://httpbin.org/status/404', retires=2, backoff_factor=0.01)

if __name__ == '__main__':
    unittest.main()

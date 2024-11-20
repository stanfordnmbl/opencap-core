from utils import makeRequestWithRetry

def test_makeRequestWithRetry(self):
    test_URL = 'http://httpbin.org/status/404'

    makeRequestWithRetry('GET', test_URL)

if __name__ == '__main__':
    unittest.main()
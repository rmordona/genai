import genai as ai

# Define the URLFrontier class wrapper in Python
class URLFrontier:
    def __init__(self, max_urls=1000, queue_address=''):
        self._urlfrontier = ai.URLFrontier(max_urls, queue_address.encode('utf-8'))

    def enqueue(self, url, priority_percentage):
        url_bytes = url.encode('utf-8')
        self._urlfrontier.enqueue(url_bytes, priority_percentage)

    def start_worker_threads(self):
        self._urlfrontier.startWorkerThreads()

# Example usage:
if __name__ == '__main__':
    url_frontier = URLFrontier(max_urls=100, queue_address='tcp://127.0.0.1:5555')

    # Enqueue URLs with priority percentages
    url_frontier.enqueue('http://example.com/page1', 80)
    url_frontier.enqueue('http://example.com/page2', 50)

    # Start worker threads
    url_frontier.start_worker_threads()

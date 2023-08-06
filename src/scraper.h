/*
 * GENAI - A distributed framework that can help build LLMs
 *
 * Copyright (c) 2023 Raymod Michael O. Ordona
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Author: Raymond Michael O. Ordona
 *
 */

 /**************************************************************************************************************************
* Notes:
*
* PUB-SUB (Publish-Subscribe): This pattern is not well-suited for a URL frontier because it involves broadcasting 
* messages to multiple subscribers. In a URL frontier, you want to push URLs to specific workers (crawlers) rather 
* than broadcasting them to multiple consumers.
*
* REQ-REP (Request-Reply): This pattern is designed for synchronous communication between a client (requester) and 
a server (replier). It might not be the best fit for a URL frontier where you want to push URLs to workers without
* waiting for a reply.
*
* PUSH-PULL: This pattern is a good fit for a URL frontier. In the PUSH-PULL pattern, a push socket (PUSH) 
* pushes messages to a pull socket (PULL), allowing you to distribute work (URLs) to multiple workers (crawlers). 
* This aligns well with the concept of a URL frontier, where you want to distribute URLs to multiple crawling threads 
* efficiently.
*
* For URL Frontier, we choose PUSH-PULL communication type. Note that we have four types of multi-threads: 
* URLFrontier threads, RouterAgent threads, and SelectorAgent threads, Crawler threads.
**************************************************************************************************************************/

/**************************************************************************************************************************
* Notes: URL Frontier in the context of Crawling
*
* URLFrontier:
* - Calculates the priority percentage of a URL using getPriorityPercentageFromUrl. 
* - Enqueues the URL to the PriorityQueueService using the enqueue function.
*
* PriorityQueueService:
* - Receives the URL and priority percentage from the URLFrontier.
* - Determines the priority level based on the provided percentage.
* - Routes the URL to the corresponding queue (PriorityQueue or PolitenessQueue) based on the priority level.
* - Enqueues URLs to the PolitenessQueueService for further politeness queue handling.
*
* PolitenessQueueService:
* - Handles the politeness queue functionality for different sites.
* - Enqueues URLs into site-based politeness queues.
* - Dequeues URLs from site-based politeness queues when requested by the SelectorAgent.
*
* RouterAgent:
* - Dequeues URLs from the PriorityQueue using the dequeueFromPriorityQueue function.
* - Routes the URLs to the PolitenessQueueService using the routeToPolitenessQueue function.
*
* SelectorAgent:
* - Dequeues URLs from the PolitenessQueueService using the dequeueFromPolitenessQueue function.
* - Selects the next URL to crawl based on the heap and timestamp.
* - Routes the selected URL to the Crawler.
**************************************************************************************************************************/



extern int getPriorityPercentageFromUrl(const std::string& url, int defaultPriorityPercentage);

class CrawlerService {
private:
    // Callback function for libcurl to write the response data into a string
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
        size_t totalSize = size * nmemb;
        response->append(static_cast<char*>(contents), totalSize);
        return totalSize;
    }

public:
    // Constructor
    CrawlerService() {
        // Initialize libcurl
        curl_global_init(CURL_GLOBAL_ALL);
    }

    // Destructor
    ~CrawlerService() {
        // Cleanup libcurl
        curl_global_cleanup();
    }

    // Implementation of crawler to crawl the URL and extract content
    // Implement your crawling logic here
    void crawl(const std::string& url) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            std::cerr << "Failed to initialize libcurl." << std::endl;
            return;
        }

        // Set the URL to fetch
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Configure libcurl to follow redirects
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        // Response data will be written to this string
        std::string response;

        // Set libcurl write callback to append the response data to the 'response' string
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // Perform the HTTP request
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "Failed to fetch URL: " << curl_easy_strerror(res) << std::endl;
        } else {
            // Do something with the 'response' string (e.g., parse content, extract data, etc.)
            std::cout << "Crawled URL: " << url << std::endl;
            std::cout << "Response: " << std::endl << response << std::endl;
        }

        // Cleanup libcurl handle
        curl_easy_cleanup(curl);
    }


};

/*************************************************************************************************
* PriorityQueueService (PrQS):
* - Receives the URL and priority percentage from the URLFrontier.
* - Determines the priority level based on the provided percentage.
* - Enqueues the URLs to to the appropriate queue based on priority level.
* - Dequeues URLs based on priority as requested by RouterAgent.
* - RouterAgent takes the URL for routing. See RouterAgent.
*************************************************************************************************/

class PriorityQueueService {
private:
    struct UrlComparator {
        bool operator()(const std::string& lhs, const std::string& rhs) const {
            return lhs < rhs; // Replace this with your custom timestamp comparison logic
        }
    };

    // Add a data structure to track the number of URLs in each priority queue
    std::vector<int> priorityQueueSizes;

    // Map to store priority-based queues and their corresponding sockets
    std::map<int, std::pair<std::unique_ptr<zmq::socket_t>, std::unique_ptr<zmq::socket_t>>> priorityQueues;

    // ZeroMQ context
    zmq::context_t context;

    // Queue Address
    std::string queueAddress = "";

    int roundRobinIndex = 0; // Initialize the roundRobinIndex to 0


    std::mutex mutex;

public:

    // The std::move is used to indicate that ownership of the resource managed is transferred.
    // Note that: ZMQ_PUSH is defined for zmq::socket_type::push
    //            ZMQ_PULL is defined for zmq::socket_type::pull
    PriorityQueueService(const std::string& queueAddress) : context(1) {
        try {
            // Preserve queueaddress
            this->queueAddress = queueAddress;
            // Create PULL sockets for each priority queue
            for (int priority = 0; priority <= 100; priority += 10) {
                std::unique_ptr<zmq::socket_t> pushSocket = std::make_unique<zmq::socket_t>(context, ZMQ_PUSH);
                std::unique_ptr<zmq::socket_t> pullSocket = std::make_unique<zmq::socket_t>(context, ZMQ_PULL);
                priorityQueues[priority] = std::make_pair(std::move(pushSocket), std::move(pullSocket));
            }
        } catch (const zmq::error_t& e) {
            throw std::runtime_error("Error initializing ZeroMQ: " + std::string(e.what()));
        }
    }

    // Destructor
    ~PriorityQueueService() {
        try {
            // Close sockets and terminate context
            for (const auto& queue : priorityQueues) {
                queue.second.first->close();
                queue.second.second->close();
            }
            context.close();
        } catch (const zmq::error_t& e) {
            // Log or handle any error during cleanup (optional)
        }
    }

    // Invoked by URLFrontier startWorkerThread
    void connectPushSockets() ;

    // Invoked by RouterAgent startWorkerThread
    void bindPullSockets() ;

    // Implementation of priority queue service enqueue
    // Add URL to the corresponding priority queue based on priorityPercentage
    // Invoked by URLFrontier for enqueueing
    void enqueue(const std::string& url, int priorityPercentage);

    // Implementation of priority queue service enqueue
    // Add URL to the corresponding priority queue based on priorityPercentage
    std::string dequeue(int priorityLevel);

    // Implementation of priority queue service isEmpty
    // Check if all priority queues are empty
    bool isEmpty() const ;

    // Implementation of priority queue service hasMoreUrls
    // Check if there are more URLs to be processed in any priority queue
    bool hasMoreUrls() const ;

    // Invoked by RouterAgent for dequeueing
    std::string getNextUrl();

};

/********************************************************************************************************************
* Politeness Queue Service
*********************************************************************************************************************/
/*************************************************************************************************
* PolitenessQueueService (PoQS):
* - Handles the politeness queue functionality for different sites.
* - Enqueues URLs into site-based politeness queues.
* - Dequeues URLs from site-based politeness queues when requested by the SelectorAgent.
*************************************************************************************************/

class PolitenessQueueService {
private:
    struct SiteInfo {
        std::string url;            // URL of the site
        std::chrono::system_clock::time_point timestamp; // Timestamp of the site

        // Constructor to initialize the SiteInfo
        SiteInfo(const std::string& u, const std::chrono::system_clock::time_point& t)
            : url(u), timestamp(t) {}
    };

    struct SiteComparator {
        bool operator()(const SiteInfo& lhs, const SiteInfo& rhs) const {
            return lhs.timestamp > rhs.timestamp;
        }
    };

    //  This typedef contains three templates: 
    //    - SiteInfo: the type of the elements stored in the priority queue
    //    - std::vector<SiteInfo>: the container type to store the elements in the priority queue.
    //    - Sitecomparator: to describe the ordering of the elements in the priority queue.
    typedef std::priority_queue<SiteInfo, std::vector<SiteInfo>, SiteComparator> SiteQueue;

    // This is the actual queue which gets transformed into a priority queue of a vector of SiteInfo
    // which is shown as the second template in the typedef above.
    std::unordered_map<std::string, SiteQueue> siteQueues;

    std::mutex mutex;
    std::condition_variable condition;

    std::string extractSiteFromUrl(const std::string& url) const {
        // Replace this with your logic to extract the site from the URL
        // In this example, we assume the site is the domain name
        std::string site = url;
        size_t pos = site.find("://");
        if (pos != std::string::npos) {
            pos += 3;
            size_t endPos = site.find('/', pos);
            if (endPos != std::string::npos) {
                site = site.substr(0, endPos);
            }
        }
        return site;
    }

public:

    PolitenessQueueService() {}

    ~PolitenessQueueService() {}

    // Enqueue a URL into the appropriate site queue
    void enqueue(const std::string& url);

    // Dequeue and exhaust URLs from a site's queue
    // Called from SelectorAgent.processSite()
    std::string dequeue(const std::string& site);

    bool isSiteQueueEmpty();

    std::string getNextSite();

};

/********************************************************************************************************************
* The RouterAgent class now takes a reference to PolitenessQueueService as a parameter in its constructor. 
* With this reference, the RouterAgent can access and use the PolitenessQueueService to enqueue URLs to the 
* corresponding politeness queue.
*
* The routeToPolitenessQueue function is responsible for extracting the site from the URL (using the 
* extractSiteFromUrl function) and then enqueuing the URL to the appropriate politeness queue using the 
* politenessQueueService.enqueue(url) call.
*
* With this setup, the RouterAgent doesn't need to handle any ZeroMQ communication directly. Instead, it
* delegates the task of handling ZeroMQ to the PolitenessQueueService, which simplifies the code and ensures 
* a clear separation of concerns. The RouterAgent can focus solely on routing URLs, while the PolitenessQueueService
* can handle the communication using ZeroMQ and managing the politeness queues.
**********************************************************************************************************************/
class RouterAgent {
private:
    PriorityQueueService& priorityQueueService;  // Reference to the PriorityQueueService
    PolitenessQueueService& politenessQueueService; // Reference to the PolitenessQueueService
    bool running;

    //  zmq::socket_t routerQueueSocket; // The ZeroMQ socket for the router's queue

    int numThreads;
    std::vector<std::thread> workerThreads;
    std::condition_variable condition;

public:

/*
    RouterAgent(const std::string& priorityQueueAddress, const std::string& politenessQueueAddress)
        : priorityQueueService(priorityQueueAddress),
        politenessQueueService(politenessQueueAddress)
    {
        routerQueueSocket = zmq::socket_t(context, zmq::socket_type::pull);
        routerQueueSocket.connect(priorityQueueAddress);

        politenessQueueSocket = zmq::socket_t(context, zmq::socket_type::push);
        politenessQueueSocket.connect(politenessQueueAddress);
    }
*/

    // Constructor that takes references to PriorityQueueService and PolitenessQueueService
    RouterAgent(PriorityQueueService& prQueueService, PolitenessQueueService& pqService) 
        : priorityQueueService(prQueueService), politenessQueueService(pqService), running(true) {
        // Create worker threads
        for (int i = 0; i < numThreads; ++i) {
            std::string identity = "RouterAgent_" + std::to_string(i);
            workerThreads.emplace_back(&RouterAgent::workerThread, this, identity);
        }

    }

    ~RouterAgent() {
        // Stop worker threads and wait for them to finish
        running = false;
        condition.notify_all();
        for (auto& thread : workerThreads) {
            thread.join();
        }
    }

    void workerThread(const std::string& identity);

    // Implementation of dequeue from the PriorityQueue
    // std::string dequeueFromPriorityQueue() ;

    // void routeURLs();

    // Worker thread function
    // void startWorkerThreads(int numThreads);
    // void startWorkerThread();

};

/*************************************************************************************************
* SelectorAgent:
* - Dequeues URLs from the PolitenessQueueService using the dequeueFromPolitenessQueue function.
* - Selects the next URL to crawl based on the heap and timestamp.
* - Routes the selected URL to the Crawler.
*************************************************************************************************/
class SelectorAgent {
private:
    PolitenessQueueService& politenessQueueService;
    CrawlerService& crawlerService;
    bool running;

    std::vector<std::thread> workerThreads;
    std::mutex mutex;
    std::condition_variable condition;
    size_t siteIndex = 0;


public:
    SelectorAgent(PolitenessQueueService& politenessQueueService,  CrawlerService& crawlerService, int numThreads)
        : politenessQueueService(politenessQueueService), crawlerService(crawlerService), running(true) {
        // Create worker threads
        for (int i = 0; i < numThreads; ++i) {
            workerThreads.emplace_back(&SelectorAgent::workerThread, this);
        }
    }

    ~SelectorAgent() {
        // Stop worker threads and wait for them to finish
        running = false;
        condition.notify_all();
        for (auto& thread : workerThreads) {
            thread.join();
        }
    }

    void workerThread();

    void processSite(std::string& site);

};

/*************************************************************************************************
* URLFrontier:
* - Calculates the priority percentage of a URL using getPriorityPercentageFromUrl. 
* - Enqueues the URL to the PriorityQueueService using the enqueue function.
*************************************************************************************************/
class URLFrontier {
private:
    int maxUrls;
    int politenessDelay = 1000;
    PriorityQueueService priorityQueueService;
    PolitenessQueueService politenessQueueService;
    CrawlerService crawlerService;
    RouterAgent routerAgent;
    SelectorAgent selectorAgent;


    const int PRIORITY_THRESHOLD = 50; // You can set the desired threshold value here

    // std::thread urlFrontierThread;
    // std::thread routerThread;
    // std::thread selectorThread;
    // std::thread selectorThread;
    std::vector<std::thread> urlFrontierThread;
    // std::vector<std::thread> routerThreads;
    // std::vector<std::thread> selectorThreads;
    int numThreads = 2;

    std::queue<std::string> urlQueue;  // Declare the urlQueue
    std::mutex mutex;  // Mutex for thread safety

    std::string extractSiteFromUrl(const std::string& url) const {
        // Replace this with your logic to extract the site from the URL
        // In this example, we assume the site is the domain name
        std::string site = url;
        size_t pos = site.find("://");
        if (pos != std::string::npos) {
            pos += 3;
            size_t endPos = site.find('/', pos);
            if (endPos != std::string::npos) {
                site = site.substr(0, endPos);
            }
        }
        return site;
    }

public:

    // Constructor with default arguments
    /****************************************************************************************************************************
    * The PriorityQueueService creates a PUSH socket and a PULL socket using the same address (priorityQueueAddress).
    * The URLFrontier uses the PUSH socket to push URLs to the queue.
    * The RouterAgent uses the PULL socket to pull URLs from the same queue, and this socket is part of the 
    *      PriorityQueueService instance passed to it.
    * This setup ensures that both the URLFrontier and the RouterAgent can communicate through the same 
    *      queue using the PUSH and PULL sockets, respectively.
    *****************************************************************************************************************************/
    URLFrontier(int maxUrls = 1000, const std::string& priorityQueueAddress = "", const std::string& politenessQueueAddress = "")
        : maxUrls(maxUrls),
        priorityQueueService(priorityQueueAddress), // the queue address for PriorityQueueService
        politenessQueueService(), // the queue address for PriorityQueueService
        crawlerService(),  // You may need to provide any necessary parameters for the Crawler constructor
        routerAgent(priorityQueueService, politenessQueueService),
        selectorAgent(politenessQueueService, crawlerService, numThreads)
    {
        // Start the worker threads for RouterAgent second since this connects to the Push socket for the PoQS
        // Also this , this binds to Pull sockets for the PrQS
        // routerAgent.startWorkerThreads(numThreads);

        // Create and start threads for URLFrontier third since this connects to the Push sockets for the PrQS
        // Start worker threads for URLFrontier
        startWorkerThreads(numThreads);

    }

    // Join the worker threads on destruction. Joining" a thread in C++ means to wait for the thread 
    // to complete its execution before moving on. 
    ~URLFrontier() {

        // for (std::thread& thread : routerThreads) {
        //    thread.join();
        // }
        // for (std::thread& thread : selectorThreads) {
        //    thread.join();
        // }
    }

    void setPolitenessDelay(int delayMilliseconds);

    void enqueue(const std::string& url);

    std::string dequeue();

    void processUrl(const std::string& url);

    void startWorkerThreads(int numThreads);

    void startWorkerThread();

};


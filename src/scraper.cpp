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

#include "genai.h"
#include "scraper.h"

/*****************************************************************************************************
* Scraper / Crawler Function
* The crawl function is responsible for crawling sites.
*****************************************************************************************************/

// Member function for the WriteCallback
size_t Scraper::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t totalSize = size * nmemb;
    output->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// Function to fetch and parse a URL using libcurl and libxml2
bool Scraper::crawl(const std::string& url, int depth) {
    if (depth > 2) {
        // Reached the maximum depth, stop crawling
        return true;
    }

    CURL* curl;
    CURLcode res;
    std::string xml_data;

    // Initialize libcurl
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) {

        // Url to craw
        py_cout << "URL: " << url.c_str() << std::endl;
        // Set the URL to fetch
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Set the user-agent (change this to your desired user-agent)
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3");

        // Set the write callback function to receive data
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &Scraper::WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &xml_data);

        // Perform the request
        res = curl_easy_perform(curl);

        // Check for errors
        if (res != CURLE_OK) {
            std::cerr << "Error fetching URL: " << curl_easy_strerror(res) << std::endl;
            curl_easy_cleanup(curl);
            curl_global_cleanup();
            return false;
        } else {
            py_cout << "XML Data fetched successfully!" << std::endl;
            // Initialize libxml2 parser
            xmlInitParser();

            // Parse the fetched XML data
            xmlDocPtr doc = xmlReadMemory(xml_data.c_str(), xml_data.size(), "noname.xml", NULL, 0);
            if (doc == NULL) {
                std::cerr << "Error parsing XML data" << std::endl;
                xmlCleanupParser();
                curl_easy_cleanup(curl);
                curl_global_cleanup();
                return false;
            } else {
                // Process the XML document here...
                // For example, you can access nodes using libxml2 APIs
                xmlNodePtr root_element = xmlDocGetRootElement(doc);
                if (root_element != NULL) {
                    // Your parsing logic goes here...
                }

                // Get the links on the current page
                std::vector<std::string> links;
                xmlNodePtr currentNode = xmlDocGetRootElement(doc);
                while (currentNode != NULL) {
                    if (currentNode->type == XML_ELEMENT_NODE && xmlStrEqual(currentNode->name, (const xmlChar*)"a")) {
                        xmlChar* href = xmlGetProp(currentNode, (const xmlChar*)"href");
                        if (href != NULL) {
                            links.push_back(reinterpret_cast<const char*>(href));
                        }
                    }
                    currentNode = currentNode->next;
                }

                // Free the XML document after processing
                xmlFreeDoc(doc);

                // Cleanup libxml2 parser
                xmlCleanupParser();

                // Fetch and parse each link recursively
                for (const auto& link : links) {
                    crawl(link, depth + 1);
                }
            }
        }

        // Cleanup libcurl
        curl_easy_cleanup(curl);
    } else {
        std::cerr << "Error initializing libcurl" << std::endl;
        curl_global_cleanup();
        return false;
    }

    // Cleanup libcurl global state
    curl_global_cleanup();
    return true;
}


/**************************************************************************************************************************
* The next few classes support URL Frontier in the context of Crawling
* PriorityQueueService:
*    This class supports Priority Queueing. It allows multiple queues, each queue corresponding to a range of priority.
*
* PolitenessQueueService:
*    This class supports Politeness Queueing. It allows multiple queues, each queue correspond to each unique site.
*
* RouterAgent:
*    This class is responsible in routing request from Priority queue to Politeness queue.
*
* SelectorAgent:
*    This class selects URL for crawling based on hash from the heap along with a timestamp.
*
* Crawler:
*    This class crawls websites.
**************************************************************************************************************************/


int getPriorityPercentageFromUrl(const std::string& url, int defaultPriorityPercentage) {
    // Parse the URL to extract query parameters
    size_t queryStartPos = url.find('?');
    if (queryStartPos == std::string::npos) {
        // No query parameters found, return the default priority percentage
        return defaultPriorityPercentage;
    }

    std::string queryParameters = url.substr(queryStartPos + 1);

    // Split query parameters by '&' to get individual parameter-value pairs
    std::unordered_map<std::string, std::string> queryParams;
    size_t startPos = 0;
    size_t endPos;
    while ((endPos = queryParameters.find('&', startPos)) != std::string::npos) {
        std::string paramValuePair = queryParameters.substr(startPos, endPos - startPos);
        size_t eqPos = paramValuePair.find('=');
        if (eqPos != std::string::npos) {
            std::string param = paramValuePair.substr(0, eqPos);
            std::string value = paramValuePair.substr(eqPos + 1);
            queryParams[param] = value;
        }
        startPos = endPos + 1;
    }

    // Handle the last parameter-value pair (if any)
    std::string lastParamValuePair = queryParameters.substr(startPos);
    size_t lastEqPos = lastParamValuePair.find('=');
    if (lastEqPos != std::string::npos) {
        std::string lastParam = lastParamValuePair.substr(0, lastEqPos);
        std::string lastValue = lastParamValuePair.substr(lastEqPos + 1);
        queryParams[lastParam] = lastValue;
    }

    // Check if the "priority" query parameter exists
    auto it = queryParams.find("priority");
    if (it != queryParams.end()) {
        try {
            // Convert the "priority" value to an integer and return it as the priority percentage
            int priorityPercentage = std::stoi(it->second);
            return priorityPercentage;
        } catch (const std::invalid_argument& e) {
            // Invalid integer, return the default priority percentage
            return defaultPriorityPercentage;
        } catch (const std::out_of_range& e) {
            // Integer out of range, return the default priority percentage
            return defaultPriorityPercentage;
        }
    }

    // If "priority" query parameter not found, return the default priority percentage
    return defaultPriorityPercentage;
}

/*************************************************************************************************
* PriorityQueueService (PrQS):
* - Receives the URL and priority percentage from the URLFrontier.
* - Determines the priority level based on the provided percentage.
* - Enqueues the URLs to to the appropriate queue based on priority level.
* - Dequeues URLs based on priority as requested by RouterAgent.
* - RouterAgent takes the URL for routing. See RouterAgent.
*************************************************************************************************/
// URLFrontier worker thread connects to each of the queue sockets.
void PriorityQueueService::connectPushSockets() {
    try {
        // Connect to PUSH sockets for each priority queue
        for (int priority = 0; priority <= 100; priority += 10) {
            priorityQueues[priority].first->connect(this->queueAddress);
        }
    } catch (const zmq::error_t& e) {
        throw std::runtime_error("Error connecting PUSH sockets: " + std::string(e.what()));
    }
}


void PriorityQueueService::bindPullSockets() {
    try {
        // Bind to PULL sockets for each priority queue
        for (int priority = 0; priority <= 100; priority += 10) {
            priorityQueues[priority].second->bind(this->queueAddress);
        }
    } catch (const zmq::error_t& e) {
        throw std::runtime_error("Error connecting PUSH sockets: " + std::string(e.what()));
    }
}

// Implementation of priority queue service enqueue
// Add URL to the corresponding priority queue based on priorityPercentage
void PriorityQueueService::enqueue(const std::string& url, int priorityPercentage) {
    try {

        // Ensure thread safety to prevent data races and other concurrency issues. 
        std::lock_guard<std::mutex> lock(mutex);  // Lock to ensure thread safety

        if (priorityPercentage < 0 || priorityPercentage > 100) {
            throw std::invalid_argument("Priority percentage must be between 0 and 100.");
        }

        // Determine priority level based on percentage
        int priorityLevel = (100 - priorityPercentage) / 10; // Higher percentage gets higher priority

        // Send a message to the PUSH socket to enqueue the URL to the corresponding priority queue
        zmq::message_t message(url.begin(), url.end());
        priorityQueues[priorityLevel].first->send(message, zmq::send_flags::none);

        // Update the priorityQueueSizes data structure
        ++priorityQueueSizes[priorityLevel];
    } catch (const zmq::error_t& e) {
        throw std::runtime_error("Error enqueuing URL: " + std::string(e.what()));
    }
}



// Implementation of priority queue service dequeue
// Remove and return the next URL from the highest priority queue.
// Called from getNextUrl().  This function is thread-safe based on getNextUrl().
std::string PriorityQueueService::dequeue(int priorityLevel) {
    try {

        // Check for a message on the PULL socket for the specific priority level
        zmq::message_t message;
        if (priorityQueues[priorityLevel].second->recv(message, zmq::recv_flags::dontwait)) {
            // A message is available on this queue, dequeue the URL
            std::string url(static_cast<const char*>(message.data()), message.size());
            return url;
        }
        return ""; // Return an empty string if no URL is available in this priority queue
    } catch (const zmq::error_t& e) {
        throw std::runtime_error("Error dequeuing URL from priority queue: " + std::string(e.what()));
    }
}



// Implementation of priority queue service hasMoreUrls
// Check if there are more URLs to be processed in any priority queue
// Invoked from RouterAgent
bool PriorityQueueService::hasMoreUrls() const {
    // Check if there are more URLs to be processed in any priority queue
    return std::any_of(priorityQueues.begin(), priorityQueues.end(), [](const auto& pq) {
        zmq::message_t message;
        return pq.second.second->recv(message, zmq::recv_flags::dontwait);
    });
}

bool PriorityQueueService::isEmpty() const {
    return priorityQueues.empty();
}

// Get the next Url
// Invoked from RouterAgent
std::string PriorityQueueService::getNextUrl() {
    // Remove and return the next URL from the highest priority queue

    // Ensure thread safety to prevent data races and other concurrency issues. 
    std::lock_guard<std::mutex> lock(mutex);  // Lock to ensure thread safety

    if (!isEmpty()) {
        int highestPriority = 100; // Start with the highest priority
        while (highestPriority >= 0) {
            std::string url = dequeue(highestPriority);
            if (!url.empty()) {
                return url;
            }
            highestPriority -= 10; // Move to the next lower priority level
        }
    }
    return ""; // Return an empty string if no URL is available
}

/*************************************************************************************************
* PolitenessQueueService (PoQS):
* - Handles the politeness queue functionality for different sites.
* - Enqueues URLs into site-based politeness queues.
* - Dequeues URLs from site-based politeness queues when requested by the SelectorAgent.
*************************************************************************************************/


    // Enqueue a URL into the appropriate site queue
void PolitenessQueueService::enqueue(const std::string& url) {
    std::string site = extractSiteFromUrl(url);
    SiteInfo siteInfo(url, std::chrono::system_clock::now());

    std::lock_guard<std::mutex> lock(mutex);
    siteQueues[site].push(siteInfo);

    // Notify waiters. See SelectorAgent.processSite()
    condition.notify_one();
}

// Dequeue and exhaust URLs from a site's queue
// Called from SelectorAgent.processSite()
std::string PolitenessQueueService::dequeue(const std::string& site) {

        std::unique_lock<std::mutex> lock(mutex);

        if (siteQueues[site].empty()) {
            return ""; // return empty.
        }

        SiteInfo siteInfo = siteQueues[site].top();
        siteQueues[site].pop();

        // Unlock the mutex before processing the URL to avoid holding the lock during processing
        lock.unlock();

        // Process the URL
        return siteInfo.url;

}

bool PolitenessQueueService::isSiteQueueEmpty() {
    if (siteQueues.empty()) return true;
    return false;
}

std::string PolitenessQueueService::getNextSite() {
    std::lock_guard<std::mutex> lock(mutex);

    if (siteQueues.empty()) {
        return ""; // Return an empty string if no sites are available
    }

    // Find the site with the earliest timestamp
    auto minSiteIt = std::min_element(
        siteQueues.begin(), siteQueues.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.second.top().timestamp > rhs.second.top().timestamp;
        }
    );

    if (minSiteIt != siteQueues.end()) {
        // Get the name of the selected site
        std::string siteName = minSiteIt->first;

        return siteName;
    } else {
        return ""; // Return an empty string if no sites have URLs
    }
}

/*************************************************************************************************
* RouterAgent (RT):
* - Get the next URLs  from the PriorityQueueService (PrQS) using the getNextUrl() function.
* - Routes the URLs to the PolitenessQueueService.
*************************************************************************************************/
void RouterAgent::workerThread(const std::string& identity) {

    while (running) {

        std::string url = priorityQueueService.getNextUrl();  

        // If the URL is empty, it means the priority queues are empty
        // Sleep for a while and continue
        if (url.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (!running) {
            return; // Thread should exit
        }

        // Process the URL (e.g., route it to the corresponding politeness queue)
        politenessQueueService.enqueue(url);

    }
}

/*************************************************************************************************
* SelectorAgent:
* - Dequeues URLs from the PolitenessQueueService using the dequeueFromPolitenessQueue function.
* - Selects the next URL to crawl based on the heap and timestamp.
* - Routes the selected URL to the Crawler.
*************************************************************************************************/
void SelectorAgent::workerThread() {
    while (running) {
        std::string site;
        std::string url;

        { // this block of code is bounded by the mutex lock

            std::unique_lock<std::mutex> lock(mutex);

            // Wait until a site queue is available
            condition.wait(lock, [this] { return !politenessQueueService.isSiteQueueEmpty() || !running; });

            if (!running) {
                return; // Thread should exit
            }

            // Pop the front site queue and URL
            site = politenessQueueService.getNextSite();

            // Now process all urls in the site.
            processSite(site);

        }

    }
}

// TODO: We can decouple The selector agent from crawler using Kafka interfaces to stream URLs
// to Crawler engine.
void SelectorAgent::processSite(std::string& site) {
    while (true) {
       std::string url = politenessQueueService.dequeue(site);
       if (url.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return;
       }
       crawlerService.crawl(url); // spin a crawler thread and detach.
    }
}

/*************************************************************************************************
* URLFrontier:
* - Calculates the priority percentage of a URL using getPriorityPercentageFromUrl. 
* - Enqueues the URL to the PriorityQueueService using the enqueue function.
*************************************************************************************************/
// Set the politeness delay
void URLFrontier::setPolitenessDelay(int delayMilliseconds) {
    if (delayMilliseconds >= 0) {
        politenessDelay = delayMilliseconds;
    }
}

// Enqueue a URL into the urlQueue
// TODO:  Perhaps write up a REST API that accepts a list of url that can be queued
//        such that the URLFrontier worker thread dequeues from urlEnqueue for prioritization.
void URLFrontier::enqueue(const std::string& url) {
    std::lock_guard<std::mutex> lock(mutex);
    urlQueue.push(url);
}

std::string URLFrontier::dequeue() {    
    // For example, you might have a list of URLs to be crawled and pop them one by one.
    // not that the mutex gets unlocked once the lock goes out of scope. So need to
    // explicitly unlock.
    std::lock_guard<std::mutex> lock(mutex);
    if (!urlQueue.empty()) {
        std::string nextUrl = urlQueue.front();
        urlQueue.pop();
        return nextUrl;
    }

    return ""; // Return an empty string if there are no more URLs to crawl.
}

// Implementation of processUrl in URLFrontier class
void URLFrontier::processUrl(const std::string& url) {
    // Determine priority percentage from the URL
    int priorityPercentage = getPriorityPercentageFromUrl(url, 50);

    // Use the PriorityQueueService to enqueue the URL with the calculated priority percentage
    priorityQueueService.enqueue(url, priorityPercentage);
}

// The urlFrontierThread is a vector of st::thread
void URLFrontier::startWorkerThreads(int numThreads) {

    // Connect to the Priority Queues (Sockets).
    priorityQueueService.connectPushSockets();
    
    // Create and start multiple threads, each with a unique identity
    for (int i = 0; i < numThreads; ++i) {
        urlFrontierThread.emplace_back(&URLFrontier::startWorkerThread, this);
    }

    // Wait for worker threads to finish
    for (auto& thread : urlFrontierThread) {
        thread.join();
    }
}

void URLFrontier::startWorkerThread() {

    while (true) {
        // Generate or fetch the next URL to be crawled.
        std::string url = dequeue();

        // If the URL is empty, sleep and continue
        if (url.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Process the URL (e.g., route it to the corresponding politeness queue)
        processUrl(url);
    }
}

/*

// Example usage:
int main() {
    URLFrontier frontier(1000);
    frontier.setPolitenessDelay(1000); // Set politeness delay to 1000 milliseconds (1 second)

    // Enqueue some URLs with priority percentages
    frontier.enqueue("https://example.com/page1", 30); // 30% priority
    frontier.enqueue("https://example.com/page2", 20); // 20% priority
    frontier.enqueue("https://example.com/page3", 50); // 50% priority
    frontier.enqueue("https://example.com/page4");      // Politeness queue

    // Crawl URLs with politeness delay and priority-based routing
    frontier.crawlUrls();

    return 0;
}
*/

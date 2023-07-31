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
            std::cout << "XML Data fetched successfully!" << std::endl;
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
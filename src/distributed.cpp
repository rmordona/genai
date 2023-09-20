
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

/******************************************************************************************************************************
OpenMPI and ZeroMQ serve different purposes and are used in different contexts:

OpenMPI (Open Message Passing Interface):

OpenMPI is a library and framework designed for high-performance computing and parallel computing on distributed memory systems, 
often used in scientific and engineering applications. It provides a standardized interface for communication between processes 
running on different nodes in a cluster, typically using a message-passing model. OpenMPI is commonly used for distributed-memory 
parallel programming, where data is distributed across multiple nodes, and computation is performed on each node independently. 

ZeroMQ (ØMQ):

ZeroMQ is a lightweight, high-performance messaging library that provides asynchronous communication between distributed 
applications. It's designed for inter-process communication and is well-suited for building scalable and distributed systems.
ZeroMQ supports various messaging patterns (e.g., request-reply, publish-subscribe, push-pull) and allows applications to 
communicate in a decoupled, flexible manner.

In summary, OpenMPI is primarily used for parallel computing and distributed-memory programming, while ZeroMQ is used for 
inter-process communication and building scalable, distributed systems. They have different use cases and are not directly 
interchangeable. If you are looking to build a distributed system for messaging and communication between components, 
ZeroMQ would be more appropriate. If you are working on high-performance computing tasks that involve parallel processing 
across multiple nodes, OpenMPI is a better choice.

If you are working on high-performance computing tasks that involve parallel processing across multiple nodes, you can consider 
using both OpenMPI and ZeroMQ in your distributed system, each serving its specific purpose:

OpenMPI:

Use OpenMPI for parallel computing tasks that require distributing computational workloads across multiple nodes in a 
cluster or supercomputer.

OpenMPI provides a powerful framework for managing communication between processes running on different nodes.
It is optimized for high-performance computing and parallel algorithms, making it an excellent choice for scientific 
and engineering applications.

ZeroMQ:

Use ZeroMQ for communication and messaging between different components of your distributed system. ZeroMQ provides a 
lightweight, asynchronous messaging library, allowing you to build scalable and loosely-coupled communication patterns. It 
complements OpenMPI by providing more flexible and decoupled communication channels between different nodes or components 
in your system.

By combining both OpenMPI and ZeroMQ, you can leverage the strengths of each library to build a distributed system that 
efficiently handles both parallel processing and communication between nodes. For example, you can use OpenMPI to manage 
the parallel computation tasks, and then use ZeroMQ to exchange results or coordinate actions between different 
nodes or components.

However, keep in mind that integrating these libraries may introduce additional complexity, and it's essential to design 
your system carefully to ensure efficient and reliable communication and parallel processing. Additionally, consider 
exploring other distributed computing frameworks or platforms (e.g., Apache Spark, Dask) that provide more integrated 
solutions for both parallel computing and distributed communication.
******************************************************************************************************************************/
#include <genai.h>

namespace py = pybind11;
using namespace py::literals;

const int NUM_REPLICAS = 3;

void DistributedKVStore::replicationSubscriberThread() {
    // Set up the queue node's Pull socket to receive replication messages from the queuePushSocket
    zmq::context_t context(1); // Initialize ZeroMQ context with one I/O thread
    zmq::socket_t queuePullSocket(context, zmq::socket_type::pull);
    queuePullSocket.connect("inproc://queue");

    while (true) {
        // Receive replication messages from the queue node's Pull socket
        zmq::message_t message;
        zmq::recv_result_t result = queuePullSocket.recv(message, zmq::recv_flags::none); // Use the correct recv signature

        if (result.has_value()) {
            // Successfully received data
            // You can access the received data in 'empty'
        } else {
            // Handle the error condition
            // You can use result.error() to get information about the error
        }

        // Parse the message (key-value pair) and use libmemcached to store the replicated data locally
        std::string messageStr(static_cast<char*>(message.data()), message.size());

        // Assuming the message is formatted as "key=value"
        size_t separatorPos = messageStr.find('=');
        if (separatorPos != std::string::npos) {
            std::string key = messageStr.substr(0, separatorPos);
            std::string value = messageStr.substr(separatorPos + 1);

            // Connect to the local memcached server
            memcached_st *memc = memcached_create(NULL);
            memcached_return rc = memcached_server_add(memc, "localhost", 11211);

            if (rc == MEMCACHED_SUCCESS) {
                // Store the replicated data locally in the memcached server
                rc = memcached_set(memc, key.c_str(), key.size(), value.c_str(), value.size(), (time_t)0, (uint32_t)0);

                if (rc != MEMCACHED_SUCCESS) {
                    // Handle the error when storing data in memcached
                    std::cerr << "Failed to store replicated data in memcached: " << memcached_strerror(memc, rc) << std::endl;
                }
            } else {
                // Handle the error when connecting to memcached
                std::cerr << "Failed to connect to memcached server: " << memcached_strerror(memc, rc) << std::endl;
            }

            // Cleanup the memcached connection
            memcached_free(memc);
        } else {
            // Handle invalid message format
            std::cerr << "Received an invalid replication message: " << messageStr << std::endl;
        }
    }
}

// Constructor to initialize ZeroMQ, set up Memcached connections, and set up the queue node
DistributedKVStore::DistributedKVStore(const std::vector<std::string>& nodeIdentifiers)
    : zmqContext(1),
      publisherSocket(zmqContext, zmq::socket_type::pub),
      queuePushSocket(zmqContext, zmq::socket_type::push),
      queuePullSocket(zmqContext, zmq::socket_type::pull),
      nodes(nodeIdentifiers.size()), // Initialize the nodes vector with the appropriate size
      subscriberThread(&DistributedKVStore::replicationSubscriberThread, this) {

    // Set up Memcached client object
    memc = memcached_create(NULL);

    // Add Memcached servers to the memcached_st structure and set user data for the client
    for (size_t i = 0; i < nodeIdentifiers.size(); ++i) {
        nodes[i].identifier = nodeIdentifiers[i];
        nodes[i].memc = memcached_clone(NULL, memc); // Use memcached_clone to clone the configuration
        memcached_server_add_with_weight(nodes[i].memc, nodeIdentifiers[i].c_str(), 11211, 0);
    }

    // Set user data for the client to access the nodes vector
    memcached_set_user_data(memc, &nodes);

    // Bind the publisher socket to its port for replication messages broadcasting
    publisherSocket.bind("tcp://*:port"); // Replace 'port' with the port number for the publisher node

    // Bind the queue node's Push socket and connect its Pull socket in the same inproc context
    queuePushSocket.bind("inproc://queue");
    queuePullSocket.connect("inproc://queue");
}

// Destructor to join the subscriber thread, clean up Memcached connections, ZeroMQ sockets, and ZeroMQ context
DistributedKVStore::~DistributedKVStore() {
    // Clean up Memcached connections
    for (auto& node : nodes) {
        memcached_free(node.memc);
    }

    // Clean up ZeroMQ sockets
    publisherSocket.close();
    queuePushSocket.close();
    queuePullSocket.close();

    // Join the subscriber thread
    subscriberThread.join();
}

size_t DistributedKVStore::getServerIndexForKey(const std::string& key) {

    // Calculate the hash value of the key
    std::hash<std::string> hasher;
    size_t hash_value = hasher(key);

    // Use consistent hashing to determine the server index
    //size_t num_virtual_nodes = 100; // You can adjust the number of virtual nodes as needed
    // size_t virtual_node_index = hash_value % num_virtual_nodes;
    size_t physical_node_index = hash_value % nodes.size();

    return physical_node_index;
}

void DistributedKVStore::remove(const std::string& key) {
    // Determine the node using consistent hashing
    size_t nodeIndex = getServerIndexForKey(key);
    Node& node = nodes[nodeIndex];

    // Use libmemcached to remove the key-value pair from the selected node
    memcached_return_t rc = memcached_delete(node.memc, key.c_str(), key.size(), 0);

    // Check if the operation was successful
    if (rc == MEMCACHED_SUCCESS) {
        std::cout << "Key removed successfully: " << key << std::endl;
    } else if (rc == MEMCACHED_NOTFOUND) {
        std::cout << "Key not found: " << key << std::endl;
    } else {
        std::cerr << "Error removing key: " << memcached_strerror(node.memc, rc) << std::endl;
    }
}

// Function to add a new node to the distributed system
void DistributedKVStore::addNode(const std::string& nodeIdentifier) {
    Node node;
    node.identifier = nodeIdentifier;
    node.memc = memcached_create(NULL);
    memcached_server_add(node.memc, nodeIdentifier.c_str(), 11211);
    nodes.push_back(node);
}

// Function to remove a node from the distributed system
void DistributedKVStore::removeNode(const std::string& nodeIdentifier) {
    auto it = std::remove_if(nodes.begin(), nodes.end(),
                             [&](const Node& node) { return node.identifier == nodeIdentifier; });

    if (it != nodes.end()) {
        // Close the connection and remove the node from the vector
        memcached_free(it->memc);
        nodes.erase(it, nodes.end());
    }
}

void DistributedKVStore::put(const std::string& key, const std::string& value) {
    // Determine the node using consistent hashing
    size_t hashValue = std::hash<std::string>{}(key);
    size_t nodeIndex = hashValue % nodes.size();
    Node& node = nodes[nodeIndex];

    // Use libmemcached to store the key-value pair on the selected node
    memcached_return_t rc = memcached_set(node.memc, key.c_str(), key.size(), value.c_str(), value.size(), 0, 0);

    if (rc != MEMCACHED_SUCCESS) {
        // Handle error
        std::cerr << "Error storing value on node: " << memcached_strerror(node.memc, rc) << std::endl;
        // You might want to add additional error handling here if needed.
    }

    // Replicate the key-value pair to other nodes
    replicateDataZMQPubSub(nodeIndex, key, value);
}

std::string DistributedKVStore::get(const std::string& key) {

    // Use consistent hashing to determine the server index
    size_t server_index = getServerIndexForKey(key);

    // Get the node associated with the selected server index
    Node& node = nodes[server_index];

    // Use libmemcached to retrieve the value for the key from the selected node
    size_t value_length;
    uint32_t flags;
    memcached_return_t rc; // Declare rc here
    char* value = memcached_get(node.memc, key.c_str(), key.size(), &value_length, &flags, &rc);
    if (rc != MEMCACHED_SUCCESS) {
        // Handle error
        std::cerr << "Error getting value from node: " << memcached_strerror(node.memc, rc) << std::endl;
        return "";
    }

    std::string result(value, value_length);
    free(value);
    return result;
}

// Placeholder implementation of getNodeForKey()
std::string DistributedKVStore::getNodeForKey(const std::string& key) {
    // In this simple implementation, we'll use a basic consistent hashing algorithm.
    // You may replace this with more sophisticated partitioning strategies.
    size_t hashValue = std::hash<std::string>{}(key);
    size_t nodeIndex = hashValue % nodes.size();
    return nodes[nodeIndex].identifier;
}


// Placeholder implementation of replicateData()
void DistributedKVStore::replicateDataMC(const std::string& nodeIdentifier, const std::string& key, const std::string& value) {
    // Connect to the memcached cluster
    memcached_st *memc = memcached_create(nullptr);
    memcached_server_st *servers = memcached_servers_parse("localhost:11211"); // Change this to your memcached cluster address
    memcached_server_push(memc, servers);
    memcached_server_list_free(servers);

    // Replicate the data to the next node in the nodes vector
    size_t nodeIndex = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].identifier == nodeIdentifier) {
            nodeIndex = (i + 1) % nodes.size();
            break;
        }
    }

    const Node& nextNode = nodes[nodeIndex];
    memcached_return_t rc = memcached_set(memc, key.c_str(), key.length(), value.c_str(), value.length(), 0, 0);

    if (rc == MEMCACHED_SUCCESS) {
        std::cout << "Data replicated to node: " << nextNode.identifier << std::endl;
    } else {
        std::cerr << "Failed to replicate data to node: " << nextNode.identifier << std::endl;
    }

    memcached_free(memc);
}

// Sample function for replication using ZeroMQ PUB/SUB pattern
void DistributedKVStore::replicateDataZMQPubSub(size_t nodeIndex, const std::string& key, const std::string& value) {
    // Get the local node identifier
    std::string localNodeIdentifier = nodes[nodeIndex].identifier;

    // Create a ZeroMQ PUB socket to send replication messages
    zmq::context_t context(1);
    zmq::socket_t pubSocket(context, zmq::socket_type::pub);

    // Bind the PUB socket to the local node's address for replication broadcasting
    pubSocket.bind("tcp://*:port"); // Replace 'port' with the port number for the publisher node

    // Wait for the PUB socket to bind before sending messages
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Serialize the key-value pair into a message
    std::string serializedData = key + ":" + value;

    // Send the message to all nodes except the local node
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i == nodeIndex) {
            continue; // Skip sending to the local node
        }

        std::string destinationNodeIdentifier = nodes[i].identifier;

        // Set the filter for the PUB socket to the destination node's identifier
        pubSocket.set(zmq::sockopt::subscribe, destinationNodeIdentifier.c_str());

        // Send the serialized data as a message to the destination node
        zmq::message_t message(serializedData.c_str(), serializedData.size());
        
        // pubSocket.send(message);
        pubSocket.send(message, zmq::send_flags::none);

        // Remove the filter for the next iteration
        pubSocket.set(zmq::sockopt::unsubscribe, destinationNodeIdentifier.c_str());
    }
}
 
void DistributedKVStore::replicateDataZMQReqRep(const std::string& nodeIdentifier, const std::string& key, const std::string& value) {
    // In this simple implementation, we'll replicate the data to the next node in the nodes vector.
    // You can modify this to handle more advanced replication strategies (e.g., quorum-based replication).
/*
    size_t nodeIndex = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].identifier == nodeIdentifier) {
            nodeIndex = (i + 1) % nodes.size();
            break;
        }
    }
*/

    zmq::message_t nodeId(nodeIdentifier.data(), nodeIdentifier.size());
    zmq::message_t empty;
    zmq::message_t keyMsg(key.data(), key.size());
    zmq::message_t valueMsg(value.data(), value.size());

    zmq::context_t context(1);
    zmq::socket_t backendSocket(context, zmq::socket_type::req);
    backendSocket.connect("inproc://backend");

    backendSocket.send(nodeId, zmq::send_flags::none);
    backendSocket.send(empty, zmq::send_flags::none);
    backendSocket.send(keyMsg, zmq::send_flags::none);
    backendSocket.send(valueMsg, zmq::send_flags::none);

    zmq::message_t reply;
    zmq::recv_result_t result = backendSocket.recv(reply, zmq::recv_flags::none);
    if (result.has_value()) {
        // Successfully received data
        // You can access the received data in 'empty'
    } else {
        // Handle the error condition
        // You can use result.error() to get information about the error
    }

}


// Placeholder implementation of handleNetworkPartition()
void DistributedKVStore::handleNetworkPartition(const std::string& failedNodeIdentifier) {
    // In this simple implementation, we'll just remove the failed node from the nodes vector.
    // You would need more sophisticated mechanisms to handle data redistribution and recovery in a real-world system.
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        if (it->identifier == failedNodeIdentifier) {
            nodes.erase(it);
            break;
        }
    }
}

// Example of recoverFromNodeFailure()
void DistributedKVStore::recoverFromNodeFailure(const std::string& failedNodeIdentifier) {
    // In this simple example, we will assume that the data on the failed node is lost.
    // We'll redistribute the data stored on the failed node to other available nodes.

    // Find a new node to receive the data
    std::string newReplica;
    for (const auto& node : nodes) {
        if (node.identifier != failedNodeIdentifier) {
            newReplica = node.identifier;
            break;
        }
    }

    // Loop through the data on the failed node and redistribute it to the new node
    std::lock_guard<std::mutex> lock(mtx);
    auto it = data.find(failedNodeIdentifier);
    if (it != data.end()) {
        for (const auto& entry : it->second) {
            // Perform some operations with 'entry' if needed
            std::string key = entry.first;        // Assuming 'first' is the key in the key-value pair
            std::string value = entry.second;     // Assuming 'second' is the value in the key-value pair

            // Log the key-value pair being redistributed
             std::cout << "Redistributing key: " << key << ", value: " << value << std::endl;

            // Redistribute the data to the new node
            redistributeData(failedNodeIdentifier, newReplica);
        }
        data.erase(it);
    }
}

// Example of redistributeData()
void DistributedKVStore::redistributeData(const std::string& fromNode, const std::string& toNode) {
    // In this simple example, we'll assume that the data transfer between nodes happens instantly.
    // We'll move data directly from 'fromNode' to 'toNode'.

    std::lock_guard<std::mutex> lock(mtx);
    auto it = data.find(fromNode);
    if (it != data.end()) {
        for (const auto& entry : it->second) {
            data[toNode][entry.first] = entry.second;
        }
        data.erase(it);
    }
}

// Example of ensureDataConsistency()
void DistributedKVStore::ensureDataConsistency() {
    // In this simple example, we'll assume an eventual consistency model.
    // We'll replicate all data to a specific number of nodes to ensure consistency.

    // Determine the required number of replicas (e.g., 3 replicas)
    int requiredReplicas = 3;

    std::lock_guard<std::mutex> lock(mtx);
    for (const auto& entry : data) {
        const std::string& nodeIdentifier = entry.first;
        const std::unordered_map<std::string, std::string>& nestedData = entry.second;

        for (const auto& innerEntry : nestedData) {
            const std::string& key = innerEntry.first;
            const std::string& value = innerEntry.second;

            // Get the nodes responsible for storing the key's data
            std::vector<std::string> targetNodes;
            for (const auto& node : nodes) {
                if (getNodeForKey(key) == node.identifier) {
                    targetNodes.push_back(node.identifier);
                }
            }

            // Replicate the data to the required number of nodes
            int replicasToCreate = requiredReplicas - targetNodes.size();
            for (int i = 0; i < replicasToCreate; ++i) {
                // Select a new replica node (randomly or using a consistent hashing algorithm)
                std::string newReplica;
                for (const auto& node : nodes) {
                    if (std::find(targetNodes.begin(), targetNodes.end(), node.identifier) == targetNodes.end()) {
                        newReplica = node.identifier;
                        break;
                    }
                }

                // Replicate the data to the new replica
                replicateDataZMQReqRep(nodeIdentifier, key, value);
                targetNodes.push_back(newReplica);
            }
        }
    }
}

bool DistributedKVStore::pingNode(const std::string& nodeIdentifier) {
    // Create a ZeroMQ REQ socket to send the ping message
    zmq::context_t context(1);
    zmq::socket_t reqSocket(context, zmq::socket_type::req);
    reqSocket.connect("tcp://" + nodeIdentifier); // Replace 'nodeIdentifier' with the actual address of the node

    // Send the ping message to the node
    std::string pingMessage = "Ping";
    zmq::message_t request(pingMessage.c_str(), pingMessage.size());
    reqSocket.send(request, zmq::send_flags::none);

    // Wait for the response from the node
    zmq::message_t response;
    zmq::recv_result_t result = reqSocket.recv(response, zmq::recv_flags::none);

    if (result.has_value()) {
        // Successfully received data
        // You can access the received data in 'empty'
    } else {
        // Handle the error condition
        // You can use result.error() to get information about the error
    }

    // Check if the response is as expected
    std::string responseMsg(static_cast<const char*>(response.data()), response.size());
    if (responseMsg == "Pong") {
        std::cout << "Node " << nodeIdentifier << " is reachable." << std::endl;
        return true;
    } else {
        std::cout << "Node " << nodeIdentifier << " did not respond as expected." << std::endl;
        return false;
    }
}

// Example of handleFaults()
void DistributedKVStore::handleFaults() {
    // In this simple example, we'll assume that a fault is detected when a node does not respond.

    for (const auto& node : nodes) {
        // Send a ping message to the node and wait for a response
        // If the node doesn't respond within a timeout, consider it as a fault.
        bool isNodeAlive = pingNode(node.identifier);

        if (!isNodeAlive) {
            // Handle the fault for the failed node
            recoverFromNodeFailure(node.identifier);

            // Handle network partition if required
            handleNetworkPartition(node.identifier);
        }
    }
}

// Start the distributed key-value store server
void DistributedKVStore::startServer() {
    zmq::context_t context(1);
    zmq::socket_t frontendSocket(context, zmq::socket_type::router);
    zmq::socket_t backendSocket(context, zmq::socket_type::router);
    frontendSocket.bind("tcp://*:8888");
    backendSocket.bind("inproc://backend");

    zmq::message_t msg;
    zmq::pollitem_t items[] = {{frontendSocket, 0, ZMQ_POLLIN, 0}, {backendSocket, 0, ZMQ_POLLIN, 0}};

    // Declare the 'replica' variable here
    std::string replica;

    while (true) {
        try {        
            zmq::poll(items, 2, std::chrono::milliseconds(-1));

            if (items[0].revents & ZMQ_POLLIN) {
                zmq::recv_result_t result1 = frontendSocket.recv(msg, zmq::recv_flags::none);
                if (result1.has_value()) {
                    // Successfully received data
                    // You can access the received data in 'empty'
                } else {
                    // Handle the error condition
                    // You can use result.error() to get information about the error
                }


                // Get the client ID (socket identity)
                std::string clientId(static_cast<char*>(msg.data()), msg.size());

                zmq::recv_result_t result2 = frontendSocket.recv(msg, zmq::recv_flags::none);
                if (result2.has_value()) {
                    // Successfully received data
                    // You can access the received data in 'empty'
                } else {
                    // Handle the error condition
                    // You can use result.error() to get information about the error
                }

                // Get the key from the client request
                std::string key(static_cast<char*>(msg.data()), msg.size());

                std::string node = getNodeForKey(key);
                std::string value = get(key);

                // Send the response back to the client
                zmq::message_t response(value.begin(), value.end());
                frontendSocket.send(zmq::const_buffer(clientId.data(), clientId.size()), zmq::send_flags::sndmore);
                frontendSocket.send(zmq::const_buffer("", 0), zmq::send_flags::sndmore);
                frontendSocket.send(response, zmq::send_flags::none);

            }

            if (items[1].revents & ZMQ_POLLIN) {
                // Receive the key-value pair for replication
                zmq::message_t nodeId;
                zmq::message_t empty;
                zmq::message_t key;
                zmq::message_t value;

                zmq::recv_result_t resultn = backendSocket.recv(nodeId, zmq::recv_flags::none);
                if (resultn.has_value()) {
                    // Successfully received data
                    // You can access the received data in 'empty'
                } else {
                    // Handle the error condition
                    // You can use result.error() to get information about the error
                }

                zmq::recv_result_t resulte = backendSocket.recv(empty, zmq::recv_flags::none);
                if (resulte.has_value()) {
                    // Successfully received data
                    // You can access the received data in 'empty'
                } else {
                    // Handle the error condition
                    // You can use result.error() to get information about the error
                }

                zmq::recv_result_t resultk = backendSocket.recv(key, zmq::recv_flags::none);
                if (resultk.has_value()) {
                    // Successfully received data
                    // You can access the received data in 'empty'
                } else {
                    // Handle the error condition
                    // You can use result.error() to get information about the error
                }

                zmq::recv_result_t resultv = backendSocket.recv(value, zmq::recv_flags::none);
                if (resultv.has_value()) {
                    // Successfully received data
                    // You can access the received data in 'empty'
                } else {
                    // Handle the error condition
                    // You can use result.error() to get information about the error
                }


                // Store the replicated data locally using libmemcached
                replica = std::string(static_cast<char*>(nodeId.data()), nodeId.size());
                std::string replicatedKey(static_cast<char*>(key.data()), key.size());
                std::string replicatedValue(static_cast<char*>(value.data()), value.size());

                // Use libmemcached to store the replicated data
                memcached_return_t rc = memcached_set(nodes[0].memc, replicatedKey.c_str(), replicatedKey.size(),
                                                     replicatedValue.c_str(), replicatedValue.size(), 0, 0);

                if (rc != MEMCACHED_SUCCESS) {
                    // Handle error
                    std::cerr << "Error storing replicated data: " << memcached_strerror(nodes[0].memc, rc) << std::endl;
                }

                // Lock the data store during replication to ensure consistency
                std::lock_guard<std::mutex> lock(mtx);
                // Store the replicated data in the local data store
                data[replica][replicatedKey] = replicatedValue;

            }
        } catch (const zmq::error_t& ex) {
            // Handle ZeroMQ errors
            std::cerr << "ZeroMQ error: " << ex.what() << std::endl;
        } catch (const std::exception& ex) {
            // Handle other exceptions
            std::cerr << "Error: " << ex.what() << std::endl;
        }
    }
}

/*
int main() {
    std::vector<std::string> initialNodes = {"192.168.1.100", "192.168.1.101", "192.168.1.102"};
    DistributedKVStore kvStore(initialNodes);

    kvStore.startServer();

    // Add nodes to the distributed system
    kvStore.addNode("node1");
    kvStore.addNode("node2");
    kvStore.addNode("node3");

    // Start the distributed key-value store server
    kvStore.startServer();

    return 0;
}
*/
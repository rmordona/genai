/*
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
 
/**************************************************************************************************
  LOGGER Class to handle logging for this module.
**************************************************************************************************/

#ifndef LOGGER_H
#define LOGGER_H

// #define FMT_COMPILE // Add this macro before including fmt/format.h
#include <fmt/format.h>  // To support Eigen::MatrixXd and Eigen::VectorXd.

#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>  // support for loading levels from the environment variable.
#include <spdlog/sinks/rotating_file_sink.h>

#include <unistd.h>

namespace spd = spdlog;

class LOGGER {
private:
    std::shared_ptr<spdlog::logger> log;
public:
    std::string filename = "spdlog.log";
    LOGGER () {

        char buffer[PATH_MAX];
        if (getcwd(buffer, sizeof(buffer))) {
            std::cout << "Current Directory: " << buffer << std::endl;
        } else {
            std::cerr << "Error: Unable to get the current directory." << std::endl;
        }

        auto maxSize = 1048576 * 50; // 50 megabytes 
        auto maxRotate = 5; // 5 rotation files only
        log = spd::rotating_logger_mt("LOGGER", filename, maxSize, maxRotate );

        spd::set_level(spd::level::debug);

        log->info("Initializing LOGGER ....");
    }

    ~LOGGER () {
        // Flush the log
        spd::shutdown();
    }

    // Variadic function Template for logging
    template <typename T, typename ...P>
    void logging(const std::string& ltype, T &&format, P &&... params)
    {
        std::string msg = fmt::format(std::forward<T>(format), std::forward<P>(params)...);
        if (ltype == "INFO")         { log->info(msg); } else
        if (ltype == "INFO_INDENT")  { log->info(msg); } else
        if (ltype == "TRACE")        { log->trace(msg); } else
        if (ltype == "DEBUG")        { log->debug(msg); } else
        if (ltype == "WARN")         { log->warn(msg); } else
        if (ltype == "ERROR")        { log->error(msg); } else
        if (ltype == "CRITICAL")     { log->critical(msg); };
        std::cout << msg << std::endl;
    }

    template <class T>
    std::string loggingEigenMatrix(const aimatrix<T>& matrix) {
        std::stringstream ss;
        for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
            ss << "                                          ";
            for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
                ss << fmt::format("{: 8.8f} ", matrix(i, j));
            }
            ss << '\n';
        }
        return ss.str();
    }

    template <class T>
    std::string loggingEigenVector(const aivector<T>& vect) {
        std::stringstream ss;
        for (Eigen::Index i = 0; i < vect.rows(); ++i) {
            ss << "                                          ";
            for (Eigen::Index j = 0; j < vect.cols(); ++j) {
                ss << fmt::format("{: 8.8f} ", vect(i, j));
            }
            ss << '\n';
        }
        return ss.str();
    }

    template <class T>
    std::string loggingEigenRowVector(const airowvector<T>& vect) {
        std::stringstream ss;
        for (Eigen::Index i = 0; i < vect.rows(); ++i) {
            ss << "                                          ";
            for (Eigen::Index j = 0; j < vect.cols(); ++j) {
                ss << fmt::format("{: 8.8f} ", vect(i, j));
            }
            ss << '\n';
        }
        return ss.str();
    }

    template <class T>
    std::string loggingEigenScalar(const aiscalar<T>& scalar) {
        std::stringstream ss;
        ss << "                                          ";
        ss << fmt::format("{: 8.8f} ", scalar);
        ss << '\n';
        return ss.str();
    }

    // Variadic function Template for logging detail
    template <typename T1, typename ...T2>
    void logging_detail(T1 &&format, T2 &&... params)
    {
        std::string indent = "{:>1}{}";
        std::string msg = fmt::format(indent, "", fmt::format( std::forward<T1>(format), std::forward<T2>(params)...));
        std::cout << "tab(xyz): " << msg << std::endl;
        log->info(msg);
    }

    template <typename T1, typename ...T2>
    void logging_wdetail(T1&& format, T2&&... params)
    {
        std::wstring indent = L"{:>1}{}";
        std::wstring msg = fmt::format(indent, L"", fmt::format(std::forward<T1>(format), std::forward<T2>(params)...));
        log->info(wstringToUtf8(msg));
    }

    template <class T>
    void eigen_matrix(const aimatrix<T>& mat) {
        aimatrix<T> tmp_mat = mat;
        std::string msg = fmt::format("{:>1}Matrix:\n{}", "", loggingEigenMatrix(tmp_mat));
        std::cout << "matrix(xyz): " << msg << std::endl;
        log->info(msg);
    }

    template <class T>
    void eigen_matrix(const aitensor<T>& mat) {
        ssize_t batches = mat.size();
        std::string msg = "";
        for (int i = 0; i < batches; i++) {
            aimatrix<T> tmp_mat = mat.at(i);  
            msg = msg + fmt::format("{:>1}Matrix:\n{}", "", loggingEigenMatrix(tmp_mat));
        }
        log->info(msg);
    }

    template <class T>
    void eigen_vector(const aivector<T>& vec) {
        aivector<T> tmp_vec = vec;
        std::string msg = fmt::format("{:>1}Vector:\n{}", "", loggingEigenVector(tmp_vec));
        log->info(msg);
    }

    template <class T>
    void eigen_rowvector(const airowvector<T>& vec) {
        airowvector<T> tmp_vec = vec;
        std::string msg = fmt::format("{:>1}Row Vector:\n{}", "", loggingEigenRowVector(tmp_vec));
        log->info(msg);
    }

    template <class T>
    void eigen_scalar(const aiscalar<T>& scalar) {
        aiscalar<T> tmp_scalar = scalar;
        std::string msg = fmt::format("{:>1}Scalar:\n{}", "", loggingEigenScalar(tmp_scalar));
        log->info(msg);
    }

    void info(const std::string& msg) { log->info(msg); }

    void trace(const std::string& msg) { log->trace(msg); }

    void debug(const std::string& msg) { log->debug(msg); }

    void warn(const std::string& msg) { log->warn(msg); }

    void error(const std::string& msg) { log->error(msg); }

    void critical(const std::string& msg) { log->critical(msg); }

    void set_tag(const std::string& msg) { } //  log->set_tag(msg); }

};

extern LOGGER* ai_log;

#define log_tag(msg) ai_log->set_tag(msg);

// Example use:   log_info("Logging position: {0} {1}", "this", "message");
//                log_info("Logging a float: {:3.2f}", 20.5);
//                log_info("Logging a integer: {:03d}", 345);
#ifdef ENABLE_INFO
#define info_tag()         ai_log->set_tag(__FUNCTION__);
#define log_info(...)      ai_log->logging("INFO", __VA_ARGS__);  
#define log_detail(...)    ai_log->logging_detail(__VA_ARGS__); 
#define log_wdetail(...)   ai_log->logging_detail(__VA_ARGS__); 
#define log_matrix(msg)    ai_log->eigen_matrix(msg); 
#define log_vector(msg)    ai_log->eigen_vector(msg); 
#define log_rowvector(msg) ai_log->eigen_rowvector(msg); 
#define log_scalar(msg)    ai_log->eigen_scalar(msg); 
#else
#define info_tag()
#define log_info(...)  
#define log_detail(...)  
#define log_wdetail(...)  
#define log_matrix(msg)
#define log_vector(msg)
#define log_rowvector(msg)
#define log_scalar(msg)
#endif

#ifdef ENABLE_TRACE
#define log_trace(...) ai_log->logging("TRACE",  __VA_ARGS__); 
#else
#define log_trace(...)  
#endif

#ifdef ENABLE_DEBUG
#define log_debug(...)  ai_log->logging("DEBUG", __VA_ARGS__); 
#else
#define log_debug(...)  
#endif

#ifdef ENABLE_WARNING
#define log_warning(...) ai_log->logging("WARNING",  __VA_ARGS__); 
#else
#define log_warning(...)  
#endif

#ifdef ENABLE_ERROR
#define log_error(...)  ai_log->logging("ERROR",  __VA_ARGS__); 
#else
#define log_error(...)  
#endif

#ifdef ENABLE_CRITICAL
#define log_critical(...)  ai_log->logging("CRITICAL",  __VA_ARGS__); 
#else
#define log_critical(...)  
#endif

// extern LOG


#endif

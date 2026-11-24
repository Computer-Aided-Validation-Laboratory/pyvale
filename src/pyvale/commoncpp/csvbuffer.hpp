#ifndef PYVALE_COMMONCPP_CSVBUFFER_HPP
#define PYVALE_COMMONCPP_CSVBUFFER_HPP

#include <charconv>
#include <fstream>
#include <string>
#include <system_error>

namespace common_util {

class CsvBuffer {
public:
    CsvBuffer(std::ofstream& file,
              const std::string& delimiter,
              int precision = 8)
        : file_(file), delimiter_(delimiter), precision_(precision) {
        buffer_.reserve(1024 * 1024);
    }

    ~CsvBuffer() { flush(); }

    void raw(const std::string& value) {
        buffer_.append(value);
        flush_if_full();
    }

    void field(const std::string& value, bool last = false) {
        raw(value);
        if (!last) raw(delimiter_);
    }

    void field(double value, bool last = false) {
        char output[64];
        const auto result = std::to_chars(output, output + sizeof(output),
                                          value, std::chars_format::scientific,
                                          precision_);
        buffer_.append(output, result.ptr);
        if (!last) buffer_.append(delimiter_);
        flush_if_full();
    }

    void field(int value, bool last = false) {
        char output[32];
        const auto result = std::to_chars(output, output + sizeof(output), value);
        buffer_.append(output, result.ptr);
        if (!last) buffer_.append(delimiter_);
        flush_if_full();
    }

    void newline() {
        buffer_.push_back('\n');
        flush_if_full();
    }

    void flush() {
        if (!buffer_.empty()) {
            file_.write(buffer_.data(),
                        static_cast<std::streamsize>(buffer_.size()));
            buffer_.clear();
        }
    }

private:
    void flush_if_full() {
        if (buffer_.size() >= 1024 * 1024) flush();
    }

    std::ofstream& file_;
    const std::string& delimiter_;
    int precision_;
    std::string buffer_;
};

} // namespace common_util

#endif // PYVALE_COMMONCPP_CSVBUFFER_HPP

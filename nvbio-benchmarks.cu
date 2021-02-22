/*
 * Copyright (c) 2021 Quim Aguado
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/vector.h>
#include <nvbio/alignment/alignment.h>
#include <nvbio/alignment/alignment_base.h>
#include <nvbio/alignment/batched.h>

#define TIMER_INIT  std::chrono::steady_clock::time_point bm_timer_begin; \
                    std::chrono::steady_clock::time_point bm_timer_end;
#define TIMER_START bm_timer_begin = std::chrono::steady_clock::now();
#define TIMER_STOP  bm_timer_end = std::chrono::steady_clock::now();

#define TIMER_MS std::chrono::duration_cast<std::chrono::milliseconds> (bm_timer_end - bm_timer_begin).count()

#define ALPHABET_SIZE 4

const uint32_t BAND_LEN = 31;

const char *USAGE_STR = "Usage:\n"
                        "nvbio-benchmark <file> <max_seq_len> <num_alignments> "
                        "<batch_size=50000>";


class Sequences {
public:
    size_t seq_len;
    size_t num_alignments;
    char* sequences_buffer;
    int* sequences_len;

    Sequences (char* filepath, int seq_len, int num_alignments) :\
                                                    seq_len(seq_len),
                                                    num_alignments(num_alignments) {
        std::cout << "Sequences object:" << std::endl
                  << "\tFile: " << filepath << std::endl
                  << "\tSequence length: " << seq_len << std::endl
                  << "\tNumber of alignments: " << num_alignments << std::endl;

        std::size_t seq_bytes_to_alloc = ((size_t)num_alignments * (size_t)seq_len * 2L);
        std::cout << "Allocating " << (seq_bytes_to_alloc / (1 << 20))
                  << "MiB of memory to store the sequences" << std::endl;
        try {
            this->sequences_buffer = new char[seq_bytes_to_alloc];
        } catch (std::bad_alloc & exception) {
            std::cerr << "bad_alloc detected: " << exception.what();
            exit(-1);
        }
        memset(this->sequences_buffer, 0, seq_bytes_to_alloc);
        this->sequences_len = new int[(size_t)num_alignments * 2L];

        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (file.fail()) {
            std::cerr << "Could not open file: \"" << filepath << "\"" << std::endl;
            // TODO
            exit(-1);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        TIMER_INIT

        TIMER_START

        std::string line;
        size_t sequences_read = 0;
        while(std::getline(file, line) && sequences_read < (num_alignments*2)) {
            strncpy(this->get_sequence(sequences_read),
                    // +1 to avoid the initial > and <
                    line.c_str() + 1,
                    seq_len);
            this->sequences_len[sequences_read] = line.length() - 1;
            sequences_read++;
        }

        TIMER_STOP
        std::cout << "Read " << sequences_read << " sequences in " << TIMER_MS
                  << "ms." << std::endl;
    };

    ~Sequences () {
        delete [] this->sequences_buffer;
        delete [] this->sequences_len;
    }

    char* get_sequence(size_t n) const {
#ifdef DEBUG
        // Only for debug purposes
        if (n >= this->num_alignments*2) {
            std::cout << "Trying to read too far... n=" << n << std::endl;
            return 0;
        }
#endif
        return this->sequences_buffer + (this->seq_len * n);
    }
};

// Function based on the example in the NVidia blog, but adapted:
//     https://developer.nvidia.com/blog/accelerating-bioinformatics-nvbio/
// Time is returned in ms
void batch_alignment_test (const Sequences &sequences, const size_t batch_offset,
                           const uint32_t batch_size, double *time) {
    using namespace nvbio;
    // build two concatenated string sets, one for the patterns, 
    // containing a concatenated sequence of strings of 100 
    // characters each, and one for the texts, 
    // containing 200 characters each
    const uint32 n_strings   = batch_size;
    const uint32 pattern_len = sequences.seq_len;
    const uint32 text_len    = sequences.seq_len;

#ifdef DEBUG
    std::cerr << "Batch size: " << n_strings << std::endl
              << "Pattern length: " << pattern_len << std::endl
              << "Text length: " << text_len << std::endl;
#endif

    // setup the strings on the host
    nvbio::vector<host_tag, uint8> h_pattern(n_strings * pattern_len);
    nvbio::vector<host_tag, uint8> h_text(n_strings * text_len);

    // Copy patterns for this batch
    for (uint32 i = 0; i < n_strings; i++) {
        //std::cout << "copying pattern "<< i << std::endl;
        memcpy((void*)&h_pattern[i * pattern_len], 
               sequences.get_sequence(batch_offset + i*2),
               sequences.seq_len);
    }
    // Copy texts for this batch
    for (uint32 i = 0; i < n_strings; i++) {
        memcpy((void*)&h_text[i * text_len], 
               sequences.get_sequence(batch_offset + (i+1)*2),
               sequences.seq_len);
    }

    TIMER_INIT

    TIMER_START
    // copy the strings storage to the device
    nvbio::vector<device_tag, uint8> d_pattern( h_pattern );
    nvbio::vector<device_tag, uint8> d_text( h_text );

    // allocate two vectors representing the string offsets
    nvbio::vector<device_tag, uint32> d_pattern_offsets( n_strings+1 );
    nvbio::vector<device_tag, uint32> d_text_offsets( n_strings+1 );

    // prepare the string offsets using Thrust's sequence() 
    // function, setting up the offset of pattern i as i * pattern_len, 
    // and the offset of text i as i * text_len
    thrust::sequence( d_pattern_offsets.begin(), 
                      d_pattern_offsets.end(), 0u, pattern_len );
    thrust::sequence( d_text_offsets.begin(), 
                      d_text_offsets.end(), 0u, text_len );

    // prepare a vector of alignment sinks
    nvbio::vector<device_tag, aln::BestSink<uint32> > 
        sinks( n_strings );

    // and execute the batch alignment, on a GPU device
    aln::batch_banded_alignment_score<BAND_LEN>(
        aln::make_edit_distance_aligner
            <aln::GLOBAL, aln::MyersTag<ALPHABET_SIZE> >(),
        make_concatenated_string_set( n_strings, 
                                      d_pattern.begin(),
                                      d_pattern_offsets.begin() ),
        make_concatenated_string_set( n_strings, 
                                      d_text.begin(), 
                                      d_text_offsets.begin() ),
        sinks.begin(),
        aln::DeviceThreadScheduler(),
        sequences.seq_len,
        sequences.seq_len );
    TIMER_STOP
    *time += TIMER_MS;
}

int main (int argc, char* argv[]) {
    char* filepath;
    size_t batch_size = 50000;
    int seq_size = 0;
    int num_alignments;

    if (argc >= 4) {
        filepath = argv[1];
        seq_size = std::atoi(argv[2]);
        num_alignments = std::atoi(argv[3]);
    }
    if (argc == 5) {
        batch_size = std::atoi(argv[4]);
    }
    if (argc < 4 || argc > 5) {
        std::cerr << USAGE_STR << std::endl;
        return EXIT_FAILURE;
    }

    if (batch_size > num_alignments) {
        std::cerr << "Batch size can not be bigger than the number of alignments"
                  << "\nChanging batch size to " << num_alignments << std::endl;
        batch_size = num_alignments;
    } else {
        std::cout << "Batch size set to " << batch_size << std::endl;
    }

    Sequences sequences(filepath, seq_size, num_alignments);

    // Total time in milliseconds
    double total_time = 0;
    size_t alignments_computed = 0;
    int cnt = 0;

    while (alignments_computed < num_alignments) {
        size_t curr_batch_size = std::min(batch_size,
                                          num_alignments - alignments_computed);

        batch_alignment_test (sequences,
                              alignments_computed,
                              curr_batch_size,
                              &total_time);
        std::cerr << "Batch " << cnt++ << " executed." << std::endl;
        alignments_computed += batch_size;
    }

    std::cout << "Executed " << num_alignments << " alignments in "
              << total_time << "ms."
              << std::endl
              << "Performance: "
              << (double)((num_alignments * (uint64_t)(seq_size*seq_size)) / (total_time/1000)) / 1000000000
              << " GCUPs" << std::endl;

    return 0;
}

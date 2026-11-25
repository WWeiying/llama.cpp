#include "llama.h"
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct logits_dump_record {
    uint32_t magic;
    uint32_t version;
    uint32_t step;
    uint32_t n_vocab;
    int32_t  target_token;
    uint32_t reserved;
};

static constexpr uint32_t LOGITS_DUMP_MAGIC = 0x5142534c;

static bool dump_logits(
        FILE * file, const float * logits, uint32_t step, uint32_t n_vocab, llama_token target_token) {
    if (file == nullptr) {
        return true;
    }
    const logits_dump_record record = {
        LOGITS_DUMP_MAGIC, 2, step, n_vocab, target_token, 0,
    };
    return logits != nullptr &&
           fwrite(&record, sizeof(record), 1, file) == 1 &&
           fwrite(logits, sizeof(*logits), n_vocab, file) == n_vocab &&
           fflush(file) == 0;
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-t n_threads] [-fa auto|on|off] [--kv-type f16|q8_0|q4_0] [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;
    // number of CPU threads; zero keeps the llama.cpp default
    int n_threads = 0;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    ggml_type kv_type = GGML_TYPE_F16;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-t") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_threads = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                    if (n_threads <= 0) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-fa") == 0) {
                if (++i >= argc) { print_usage(argc, argv); return 1; }
                if (!strcmp(argv[i], "on")) flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                else if (!strcmp(argv[i], "off")) flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                else if (!strcmp(argv[i], "auto")) flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "--kv-type") == 0) {
                if (++i >= argc) { print_usage(argc, argv); return 1; }
                if (!strcmp(argv[i], "f16")) kv_type = GGML_TYPE_F16;
                else if (!strcmp(argv[i], "q8_0")) kv_type = GGML_TYPE_Q8_0;
                else if (!strcmp(argv[i], "q4_0")) kv_type = GGML_TYPE_Q4_0;
                else { print_usage(argc, argv); return 1; }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(vocab);
    const bool teacher_force = std::getenv("LLAMA_SIMPLE_TEACHER_FORCE") != nullptr;
    FILE * logits_file = nullptr;
    if (const char * logits_path = std::getenv("LLAMA_SIMPLE_LOGITS_FILE")) {
        logits_file = std::fopen(logits_path, "wb");
        if (logits_file == nullptr) {
            std::fprintf(stderr, "%s: unable to open logits dump %s\n",
                         __func__, logits_path);
            llama_model_free(model);
            return 1;
        }
    }
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.flash_attn_type = flash_attn;
    ctx_params.type_k = kv_type;
    ctx_params.type_v = kv_type;
    // n_ctx is the context size
    ctx_params.n_ctx = teacher_force ? n_prompt : n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    if (n_threads > 0) {
        ctx_params.n_threads = n_threads;
        ctx_params.n_threads_batch = n_threads;
    }
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    if (teacher_force) {
        if (n_prompt < 2) {
            fprintf(stderr, "%s: teacher-forced evaluation needs at least two tokens\n", __func__);
            return 1;
        }

        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        batch.n_tokens = n_prompt;
        for (int pos = 0; pos < n_prompt; ++pos) {
            batch.token[pos] = prompt_tokens[pos];
            batch.pos[pos] = pos;
            batch.n_seq_id[pos] = 1;
            batch.seq_id[pos][0] = 0;
            batch.logits[pos] = pos + 1 < n_prompt;
        }

        const auto t_eval_start = ggml_time_us();
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s: teacher-forced decode failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }

        for (int pos = 0; pos + 1 < n_prompt; ++pos) {
            if (!dump_logits(logits_file, llama_get_logits_ith(ctx, pos), pos,
                             n_vocab, prompt_tokens[pos + 1])) {
                fprintf(stderr, "%s: failed to write teacher-forced logits\n", __func__);
                llama_batch_free(batch);
                return 1;
            }
        }
        const auto t_eval_end = ggml_time_us();

        printf("LLAMA_TEACHER_RECORDS=%d\n", n_prompt - 1);
        fprintf(stderr, "%s: evaluated %d fixed next-token targets in %.2f s\n",
                __func__, n_prompt - 1,
                (t_eval_end - t_eval_start) / 1000000.0f);
        llama_perf_context_print(ctx);

        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        if (logits_file != nullptr) {
            std::fclose(logits_file);
        }
        return 0;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        if (!dump_logits(logits_file, llama_get_logits_ith(ctx, -1), n_decode,
                         n_vocab, LLAMA_TOKEN_NULL)) {
            fprintf(stderr, "%s: failed to write logits dump\n", __func__);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    if (logits_file != nullptr) {
        std::fclose(logits_file);
    }

    return 0;
}

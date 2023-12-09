from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import streamlit as st
import os
import time
import torch
import speedtest

print("\n--------------------> Reload")

# export LLM_RUN_MODE=gpu         for Inference
# export LLM_RUN_MODE=cpu         for Inference
# echo $LLM_RUN_MODE

# Supported models:
# 01-ai/Yi-6B-Chat
# 01-ai/Yi-6B-Chat-4bits
# 01-ai/Yi-34B-Chat-4bits

# export LLM_MODEL_NAME=01-ai/Yi-6B-Chat-4bits        only GPU
# export LLM_MODEL_NAME=01-ai/Yi-34B-Chat-4bits       only GPU
# export LLM_MODEL_NAME=01-ai/Yi-6B-Chat              GPU or CPU
# echo $LLM_MODEL_NAME

# export LLM_HEADER_NAME=LLM
# echo $LLM_HEADER_NAME

# streamlit run app1.py --server.port=8000

run_mode = os.getenv("LLM_RUN_MODE",default="gpu")
model_name = os.getenv("LLM_MODEL_NAME",default="01-ai/Yi-6B-Chat-4bits")
if model_name != "01-ai/Yi-6B-Chat": # the quantized model can only by run by GPU
    run_mode = "gpu"
header_name = os.getenv("LLM_HEADER_NAME",default="")
if header_name == "":
    header_name = model_name

st.sidebar.subheader("Settings")
streaming_mode = st.sidebar.slider("Enable Streaming Mode", min_value=0, max_value=1, value=1, step=1)
memory_mode = st.sidebar.slider("Enable Conversational Memory", min_value=0, max_value=1, value=1, step=1)
max_memory_ability = st.sidebar.slider("Memory Ability (sentences)", min_value=11, max_value=99, value=11, step=2)
generate_max_len = st.sidebar.number_input("generate_max_len (tokens)", min_value=0, max_value=512, value=64, step=1)
top_k = st.sidebar.slider("top_k", min_value=0, max_value=10, value=3, step=1)
top_p = st.sidebar.number_input("top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
temperature = st.sidebar.number_input("temperature", min_value=0.0, max_value=100.0, value=1.0, step=0.1)

print("Header: {}, Model Name: {}, Run Mode: {}, Streaming: {}, Memory: {}, Memory Ability: {}".format(
                  header_name,model_name,run_mode,streaming_mode, memory_mode,max_memory_ability  ) )

st.header(header_name)
# st.subheader("Data Analytics")

@st.cache_resource
def load_global_history():
    return [{'role': 'system', 'content': 'You are a helpful assistant.'}]

def load_local_history():
    return [{'role': 'system', 'content': 'You are a helpful assistant.'}]

global_history = load_global_history()  # Global conversational memory
if len(global_history) > max_memory_ability:
    print("\nDelete these sentences from the memory:")
while (len(global_history) > max_memory_ability ):
    print(global_history[1])
    del global_history[1]

local_history = load_local_history()  # The memory for current round

@st.cache_resource
def load_model():
    if(run_mode == "cpu"):
        tempModel = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto").eval()
    else:
        tempModel = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", torch_dtype="auto").eval()
    tempTokenizer = AutoTokenizer.from_pretrained(model_name)
    return tempModel,tempTokenizer

model,tokenizer = load_model()

inputs = st.text_area("Please input", max_chars=512)

if st.button("Generate"):
    inputs_message = st.empty()
    if inputs != "":
        inputs_message.write("Working to generate the answer, please wait ...")

        if memory_mode == 1:
            history = global_history
        else:
            history = local_history
        history.append({'role': 'user', 'content': inputs})

        if streaming_mode == 1:
            print("\n----------------------------------------> Generation with Streaming")

            input_string_with_st = tokenizer.apply_chat_template( conversation=history, tokenize=False,
                                                                  add_generation_prompt=True) # 只是应用模版
            tempInputs = tokenizer(input_string_with_st, return_tensors='pt') # including special tokens
            input_string = tokenizer.decode(tempInputs.input_ids[0], skip_special_tokens=True)

            print("--------------------> Input")
            print("The length of input sequence: {}".format(tempInputs.input_ids.shape[1]))
            print(tempInputs.input_ids) # including special token IDs
            print("------> input words (history and prompt) with special tokens")
            print(input_string_with_st)  # including special tokens
            #print("------> input words (history and prompt) without special tokens")
            #print(input_string)

            real_length = generate_max_len + tempInputs.input_ids.shape[1]

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

            start_time = time.time()
            if (run_mode == "cpu"):
                generation_kwargs = dict(
                    tempInputs,streamer=streamer,
                    max_length=real_length, eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,repetition_penalty=1.3,no_repeat_ngram_size=5,
                    temperature=temperature,top_k=top_k,top_p=top_p)
            else:
                generation_kwargs = dict(
                    tempInputs.to('cuda'),streamer=streamer,
                    max_length=real_length, eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,repetition_penalty=1.3,no_repeat_ngram_size=5,
                    temperature=temperature,top_k=top_k,top_p=top_p)

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            outputs_message = st.empty()
            output_string = ""
            print("--------------------> Output")
            for new_text in streamer:
                print(new_text)
                if ('<|im_end|>' in new_text):
                    output_string += new_text.replace('<|im_end|>','')
                    print("!!!!!!!!!! REMOVE <|im_end|> from： " + new_text)
                else:
                    output_string += new_text
                outputs_message.write(output_string)

            end_time = time.time()
            inputs_message.write("Completed and took {:.3f} s".format(end_time - start_time))

            print("------> output words (no history and prompt) without special tokens")
            print(output_string)

            if(len(output_string) > 256):
                output_string = output_string[0:256]
            history.append({'role': 'assistant', 'content': output_string})

        else:
            print("\n----------------------------------------> Generation without Streaming")

            input_ids = tokenizer.apply_chat_template(conversation=history, tokenize=True,
                                                      add_generation_prompt=True,
                                                      return_tensors='pt')
            input_string = tokenizer.decode(input_ids[0], skip_special_tokens=True) # not including special tokens
            input_string_with_st = tokenizer.decode(input_ids[0]) # can only decode a sample, not a batch

            print("--------------------> Input")
            print("The length of input sequence: {}".format(input_ids.shape[1]))
            print(input_ids) # including special token IDs
            print("------> input words (history and prompt) with special tokens")
            print(input_string_with_st)
            #print("------> input words (history and prompt) without special tokens")
            #print(input_string) # not including special tokens

            real_length = generate_max_len + input_ids.shape[1]

            start_time = time.time()
            if (run_mode == "cpu"):
                outputs = model.generate(
                    input_ids,
                    max_length=real_length,eos_token_id=tokenizer.eos_token_id,
                    do_sample=True, repetition_penalty=1.3, no_repeat_ngram_size=5,
                    temperature=temperature, top_k=top_k, top_p=top_p)
            else:
                outputs = model.generate(
                    input_ids.to('cuda'),
                    max_length=real_length, eos_token_id=tokenizer.eos_token_id,
                    do_sample=True, repetition_penalty=1.3, no_repeat_ngram_size=5,
                    temperature=temperature, top_k=top_k, top_p=top_p)
            end_time = time.time()
            inputs_message.write("Completed and took {:.3f} s".format(end_time - start_time))

            output_string = tokenizer.decode(outputs[0],skip_special_tokens=True)  # not including special tokens
            output_string_with_st = tokenizer.decode(outputs[0])

            result = output_string.replace(input_string, "") # remove the input from the output

            print("--------------------> Output")
            print("The length of output sequence: {}".format(outputs.shape[1]))
            print(outputs[0]) # including special token IDs
            #print("------> output words (history and prompt) with special tokens")
            #print(output_string_with_st)
            print("------> output words (no history and prompt) without special tokens")
            print(result) # not including special tokens

            outputs_message = st.empty()
            outputs_message.write(result)

            if(len(result) > 256):
                result = result[0:256]
            history.append({'role': 'assistant', 'content': result})

        #print("--------------------> History")
        #for x in history:
        #    print(x)

    else:
        inputs_message.write("The inputs cannot be empty.")
        # st.stop()
#else:
#    st.stop()

if st.button("Conversational Memory"):
    history_message = st.empty()
    history_message.write(global_history)
#else:
#   st.stop()

if st.button("Environment"):
    st.write( "PyTorch and CUDA Version: {}".format(torch.__version__) + ", cdDNN Version: {}".format(torch.backends.cudnn.version()) )
    gpu_number = torch.cuda.device_count()
    st.write( "GPU Device Number: {}".format(gpu_number) )

    temp1gb = 1024**3
    for i in range(gpu_number):
        st.write("Device {} : {}".format(i,torch.cuda.get_device_name(i)))
        t = torch.cuda.get_device_properties(i).total_memory / temp1gb
        r = torch.cuda.memory_reserved(i) / temp1gb
        a = torch.cuda.memory_allocated(i) / temp1gb
        f = (r - a) / temp1gb  # free inside reserved
        st.write("Memory of Device {}: total {:.2f} GB, reserved {:.2f} GB, allocated {:.2f} GB, free {:.2f} GB".format(i,t,r,a,f))

if st.button("Network Test"):
    speed_test = speedtest.Speedtest()
    bserver = speed_test.get_best_server()

    download_speed = speed_test.download() / 1000000  # Convert to Mbps
    upload_speed = speed_test.upload() / 1000000  # Convert to Mbps

    result = {
        "Test Server Location": bserver['name'],
        "Country": bserver['country'],
        "Latency": '{:.2f} ms'.format(bserver['latency']),
        "Download Speed": "{:.2f} Mbps".format(download_speed),
        "Upload Speed": "{:.2f} Mbps".format(upload_speed)
    }

    network_message = st.empty()
    network_message.write(result)
#else:
#   st.stop()
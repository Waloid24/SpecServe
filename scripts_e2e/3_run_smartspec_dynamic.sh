# cd ../
# bash compile.sh
# cd scripts

kill -9 `ps -aux|grep multi|awk '{print $2}'`

# pip uninstall -y vllm-flash-attn

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=on
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"

# pgrep -f 'api_server' | xargs kill -9

gpu_id=0
gpu_memory_utilizations=(0.8)

gpu_count=$(echo "$gpu_id" | awk -F, '{print NF}')
draft_models=(JackFram/llama-160m)
target_models=(lmsys/vicuna-7b-v1.3)
# draft_models=(JackFram/llama-160m)
# target_models=(lmsys/vicuna-33b-v1.3)
rate_change_interval=1
# request_rates=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
# duration=60
# num_prompt=630
duration=15
scale=1
req_rates_csv='../benchmarks/specserve/traces/AzureLLMInferenceTrace_conv_1week.csv'  
request_rates=(     $(tail -n +2 "$req_rates_csv" | \
    cut -d',' -f1 | \
    sed 's/\..*$//' | \
    uniq -c | \
    awk '{print $1}') )

request_rates=("${request_rates[@]:0:$duration}")
echo "request_rates=(${request_rates[@]})"
request_rates_str=$(printf "%s," "${request_rates[@]}")
request_rates_str=${request_rates_str%,} 
num_prompt=0
for ((i=0; i<${#request_rates[@]}; i++)); do
    num_prompt=$((num_prompt + request_rates[i]))
done
echo "num_prompt: $num_prompt"
echo "avg request rate: $((num_prompt / duration))"

num_speculative_tokens=(10)
max_num_seqs=(256)
dataset_path=./dataset/sharegpt_prompt_gt10x_generated.json
dataset_name=p10g
repeats=3

proposal_len_selection_policys=('specserve')
log_path=dataset/${proposal_len_selection_policys}/dynamic/${dataset_name}
if [ ! -d ${log_path} ]; then
    mkdir -p ${log_path}
fi

wait_for_server() {
    local port=$1
    while true; do
        if netstat -tulnp | grep -q "${port}"; then
            echo "server is running on port ${port}"
            break
        else
            echo "server is not running on port ${port}"
            nvidia-smi
            sleep 5
        fi
    done
}

for repeat in $(seq 1 $repeats); do
    for i in "${!draft_models[@]}"; do
        draft_model="${draft_models[$i]}"
        gpu_memory_utilization="${gpu_memory_utilizations[$i]}"
        draft_model_name=$(echo "$draft_model" | tr '/' '_')
        target_model_name=$(echo "${target_models[$i]}" | tr '/' '_')
        for num_speculative_token in ${num_speculative_tokens[@]}; do
            for max_num_seq in ${max_num_seqs[@]}; do
                for proposal_len_selection_policy in ${proposal_len_selection_policys[@]}; do
                    CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
                        --speculative-model ${draft_model} \
                        --model ${target_models[$i]} \
                        --port 8089 \
                        --speculative-draft-tensor-parallel-size 1 \
                        --tensor-parallel-size ${gpu_count} \
                        --swap-space 4 \
                        --gpu-memory-utilization ${gpu_memory_utilization} \
                        --max-num-seqs ${max_num_seq} \
                        --use-v2-block-manager \
                        --ssd \
                        --rsd \
                        --num-speculative-tokens ${num_speculative_token} \
                        --disable-log-requests > ${log_path}/${draft_model_name}_${target_model_name}_${duration}_${num_speculative_token}_${max_num_seq}_${proposal_len_selection_policy}_${num_prompt}_server.log 2>&1 &
                    pid=$!    
                    wait_for_server 8089
                    sleep 1
                    
                    python3 ../benchmarks/benchmark_serving_dynamic.py \
                        --port 8089 \
                        --model ${target_models[$i]} \
                        --dataset ${dataset_path} \
                        --request-rates ${request_rates} \
                        --rate-change-interval ${rate_change_interval} \
                        --num-prompts ${num_prompt} \
                        --tokenizer ${target_models[$i]} >> ${log_path}/${draft_model_name}_${target_model_name}_${duration}_${num_speculative_token}_${max_num_seq}_${proposal_len_selection_policy}_${num_prompt}_client.log 
                    kill -9 $pid 
                    sleep 1
                    pkill -f 'multiprocessing-fork'
                done
            done   
        done
    done
done    
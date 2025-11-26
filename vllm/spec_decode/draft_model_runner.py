import sys
import time
from typing import List, Optional, Dict

import torch

from vllm.forward_context import set_forward_context
from vllm.sequence import SamplerOutput
from vllm.spec_decode.profile import SpecProfiler
import numpy as np

scaled_slo_tpot = 0.04 # SLO contraint used in function "_predict_goodput_with_constraint"

try:
    try:
        from vllm.attention.backends.flash_attn import FlashAttentionMetadata
    except (ModuleNotFoundError, ImportError):
        # vllm_flash_attn is not installed, try the ROCm FA metadata
        from vllm.attention.backends.rocm_flash_attn import (
            ROCmFlashAttentionMetadata as FlashAttentionMetadata)
except (ModuleNotFoundError, ImportError) as err:
    raise RuntimeError(
        "Draft model speculative decoding currently only supports"
        "CUDA and ROCm flash attention backend.") from err

from vllm.logger import init_logger
from vllm.multimodal import MultiModalInputs
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner, _get_graph_batch_size)

logger = init_logger(__name__)

# A flag to enable debug prints for the updated input tensors
# before each step.
debug_advance_input = False
# A flag to allow GPU advance step for draft model runner.
# Set to False for debugging.
allow_gpu_advance_step = True


class TP1DraftModelRunner(ModelRunner):
    """Specialized model runner for speculative decoding draft model.
    Since the draft model always execute k forward passes consecutively to
    generate k speculative tokens in a single speculative decoding step,
    we could get rid of most CPU-GPU synchronization and data transfer
    overheads by keeping model input and output tensors on GPU all the time.

    TODOs:
    1. Currently supports only flash-attn, add support for other attn_backends.
    2. Support TP > 1 (this requires some designs because we do not expect
       any broadcasting inside execute_model).
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("return_hidden_states"):
            raise ValueError(
                "return_hidden_states is not supported for TP1DraftModelRunner."
            )

        super().__init__(*args, **kwargs)

        self.indices_of_seq_with_bonus_tokens = None

        self.draft_seq_lens = 0
        self.target_seq_lens = 0

        if self.speculative_config.ssd:
            self.draft_models = None
            self.target_models = None
            self.goodput = -1
            self.draft_overhead = []
            self.target_overhead = []
            self.draft_input_len_list = []

            self.total_confidence = 0.0
            self.total_count = 0
            self.default_confidence = 0.6

    def _update_sampling_metadata(self, sampling_metadata, num_seqs,
                                  num_queries):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (
            num_queries,)
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        # Verify that all sequences are decodes
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple

    def _gpu_advance_step(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            last_output: SamplerOutput
    ) -> ModelInputForGPUWithSamplingMetadata:
        # Currently, we expect "decode mode" only
        assert not model_input.is_prompt

        # Get num_seqs
        num_seqs = len(model_input.seq_lens)
        num_queries = len(model_input.query_lens)

        # Get output tokens GPU tensor
        sampled_token_ids = last_output.sampled_token_ids
        assert sampled_token_ids is not None

        # Update attn_metadata
        attn_metadata = model_input.attn_metadata
        assert isinstance(attn_metadata, FlashAttentionMetadata)

        attn_metadata.advance_step(model_input, sampled_token_ids,
                                   self.block_size, num_seqs, num_queries)

        # Update sampling_metadata
        sampling_metadata = model_input.sampling_metadata
        self._update_sampling_metadata(sampling_metadata, num_seqs,
                                       num_queries)

        # Create new input
        new_model_input = self._model_input_cls(
            input_tokens=model_input.input_tokens,
            input_positions=model_input.input_positions,
            attn_metadata=attn_metadata,
            seq_lens=attn_metadata.seq_lens,
            query_lens=model_input.query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            sampling_metadata=model_input.sampling_metadata,
            is_prompt=False,
        )

        # Ensure we skip CPU samples
        assert new_model_input.sampling_metadata.skip_sampler_cpu_output is True
        # We can reuse sampling tensors since every decode iteration is the same
        new_model_input.sampling_metadata.reuse_sampling_tensors = True

        if debug_advance_input:
            logger.debug("NEW INPUT: ")
            logger.debug("  input_tokens = %s", new_model_input.input_tokens)
            logger.debug("  input_positions = %s",
                         new_model_input.input_positions)
            logger.debug("  seq_lens = %d", new_model_input.seq_lens)
            logger.debug("  query_lens = %d", new_model_input.query_lens)
            logger.debug("  attn_metadata:")
            logger.debug("    seq_lens_tensor: %s",
                         attn_metadata.seq_lens_tensor)
            logger.debug("    slot_mapping: %s", attn_metadata.slot_mapping)
            logger.debug("    block_tables: %s", attn_metadata.block_tables)

        return new_model_input

    def supports_gpu_multi_step(self, execute_model_req: ExecuteModelRequest):
        """Determines if draft_model_runner GPU multi-step can be used.
        Currently required conditions are:
            1. Only decodes 
            2. Only flash-attn
            3. No LORA
            4. No prompt_adapter_config
        """
        if not allow_gpu_advance_step:
            return False

        # We allow multi-step GPU only in decode mode
        for seq_group in execute_model_req.seq_group_metadata_list:
            if seq_group.is_prompt:
                return False

        # TODO: Add support for other attn backends
        if self.attn_backend.get_name() != "FLASH_ATTN":
            return False

        # TODO: Add support for LORA
        if self.lora_config:
            return False

        # TODO: Add soft-tuning prompt adapter support
        return not self.prompt_adapter_config

    def set_indices_of_seq_with_bonus_tokens(self,
                                             indices_of_seq_with_bonus_tokens):
        self.indices_of_seq_with_bonus_tokens = indices_of_seq_with_bonus_tokens

    @torch.inference_mode()
    def execute_model(
            self,
            model_input: ModelInputForGPUWithSamplingMetadata,
            kv_caches: List[torch.Tensor],
            previous_hidden_states: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        """Executes num_steps forward passes with advacement of input tensors 
        on the GPU. Look at supports_gpu_multi_step(..) for pre-conditions.

        Optimizations used:
            1. Input tensors are updated on the GPU directly
            2. Skips GPU=>CPU serialization of sampler outputs (we don't need 
                them since we do batch expansion later that uses GPU outputs)
            3. Reuses sampling tensors (since we run only decodes and they have
                a repeating sampling logic)
        """

        # When num_steps == 1, we execute the fallback here for the GPU
        # advance_step, which runs prepare_inputs on CPU and for each spec
        # iteration invokes this function only once
        # (Look at multi-step-worker code)
        is_fallback = num_steps == 1
        if not is_fallback:
            # Since we do not broadcast data inside execute_model anymore,
            # we need to figure out the best way to support TP > 1 in this
            # case, because we will at least need to broadcast the sampled
            # tokens to all workers.
            if not self.is_driver_worker:
                raise ValueError("TP1DraftModelRunner only supports TP=1.")

            # Sanity
            if self.lora_config is not None:
                raise ValueError("TP1DraftModelRunner has no support for LORA")
            if self.prompt_adapter_config is not None:
                raise ValueError("TP1DraftModelRunner has no support for "
                                 "prompt_adapter_config")
            if model_input.multi_modal_kwargs:
                raise ValueError(
                    "TP1DraftModelRunner has no support for multi_modal_kwargs"
                )
        else:
            if self.lora_config:
                assert model_input.lora_requests is not None
                assert model_input.lora_mapping is not None
                self.set_active_loras(model_input.lora_requests,
                                      model_input.lora_mapping)

            if self.prompt_adapter_config:
                assert model_input.prompt_adapter_requests is not None
                assert model_input.prompt_adapter_mapping is not None
                self.set_active_prompt_adapters(
                    model_input.prompt_adapter_requests,
                    model_input.prompt_adapter_mapping)

            self.attn_state.begin_forward(model_input)

        # Detect exec mode
        assert model_input.attn_metadata is not None
        use_cuda_graph = False
        if model_input.attn_metadata.num_prefills > 0:
            # In this case, execute_model(..) was called directly
            if num_steps > 1:
                raise ValueError(
                    "execute_model(..) of draft_model_runner can be called "
                    "directly only with a single-step prefill")
        else:
            # We can skip CPU samples for spec token generation.
            # (We do allow CPU samples for num_steps == 1 to support the
            # fallback case, where supports_gpu_multi_step(..) does not pass)
            model_input.sampling_metadata.skip_sampler_cpu_output = (
                not is_fallback)

            # Attn attr defines if we use cuda graphs
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph

        # Get model
        if use_cuda_graph:
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = (self.graph_runners[model_input.virtual_engine]
            [graph_batch_size])

            if previous_hidden_states is not None:
                hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                        dtype=previous_hidden_states.dtype,
                        device=previous_hidden_states.device)
                ])
            else:
                hidden_states = None
        else:
            model_executable = self.model
            hidden_states = previous_hidden_states

        self.draft_batch_size = len(model_input.query_lens)
        if self.speculative_config.ssd:
            self.indices_set = set(self.indices_of_seq_with_bonus_tokens)
        if self.speculative_config.ssd:
            start_ts = time.perf_counter()
            self.draft_input_len_list = model_input.seq_lens[:self.draft_batch_size]
            self.draft_seq_lens = 0
            for i in self.draft_input_len_list:
                self.draft_seq_lens += i

            end_ts = time.perf_counter()
            SpecProfiler.ssd_overhead[-1] += (end_ts - start_ts)

        outputs: List[SamplerOutput] = []
        flag = False
        for step in range(num_steps):
            # predict the output of the next step
            if self.speculative_config.ssd:
                start_ts = time.perf_counter()
                if step > 0:
                    for i in range(self.target_batch_size):
                        self.confidence_list[i].append(self.confidence_list[i][-1] * self.default_confidence)
                    cur_goodput = self._predict_goodput_with_constraint(step + 1, scaled_slo_tpot)

                    for i in range(self.target_batch_size):
                        self.confidence_list[i].pop()

                    # logger.info("predict_goodput with proposal len %d: %f", step + 1, cur_goodput)
                    if cur_goodput < self.goodput:
                        end_ts = time.perf_counter()
                        SpecProfiler.ssd_overhead[-1] += (end_ts - start_ts)
                        # logger.info("Early stopping at step %d, goodput: %f, batch size: %f", step, cur_goodput, self.target_batch_size)
                        break

                end_ts = time.perf_counter()
                SpecProfiler.ssd_overhead[-1] += (end_ts - start_ts)

            multi_modal_kwargs = model_input.multi_modal_kwargs or {}

            kwargs = {"previous_hidden_states": hidden_states} \
                if previous_hidden_states is not None else {}

            # Run model
            # with set_forward_context(model_input.attn_metadata):
            hidden_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **MultiModalInputs.as_kwargs(multi_modal_kwargs,
                                                device=self.device),
                **kwargs,
            )

            # Compute the logits.
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)

            # Sample the next token.
            output = self.model.sample(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )

            # SSD ：Step-level Speculative Decoding, with different Speculative lengths for each step
            if self.speculative_config.ssd:
                start_ts = time.perf_counter()
                probs = output.probs  # (batch_size, vocab_size)

                selected_probs = probs.gather(1, output.sampled_token_ids).squeeze(-1).tolist()  # (batch_size, 1)
                index = 0
                confidence = 0
                if step == 0:
                    for idx, prob in enumerate(selected_probs):
                        if idx in self.indices_set:
                            self.confidence_list[index].append(prob)
                            confidence += prob
                            index += 1
                else:
                    for idx, prob in enumerate(selected_probs):
                        if idx in self.indices_set:
                            confidence += prob
                            self.confidence_list[index].append(prob * self.confidence_list[index][-1])
                            index += 1

                cur_goodput = self._predict_goodput_with_constraint(step + 1, scaled_slo_tpot)
                if cur_goodput < self.goodput:
                    # logger.info("error predict, actual goodput is %f", cur_goodput)

                    flag = True

                    # eliminate the last step
                    # for i in range(self.target_batch_size):
                    #     self.confidence_list[i].pop()
                    #
                    # if outputs == []:
                    #     break

                    # retain the output of the last step
                    self.goodput = cur_goodput
                    outputs.append(output)

                else:
                    self.goodput = cur_goodput
                    outputs.append(output)
                    self.total_count += self.target_batch_size
                    self.total_confidence += confidence
                    self.default_confidence = self.total_confidence / self.total_count
                    SpecProfiler.avg_confidence = self.default_confidence
                    # logger.info("default confidence: %f", self.default_confidence)
                end_ts = time.perf_counter()
                SpecProfiler.ssd_overhead[-1] += (end_ts - start_ts)
            else:
                outputs.append(output)

            if model_input.attn_metadata.num_prefills == 0 \
                    and self.indices_of_seq_with_bonus_tokens is not None:
                assert output.sampled_token_ids is not None
                # output.sampled_token_ids should be of shape (num_seqs, 1)
                nums_seqs, num_tokens_per_seq = output.sampled_token_ids.shape
                # assert num_tokens_per_seq == 1
                count = 0
                for i in range(nums_seqs):
                    bonus_seq_idx = self.indices_of_seq_with_bonus_tokens[
                        count]
                    if i != bonus_seq_idx:
                        # The following might cause a cpu->gpu sync
                        # However, the performance impact is negligible as we
                        # benchmarked on H100.
                        output.sampled_token_ids[
                        i, :] = model_input.input_tokens[bonus_seq_idx]
                    else:
                        count += 1

            # SSD: break the loop if the goodput is lower than the previous step
            if flag:
                break

            # Prepare inputs for the next step
            if step != num_steps - 1:
                model_input = self._gpu_advance_step(model_input, outputs[-1])

        return outputs

    def _pre_predict_goodput(self, batch: ExecuteModelRequest):
        start_ts = time.perf_counter()
        self.draft_batch_size = len(batch.seq_group_metadata_list)
        self.target_batch_size = self.draft_batch_size
        self._init_ssd(self.target_batch_size)

        self.draft_input_len_list = [seq_metadata.seq_data[list(seq_metadata.seq_data.keys())[0]].get_len() for seq_metadata in batch.seq_group_metadata_list]
        self.target_input_len_list = self.draft_input_len_list

        self.draft_seq_lens = sum(self.draft_input_len_list)
        self.target_seq_lens = sum(self.target_input_len_list)

        self.goodput = self._predict_goodput_with_constraint(0, scaled_slo_tpot)
        for i in range(self.target_batch_size):
            self.confidence_list[i].append(self.default_confidence)
        cur_goodput = self._predict_goodput_with_constraint(1, scaled_slo_tpot)

        for i in range(self.target_batch_size):
            self.confidence_list[i].pop()

        end_ts = time.perf_counter()
        SpecProfiler.ssd_overhead.append(end_ts - start_ts)
        return cur_goodput > self.goodput


    def _eliminate_spec_lens_cuda_graph(self, spec_len_list):
        eliminated = False
        batch_size = int((np.mean(spec_len_list) + 1) * self.target_batch_size)
        if batch_size <= 32:
            delta = batch_size - (1 << ((batch_size - 1).bit_length() - 1))
        else:
            delta = batch_size - ((batch_size - 1) // 32 * 32)
        while(self.confidence_list and delta):
            min_value = float('inf')
            sub_list_index = -1

            # Find the minimum last element
            for i, sub_confidence_list in enumerate(self.confidence_list):
                if sub_confidence_list and spec_len_list[i]:  # Ensure sublist is not empty
                    current_value = sub_confidence_list[-1]  # Take the last element
                    if current_value < min_value:
                        min_value = current_value
                        sub_list_index = i

            # Remove the found minimum value
            if sub_list_index != -1 and spec_len_list[sub_list_index] > 0:
                spec_len_list[sub_list_index] -= 1
                delta -= 1

        if self._predict_goodput_rsd(spec_len_list) >= self.goodput:
            eliminated = True

        return spec_len_list, eliminated

    def _predict_goodput_rsd(self, spec_len_list):
        start_ts = time.perf_counter()
        accept_lens = self._solve_draft_list(spec_len_list, True, True) + len(self.confidence_list)
        batch_time = self._get_batch_proposal_verify_time_rsd(spec_len_list)
        end_ts = time.perf_counter()
        SpecProfiler.goodput_overhead.append(end_ts - start_ts)
        return accept_lens / batch_time

    def _predict_goodput_with_constraint(self, proposal_len, constraint):
        start_ts = time.perf_counter()
        batch_time = self._get_batch_proposal_verify_time(proposal_len)
        # SLO constraint
        # if proposal_len != 0 and batch_time > constraint:
        #     end_ts = time.perf_counter()
        #     SpecProfiler.goodput_overhead.append(end_ts - start_ts)
        #     SpecProfiler.slo += 1
        #     return -1
        accept_lens = sum([sum(sub_list) for sub_list in self.confidence_list]) + self.target_batch_size
        end_ts = time.perf_counter()
        SpecProfiler.goodput_overhead.append(end_ts - start_ts)
        return accept_lens / batch_time

    def _init_ssd(self, batch_size: int):
        if self.draft_models is None:
            self.draft_models = self.speculative_config.rsd_draft_model
            self.target_models = self.speculative_config.rsd_target_model

            self.draft_overhead = self.speculative_config.rsd_map['draft_times_map']['overhead']
            self.speculative_config.rsd_map['draft_times_map'].pop('overhead')

            self.target_overhead = self.speculative_config.rsd_map['target_times_map']['overhead']
            self.speculative_config.rsd_map['target_times_map'].pop('overhead')

        self.goodput = -1
        self.confidence_list = [[] for _ in range(batch_size)]

    def _get_bucket_seq_len(self, times_map: Dict[int, Dict[int, float]],
                            seq_len: int) -> int:
        all_seq_lens = list(times_map.keys())
        for i in range(len(all_seq_lens) - 1):
            if all_seq_lens[i] <= seq_len and seq_len < all_seq_lens[i + 1]:
                return all_seq_lens[i]
        return all_seq_lens[-1]

    def _get_batch_latency(self, times_map: Dict[int, Dict[int, float]],
                           seq_len: int, batch_size: int, models) -> float:
        batch_latencies = times_map[seq_len]
        if batch_size <= max(batch_latencies.keys()):
            return batch_latencies[batch_size]

        model = models[seq_len]
        return model.predict(np.array(batch_size).reshape(-1, 1))[0]

    def _get_batch_proposal_verify_time_rsd(self, spec_len_list: List[int]) -> float:
        draft_graph_batch_size = _get_graph_batch_size(self.draft_batch_size)

        draft_time = self.draft_time

        num_batched_token = int((np.mean(spec_len_list) + 1) * self.target_batch_size)
        target_graph_batch_size = _get_graph_batch_size(num_batched_token)
        sum = 0
        for i in range(len(spec_len_list)):
            spec_len = spec_len_list[i]
            sum += (spec_len ** 2 + spec_len) / 2 + (spec_len + 1) * self.target_input_len_list[i]
        avg_target_seq_len = sum // target_graph_batch_size
        target_seq_len = self._get_bucket_seq_len(self.speculative_config.rsd_map['target_times_map'],
                                                  avg_target_seq_len)

        target_time = self._get_batch_latency(self.speculative_config.rsd_map['target_times_map'], target_seq_len,
                                              target_graph_batch_size,
                                              self.target_models)
        if max(spec_len_list) > 0:
            draft_time += self.draft_overhead[draft_graph_batch_size]

        target_time += self.target_overhead[target_graph_batch_size]
        return draft_time + target_time

    def _get_batch_proposal_verify_time(self, proposal_len: int) -> float:
        draft_graph_batch_size = _get_graph_batch_size(self.draft_batch_size)

        # avg_draft_seq_len = self.draft_seq_lens // self.draft_batch_size + proposal_len
        avg_draft_seq_len = (self.draft_seq_lens + proposal_len * self.draft_batch_size) // draft_graph_batch_size

        self.draft_seq_len = self._get_bucket_seq_len(self.speculative_config.rsd_map['draft_times_map'], avg_draft_seq_len)

        self.single_draft_time = self._get_batch_latency(self.speculative_config.rsd_map['draft_times_map'],
                                                    self.draft_seq_len,
                                                    draft_graph_batch_size,
                                                    self.draft_models)
        draft_time = self.single_draft_time * proposal_len
        self.draft_time = draft_time

        num_batched_token = (proposal_len + 1) * self.target_batch_size
        target_graph_batch_size = _get_graph_batch_size(num_batched_token)
        avg_target_seq_len = ((proposal_len + 1) * self.target_seq_lens + self.target_batch_size * (
                    proposal_len ** 2 + proposal_len) / 2) // target_graph_batch_size
        target_seq_len = self._get_bucket_seq_len(self.speculative_config.rsd_map['target_times_map'],
                                                  avg_target_seq_len)

        target_time = self._get_batch_latency(self.speculative_config.rsd_map['target_times_map'], target_seq_len,
                                              target_graph_batch_size,
                                              self.target_models)
        if proposal_len > 0:
            draft_time += self.draft_overhead[draft_graph_batch_size]
        target_time += self.target_overhead[target_graph_batch_size]
        return draft_time + target_time

    def _solve_draft_list(self, spec_len_list, sum_flag, eliminate_flag=False):
        # 1. Truncate each sub_list of confidence_or_accepted-list according to spec_len_st, so that the length of each sub_list is equal to the corresponding spec_len
        extracted_elements = []
        for sub_list, length in zip(self.confidence_list, spec_len_list):
            extracted = sub_list[:length]
            extracted_elements.append(extracted)
        # print(extracted_elements)

        if eliminate_flag:
            return sum([sum(sub_list) for sub_list in extracted_elements])

        # 2. When entering confidence_list, the cumulative result is the contribution of each draft token to the expected number of accepted tokens;
        cumulative_product_list = []
        for sub_list in extracted_elements:
            cumulative_product = []
            current_product = 1
            for element in sub_list:
                current_product *= element
                cumulative_product.append(current_product)
            cumulative_product_list.append(cumulative_product)

        # 3. According to the summary flag, whether to execute and return the sum of cumulive-products_list or the cumulive-products_list itself
        if sum_flag:
            total_sum = sum([sum(sub_list) for sub_list in cumulative_product_list])
            return total_sum  # return accepted in each request
        else:
            return cumulative_product_list

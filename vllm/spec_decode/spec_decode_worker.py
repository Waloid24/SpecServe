import copy
import time
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
from sklearn.linear_model import LinearRegression
import numpy as np
from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.sequence import SamplerOutput
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler, SpecDecodeStochasticBaseSampler)
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler)
from vllm.sequence import (VLLM_INVALID_TOKEN_ID,
                           CompletionSequenceGroupOutput, ExecuteModelRequest,
                           HiddenStates, SequenceGroupMetadata,
                           get_all_seq_ids_and_request_ids)
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.dsd import DSD
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.medusa_worker import MedusaWorker
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.mlp_speculator_worker import MLPSpeculatorWorker
from vllm.spec_decode.mqa_scorer import MQAScorer
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.spec_decode.util import (Timer, create_logprobs_output,
                                   create_sequence_group_output,
                                   get_all_num_logprobs,
                                   get_sampled_token_logprobs, nvtx_range,
                                   split_batch_by_proposal_len)
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase
from vllm.spec_decode.profile import SpecProfiler

logger = init_logger(__name__)

def create_spec_worker(*args, **kwargs) -> "SpecDecodeWorker":
    """Helper method that is the entrypoint for Executors which use
    WorkerWrapper. It constructs a SpecDecodeWorker from the speculative config.
    """
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    speculative_config: SpeculativeConfig = vllm_config.speculative_config
    assert speculative_config is not None

    draft_worker_kwargs = kwargs.copy()

    kwargs["model_runner_cls"] = TargetModelRunner
    target_worker = Worker(*args, **kwargs)

    # Set the disable_logprobs variable in the TargetModelRunner instance
    # as per its value specified in the SpeculativeConfig.
    target_worker.model_runner.disable_logprobs = \
        speculative_config.disable_logprobs

    draft_worker_config = copy.deepcopy(vllm_config)
    draft_worker_config.model_config = speculative_config.draft_model_config
    draft_worker_config.quant_config = VllmConfig._get_quantization_config(
        draft_worker_config.model_config,
        vllm_config.load_config,
    )
    draft_worker_config.parallel_config = speculative_config.draft_parallel_config  # noqa
    # TODO allow draft-model specific load config.

    # Override draft-model specific worker args.
    draft_worker_kwargs.update(
        vllm_config=draft_worker_config,
        ngram_prompt_lookup_max=speculative_config.ngram_prompt_lookup_max,
        ngram_prompt_lookup_min=speculative_config.ngram_prompt_lookup_min,
    )

    spec_decode_worker = SpecDecodeWorker.create_worker(
        scorer_worker=target_worker,
        draft_worker_kwargs=draft_worker_kwargs,
        disable_mqa_scorer=speculative_config.speculative_disable_mqa_scorer,
        disable_by_batch_size=speculative_config.
        speculative_disable_by_batch_size,
        draft_token_acceptance_method=speculative_config.
        draft_token_acceptance_method,
        typical_acceptance_sampler_posterior_threshold=speculative_config.
        typical_acceptance_sampler_posterior_threshold,
        typical_acceptance_sampler_posterior_alpha=speculative_config.
        typical_acceptance_sampler_posterior_alpha,
        disable_logprobs=speculative_config.disable_logprobs,
        disable_log_stats=speculative_config.disable_log_stats,
        dsd=speculative_config.dsd,
        ssd=speculative_config.ssd,
        rsd=speculative_config.rsd,
        specinfer=speculative_config.specinfer,
        dynamic_spec=speculative_config.dynamic_spec,
    )

    return spec_decode_worker


# Reminder: Please update docs/source/serving/compatibility_matrix.rst
# If the feature combo become valid
class SpecDecodeWorker(LoraNotSupportedWorkerBase):
    """Worker which implements speculative decoding.

    Speculative decoding reduces decoding per-token latency by using a proposal
    method, such as a small draft model, to speculate ahead of a larger LLM. The
    probabilities of the speculative tokens are then determined by the larger
    LLM, after which some verification routine determines which (if any) of the
    speculative tokens are accepted by the larger LLM.

    See https://github.com/vllm-project/vllm/pull/2188 and
    https://github.com/vllm-project/vllm/pull/3103 for more info.

    The current implementation has the following limitations:
    * Only draft-model proposal is implemented (contributions for more forms are
        welcome!).
    * Only top-1 proposal and scoring are implemented. Tree-attention is left as
        future work.
    * All sequences in a batch must have the same proposal length, or zero. This
        can be improved by having per-sequence speculation in the future.
    * The scoring forward pass is done without an MQA kernel, which is
        suboptimal especially as the batch size, proposal length, and sequence
        lengths grow. Contributions to add a MQA scoring are welcome once
        correctness tests pass.
        More info here https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit.
    """

    @classmethod
    def create_worker(
            cls,
            scorer_worker: Worker,
            draft_worker_kwargs: Dict[str, Any],
            disable_mqa_scorer: bool,
            disable_by_batch_size: Optional[int],
            draft_token_acceptance_method: str,
            typical_acceptance_sampler_posterior_threshold: float,
            typical_acceptance_sampler_posterior_alpha: float,
            disable_logprobs: bool,
            disable_log_stats: bool,
            dsd: bool,
            ssd: bool,
            rsd: bool,
            specinfer: bool,
            dynamic_spec: bool,
    ) -> "SpecDecodeWorker":

        allow_zero_draft_token_step = True
        ngram_prompt_lookup_max = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_max"))
        ngram_prompt_lookup_min = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_min"))
        draft_model_config = draft_worker_kwargs["vllm_config"].model_config
        draft_parallel_config: ParallelConfig = draft_worker_kwargs[
            'vllm_config'].parallel_config
        if ngram_prompt_lookup_max > 0:
            proposer_worker = NGramWorker(**draft_worker_kwargs)
            proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                                  ngram_prompt_lookup_max)
        else:
            draft_tp = draft_parallel_config.tensor_parallel_size
            target_tp = scorer_worker.parallel_config.tensor_parallel_size

            if draft_model_config.hf_config.model_type == "mlp_speculator":
                proposer_worker = MLPSpeculatorWorker(**draft_worker_kwargs)
            elif draft_model_config.hf_config.model_type == "medusa":
                proposer_worker = MedusaWorker(**draft_worker_kwargs)
            else:
                if draft_tp == 1:
                    draft_worker_kwargs[
                        "model_runner_cls"] = TP1DraftModelRunner
                else:
                    if draft_model_config.hf_config.model_type == "eagle":
                        raise NotImplementedError(
                            "EAGLE does not support TP > 1 yet")

                    allow_zero_draft_token_step = False
                proposer_worker = MultiStepWorker(**draft_worker_kwargs)

            proposer_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                proposer_worker, draft_tp, target_tp)

        logger.info("Configuring SpecDecodeWorker with proposer=%s",
                    type(proposer_worker))

        spec_decode_sampler: SpecDecodeBaseSampler = None
        if draft_token_acceptance_method == "rejection_sampler":
            spec_decode_sampler = RejectionSampler()
        elif draft_token_acceptance_method == "typical_acceptance_sampler":
            spec_decode_sampler = TypicalAcceptanceSampler(
                posterior_threshold= \
                    typical_acceptance_sampler_posterior_threshold,
                posterior_alpha=typical_acceptance_sampler_posterior_alpha,
            )
        logger.info(
            "[Speculative Decoding] Configuring"
            " SpecDecodeWorker with sampler=%s", type(spec_decode_sampler))

        if (not disable_mqa_scorer):
            if scorer_worker.model_runner.attn_backend.get_name(
            ) != "FLASH_ATTN":
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "MQA is only available with flash attn backend.")

            if draft_model_config and \
                    draft_model_config.max_model_len < \
                    scorer_worker.model_config.max_model_len:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "draft model max_model_len is smaller than the target "
                    "model max_model_len.")

            if not scorer_worker.model_runner.model_config.enforce_eager:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "target model is not running in eager mode.")

        return SpecDecodeWorker(
            proposer_worker,
            scorer_worker,
            disable_mqa_scorer=disable_mqa_scorer,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            disable_by_batch_size=disable_by_batch_size,
            spec_decode_sampler=spec_decode_sampler,
            allow_zero_draft_token_step=allow_zero_draft_token_step,
            use_dsd=dsd,
            use_ssd=ssd,
            use_rsd=rsd,
            use_specinfer=specinfer,
            use_dynamic_spec=dynamic_spec,
        )

    def __init__(
            self,
            proposer_worker: ProposerWorkerBase,
            scorer_worker: WorkerBase,
            spec_decode_sampler: SpecDecodeBaseSampler,
            disable_mqa_scorer: bool = False,
            disable_logprobs: bool = False,
            disable_log_stats: bool = False,
            metrics_collector: Optional[AsyncMetricsCollector] = None,
            disable_by_batch_size: Optional[int] = None,
            allow_zero_draft_token_step: Optional[bool] = True,
            use_dsd: Optional[bool] = None,
            use_ssd: Optional[bool] = None,
            use_rsd: Optional[bool] = None,
            use_specinfer: Optional[bool] = None,
            use_dynamic_spec: Optional[bool] = None,
    ):
        """
        Create a SpecDecodeWorker.

        Args:
            proposer_worker: A worker that can produce speculative tokens for
                sequences.
            scorer_worker: A worker that produces probabilities of speculative
                tokens according to some base model. Typically a vanilla vLLM
                Worker.
            spec_decode_sampler: A Torch module used to perform acceptance
                sampling of the draft tokens in the verification step of
                speculative decoding. Currently we support two different 
                types of sampler namely RejectionSampler and
                TypicalAcceptanceSampler. 'spec_decode_sampler' is either an
                instance of RejectionSampler or TypicalAcceptanceSampler.
            disable_mqa_scorer: If set to True, disable the MQA scorer and use
                the BatchExpansionTop1Scorer instead.
            disable_logprobs: If set to True, token log probabilities will
                not be output in both the draft worker and the target worker.
                If set to False, log probabilities will be output by both.
            disable_log_stats: If set to True, disable periodic printing of
                speculative stage times.
            disable_by_batch_size: If the batch size is larger than this,
                disable speculative decoding for new incoming requests.
            metrics_collector: Helper class for collecting metrics; can be set
                for testing purposes.
            allow_zero_draft_token_step: whether to allow a step where the draft
                model generates no draft token; should disallow when the tp of
                draft model is larger than 1 (TODO: #5814)
        """
        self.proposer_worker = proposer_worker
        self.scorer_worker = scorer_worker
        scorer_runner = getattr(self.scorer_worker, "model_runner", None)
        self.generators = scorer_runner.get_generators(
        ) if scorer_runner else None
        self.disable_by_batch_size = disable_by_batch_size or float("inf")
        self.spec_decode_sampler = spec_decode_sampler
        self._allow_zero_draft_token_step = allow_zero_draft_token_step
        self._metrics = AsyncMetricsCollector(
            self.spec_decode_sampler
        ) if metrics_collector is None else metrics_collector
        # Tracks the sequence IDs that received a bonus token ID in
        # their last forward pass. Needed only if KV cache is being
        # used for token generation such as in the case of MultiStepWorker.
        self._seq_with_bonus_token_in_last_step: Set[int] = set()
        # Tracks the currently active request ids and the sequence IDs
        # corresponding to them
        self._request_id_seq_id_mapping: Dict[str, Set[int]] = defaultdict(set)
        # Tracks if the proposer worker uses the KV cache or not.

        self.probs_dtype = self.spec_decode_sampler.probs_dtype
        self.token_id_dtype = self.spec_decode_sampler.token_id_dtype
        # Lazy initialization.
        self.scorer: SpeculativeScorer
        self.disable_mqa_scorer = disable_mqa_scorer

        # Hidden states from target model to pass to proposer
        # in the subsequent step.
        self.previous_hidden_states: Optional[HiddenStates] = None
        self._disable_logprobs = disable_logprobs
        self._disable_log_stats = disable_log_stats
        self.use_dsd = use_dsd
        if self.use_dsd:
            logger.info("[Speculative Decoding] DSD is enabled.")
        self.use_ssd = use_ssd
        self.use_rsd = use_rsd
        if self.use_rsd:
            logger.info("[Speculative Decoding] RSD is enabled.")
        elif self.use_ssd:
            logger.info("[Speculative Decoding] SSD is enabled.")
        self.use_specinfer = use_specinfer
        if self.use_specinfer:
            logger.info("[Speculative Decoding] SpecInfer is enabled.")
        self.use_dynamic_spec = use_dynamic_spec
        if self.use_dynamic_spec:
            logger.info("[Speculative Decoding] Dynamic Spec is enabled.")

    def init_device(self) -> None:
        """Initialize both scorer and proposer models.
        """
        # The scorer worker model is initialized first in case the proposer
        # model has a smaller TP degree than the target worker.
        self.scorer_worker.init_device()
        self.proposer_worker.init_device()

        # NOTE(cade): load_model is not part of the WorkerBase interface.
        self.scorer_worker.load_model()
        self.proposer_worker.load_model()

        self._metrics.init_gpu_tensors(self.rank)
        self.spec_decode_sampler.init_gpu_tensors(self.rank)

        scorer_cls: Type[SpeculativeScorer]
        if self.disable_mqa_scorer:
            scorer_cls = BatchExpansionTop1Scorer
            logger.info("[Speculative Decoding] Use batch "
                        "expansion for scoring proposals.")
        else:
            scorer_cls = MQAScorer
            logger.info(
                "[Speculative Decoding] Use MQA scorer for scoring proposals.")

        self.scorer = scorer_cls(scorer_worker=self.scorer_worker,
                                 device=self.device,
                                 vocab_size=self._vocab_size)

        self._configure_model_sampler_for_spec_decode()

    def load_model(self, *args, **kwargs):
        pass

    def _configure_model_sampler_for_spec_decode(self):
        """Configure model sampler to emit GPU tensors. This allows spec decode
        to keep data on device without transferring to CPU and serializing,
        which significantly reduces overhead of sampling during verification.

        NOTE(cade): This breaks abstraction boundaries pretty badly. The better
        design is to have the "move to CPU and serialize" sampling decision be
        done outside of the model/sampler; this way the "last-mile" worker
        object which interfaces with the scheduler can serialize and incur the
        performance hit as necessary. This allows us to run the worker several
        iterations in a row without incurring the "move to CPU and serialize"
        performance penalty.

        Since this requires a large change to vLLM, we defer it to later and
        temporarily accept this broken abstraction boundary.

        NOTE(cade): This will require a special check if the proposer worker
        does not have a sampler (e.g. ngram speculation).
        """
        (self.scorer_worker.model_runner.model.sampler.include_gpu_probs_tensor
         ) = True
        (self.scorer_worker.model_runner.model.sampler.
         should_modify_greedy_probs_inplace) = True
        self.proposer_worker.set_should_modify_greedy_probs_inplace()
        self.proposer_worker.set_include_gpu_probs_tensor()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of cache blocks to use.

        This is done by profiling the scorer model (which is typically the
        larger of the two). Then the total memory which would be used by the
        scorer cache is divided evenly between the proposer and scorer model KV,
        such that the number of blocks is equal in both KV caches.
        """
        num_gpu_blocks, num_cpu_blocks = (
            self.scorer_worker.determine_num_available_blocks())

        scorer_cache_block_size_bytes = (
            self.scorer_worker.get_cache_block_size_bytes())
        proposer_cache_block_size_bytes = (
            self.proposer_worker.get_cache_block_size_bytes())

        new_num_gpu_blocks = split_num_cache_blocks_evenly(
            scorer_cache_block_size_bytes, proposer_cache_block_size_bytes,
            num_gpu_blocks)
        return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the cache engine of the scorer and proposer workers.
        """
        self.scorer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                            num_cpu_blocks=num_cpu_blocks)
        self.proposer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)

        # Special handling for SSD/RSD
        if self.use_ssd:
            target_times_map = self.scorer_worker.profile_exec_time()

            if type(self.proposer_worker) == MultiStepWorker:
                self.proposer_worker.speculative_config.ssd = False
                draft_times_map = self.proposer_worker.profile_exec_time()
                self.proposer_worker.speculative_config.ssd = True
                rsd_map = {"draft_times_map": draft_times_map, "target_times_map": target_times_map}
                self.proposer_worker.speculative_config.rsd_draft_model = self._fit_latency_models(draft_times_map)
                self.proposer_worker.speculative_config.rsd_target_model = self._fit_latency_models(target_times_map)
                self.proposer_worker.speculative_config.rsd_map = rsd_map
            else:
                self.proposer_worker._worker.speculative_config.ssd = False
                draft_times_map = self.proposer_worker.profile_exec_time()
                self.proposer_worker._worker.speculative_config.ssd = True
                rsd_map = {"draft_times_map": draft_times_map, "target_times_map": target_times_map}
                if draft_times_map is not None:
                    self.proposer_worker._worker.speculative_config.rsd_draft_model = self._fit_latency_models(
                        self.proposer_worker.profile_exec_time())
                if target_times_map is not None:
                    self.proposer_worker._worker.speculative_config.rsd_target_model = self._fit_latency_models(
                        self.scorer_worker.profile_exec_time())
                self.proposer_worker._worker.speculative_config.rsd_map = rsd_map

            self.scorer_worker.speculative_config.ssd = True

    # Adapted from SmartSpec: https://github.com/LiuXiaoxuanPKU/vllm/blob/dsd
    def _fit_latency_models(
            self, seq_data_dict: Dict[int,
            Dict[int,
            float]]) -> Dict[int, LinearRegression]:
        models = {}
        for seq_len in seq_data_dict:
            data_dict = seq_data_dict[seq_len]
            model, r2 = self._fit_predict_latency(data_dict)
            print(f"Seq len: {seq_len}, R2 score: {r2}")
            models[seq_len] = model
        return models

    def _fit_predict_latency(
            self, data_dict: Dict[int,
            float]) -> Tuple[LinearRegression, float]:
        """
        Fit a linear regression model to predict batch latency from batch size.

        Parameters:
        data_dict (dict): Dictionary with batch_size and batch_latency pairs

        Returns:
        tuple: (model, r2_score)
        """
        # Convert dictionary to arrays
        X = np.array(list(data_dict.keys())).reshape(-1, 1)  # batch sizes
        y = np.array(list(data_dict.values()))  # latencies

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate R-squared score
        r2_score = model.score(X, y)
        return model, r2_score

    @torch.inference_mode()
    def execute_model(
            self,
            execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform speculative decoding on the input batch.
        """
        if self.rank != self._driver_rank:
            self._run_non_driver_rank()
            return []

        if execute_model_req is None:
            # This signals that there's no more requests to process for now.
            # All workers are running infinite loop with broadcast_tensor_dict,
            # and it stops the loop when the driver broadcasts an empty input.
            # Send an empty input to notify all other workers to stop their
            # execution loop.
            broadcast_tensor_dict({}, src=0)
            return []

        self._track_finished_requests(execute_model_req)
        disable_all_speculation = self._should_disable_all_speculation(
            execute_model_req)
        num_lookahead_slots = execute_model_req.num_lookahead_slots

        # Speculative decoding is disabled in the following cases:
        # 1. Prefill phase: Speculative decoding is not
        #    used during the prefill phase.
        # 2. Auto-disable enabled: The running queue size exceeds
        #    the specified threshold.
        # 3. No request: There are no requests in the batch, or
        #    none of the requests in the batch have spec decoding enabled.
        # In any of these cases, the proposer and scorer workers
        # are called normally.
        no_spec = num_lookahead_slots == 0 or disable_all_speculation or all(
            sgm.num_speculative_tokens == 0
            for sgm in execute_model_req.seq_group_metadata_list)

        # Broadcast how many lookahead slots are scheduled for this step, and
        # whether all speculation is disabled, to all non-driver workers.

        # This is required as if the number of draft model runs changes
        # dynamically, the non-driver workers won't know unless we perform a
        # communication to inform them.

        # no_spec is used to signal non-driver worker about prefill vs decode
        # stage. This is needed to ensure that order of execution of proposer
        # and scorer is same in both driver and non-driver workers (i.e.,
        # scorer -> proposer for prefill and proposer -> scorer in decode). This
        # order is needed to support models like EAGLE that take scorer states
        # as inputs.
        broadcast_dict = dict(
            num_lookahead_slots=num_lookahead_slots,
            no_spec=no_spec,
            disable_all_speculation=disable_all_speculation,
        )
        broadcast_tensor_dict(broadcast_dict, src=self._driver_rank)

        assert execute_model_req.seq_group_metadata_list is not None, (
            "speculative decoding requires non-None seq_group_metadata_list")

        self._maybe_disable_speculative_tokens(
            disable_all_speculation, execute_model_req.seq_group_metadata_list)

        if no_spec:
            return self._run_no_spec(execute_model_req,
                                     skip_proposer=disable_all_speculation)
        return self._run_speculative_decoding_step(execute_model_req,
                                                   num_lookahead_slots)

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop to perform speculative decoding
        in parallel worker."""
        while self._run_non_driver_rank():
            pass

    def _should_disable_all_speculation(
            self, execute_model_req: ExecuteModelRequest) -> bool:
        # When the batch size is too large, disable speculative decoding
        # to stop trading off throughput for latency.
        return (execute_model_req.running_queue_size >=
                self.disable_by_batch_size)

    def _maybe_disable_speculative_tokens(
            self, disable_all_speculation: bool,
            seq_group_metadata_list: List[SequenceGroupMetadata]) -> None:
        if not disable_all_speculation:
            return

        for seq_group_metadata in seq_group_metadata_list:
            # Once num_speculative_tokens is set to 0, the spec decode
            # of this request will be disabled forever.
            # TODO(comaniac): We currently store spec decoding specific
            # state in the global data structure, but we should maintain
            # this state within spec decode worker.
            seq_group_metadata.num_speculative_tokens = 0

    def _serialize_sampler_output_no_logprobs(
            self, execute_model_req: ExecuteModelRequest,
            sampler_output: SamplerOutput) -> SamplerOutput:
        """
        Creates and returns a `SamplerOutput` with only the token IDs being
        serialized to CPU and populated in `CompletionSequenceGroupOutput`.
        All other parameters in `CompletionSequenceGroupOutput` related to log
        probabilities are skipped.

        Args:
            execute_model_req (ExecuteModelRequest): The model request that
            was executed.
            sampler_output (SamplerOutput): The output from the sampler with
            only GPU tensors populated.

        Returns:
            SamplerOutput: A new `SamplerOutput` instance containing a list of
            `CompletionSequenceGroupOutput` objects with only token IDs
            populated.
        """
        seq_output_prompt_logprobs = [
            seq.is_prompt and seq.sampling_params.prompt_logprobs is not None
            and seq.sampling_params.prompt_logprobs > 0
            for seq in execute_model_req.seq_group_metadata_list
        ]
        # ignore slots for prompt tokens that are filled with INVALID_TOKEN_ID
        sampled_token_ids_list = (sampler_output.sampled_token_ids[torch.where(
            # subtracting is faster than testing for equality
            sampler_output.sampled_token_ids - VLLM_INVALID_TOKEN_ID)[0]] \
                                      if any(seq_output_prompt_logprobs) else \
                                      sampler_output.sampled_token_ids).tolist()

        seq_data_entries = (
            (seq_id, seq_data) for sg in \
            execute_model_req.seq_group_metadata_list \
            for seq_id, seq_data in sg.seq_data.items()
        )
        completion_seq_group_output_list: List[
            CompletionSequenceGroupOutput] = []
        for index, ((seq_id, seq_data), needs_prompt_logprobs) in \
                enumerate(zip(seq_data_entries, seq_output_prompt_logprobs)):
            if needs_prompt_logprobs:
                prompt_token_ids = seq_data.get_prompt_token_ids()
                prompt_logprobs = [
                    create_logprobs_output(
                        token_id=p_token_id,
                        token_id_logprob_rank=-1,
                        token_id_logprob=0.0,
                        topk_token_ids=[],
                        topk_logprobs=[],
                    )
                    # no prompt logprobs for the first token
                    for p_token_id in prompt_token_ids[1:]
                ]
            else:
                prompt_logprobs = None

            completion_seq_group_output_list.append(
                create_sequence_group_output(
                    token_id=sampled_token_ids_list[index][0],
                    token_id_logprob_rank=-1,
                    token_id_logprob=0.0,
                    seq_id=seq_id,
                    topk_token_ids=[],
                    topk_logprobs=[],
                    prompt_logprobs=prompt_logprobs))
        return SamplerOutput(outputs=completion_seq_group_output_list)

    @nvtx_range("spec_decode_worker._run_no_spec")
    def _run_no_spec(self, execute_model_req: ExecuteModelRequest,
                     skip_proposer: bool) -> List[SamplerOutput]:
        """Run a single generation step without any speculation. The input is
        sent to the proposer and scorer model so that the KV cache is consistent
        between the two. When skip_proposer is True, the proposer model is
        not called, meaning that the kv-cache in proposer for requests is not
        updated, so they cannot enable spec decode in the rest decoding.
        """

        sampler_output = self.scorer_worker.execute_model(execute_model_req)
        assert len(sampler_output) == 1
        sampler_output = sampler_output[0]

        # Store hidden states from target model execution.
        hidden_states = sampler_output.hidden_states
        if hidden_states is not None:
            # remove hidden_states for prompt tokens
            if any(seq.is_prompt
                   for seq in execute_model_req.seq_group_metadata_list):
                hidden_states = hidden_states[
                    torch.where(sampler_output.sampled_token_ids -
                                VLLM_INVALID_TOKEN_ID)[0]]
            if self.previous_hidden_states is None:
                self.previous_hidden_states = HiddenStates(
                    hidden_states, execute_model_req.seq_group_metadata_list)
            else:
                self.previous_hidden_states.update(
                    hidden_states, execute_model_req.seq_group_metadata_list)

        if not skip_proposer:
            # We prepare the prefill hidden states here so that there no
            # additional complexity in worker for spec_decode vs non_spec_decode
            # flow and execute_model doesn't need additional modifications.
            execute_model_req.previous_hidden_states = \
                prepare_prefill_hidden_states(
                    sampler_output.prefill_hidden_states)

            # Special handling for SSD/RSD
            if type(self.proposer_worker) == MultiStepWorker:
                self.proposer_worker.speculative_config.dynamic_spec = False
                self.proposer_worker.speculative_config.ssd = False
                self.proposer_worker.execute_model(execute_model_req)
                self.proposer_worker.speculative_config.ssd = self.use_ssd
                self.proposer_worker.speculative_config.dynamic_spec = self.use_dynamic_spec
            else:
                self.proposer_worker._worker.speculative_config.dynamic_spec = False
                self.proposer_worker._worker.speculative_config.ssd = False
                self.proposer_worker.execute_model(execute_model_req)
                self.proposer_worker._worker.speculative_config.ssd = self.use_ssd
                self.proposer_worker._worker.speculative_config.dynamic_spec = self.use_dynamic_spec

        sampler_output_to_return = (self._serialize_sampler_output_no_logprobs(
            execute_model_req=execute_model_req, sampler_output=sampler_output)
                                    if self._disable_logprobs else
                                    sampler_output)

        # Clear device tensors from sampler output. This reduces communication
        # overhead when the engine runs in a different process than the workers.
        sampler_output.sampled_token_probs = None
        sampler_output.sampled_token_ids = None
        sampler_output.logprobs = None
        return [sampler_output_to_return]

    def _run_non_driver_rank(self) -> bool:
        """Run proposer and verifier model in non-driver workers. This is used
        for both speculation cases (num_lookahead_slots>0) and non-speculation
        cases (e.g. prefill).

        Returns True if there are remaining sequences to process.
        """
        assert self.rank != self._driver_rank

        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:
            return False
        num_lookahead_slots = data["num_lookahead_slots"]

        # In case of prefill, scorer_worker has to be run before proposer so
        # that the hidden states can be propagated to proposer when needed.
        if data["no_spec"]:
            self.scorer_worker.execute_model()

        if not data["disable_all_speculation"]:
            # Even if num_lookahead_slots is zero, we want to run the
            # proposer model as it may have KV.
            #
            # We run the proposer once per lookahead slot. In the future we
            # should delegate how many times it runs to the proposer.
            for _ in range(max(num_lookahead_slots, 1)):
                self.proposer_worker.execute_model()

        if not data["no_spec"]:
            self.scorer_worker.execute_model()

        return True

    @nvtx_range("spec_decode_worker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
            self, execute_model_req: ExecuteModelRequest,
            num_lookahead_slots: int) -> List[SamplerOutput]:
        """Execute a single step of speculative decoding.

        This invokes the proposer worker to get k speculative tokens for each
        sequence, then scores each speculative token using the scoring worker.

        Returns a list of SamplerOutput, each containing a single token per
        sequence.
        """
        SpecProfiler.steps += 1
        assert num_lookahead_slots == execute_model_req.num_lookahead_slots

        # Pass last hidden states from target model to proposer
        execute_model_req.previous_hidden_states = self.previous_hidden_states
        self.previous_hidden_states = None

        if self.use_dsd:
            start_ts = time.perf_counter()
            proposal_len = self.dsd.get_propose_len(execute_model_req)
            end_ts = time.perf_counter()
            SpecProfiler.dsd_overhead.append(end_ts - start_ts)
            if proposal_len == 0:
                SpecProfiler.batch.append(len(execute_model_req.seq_group_metadata_list))
                SpecProfiler.draft.append(0)
                for i in range(len(execute_model_req.seq_group_metadata_list)):
                    SpecProfiler.speculative_len.append(0)
                    SpecProfiler.acc.append(0)

                for seq_group_metadata \
                        in execute_model_req.seq_group_metadata_list:
                    seq_group_metadata.num_speculative_tokens = 0
                return self._run_no_spec(execute_model_req, skip_proposer=True)
        elif self.use_ssd:
            if type(self.proposer_worker) == MultiStepWorker:
                runner = self.proposer_worker.model_runner
            else:
                runner = self.proposer_worker._worker.model_runner
            if not runner._pre_predict_goodput(execute_model_req):
                SpecProfiler.batch.append(len(execute_model_req.seq_group_metadata_list))
                SpecProfiler.draft.append(0)
                
                for i in range(len(execute_model_req.seq_group_metadata_list)):
                    SpecProfiler.speculative_len.append(0)
                    SpecProfiler.acc.append(0)

                for seq_group_metadata \
                        in execute_model_req.seq_group_metadata_list:
                    seq_group_metadata.num_speculative_tokens = 0
                return self._run_no_spec(execute_model_req, skip_proposer=True)
            else:
                proposal_len = num_lookahead_slots
        else:
            proposal_len = num_lookahead_slots

        specinfer_k = 3
        if self.use_specinfer:
            execute_model_req = MultiStepWorker._specinfer_expand_execute_model_request(execute_model_req, specinfer_k)

        with Timer() as proposal_timer:
            # Generate proposals using draft worker.
            proposals = self.proposer_worker.get_spec_proposals(
                execute_model_req,
                self._seq_with_bonus_token_in_last_step,
                proposal_len,
            )
        SpecProfiler.batch.append(len(execute_model_req.seq_group_metadata_list))
        SpecProfiler.draft.append(int(np.sum(proposals.proposal_lens.tolist())))
        
        for i in range(len(execute_model_req.seq_group_metadata_list)):
            SpecProfiler.speculative_len.append(proposals.proposal_lens[i].item())

        if not self._allow_zero_draft_token_step and proposals.no_proposals:
            # TODO: Fix it #5814
            raise RuntimeError("Cannot handle cases where distributed draft "
                               "workers generate no tokens")

        execute_model_req.previous_hidden_states = None

        if self.use_dsd:
            verify_len = self.dsd.get_verify_len(execute_model_req, proposals)
        elif self.use_ssd:
            verify_len = proposals.proposal_lens.max()
        elif self.use_dynamic_spec:
            verify_len = proposals.proposal_lens.max()
        else:
            verify_len = proposal_len

        with Timer() as scoring_timer:
            proposal_scores = self.scorer.score_proposals(
                execute_model_req, proposals)

        with Timer() as verification_timer:
            accepted_token_ids, target_logprobs = self._verify_tokens(
                execute_model_req.seq_group_metadata_list, proposal_scores,
                proposals, verify_len)

        stage_times = (proposal_timer.elapsed_time_ms / num_lookahead_slots,
                       scoring_timer.elapsed_time_ms,
                       verification_timer.elapsed_time_ms)
        # global draft_time, score_time
        # draft_time.append(proposal_timer.elapsed_time_ms)
        # score_time.append(scoring_timer.elapsed_time_ms)

        if self.use_specinfer:
            min_minus_one_subtensors = []
            subtensor_indices = []
            group_len = len(accepted_token_ids) // specinfer_k
            for i in range(0, group_len):
                group_indice = [i + group_len * j for j in range(specinfer_k)]
                group = accepted_token_ids[group_indice]
                min_count = float('inf')
                min_index = -1
                for k, tensor in enumerate(group):
                    count = torch.sum(tensor == -1).item()
                    if count < min_count:
                        min_count = count
                        min_index = i + group_len * k
                min_minus_one_subtensors.append(accepted_token_ids[min_index])
                subtensor_indices.append(min_index)
            accepted_token_ids = torch.stack(min_minus_one_subtensors)
            execute_model_req.seq_group_metadata_list = [execute_model_req.seq_group_metadata_list[i] for i in
                                                         subtensor_indices]

        # global acc
        acc_list = accepted_token_ids.tolist()
        for index, list_ in enumerate(acc_list):
            SpecProfiler.acc.append(int(sum(1 for element in list_ if element != -1) -1))
        return self._create_output_sampler_list(
            execute_model_req.seq_group_metadata_list,
            accepted_token_ids,
            target_logprobs=target_logprobs,
            k=verify_len,
            stage_times=stage_times,
            proposal_lens=proposals.proposal_lens)

    @nvtx_range("spec_decode_worker._verify_tokens")
    def _verify_tokens(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            proposal_scores: SpeculativeScores,
            proposals: SpeculativeProposals,
            max_proposal_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine which speculative tokens are accepted using the
        probabilities of each token according to the proposer and scorer models.

        Returns a tuple of Tensors, one for the accepted token ids and one for
        the logprobs according to the scoring model.
        """
        proposal_lens_list = proposals.proposal_lens.tolist()

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        (_, spec_indices), (_, non_spec_indices) = split_batch_by_proposal_len(
            seq_group_metadata_list, proposal_lens_list)
        original_indices = spec_indices + non_spec_indices

        # Get probabilities of target model, including bonus tokens.
        proposal_verifier_probs = proposal_scores.probs[spec_indices]

        # Get non-speculative sampled tokens from target model.
        non_spec_token_ids = proposal_scores.token_ids[non_spec_indices]

        # Get bonus tokens from target model.
        bonus_token_ids = proposal_scores.token_ids[spec_indices, -1:]

        # Get probabilities according to proposal method.
        proposal_probs = proposals.proposal_probs[spec_indices]

        # Get proposed tokens.
        proposal_token_ids = proposals.proposal_token_ids[spec_indices]

        # Sampler arguments
        sampler_extra_kwargs: Dict[str, Any] = {}
        if self.generators and isinstance(self.spec_decode_sampler,
                                          SpecDecodeStochasticBaseSampler):
            sampler_extra_kwargs["seeded_seqs"] = {
                idx: self.generators[sgm.request_id]
                for idx, sgm in enumerate(seq_group_metadata_list)
                if sgm.sampling_params.seed is not None
            }

        target_token_ids = proposal_scores.token_ids[spec_indices]

        accepted_token_ids = self.spec_decode_sampler(
            target_with_bonus_probs=proposal_verifier_probs,
            bonus_token_ids=bonus_token_ids,
            draft_probs=proposal_probs,
            draft_token_ids=proposal_token_ids,
            target_token_ids=target_token_ids,
            **sampler_extra_kwargs,
        )

        # if len(cur_step_trace.batched_requests) != accepted_token_ids.shape[0]:
        #     print(len(cur_step_trace.batched_requests))
        #     print(accepted_token_ids.shape)
        # assert len(
        #     cur_step_trace.batched_requests) == accepted_token_ids.shape[0]

        # Append output tokens from non-speculative sequences to
        # the accepted token ids tensor.
        # if non_spec_token_ids.shape[1] <= max_proposal_len + 1:
        #     non_spec_token_ids = non_spec_token_ids.expand(-1, max_proposal_len +
        #                                                    1).clone()
        try:
            non_spec_token_ids = non_spec_token_ids.expand(-1, max_proposal_len +
                                                           1).clone()
        except:
            pass
        non_spec_token_ids[:, 1:] = -1
        accepted_token_ids = torch.cat(
            [accepted_token_ids, non_spec_token_ids])

        logprobs = proposal_scores.logprobs
        # Rearrange so that results are in the order of the original seq group
        # metadata.
        accepted_token_ids[original_indices] = accepted_token_ids.clone()

        hidden_states = proposal_scores.hidden_states
        if hidden_states is not None:
            # Contract hidden states based on accepted tokens
            hs_size = hidden_states.shape[-1]

            accepted_index = accepted_token_ids + 1  # Convert -1 to 0
            accepted_index = accepted_index.count_nonzero(dim=1).add_(-1)
            index = accepted_index[:, None, None].expand(-1, 1, hs_size)
            second_last_token_hidden_states = hidden_states[:, -2]  # b x d
            hidden_states = hidden_states.gather(1, index).squeeze(1)  # b x d
            # Store hidden states from target model for subsequent decode step
            self.previous_hidden_states = HiddenStates(
                hidden_states, seq_group_metadata_list,
                second_last_token_hidden_states)

        return accepted_token_ids, logprobs

    def _create_output_sampler_list(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            accepted_token_ids: torch.Tensor,  # shape: [batch_size, k+1]
            target_logprobs: torch.Tensor,  # shape: [batch_size, k+1, vocab_size]
            k: int,
            stage_times: Tuple[float, float, float],
            proposal_lens: Optional[torch.Tensor] = None,
    ) -> List[SamplerOutput]:
        """Given the accepted token ids, create a list of SamplerOutput.

        The output is padded with -1 tokens such that each sequence has
        the same number of outputs.
        """
        batch_size, num_steps = accepted_token_ids.shape
        accepted_token_ids_by_step = accepted_token_ids.transpose(0, 1)
        if self._disable_logprobs:
            # We are skipping the logprobs. Hence don't serialize the
            # logprobs related tensors from the GPU. Instead create
            # empty/dummy lists.
            (accepted_token_id_ranks_by_step,
             accepted_token_id_logprobs_by_step,
             topk_logprobs_by_step, topk_indices_by_step) = \
                self._create_dummy_logprob_lists(
                    batch_size, num_steps,
                    self.scorer_worker.model_config.max_logprobs)
        else:
            # Organize input tensors by step instead of by sequence.
            target_logprobs_by_step = target_logprobs.transpose(0, 1)
            # Serialize all tensors into Python lists.
            (accepted_token_id_ranks_by_step,
             accepted_token_id_logprobs_by_step,
             topk_logprobs_by_step, topk_indices_by_step) = \
                self._create_logprob_lists_from_tensors(
                    target_logprobs_by_step, accepted_token_ids_by_step,
                    self.scorer_worker.model_config.max_logprobs)

        # Get the sequence ids and num_logprobs (sampling parameter) in the
        # batch.
        seq_ids, request_ids_seq_ids_mapping = get_all_seq_ids_and_request_ids(
            seq_group_metadata_list)

        num_logprobs_per_seq = get_all_num_logprobs(seq_group_metadata_list)

        # Serialize tensor to CPU Python list.
        accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

        # Construct the output on a per-step, per-sequence basis.
        sampler_output_list: List[SamplerOutput] = []
        for step_index in range(num_steps):
            if all(token_id == -1
                   for token_id in accepted_token_ids_by_step[step_index]):
                break

            step_output_token_ids: List[CompletionSequenceGroupOutput] = []
            for sequence_index in range(batch_size):
                # Each sequence may have a different num_logprobs; retrieve it.
                num_logprobs = num_logprobs_per_seq[sequence_index]
                step_output_token_ids.append(
                    create_sequence_group_output(
                        token_id=accepted_token_ids_by_step[step_index]
                        [sequence_index],
                        token_id_logprob_rank=accepted_token_id_ranks_by_step[
                            step_index][sequence_index],
                        token_id_logprob=accepted_token_id_logprobs_by_step[
                            step_index][sequence_index],
                        seq_id=seq_ids[sequence_index],
                        topk_token_ids=topk_indices_by_step[step_index]
                                       [sequence_index][:num_logprobs],
                        topk_logprobs=topk_logprobs_by_step[step_index]
                                      [sequence_index][:num_logprobs],
                    ))
            sampler_output_list.append(
                SamplerOutput(outputs=step_output_token_ids))

        if self.use_dsd:
            start_ts = time.perf_counter()
            self.dsd.set_token_acceptance_rate(
                self.spec_decode_sampler.num_accepted_tokens /
                self.spec_decode_sampler.num_draft_tokens)
            end_ts = time.perf_counter()
            SpecProfiler.dsd_overhead[-1] += end_ts - start_ts

        # Populate the data structures needed to keep track of sequences with
        # bonus tokens.
        self._track_sequences_with_bonus_tokens(seq_ids,
                                                request_ids_seq_ids_mapping,
                                                accepted_token_ids_by_step,
                                                proposal_lens.tolist())
        maybe_rejsample_metrics = (
            self._metrics.maybe_collect_rejsample_metrics(k))
        # self._maybe_log_stage_times(*stage_times)
        if maybe_rejsample_metrics is not None:
            sampler_output_list[
                0].spec_decode_worker_metrics = maybe_rejsample_metrics

            # Log time spent in each stage periodically.
            # This is periodic because the rejection sampler emits metrics
            # periodically.
            self._maybe_log_stage_times(*stage_times)
            # print(maybe_rejsample_metrics)

        return sampler_output_list

    def _maybe_log_stage_times(self, average_time_per_proposal_tok_ms: float,
                               scoring_time_ms: float,
                               verification_time_ms: float) -> None:
        """Log the speculative stage times. If stat logging is disabled, do
        nothing.
        """
        # if self._disable_log_stats:
        #     return

        logger.info(
            "SpecDecodeWorker stage times: "
            "average_time_per_proposal_tok_ms=%.02f "
            "scoring_time_ms=%.02f verification_time_ms=%.02f",
            average_time_per_proposal_tok_ms, scoring_time_ms,
            verification_time_ms)

    def _create_dummy_logprob_lists(
            self,
            batch_size: int,
            num_steps: int,
            num_top_k: int,
    ) -> Tuple[List[List[int]], List[List[float]],
    List[List[List[Optional[float]]]],
    List[List[List[Optional[int]]]]]:
        """
        Creates and returns four dummy lists representing token probabilities
        and their ranks.

        This method initializes and returns:
            - The ranks of the accepted tokens, shaped (num_steps, batch_size)
            - The log probabilities of the accepted tokens,
              shaped (num_steps, batch_size)
            - The log probabilities of the top k tokens,
              shaped (num_steps, batch_size, num_top_k)
            - The token IDs of the top k tokens,
              shaped (num_steps, batch_size, num_top_k)

        Args:
            batch_size (int): The size of the batch.
            num_steps (int): The number of steps in the sequence.
            num_top_k (int): The number of top-k token log probabilities to
            return.

        Returns:
            A tuple containing four dummy lists as described above.
        """
        accepted_token_id_ranks_by_step = [[-1] * batch_size
                                           for _ in range(num_steps)]
        accepted_token_id_logprobs_by_step = [[0.0] * batch_size
                                              for _ in range(num_steps)]
        topk_logprobs_by_step: List[List[List[Optional[float]]]] = [[
            [None] * num_top_k for _ in range(batch_size)
        ] for _ in range(num_steps)]
        topk_indices_by_step: List[List[List[Optional[int]]]] = [[
            [None] * num_top_k for _ in range(batch_size)
        ] for _ in range(num_steps)]
        return (accepted_token_id_ranks_by_step,
                accepted_token_id_logprobs_by_step, topk_logprobs_by_step,
                topk_indices_by_step)

    def _create_logprob_lists_from_tensors(
            self,
            target_logprobs_by_step: torch.Tensor,
            accepted_token_ids_by_step: torch.Tensor,
            num_top_k: int,
    ) -> Tuple[List[List[int]], List[List[float]],
    List[List[List[Optional[float]]]],
    List[List[List[Optional[int]]]]]:
        """
        Creates and returns four lists representing token probabilities and
        their ranks.

        This method initializes and returns four lists containing:
            - The ranks of the accepted tokens, shaped (num_steps, batch_size)
            - The log probabilities of the accepted tokens,
              shaped (num_steps, batch_size)
            - The log probabilities of the top k tokens,
              shaped (num_steps, batch_size, num_top_k)
            - The token IDs of the top k tokens,
              shaped (num_steps, batch_size, num_top_k)

        Args:
            target_logprobs_by_step (torch.Tensor): Tensor representing the
            log probabilities of the target model,
            shaped (num_steps, batch_size, vocab_size)
            accepted_token_ids_by_step (torch.Tensor): Tensor representing
            the accepted  token_ids, shaped (num_steps, batch_size)
            num_top_k (int): The number of top-k token log probabilities to
            return.

        Returns:
            A tuple containing the lists as described above.
        """
        # Serialize all tensors to CPU Python lists.
        # Get the logprobs/rank of the accepted tokens.
        (accepted_token_id_ranks_by_step_tensor,
         accepted_token_id_logprobs_by_step_tensor
         ) = get_sampled_token_logprobs(
            logprob_tensor=target_logprobs_by_step,
            sampled_token_ids=accepted_token_ids_by_step,
        )
        # Get the top-k logprobs (which may or may not include the
        # logprob of the accepted token).
        (topk_logprobs_by_step_tensor,
         topk_indices_by_step_tensor) = target_logprobs_by_step.topk(
            k=num_top_k,
            dim=-1,
        )
        accepted_token_id_ranks_by_step = (
            accepted_token_id_ranks_by_step_tensor.tolist())
        accepted_token_id_logprobs_by_step = (
            accepted_token_id_logprobs_by_step_tensor.tolist())
        topk_logprobs_by_step = topk_logprobs_by_step_tensor.tolist()
        topk_indices_by_step = topk_indices_by_step_tensor.tolist()
        return (accepted_token_id_ranks_by_step,
                accepted_token_id_logprobs_by_step, topk_logprobs_by_step,
                topk_indices_by_step)

    def _track_finished_requests(self, execute_model_req: ExecuteModelRequest):
        """
        Removes the finished requests and their associated sequence ids from
        internal book keeping data structures.
        """
        for finished_request in execute_model_req.finished_requests_ids:
            for seq_id in self._request_id_seq_id_mapping[finished_request]:
                self._seq_with_bonus_token_in_last_step.discard(seq_id)
            del self._request_id_seq_id_mapping[finished_request]

    def _track_sequences_with_bonus_tokens(
            self, seq_ids: List[int],
            request_ids_seq_ids_mapping: Dict[str, Set[int]],
            accepted_token_ids_by_step: List[List[int]],
            proposal_lens: List[int] = None) -> None:
        """
        Updates the internal data structures which keep track of sequences
        which have been assigned bonus tokens in their last forward pass.
        """
        max_proposal_len = max(proposal_lens)
        for seq_index, seq_id in enumerate(seq_ids):
            # last_token_id = accepted_token_ids_by_step[-1][seq_index]
            last_token_id = accepted_token_ids_by_step[-1 - (max_proposal_len - proposal_lens[seq_index])][seq_index]
            if last_token_id == -1:
                self._seq_with_bonus_token_in_last_step.discard(seq_id)
            else:
                self._seq_with_bonus_token_in_last_step.add(seq_id)
        for request_id, sequences in request_ids_seq_ids_mapping.items():
            self._request_id_seq_id_mapping[request_id].update(sequences)

    @cached_property
    def _vocab_size(self) -> int:
        """Get the vocab size of the model and make sure it's consistent between
        draft and target workers.
        """
        vocab_sizes = [
            worker.vocab_size
            for worker in [self.proposer_worker, self.scorer_worker]
        ]
        assert all(vocab_sizes[0] == vocab_size for vocab_size in vocab_sizes)
        return vocab_sizes[0]

    @property
    def rank(self):
        return self.scorer_worker.rank

    @property
    def device(self):
        return self.scorer_worker.device

    @property
    def _driver_rank(self) -> int:
        return 0

    def get_cache_block_size_bytes(self):
        """Return the size of a cache block in bytes.

        This function is only used to compose workers within a SpecDecodeWorker.
        We leave composing a SpecDecodeWorker within a SpecDecodeWorker
        undefined for now, although it could be implemented in the future.
        See https://arxiv.org/abs/2308.04623.
        """
        raise NotImplementedError

    def start_profile(self):
        if isinstance(self.scorer_worker, Worker):
            self.scorer_worker.start_profile()

    def stop_profile(self):
        if isinstance(self.scorer_worker, Worker):
            self.scorer_worker.stop_profile()


def split_num_cache_blocks_evenly(scorer_cache_block_size_bytes: int,
                                  proposer_cache_block_size_bytes: int,
                                  total_num_gpu_blocks: int) -> int:
    """Given total_num_gpu_blocks, the number of GPU blocks that could be
    allocate to the target model, this function calculates how many blocks
    should be given to the draft and target model.

    Note that usually the block size, in bytes, of each model is different,
    as it's a function of number of KV/layer, number of heads, and hidden
    dimension size.

    Since the target and draft models allocate the same number of blocks, we
    simply calculate the number of blocks where if allocated by both models,
    the total memory usage from KV cache is no larger than the number of
    blocks allocatable by the target model alone.
    """
    new_num_gpu_blocks = int(
        total_num_gpu_blocks * scorer_cache_block_size_bytes /
        (proposer_cache_block_size_bytes + scorer_cache_block_size_bytes))

    return new_num_gpu_blocks


def prepare_prefill_hidden_states(
        prefill_hidden_states: torch.Tensor) -> HiddenStates:
    # For prefill step in proposer, we run the model for N-1 tokens
    # because Nth token will be processed in the first decode step. For
    # N-1 tokens, the input should be 0:N-1 hidden states which should
    # be concatanated with 1:N token (since output of scorer has to be
    # the input for proposer). Therefore, we shift the hidden states to
    # align n-1th hidden state with nth token.
    return HiddenStates(prefill_hidden_states.roll(
        shifts=1, dims=0)) if prefill_hidden_states is not None else None

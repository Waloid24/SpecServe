import pickle
import json
import numpy as np
from datetime import datetime
from vllm.logger import init_logger

logger = init_logger(__name__)

class SingletonProfiler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.goodput_overhead = []
            self.ssd_overhead = []
            self.draft = []
            self.acc = []
            self.dsd_overhead = []
            self.rsd_overhead = []
            self.batch = []
            self.steps = 0
            self.avg_confidence = 0
            self.speculative_len = []
            self.slo = 0
            self._initialized = True

    def profile(self):
        if not self.draft:
            return
        data = {
            "draft": self.draft,
            "acc": self.acc,
            "steps": self.steps,
            "avg_confidence": self.avg_confidence,
            "speculative_len": self.speculative_len,
            "batch": self.batch,
            "goodput_overhead": self.goodput_overhead
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'_{timestamp}.json'
        print("==============Start Breakdown==============")
        logger.info("breakdown")
        if self.rsd_overhead:
            logger.info("use rsd")
            logger.info("ssd overhead: %f ms" % (np.mean(self.ssd_overhead) * 1000))
            logger.info("rsd overhead: %f ms" % (np.mean(self.rsd_overhead) * 1000))
            logger.info("SLO counts: %f", self.slo)
            logger.info("average confidence: %f", self.avg_confidence)
            data['ssd_overhead'] = self.ssd_overhead
            data['rsd_overhead'] = self.rsd_overhead
            data['slo'] = self.slo
            data['avg_confidence'] = self.avg_confidence
            filename = 'rsd' + filename
        elif self.ssd_overhead:
            logger.info("use ssd")
            logger.info("ssd overhead: %f ms" % (np.mean(self.ssd_overhead) * 1000))
            logger.info("SLO counts: %f", self.slo)
            logger.info("average confidence: %f", self.avg_confidence)
            data['ssd_overhead'] = self.ssd_overhead
            data['slo'] = self.slo
            data['avg_confidence'] = self.avg_confidence
            filename = 'ssd' + filename
        elif self.dsd_overhead:
            logger.info("use dsd")
            logger.info("dsd overhead: %f ms" % (np.mean(self.dsd_overhead) * 1000))
            data['dsd_overhead'] = self.dsd_overhead
            filename = 'dsd' + filename
        else:
            logger.info("use standard speculative decoding")
            filename = f"sd{int(np.mean(self.speculative_len))}" + filename
        logger.info("goodput overhead: %f ms" % (np.mean(self.goodput_overhead) * 1000))
        logger.info("goodput counts: %d" % len(self.goodput_overhead))
        logger.info("total steps: %d" % self.steps)
        logger.info("draft tokens: %f" % np.sum(self.draft))
        logger.info("accept tokens: %f" % np.sum(self.acc))
        logger.info("average accept lens: %f" % np.mean(self.acc))
        logger.info("average accept rate: %f" % (np.sum(self.acc) / np.sum(self.draft)))
        logger.info("speculative length: %f" % np.mean(self.speculative_len))
        logger.info("average batch size: %f" % np.mean(self.batch))
        print("==============End Breakdown==============")

        # Save results to a file using json
        with open("bench_results/" + filename, 'w') as f:
            json.dump(data, f)

SpecProfiler = SingletonProfiler()
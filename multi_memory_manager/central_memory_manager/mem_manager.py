import argparse
import pickle
import time
from typing import List

import pynvml
import sysv_ipc

MB = 1024 * 1024
MANAGER_MQ_BASE = 300
REPLY_MQ_BASE = 1300


class CentralMemManager:

    def __init__(self, client_id: int = 0, device_ids: List[int] = [0]):
        self.client_id = client_id
        self.device_ids = device_ids
        self.manager_key = MANAGER_MQ_BASE - client_id
        self.manager_mq = sysv_ipc.MessageQueue(self.manager_key,
                                                sysv_ipc.IPC_CREAT)
        pynvml.nvmlInit()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(device_id)
            for device_id in device_ids
        ]

    def get_gpu_memory_info(self):
        memory_infos = [
            pynvml.nvmlDeviceGetMemoryInfo(handle) for handle in self.handles
        ]
        free_memory = min(memory_info.free for memory_info in memory_infos)
        used_memory = max(memory_info.used for memory_info in memory_infos)
        total_memory = max(memory_info.total for memory_info in memory_infos)
        return free_memory, used_memory, total_memory

    def run(self):
        last_free_gpu_memory = 0
        last_alloc_memory = 0
        last_ts = 0.0
        while True:
            try:
                raw_data, _ = self.manager_mq.receive(block=False)
            except sysv_ipc.BusyError:
                continue

            request = pickle.loads(raw_data)
            client_key = REPLY_MQ_BASE + request["client_id"]
            reply_mq = sysv_ipc.MessageQueue(client_key, sysv_ipc.IPC_CREAT)
            req_mem = request["req_mem"]
            now = time.time()
            print(f"mem_manager recv: client={request['client_id']} req_mem={req_mem}MB",
                  flush=True)

            current_free_memory, current_used_memory, total_memory = (
                self.get_gpu_memory_info())
            if (now - last_ts < 0.15 and last_free_gpu_memory > current_free_memory
                    and last_free_gpu_memory - current_free_memory <
                    last_alloc_memory * MB):
                free_gpu_memory = last_free_gpu_memory - last_alloc_memory * MB
            else:
                free_gpu_memory = current_free_memory

            buffer_memory = 2 * 1024 * MB
            required_memory = req_mem * MB
            if free_gpu_memory > required_memory + buffer_memory:
                response = {
                    "res": "yes",
                    "size": req_mem,
                    "max_free_mem": free_gpu_memory // MB - req_mem,
                }
                last_alloc_memory = req_mem
            else:
                response = {
                    "res": "no",
                    "size": free_gpu_memory // MB,
                    "max_free_mem": 0,
                }
                last_alloc_memory = 0

            reply_mq.send(pickle.dumps(response))
            print(f"mem_manager reply: {response}", flush=True)
            last_free_gpu_memory = free_gpu_memory
            last_ts = now


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic KV cache admission controller")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    manager = CentralMemManager(args.client_id, args.devices)
    try:
        manager.run()
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

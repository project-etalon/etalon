from threading import Thread, Lock
from multiprocessing import Queue as MPQueue

from etalon.core.llm_clients import construct_client


class RequestsManager:
    """Manages requests for single LLM API client."""

    def __init__(
        self,
        client_id: int,
        model: str,
        tokenizer_name: str,
        llm_api: str,
        max_concurrent_requests: int,
        input_queue: MPQueue,
        output_queue: MPQueue,
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.result_lock = Lock()
        self.results = []
        # just create a single client per manager
        self.llm_client = construct_client(
            model_name=model,
            tokenizer_name=tokenizer_name,
            llm_api=llm_api,
        )
        self.client_id = client_id
        self.start_tasks()

    async def start_tasks(self):
        """Starts the tasks to handle requests.

        Returns:
            None
        """
        self.client_threads = [
            Thread(target=self.process_requests)
            for i in range(self.max_concurrent_requests)
        ]

        for thread in self.client_threads:
            thread.start()

    async def process_requests(self) -> None:
        while True:
            request_config = await self.input_queue.get()
            if request_config is None:
                break
            result = self.llm_client.send_llm_request(request_config)
            self.output_queue.put(result)

from multiprocessing import Queue as MPQueue
from threading import Lock, Thread

from etalon.config import ClientConfig
from etalon.core.llm_clients import construct_client


class RequestsManager:
    """Manages requests for single LLM API client."""

    def __init__(
        self,
        client_id: int,
        client_config: ClientConfig,
        input_queue: MPQueue,
        output_queue: MPQueue,
    ):
        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.result_lock = Lock()
        self.results = []
        # just create a single client per manager
        self.llm_client = construct_client(
            model_name=client_config.model,
            tokenizer_name=client_config.tokenizer,
            llm_api=client_config.llm_api,
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
            for i in range(self.client_config.num_concurrent_requests_per_client)
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

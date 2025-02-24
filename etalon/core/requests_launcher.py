from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from threading import Thread
from typing import Dict

from etalon.config.config import ClientConfig
from etalon.core.llm_clients import construct_client
from etalon.core.llm_clients.base_llm_client import BaseLLMClient


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        client_config: ClientConfig,
        input_queue: MPQueue,
        output_queue: MPQueue,
    ):
        self.clients = []
        self.llm_clients: Dict[int, BaseLLMClient] = {}

        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue

        for client_id in range(self.client_config.num_clients):
            client = Process(
                target=self.run_client,
                args=(client_id,),
            )
            self.clients.append(client)

    def start(self) -> None:
        """Start the clients."""
        for client in self.clients:
            client.start()

    def run_client(self, client_id: int) -> None:
        """Run the client."""
        assert self.client_config.tokenizer is not None
        assert self.client_config.model is not None

        self.llm_clients[client_id] = construct_client(
            model_name=self.client_config.model,
            tokenizer_name=self.client_config.tokenizer,
            llm_api=self.client_config.llm_api,
        )
        self.start_threads(client_id=client_id)

    def start_threads(self, client_id: int) -> None:
        """Start the threads."""
        client_threads = [
            Thread(target=self.process_requests, args=(client_id,))
            for _ in range(self.client_config.num_concurrent_requests_per_client)
        ]

        for thread in client_threads:
            thread.start()

    def process_requests(self, client_id: int) -> None:
        while True:
            request_config = self.input_queue.get()
            if request_config is None:
                break
            result = self.llm_clients[client_id].send_llm_request(request_config)
            self.output_queue.put(result)

    def complete_tasks(self) -> None:
        """Complete the clients."""
        # put None to indicate that client should stop
        for _ in range(
            self.client_config.num_clients
            * self.client_config.num_concurrent_requests_per_client
        ):
            self.input_queue.put(None)

        for client in self.clients:
            client.join()

    def kill_clients(self) -> None:
        """Kill all the clients."""
        for client in self.clients:
            client.terminate()
            client.join(30)
            client.kill()
            client.close()

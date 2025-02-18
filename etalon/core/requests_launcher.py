from multiprocessing import (
    Process,
    Queue as MPQueue,
)

from etalon.core.requests_manager import RequestsManager


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        model: str,
        tokenizer_name: str,
        llm_api: str,
        num_clients: int,
        num_concurrent_requests_per_client: int,
        input_queue: MPQueue,
        output_queue: MPQueue,
    ):
        self.clients = []

        self.model = model
        self.tokenizer_name = tokenizer_name
        self.llm_api = llm_api
        self.num_clients = num_clients
        self.max_concurrent_requests = num_concurrent_requests_per_client
        self.input_queue = input_queue
        self.output_queue = output_queue

        for client_id in range(num_clients):
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
        requests_manager = RequestsManager(
            client_id=client_id,
            model=self.model,
            tokenizer_name=self.tokenizer_name,
            llm_api=self.llm_api,
            max_concurrent_requests=self.max_concurrent_requests,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
        )
        requests_manager.start_tasks()

    def complete_tasks(self) -> None:
        """Complete the clients."""
        # put None to indicate that client should stop
        for _ in range(self.num_clients * self.max_concurrent_requests):
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

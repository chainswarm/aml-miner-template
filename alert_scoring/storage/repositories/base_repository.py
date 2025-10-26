from abc import ABC, abstractmethod
import time
from clickhouse_connect.driver import Client


class BaseRepository(ABC):
    def __init__(self, client: Client, partition_id: int = None):
        self.client = client
        self.partition_id = partition_id
    
    def _generate_version(self) -> int:
        base_version = int(time.time() * 1000000)
        if self.partition_id is not None:
            return base_version + self.partition_id
        return base_version
    
    @classmethod
    @abstractmethod
    def schema(cls) -> str:
        pass
    
    @classmethod
    @abstractmethod
    def table_name(cls) -> str:
        pass
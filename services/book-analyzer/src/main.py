import asyncio
import logging
import signal

from .config import Config
from .worker import Worker


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config.from_env()
    worker = Worker(config)
    await worker.start()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.stop()))

    await worker.wait()


if __name__ == "__main__":
    asyncio.run(main())

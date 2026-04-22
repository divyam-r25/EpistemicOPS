# EpistemicOps Drift Injector
# ============================
# Connects to mock API internal endpoints to trigger drift mid-era.

import os
import time
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift-injector")

def main():
    logger.info("Drift Injector started. Waiting for orchestration commands...")
    # Keep container alive, real logic will be driven by the environment engine
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()

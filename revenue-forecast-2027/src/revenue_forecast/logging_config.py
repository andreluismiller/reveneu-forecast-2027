import logging

def configure(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

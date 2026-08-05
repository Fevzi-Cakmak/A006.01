# run_borsapy.py
import argparse
import logging

from engine.adapters import BorsapyDataFetcher
from engine.core import config
from engine.runner import run_backtest
import borsapy as bp


def main():
    p = argparse.ArgumentParser(description="VIOS A003 Borsapy API")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    kodlar = sorted(set(bp.Index(config.evren_endeks).component_symbols))
    b100 = set(bp.Index("XU100").component_symbols)
    b50 = set(bp.Index("XU050").component_symbols)
    b30 = set(bp.Index("XU030").component_symbols)
    fetcher = BorsapyDataFetcher(config)
    run_backtest(
        fetcher,
        kodlar,
        b100,
        b50,
        b30,
        "BORSAPY API",
        "A003_BORSAPY",
        2,
        config,
    )


if __name__ == "__main__":
    main()
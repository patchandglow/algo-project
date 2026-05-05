"""Out-of-sample data lockbox.

Splits data into train/test. Test partition raises if accessed during development.
Touched exactly once at the end.
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime


LOCK_FILE = Path(__file__).parent.parent / "data" / "processed" / "lockbox_state.json"


class OutOfSampleLockbox:
    """Hard separator between in-sample (development) and out-of-sample (test) data.

    Usage:
        lockbox = OutOfSampleLockbox(data, train_frac=0.70)
        train = lockbox.train  # use freely
        test = lockbox.unlock_test(strategy_name="h1_overnight_gap")  # logs access
    """

    def __init__(self, data: pd.DataFrame, train_frac: float = 0.70):
        if not 0.5 <= train_frac < 1.0:
            raise ValueError("train_frac must be in [0.5, 1.0)")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("data must have DatetimeIndex")

        n = len(data)
        split_idx = int(n * train_frac)
        self._train = data.iloc[:split_idx].copy()
        self._test = data.iloc[split_idx:].copy()
        self.train_frac = train_frac
        self.split_date = data.index[split_idx]

    @property
    def train(self) -> pd.DataFrame:
        return self._train

    def unlock_test(self, strategy_name: str, justification: str = "") -> pd.DataFrame:
        """Access test data. Logs the access — every unlock is permanent record."""
        self._log_access(strategy_name, justification)
        print(f"⚠️  OUT-OF-SAMPLE TEST UNLOCKED for: {strategy_name}")
        print(f"   This access is logged. Re-running this for parameter tuning")
        print(f"   defeats the purpose of OOS validation.")
        return self._test.copy()

    def _log_access(self, strategy_name: str, justification: str) -> None:
        log = []
        if LOCK_FILE.exists():
            with open(LOCK_FILE) as f:
                log = json.load(f)
        log.append({
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "justification": justification,
            "split_date": str(self.split_date),
        })
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCK_FILE, "w") as f:
            json.dump(log, f, indent=2)

    def access_log(self):
        if not LOCK_FILE.exists():
            return []
        with open(LOCK_FILE) as f:
            return json.load(f)

    def __repr__(self):
        return (f"Lockbox(train: {self._train.index[0].date()} to "
                f"{self._train.index[-1].date()}, "
                f"test: {self._test.index[0].date()} to "
                f"{self._test.index[-1].date()})")

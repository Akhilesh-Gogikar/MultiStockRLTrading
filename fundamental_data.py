from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class FundamentalDataIntegrator:
    """Loads and aligns daily/periodic fundamental data with intraday OHLCV bars."""

    SYMBOL_CANDIDATES = [
        "symbol",
        "ticker",
        "name",
        "stock",
        "company",
        "tradingsymbol",
        "security",
    ]
    DATE_CANDIDATES = [
        "date",
        "datetime",
        "report_date",
        "fiscal_date",
        "as_of_date",
        "published_at",
    ]

    def __init__(self, repo_path: Optional[str] = None, max_files: int = 250) -> None:
        self.repo_path = self._resolve_repo_path(repo_path)
        self.max_files = max_files
        self.feature_names: List[str] = []
        self._fundamentals_by_symbol: Dict[str, pd.DataFrame] = {}

        if self.repo_path is not None:
            self._fundamentals_by_symbol, self.feature_names = self._load()

    @property
    def is_enabled(self) -> bool:
        return self.repo_path is not None and len(self.feature_names) > 0

    def merge_with_price_data(
        self, price_df: pd.DataFrame, symbol: str, datetime_col: str = "datetime"
    ) -> pd.DataFrame:
        """Merge normalized date-level fundamentals with each bar for a symbol."""
        if not self.is_enabled:
            return price_df

        symbol_key = str(symbol).upper()
        fundamentals = self._fundamentals_by_symbol.get(symbol_key)
        merged = price_df.copy()

        if datetime_col not in merged.columns:
            return merged

        merged["_fundamental_date"] = pd.to_datetime(merged[datetime_col], errors="coerce").dt.tz_localize(None).dt.normalize()

        if fundamentals is None:
            for feature in self.feature_names:
                merged[feature] = 0.0
            merged.drop(columns=["_fundamental_date"], inplace=True)
            return merged

        merged = merged.merge(
            fundamentals,
            how="left",
            left_on="_fundamental_date",
            right_index=True,
        )

        for feature in self.feature_names:
            if feature not in merged.columns:
                merged[feature] = 0.0

        merged[self.feature_names] = merged[self.feature_names].ffill().bfill().fillna(0.0)
        merged.drop(columns=["_fundamental_date"], inplace=True)
        return merged

    def _resolve_repo_path(self, repo_path: Optional[str]) -> Optional[Path]:
        if repo_path:
            path = Path(repo_path).expanduser().resolve()
            return path if path.exists() else None

        cwd = Path.cwd()
        candidates = [
            cwd / "indian_stock_market_data",
            cwd / "indian-stock-market-data",
            cwd / "indian stock market data",
            cwd.parent / "indian_stock_market_data",
            cwd.parent / "indian-stock-market-data",
            cwd.parent / "indian stock market data",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        return None

    def _load(self) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        merged_fundamentals = self._load_repo_fundamentals(self.repo_path)
        if merged_fundamentals.empty:
            return {}, []

        feature_names = [column for column in merged_fundamentals.columns if column not in ("symbol", "date")]
        by_symbol: Dict[str, pd.DataFrame] = {}
        for symbol, group in merged_fundamentals.groupby("symbol"):
            symbol_df = group.set_index("date")[feature_names].sort_index()
            by_symbol[symbol] = symbol_df

        return by_symbol, feature_names

    def _load_repo_fundamentals(self, root: Path) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        files = list(root.rglob("*.csv")) + list(root.rglob("*.parquet"))
        for path in files[: self.max_files]:
            parsed = self._parse_fundamental_file(path)
            if parsed is not None and not parsed.empty:
                frames.append(parsed)

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            value_columns = [column for column in frame.columns if column not in ("symbol", "date")]
            merged = merged.merge(frame, on=["symbol", "date"], how="outer")
            merged[value_columns] = merged[value_columns].fillna(0.0)

        merged.sort_values(["symbol", "date"], inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def _parse_fundamental_file(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            if path.suffix.lower() == ".csv":
                sample = pd.read_csv(path, nrows=50)
                if sample.empty:
                    return None
                symbol_col, date_col = self._detect_columns(sample)
                if symbol_col is None or date_col is None:
                    return None
                full_df = pd.read_csv(path)
            else:
                sample = pd.read_parquet(path)
                if sample.empty:
                    return None
                symbol_col, date_col = self._detect_columns(sample)
                if symbol_col is None or date_col is None:
                    return None
                full_df = sample
        except Exception:
            return None

        if full_df.empty:
            return None

        full_df[date_col] = pd.to_datetime(full_df[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
        full_df[symbol_col] = full_df[symbol_col].astype(str).str.upper().str.strip()

        numeric_cols = full_df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in [symbol_col, date_col]]
        if not numeric_cols:
            return None

        cleaned = full_df[[symbol_col, date_col] + numeric_cols].dropna(subset=[symbol_col, date_col]).copy()
        if cleaned.empty:
            return None

        prefixed_columns = {col: f"fundamental_{path.stem}_{col}" for col in numeric_cols}
        cleaned.rename(columns=prefixed_columns, inplace=True)

        grouped = (
            cleaned.groupby([symbol_col, date_col], as_index=False)
            .mean(numeric_only=True)
            .rename(columns={symbol_col: "symbol", date_col: "date"})
        )
        return grouped

    def _detect_columns(self, frame: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        columns = {str(column).lower(): column for column in frame.columns}

        symbol_col = None
        date_col = None

        for candidate in self.SYMBOL_CANDIDATES:
            if candidate in columns:
                symbol_col = columns[candidate]
                break

        for candidate in self.DATE_CANDIDATES:
            if candidate in columns:
                date_col = columns[candidate]
                break

        return symbol_col, date_col

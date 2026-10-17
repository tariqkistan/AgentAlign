PRICE_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
}


def estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    if model not in PRICE_PER_1M_TOKENS:
        raise ValueError(f"Unknown model: {model}")
    price_in, price_out = PRICE_PER_1M_TOKENS[model]
    cost = (tokens_in * price_in / 1_000_000) + (tokens_out * price_out / 1_000_000)
    return round(cost, 6)

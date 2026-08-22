import re
from typing import List, Dict, Any

class ContextualCompressor:
    """
    Strips uninformative filler phrases, conversational noise, and repetitive tokens
    from retrieved chunks to maximize LLM context window efficiency.
    """
    FILLER_PATTERNS = [
        r'\b(as mentioned previously|it is important to note that|in terms of|furthermore|additionally|in this context)\b',
        r'\s{2,}'
    ]

    def compress(self, text: str) -> str:
        compressed = text
        for pat in self.FILLER_PATTERNS:
            compressed = re.sub(pat, ' ', compressed, flags=re.IGNORECASE)
        return compressed.strip()

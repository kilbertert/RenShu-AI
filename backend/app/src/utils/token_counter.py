import re

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Rule:
    - 1 Chinese character ≈ 0.6 tokens
    - 1 English character ≈ 0.3 tokens
    """
    if not text:
        return 0
        
    # Count Chinese characters (including punctuation roughly)
    # Range for CJK Unified Ideographs
    cn_pattern = re.compile(r'[\u4e00-\u9fa5]')
    cn_count = len(cn_pattern.findall(text))
    
    # Count total length and subtract Chinese characters to get roughly English/ASCII count
    total_len = len(text)
    en_count = total_len - cn_count
    
    # Calculate tokens
    tokens = int(cn_count * 0.6 + en_count * 0.3)
    
    # Ensure at least 1 token if text is not empty
    return max(1, tokens) if tokens == 0 else tokens

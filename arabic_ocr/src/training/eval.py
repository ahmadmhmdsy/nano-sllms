"""Evaluation metrics for Arabic OCR."""
import torch


def compute_cer(predictions, targets):
    """Compute Character Error Rate (CER).
    
    Args:
        predictions: list of strings
        targets: list of strings
    
    Returns:
        CER as a float (0-1)
    """
    total_chars = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        # Levenshtein distance
        dist = levenshtein_distance(pred, target)
        total_errors += dist
        total_chars += len(target)
    
    if total_chars == 0:
        return 0.0
    return total_errors / total_chars


def compute_wer(predictions, targets):
    """Compute Word Error Rate (WER).
    
    Args:
        predictions: list of strings
        targets: list of strings
    
    Returns:
        WER as a float (0-1)
    """
    total_words = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        # Levenshtein distance on words
        dist = levenshtein_distance(pred_words, target_words)
        total_errors += dist
        total_words += len(target_words)
    
    if total_words == 0:
        return 0.0
    return total_errors / total_words


def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def decode_greedy(logits, idx_to_char, max_len=64):
    """Greedy decoding for transformer.
    
    Args:
        logits: (N, S, C) or (S, C)
        idx_to_char: dict mapping index to character
        max_len: maximum sequence length
    
    Returns:
        list of decoded strings
    """
    if logits.dim() == 3:
        # (N, S, C) -> argmax over C
        indices = logits.argmax(dim=-1)  # (N, S)
        results = []
        for i in range(indices.size(0)):
            chars = []
            for idx in indices[i]:
                idx = idx.item()
                if idx in (0, 2):  # PAD or EOS
                    break
                if idx == 1:  # SOS
                    continue
                chars.append(idx_to_char.get(idx, ''))
            results.append(''.join(chars))
        return results
    else:
        # (S, C) -> argmax over C
        indices = logits.argmax(dim=-1)
        chars = []
        for idx in indices:
            idx = idx.item()
            if idx in (0, 2):
                break
            if idx == 1:
                continue
            chars.append(idx_to_char.get(idx, ''))
        return [''.join(chars)]


def decode_token_ids(tokens, idx_to_char):
    """Decode generated token ids (N, L) to strings.

    Stops at EOS, skips SOS/PAD. This is the honest decoding path: the tokens
    come from autoregressive generation, not from teacher forcing.
    """
    results = []
    for row in tokens:
        chars = []
        for idx in row.tolist():
            if idx == 2:  # EOS
                break
            if idx in (0, 1):  # PAD, SOS
                continue
            chars.append(idx_to_char.get(idx, ''))
        results.append(''.join(chars))
    return results


def decode_beam_search(logits, idx_to_char, beam_width=5, max_len=64):
    """Beam search decoding for transformer.
    
    Args:
        logits: (N, S, C) or (S, C)
        idx_to_char: dict mapping index to character
        beam_width: beam width
        max_len: maximum sequence length
    
    Returns:
        list of decoded strings
    """
    if logits.dim() == 3:
        # (N, S, C)
        N, S, C = logits.shape
        results = []
        for i in range(N):
            # Initialize beams: (score, sequence)
            beams = [(0.0, [])]
            
            for t in range(S):
                new_beams = []
                for score, seq in beams:
                    # Get top-k predictions
                    log_probs = torch.log_softmax(logits[i, t], dim=-1)
                    topk_probs, topk_indices = log_probs.topk(beam_width)
                    
                    for prob, idx in zip(topk_probs, topk_indices):
                        idx = idx.item()
                        if idx in (0, 2):  # PAD or EOS
                            continue
                        if idx == 1:  # SOS
                            new_beams.append((score + prob.item(), seq))
                            continue
                        new_beams.append((score + prob.item(), seq + [idx]))
                
                # Keep top beams
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_width]
            
            # Get best beam
            if beams:
                best_seq = beams[0][1]
                text = ''
                for idx in best_seq:
                    char = idx_to_char.get(idx, '')
                    text += char
                results.append(text)
            else:
                results.append('')
        return results
    else:
        # (S, C)
        S, C = logits.shape
        beams = [(0.0, [])]
        
        for t in range(S):
            new_beams = []
            for score, seq in beams:
                log_probs = torch.log_softmax(logits[t], dim=-1)
                topk_probs, topk_indices = log_probs.topk(beam_width)
                
                for prob, idx in zip(topk_probs, topk_indices):
                    idx = idx.item()
                    if idx in (0, 2):
                        continue
                    if idx == 1:
                        new_beams.append((score + prob.item(), seq))
                        continue
                    new_beams.append((score + prob.item(), seq + [idx]))
            
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]
        
        if beams:
            best_seq = beams[0][1]
            text = ''
            for idx in best_seq:
                char = idx_to_char.get(idx, '')
                text += char
            return [text]
        else:
            return ['']


def decode_ctc(logits, idx_to_char, blank=0):
    """CTC decoding (greedy).
    
    Args:
        logits: (T, N, C) or (T, C)
        idx_to_char: dict mapping index to character
        blank: blank index
    
    Returns:
        list of decoded strings
    """
    if logits.dim() == 3:
        # (T, N, C)
        indices = logits.argmax(dim=-1)  # (T, N)
        results = []
        for i in range(indices.size(1)):
            chars = []
            prev = -1
            for t in range(indices.size(0)):
                idx = indices[t, i].item()
                if idx != blank and idx != prev:
                    chars.append(idx_to_char.get(idx, ''))
                prev = idx
            results.append(''.join(chars))
        return results
    else:
        # (T, C)
        indices = logits.argmax(dim=-1)
        chars = []
        prev = -1
        for idx in indices:
            idx = idx.item()
            if idx != blank and idx != prev:
                chars.append(idx_to_char.get(idx, ''))
            prev = idx
        return [''.join(chars)]
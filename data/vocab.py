from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        
        char =  sorted(list(set(text)))

        vocab = {}
        for i in range(len(char)):
            vocab[char[i]] = i;
        reverse_dict = {v:k for k, v in vocab.items()}

        return (vocab, reverse_dict)


    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        stois = []
        for char in text:
            stois.append(stoi[char])
        return stois

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        ito = []
        for i in ids:
            ito.append(itos[i])
        return "".join(ito)
        


            
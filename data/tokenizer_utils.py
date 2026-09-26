from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.  
        result = []
        for number in numbers:
            s = str(number)
            token = []
            pos = 0
            while(pos < len(s)):
                best = None

                for  end in range(pos + 1, len(s) +1):
                    candidate = s[pos:end]

                    if candidate in vocab:
                        best = candidate
                
                token.append(best)
                pos+=len(best)
            
            result.append(token)
        return result
                
            

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy token ization.
        # Use greedy left-to-right longest match.
        pos = 0
        token = []
        while pos < len(text):
            best  = None

            for end in range(pos + 1, len(text) + 1):
                candidate = text[pos:end]

                if candidate in vocab:
                    best = candidate
            
            token.append(best)
            pos += len(best)
        
        return len(token)


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split(" ") 
        fertility_res = self.count_tokens(text, vocab) / len(words)
        return round(fertility_res, 4)

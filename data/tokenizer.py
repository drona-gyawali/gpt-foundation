from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        chars = [*corpus]
        merges = []
        for _ in range(num_merges):


            freq = {}
            for i in range(len(chars) - 1):
                token = (chars[i]), chars[i+1]
                if token not in freq:
                    freq[token] = 1
                else:
                    freq[token] += 1
            
            if not freq:break

            best_pair = min(
                freq,
                key = lambda pair : (-freq[pair], pair)
            )
            merges.append([best_pair[0], best_pair[1]])
            i = 0
            new_token = []
            while ( i < len(chars)):
                if i + 1 < len(chars) and (chars[i], chars[i+1]) == best_pair:
                    new_token.append(best_pair[0] + best_pair[1])
                    i+=2
                else:
                    new_token.append(chars[i])
                    i+=1
                
            
            chars = new_token
        
        return merges


            



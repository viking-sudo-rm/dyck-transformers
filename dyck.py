from random import random, seed


class Dyck2Generator:

    """Follow from https://arxiv.org/pdf/1911.03329.pdf"""

    def __init__(self, p, q, max_depth=100):
        self.p = p
        self.q = q
        self.max_depth = max_depth

    def fill(self, buffer, depth=0):
        """Depth is recursive call depth, not actual depth."""
        if depth > self.max_depth:
            return

        v = random()
        if v <= self.p / 2:
            buffer.append("(")
            self.fill(buffer, depth + 1)
            buffer.append(")")
        elif v <= self.p:
            buffer.append("[")
            self.fill(buffer, depth + 1)
            buffer.append("]")
        elif v < self.p + self.q:
            self.fill(buffer, depth + 1)
            self.fill(buffer, depth + 1)
    
    def generate(self):
        string = []
        self.fill(string)
        return string


if __name__ == "__main__":
    dyck = Dyck2Generator(.5, .25)
    string = []
    dyck.fill(string)
    print("".join(string))
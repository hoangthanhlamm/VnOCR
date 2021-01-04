class Node:
    """class representing nodes in a prefix tree"""

    def __init__(self):
        self.children = {}  # all child elements beginning with current prefix
        self.isWord = False  # does this prefix represent a word

    def __str__(self):
        s = ''
        for k in self.children.keys():
            s += k
        return 'isWord: ' + str(self.isWord) + '; children: ' + s


class PrefixTree:
    """prefix tree"""

    def __init__(self):
        self.root = Node()

    def add_word(self, text):
        """add word to prefix tree"""
        node = self.root
        for i in range(len(text)):
            c = text[i]  # current char
            if c not in node.children:
                node.children[c] = Node()
            node = node.children[c]
            isLast = (i + 1 == len(text))
            if isLast:
                node.is_word = True

    def add_words(self, words):
        for w in words:
            self.add_word(w)

    def get_node(self, text):
        """get node representing given text"""
        node = self.root
        for c in text:
            if c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def is_word(self, text):
        node = self.get_node(text)
        if node:
            return node.isWord
        return False

    def get_next_chars(self, text):
        """get all characters which may directly follow given text"""
        chars = []
        node = self.get_node(text)
        if node:
            for k in node.children.keys():
                chars.append(k)
        return chars

    def get_next_words(self, text):
        """get all words of which given text is a prefix (including the text itself, it is a word)"""
        words = []
        node = self.get_node(text)
        if node:
            nodes = [node]
            prefixes = [text]
            while len(nodes) > 0:
                # put all children into list
                for k, v in nodes[0].children.items():
                    nodes.append(v)
                    prefixes.append(prefixes[0] + k)

                # is current node a word
                if nodes[0].isWord:
                    words.append(prefixes[0])

                # remove current node
                del nodes[0]
                del prefixes[0]

        return words

    def dump(self):
        nodes = [self.root]
        while len(nodes) > 0:
            # put all children into list
            for v in nodes[0].children.values():
                nodes.append(v)

            # dump current node
            print(nodes[0])

            # remove from list
            del nodes[0]

from email.policy import default
import os
from typing import BinaryIO, Iterator, Iterable
from multiprocessing import Process, Queue
import regex as re
from collections import defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# ## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


def split_special_token(
    chunk: str,
    special_token: list[str],
) -> list[str]:
    """
    Split a chunk of text by a special token.

    Example:
    >>> split_special_token("Hello World! <|endoftext|> bye", "<|endoftext|>")
    ["Hello World! ", "<|endoftext|>", " bye"]
    """
    special_token = sorted(special_token, key=lambda x: -len(x))
    if not special_token:
        return [chunk]
    else:
        # 使用 "|" 连接分隔多个特殊 Token，将其作为正则表达式的捕获组
        # 在两边加上括号以在最后输出中保留 Special Token
        # 使用 re.escape 转义 Special Token 中本来就有的 "|"，以免被误认为是正则表达式的“或”操作符
        pattern = "(" + "|".join(re.escape(token) for token in special_token) + ")"
        return re.split(pattern, chunk)


# 为了方便后面的 BPE Encode 复用，选择一种简单的策略，返回 list 而不是 dict{token, count}。
# 后者本可以在 Training 中更加高效（统计 Pairs 频率时乘上系数即可）
def pre_tokenize_chunk(
    chunk: str,
    special_tokens: list[str] = [],
    drop_special_tokens: bool = True,
) -> list[bytes]:
    """
    Pre-tokenize a chunk of text by removing special tokens and counting occurrences.
    Returns a list of bytes tokens.
    """
    # 4.1 split by special tokens
    parts = split_special_token(chunk, special_tokens)

    # 4.2 collect Pre-tokens in non-special-token parts
    PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    token_counts = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_tokens:
                token_bytes = part.encode("utf-8")
                token_counts.append(token_bytes)
        else:
            # Pre-tokenize
            pre_tokens = re.findall(PATTERN, part)
            for token in pre_tokens:
                token_bytes = token.encode("utf-8")
                token_counts.append(token_bytes)
    return token_counts


def worker(
    chunk: str,
    special_tokens: list[str],
    queue: Queue,
) -> None:
    """
    Worker function to pre-tokenize a chunk and put the result in a queue.
    """
    token_counts = pre_tokenize_chunk(chunk, special_tokens)
    queue.put(token_counts)


# From https://github.com/Spectual/stanford-cs336-a1/blob/main/cs336_basics/BPETokenizer.py#L197
def merge(
    counts: dict[tuple[int, int], int],
    counts_idx: dict[tuple[int, int], set[int]],
    pre_tokens: list[list[int]],
    best_pair: tuple[int, int],
    new_index: int,
) -> None:
    """Merge the pairs with highest frequency and update counts, counts_idx"""
    index_set = counts_idx[best_pair]

    for i in index_set:
        pretoken = pre_tokens[i]
        new_pretoken = []

        pos_list = []  # Store positions of best_pair for each new pretoken after merge
        pos = 0
        j = 0

        # Replace best_pair with new_index in each pretoken
        while j < len(pretoken):
            if (j < len(pretoken) - 1) and (
                (pretoken[j], pretoken[j + 1]) == best_pair
            ):
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1
            pos += 1

        # Update counts and counts_idx
        for pos in pos_list:
            counts[best_pair] -= 1

            if pos > 0:
                if new_pretoken[pos - 1] == new_index:
                    counts[(best_pair[1], best_pair[0])] -= 1
                else:
                    counts[(new_pretoken[pos - 1], best_pair[0])] -= 1

                counts[(new_pretoken[pos - 1], new_pretoken[pos])] += 1
                counts_idx[(new_pretoken[pos - 1], new_pretoken[pos])].add(i)

            if pos < len(new_pretoken) - 1:
                if new_pretoken[pos + 1] == new_index:
                    counts[(best_pair[1], best_pair[0])] -= 1
                else:
                    counts[(best_pair[1], new_pretoken[pos + 1])] -= 1

                counts[(new_pretoken[pos], new_pretoken[pos + 1])] += 1
                counts_idx[(new_pretoken[pos], new_pretoken[pos + 1])].add(i)

        pre_tokens[i] = new_pretoken


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input file.
    Returns a vocabulary mapping and a list of merges.

    1. read the file
    2. chunk the file @find_chunk_boundaries
    3. remove special tokens inside the chunks
    4. pre-tokenize the chunks
    5. train BPE on the pre-tokenized chunks
    6. return the vocabulary and merges
    """

    vocab = {}
    merges = []

    # 1. init vocab
    vocab = {x: bytes([x]) for x in range(256)}
    # tips: b"a" 是 bytes 的字面量. 只允许
    #       bytes(x) 是 x 个 0x00 字节组成的 bytes
    #       bytes([x]) 将 ID (0 <= x <= 255) 转换为单字节的 bytes
    # example: >>> bytes([65]) == b"A"
    #          True

    # 2. add special tokens to vocab
    special_tokens = special_tokens or []
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    num_processes = 4
    chunks = []
    # 3. read the file and find chunk boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(
            boundaries[:-1], boundaries[1:]
        ):  # (b[0], b[1]), (b[1], b[2])...
            f.seek(start)
            chunk = f.read(end - start).decode(
                "utf-8", errors="ignore"
            )  # f.read() returns bytes, decode to str
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunks.append(chunk)

    # 4. pre-tokenize the chunks (remove special tokens) parallelized
    processes = []
    pre_tokens = []  # final result of stage 4 (pre-tokenized tokens)
    q = Queue()
    for chunk in chunks:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        processes.append(p)
        p.start()

    # Collect pre-tokenization results from all processes
    pre_token_lists = [q.get() for _ in processes]

    for p in processes:
        p.join()

    pre_tokens = [token for sublist in pre_token_lists for token in sublist]

    # 5. train BPE on the pre-tokenized chunks
    # 5.1 全局统计一次 bytes pair 出现次数
    counts = defaultdict(int)  # 统计不同 bytes pair 出现的次数
    counts_idx = defaultdict(set)  # 统计不同 bytes pair 出现在哪些单词中

    for i, token in enumerate(pre_tokens):
        for token1, token2 in zip(token[:-1], token[1:]):
            counts[(token1, token2)] += 1  # {(int, int): int}
            counts_idx[(token1, token2)].add(
                i
            )  # {(int, int): set(int)} # 记录在哪个单词中，要不要记录在单词中的偏移？

    while len(vocab) < vocab_size:
        if not counts:
            break

        # 5.2 找到出现次数最多的 bytes pair。如果次数相同，选择字典序最大的合并
        best_pair = max(  # 使用最大堆维护最大值？
            counts.items(),
            key=lambda x: (
                x[1],
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore"),
            ),
        )[0]

        # 5.3 将该 bytes pair 合并为一个新的 token，加入 vocab
        new_token = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[len(vocab)] = new_token
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 5.4 更新 pre_tokens 和 counts 频率
        merge(counts, counts_idx, pre_tokens, best_pair, len(vocab) - 1)

    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        self.special_token_bytes = [
            token.encode("utf-8") for token in self.special_tokens
        ]  # Convert special tokens to bytes

        for token in self.special_token_bytes:
            if token not in self.vocab_reverse:
                new_id = len(self.vocab)
                self.vocab[new_id] = token
                self.vocab_reverse[token] = new_id

    # From https://github.com/heng380/cs336-assignment1/blob/main/cs336_basics/Tokenizer/BPE_tokenizer.py#L208
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        import json

        with open(vocab_filepath, "r") as vf:
            vocab_data = json.load(vf)
            vocab = {int(i): bytes(v, "latin1") for v, i in vocab_data.items()}

        merges = []
        with open(merges_filepath, "r") as mf:
            for line in mf:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append(
                            (bytes(parts[0], "latin1"), bytes(parts[1], "latin1"))
                        )

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """

        # 1. Pre-tokenize
        pre_tokens = pre_tokenize_chunk(
            text, self.special_tokens, drop_special_tokens=False
        )  # list[bytes]

        # 1.1 map pre-tokens to IDs
        pre_tokens_ids = []  # list[list[int]]
        for pretoken in pre_tokens:
            token_list = []
            if pretoken in self.special_token_bytes:
                token_list.append(
                    self.vocab_reverse[pretoken]
                )  # Special tokens are already in vocab
            else:
                # Convert pre-token to IDs
                token_list = [self.vocab_reverse[bytes([x])] for x in pretoken]
            pre_tokens_ids.append(token_list)

        # 2. Apply BPE merges

        # 2.1 对每一个 merge，遍历每个 pre-token 内是否可以使用该 merge
        # for pretoken in pre_tokens_ids:
        #     for merge in self.merges:
        #         new_token = []
        #         i = 0

        #         while i < len(pretoken):
        #             if (i < len(pretoken) - 1) and (
        #                 (self.vocab[pretoken[i]], self.vocab[pretoken[i + 1]]) == merge
        #             ):
        #                 new_token.append(self.vocab_reverse[merge[0] + merge[1]])
        #                 i += 2  # Skip the next token since it's merged
        #             else:
        #                 new_token.append(pretoken[i])
        #                 i += 1
        #         pretoken[:] = new_token  # Update the pretoken in place

        # 2.2 只遍历存在的 merges
        # 2.2.1 定义一个函数来获取当前 pre-token 中的相邻 token 对
        pairs = lambda w: set(
            (self.vocab[w[i]], self.vocab[w[i + 1]]) for i in range(len(w) - 1)
        )
        for pretoken in pre_tokens_ids:
            flag = True
            i = 0
            while flag and i < len(self.merges):
                # 2.2.2 先行探索一下是否有可以 merge 的对
                candidate_pairs = pairs(pretoken)
                new_token = []

                flag = False
                while i < len(self.merges):
                    if self.merges[i] in candidate_pairs:
                        flag = True
                        merge = self.merges[i]

                        # 2.2.3 找到了，合并
                        j = 0
                        while j < len(pretoken):
                            if (
                                (j < len(pretoken) - 1)
                                and self.vocab[pretoken[j]] == merge[0]
                                and self.vocab[pretoken[j + 1]] == merge[1]
                            ):
                                new_token.append(
                                    self.vocab_reverse[merge[0] + merge[1]]
                                )
                                j += 2
                            else:
                                new_token.append(pretoken[j])
                                j += 1
                        pretoken[:] = new_token  # Update the pretoken in place
                        break
                    i += 1

        # 2.2 Flatten the list of pre-tokens into a single list of IDs
        token_ids = [x for sublist in pre_tokens_ids for x in sublist]
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        replace_char = "\ufffd"
        tokens = bytes()

        for token in ids:
            if token in self.vocab:
                tokens += self.vocab[token]
            else:
                # If the token is not in the vocab, replace it with a placeholder
                tokens += replace_char.encode("utf-8")  # Use U+FFFD as a placeholder

        return tokens.decode(
            "utf-8", errors="replace"
        )  # Use 'replace' to handle decoding errors

import os
import json
import zstandard as zstd
import io
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


"""

Ugly/inefficient dataloader. Would not recommend looking too hard at this.

"""


class LiveDataLoader:
    def __init__(self, batch_size, mistral_models_path):
        self.batch_size = batch_size

        folder_name = ""  # folder with jsonl.zst files
        filenames = os.listdir(folder_name)
        filenames = sorted(filenames)
        filenames = [os.path.join(folder_name, f) for f in filenames]
        self.filenames = filenames

        self.tokenizer = MistralTokenizer.from_file(
            f"{mistral_models_path}/tokenizer.model.v3"
        )

        self.curr_file = 9
        self.file_path = self.filenames[self.curr_file]
        self.batch_size = batch_size
        self.file = None
        self.dctx = None
        self.stream_reader = None
        self.text_stream = None
        self._open_file()

    def _open_file(self):
        self.file = open(self.file_path, "rb")
        self.dctx = zstd.ZstdDecompressor()
        self.stream_reader = self.dctx.stream_reader(self.file)
        self.text_stream = io.TextIOWrapper(self.stream_reader, encoding="utf-8")

    def next_batch(self):
        batch = []
        for _ in range(self.batch_size * 5):
            try:
                line = next(self.text_stream)
                obj = json.loads(line.strip())
                line_text = obj["text"]
                line_text = line_text[: len(line_text) // 5]
                completion_request_u = ChatCompletionRequest(
                    messages=[UserMessage(content=line_text)]
                )
                tokens = self.tokenizer.encode_chat_completion(
                    completion_request_u
                ).tokens[:-1]
                batch.append(tokens)
            except StopIteration:
                break

        if len(batch) < self.batch_size:
            # If at end of json file, go to new one
            self.curr_file = (self.curr_file + 1) % len(self.filenames)
            self.file_path = self.filenames[self.curr_file]
            self._open_file()
            return self.next_batch()

        return batch

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
